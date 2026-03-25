#!/usr/bin/env python3
"""Joint-space PD torque control for UR5e with figure-8 EE tracking & NMPC orientation.

Control law:
    tau = Kp * (q_des - q) + Kd * (dq_des - dq)  (For joints 0, 1, 2)
    tau = NMPC_opt                               (For joints 3, 4, 5)

First 3 joints are locked to initial positions. Last 3 track orientation.
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import mujoco
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from Controller import PDJointController, NMPCController
from visualization import VisualizationWorker, UDPLogger


VIS_FREQ = 50.0


def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    return x_des_pos_init + np.array([
        2.0 * circle_radius * np.sin(circle_omega * t),
        circle_radius * np.sin(2.0 * circle_omega * t),
        0.5 * circle_radius * np.sin(circle_omega * t),
    ])


def get_target_ori(
    t: float,
    x_des_pos_init: np.ndarray,
    circle_radius: float,
    circle_omega: float,
    dt: float = 1e-3,
) -> np.ndarray:
    """Build orientation aligned with trajectory tangent."""
    p0 = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    p1 = get_traj_pos(t + dt, x_des_pos_init, circle_radius, circle_omega)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return np.eye(3)
    tangent /= norm
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, tangent)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm if y_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    z_axis = np.cross(tangent, y_axis)
    return np.column_stack([tangent, y_axis, z_axis])


def get_traj(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> tuple[np.ndarray, np.ndarray]:
    x_des_pos = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_mat = get_target_ori(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_quat = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
    if x_des_quat[0] < 0:
        x_des_quat *= -1.0
    return x_des_pos, x_des_quat


def build_reference_batch(
    t_traj: float,
    x_des_pos_init: np.ndarray,
    circle_radius: float,
    circle_omega: float,
    horizon_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build N+1 task references in MuJoCo frame for NMPC."""
    ref_pos_batch = np.zeros((3, horizon_steps + 1))
    ref_rot_batch = np.zeros((9, horizon_steps + 1))
    for k in range(horizon_steps + 1):
        tk = t_traj + k * dt
        pk = get_traj_pos(tk, x_des_pos_init, circle_radius, circle_omega)
        rk = get_target_ori(tk, x_des_pos_init, circle_radius, circle_omega)
        ref_pos_batch[:, k] = pk
        ref_rot_batch[:, k] = rk.flatten(order="F")
    return ref_pos_batch, ref_rot_batch


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Hybrid Control (PD + NMPC)")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="Initial joint positions file")
    parser.add_argument("--no-vis", action="store_true", help="Disable MuJoCo visualization thread")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="UDP destination IP for PlotJuggler")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP destination port for PlotJuggler")
    parser.add_argument("--udp-div", type=int, default=2, help="Send one packet every N control loops")
    return parser.parse_args()


def load_init_q(init_pos_path: Path):
    if not init_pos_path.exists():
        return None
    try:
        with open(init_pos_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if not line:
            return None
        q_init = np.array([float(x) for x in line.split()], dtype=float)
        if q_init.shape[0] != 6:
            return None
        return q_init
    except Exception as e:
        print(f"[WARN] Failed to read init_pos file: {e}")
        return None


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dt = float(cfg.get("trajectory", {}).get("control_dt", 0.01))
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    safety_cfg = cfg.get("safety", {})
    torque_limits = np.array(safety_cfg.get("torque_limits", [20, 20, 20, 10, 10, 10]), dtype=float)

    traj_cfg = cfg.get("trajectory", {})
    circle_radius = float(traj_cfg.get("circle_radius", 0.08))
    circle_omega = float(traj_cfg.get("circle_omega", 0.8))

    nmpc_cfg = cfg.get("nmpc_controller")
    if nmpc_cfg is None:
        raise ValueError("Missing 'nmpc_controller' section in ctrl_config.yaml")
    horizon_steps = int(nmpc_cfg["horizon_steps"])
    rebuild_solver = bool(nmpc_cfg.get("rebuild", False))

    pdjoint_cfg = cfg.get("pdjoint_controller")
    if pdjoint_cfg is None:
        raise ValueError("Missing 'joint_pd' section in ctrl_config.yaml")

    kp = np.array(pdjoint_cfg["kp"], dtype=float)
    kd = np.array(pdjoint_cfg["kd"], dtype=float)

    xml_path = Path("script/ur5e_gripper/scene.xml")
    if not xml_path.exists():
        xml_path = Path("script/universal_robots_ur5e/scene.xml")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)
    
    joint_controller = PDJointController(model=model, kp=kp, kd=kd, torque_limits=torque_limits)
    nmpc_controller = NMPCController(model=model, config_path=args.config, rebuild=rebuild_solver)
    
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    visualizer = None
    if not args.no_vis:
        visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)

    logger = UDPLogger(args.udp_ip, args.udp_port, send_every_n=args.udp_div)

    init_pos_path = Path("ur_client_library/init_pos.txt")
    if not init_pos_path.exists():
        init_pos_path = Path(args.init_pos)

    q_init = load_init_q(init_pos_path)
    q_actual0 = np.array(rtde_r.getActualQ(), dtype=float)

    if q_init is not None:
        q_target = q_init.copy()
        print(f"[INFO] Loaded init joint position from {init_pos_path}")
    else:
        q_target = q_actual0.copy()
        print("[WARN] init_pos not found/invalid, holding current joints")

    data.qpos[:6] = q_target
    mujoco.mj_forward(model, data)
    x_des_pos_init = data.site_xpos[ee_site_id].copy()
    x_des_quat_init = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat_init, data.site_xmat[ee_site_id].flatten())

    # =========================================================================
    # 增加 NMPC 暖启动 (Warm Start)
    # =========================================================================
    print("[INFO] Warm-starting NMPC solver...")
    x0_warm = np.concatenate([q_target, np.zeros(6)])
    for k in range(horizon_steps + 1):
        nmpc_controller.solver.set(k, "x", x0_warm)
    for k in range(horizon_steps):
        # 使用真实的重力补偿项作为控制猜测，避免解退化
        nmpc_controller.solver.set(k, "u", data.qfrc_bias[:6])
    print("[INFO] Warm-start completed.")
    # =========================================================================

    STABILIZE_END = 1.0
    RESET_END = 5.0
    RESET_BUFFER = 1.0
    q_start = None

    traj_started = False
    trajectory_points = []
    target_trajectory = []
    max_points = 200
    traj_sample_period = 0.05
    next_traj_sample = 0.0

    start_time = time.perf_counter()
    next_tick = time.perf_counter()
    total_loop_count = 0
    freq_start_time = time.perf_counter()
    freq_loop_count = 0

    try:
        while True:
            if visualizer is not None and not visualizer.is_running():
                print("[INFO] Visualization closed, stopping controller loop.")
                break

            t_now = time.perf_counter() - start_time

            q = np.array(rtde_r.getActualQ(), dtype=float)
            dq = np.array(rtde_r.getActualQd(), dtype=float)

            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            x_curr_pos = data.site_xpos[ee_site_id].copy()
            x_curr_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()

            tau_nmpc = np.zeros(6)

            if t_now <= STABILIZE_END:
                x_des_pos = x_curr_pos.copy()
                x_des_mat = x_curr_mat.copy()
                q_des = q.copy()
                dq_des = np.zeros(6)
                q_start = q.copy()
            elif t_now <= RESET_END:
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)

                # Reset phase: track q_target in joint space via PD.
                q_des = (1.0 - alpha) * q_start + alpha * q_target
                dq_des = np.zeros(6)

                data_plan.qpos[:6] = q_des
                mujoco.mj_forward(model, data_plan)
                x_des_pos = data_plan.site_xpos[ee_site_id].copy()
                x_des_mat = data_plan.site_xmat[ee_site_id].reshape(3, 3).copy()
            else:
                if not traj_started:
                    traj_started = True

                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, circle_radius, circle_omega)
                x_des_mat = np.zeros(9)
                mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
                x_des_mat = x_des_mat.reshape(3, 3)

                ref_pos_batch, ref_rot_batch = build_reference_batch(
                    t_traj,
                    x_des_pos_init,
                    circle_radius,
                    circle_omega,
                    horizon_steps,
                    dt,
                )
                
                tau_nmpc = nmpc_controller.compute_torque(data, q, dq, ref_pos_batch, ref_rot_batch)
                
                # =============================================================
                # 锁定前三轴，使用初始位置目标
                # =============================================================
                q_des = q_target.copy()
                dq_des = np.zeros(6)

            # 计算 PD 控制力矩 (不含重力)
            tau_pd = joint_controller.compute_torque(q_des, q, dq_des, dq)

            # =================================================================
            # 拼接力矩: 前三轴 PD, 后三轴 NMPC
            # =================================================================
            if traj_started:
                tau = np.concatenate([tau_pd[:3], tau_nmpc[3:]])
            else:
                tau = tau_pd

            if t_now > STABILIZE_END:
                # 下发给机器人的 directTorque 默认已经包含机器人自带的重力补偿。
                ok = rtde_c.directTorque(tau.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

            total_loop_count += 1
            logger.update(
                total_loop_count,
                q_des,
                q,
                x_des_pos,
                x_curr_pos,
                tau,
                extra={
                    "tau_cmd": tau,
                    "tau_pd": tau_pd,
                    "tau_nmpc": tau_nmpc,
                    "traj_started": float(traj_started),
                },
            )

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)

            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                print(f"[INFO] Control Frequency: {freq_loop_count / elapsed:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        if visualizer is not None:
            visualizer.stop()
        logger.stop()
        rtde_c.stopScript()
        print("[INFO] Script stopped")


if __name__ == "__main__":
    main()