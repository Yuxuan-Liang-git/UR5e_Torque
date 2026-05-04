#!/usr/bin/env python3
"""Task-space PD torque control for UR5e with figure-8 EE tracking.
(Converted from torque_ISM.py to JointSpace form)

Control law:
    tau = J^T * (Kp * e - Kd * (J * dq))

Joint references are generated from end-effector figure-8 trajectory using
Jacobian-based resolved-rate mapping, then tracked by joint-space PD.
"""

import argparse
import time
import signal
import threading
from pathlib import Path

import numpy as np
import mujoco
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from visualization import VisualizationWorker, UDPLogger

# --- 硬编码控制器参数 ---
# KP = np.array([1500.0, 1500.0, 1500.0, 150.0, 150.0, 10.0], dtype=float)
# KD = np.array([40.0, 40.0, 40.0, 5.0, 5.0, 1.0], dtype=float)

KP = np.array([1500.0, 1500.0, 1500.0, 150.0, 150.0, 20.0], dtype=float)
KD = np.array([40.0, 40.0, 40.0, 5.0, 5.0, 1.0], dtype=float)
TORQUE_LIMITS = np.array([50.0, 50.0, 50.0, 15.0, 15.0, 15.0], dtype=float)

# --- 关节测试参数 ---
JOINT_TEST = {1}  # 测试的关节索引集合
SINE_AMP = 0.15
SINE_FREQ = 0.4

# --- ISM 参数 ---
# sigma = S1 * (q - q_t0 - ∫dq) + S2 * (dq - dq_t0 - ∫u)
S1_ISM = 1.0 * np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
# UMAX_ISM = 1.0 * np.array([30.0, 15.0, 20.0, 50.0, 200.0, 20000.0], dtype=float)
UMAX_ISM = 1.0 * np.array([0.0, 0.0, 20.0, 150.0, 200.0, 5000.0], dtype=float)


K_TANH = 20.0 

# --- 全局停止事件 ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    stop_event.set()

# --- 轨迹参数 ---
CIRCLE_RADIUS = 0.06
CIRCLE_OMEGA = 1.2
CONTROL_DT = 0.002 
VIS_FREQ = 50.0

def get_joint_traj(t: float, q_init: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成关节空间正弦轨迹"""
    q_des = q_init.copy()
    dq_des = np.zeros(6)
    ddq_des = np.zeros(6)
    
    for j in JOINT_TEST:
        w = 2.0 * np.pi * SINE_FREQ
        q_des[j] = q_init[j] + SINE_AMP * np.sin(w * t)
        dq_des[j] = SINE_AMP * w * np.cos(w * t)
        ddq_des[j] = -SINE_AMP * (w**2) * np.sin(w * t)
        
    return q_des, dq_des, ddq_des

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Joint-Space PD Torque Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--no-vis", action="store_true", help="Disable MuJoCo visualization thread")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="UDP destination IP for PlotJuggler")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP destination port for PlotJuggler")
    parser.add_argument("--udp-div", type=int, default=2, help="Send one packet every N control loops")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- MuJoCo 环境加载 ---
    xml_path = Path("ur5e_gripper/scene.xml").absolute()
    if not xml_path.exists():
        xml_path = Path("universal_robots_ur5e/scene.xml").absolute()
    if not xml_path.exists():
        xml_path = Path("scene.xml").absolute()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    visualizer = None
    if not args.no_vis:
        visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)

    logger = UDPLogger(args.udp_ip, args.udp_port, send_every_n=args.udp_div)

    # 获取初始位姿
    q_actual0 = np.array(rtde_r.getActualQ(), dtype=float)
    q_target = np.array([1.5326, -1.30735, -2.06397, -1.36756, 1.59805, -1.55037], dtype=float)

    data.qpos[:6] = q_target
    mujoco.mj_forward(model, data)
    x_des_pos_init = data.site_xpos[ee_site_id].copy()

    STABILIZE_END = 1.0
    RESET_END = 5.0
    RESET_BUFFER = 1.0
    traj_started = False
    ism_init = False
    ism_q_t0 = np.zeros(6)
    ism_dq_t0 = np.zeros(6)
    ism_int_dq = np.zeros(6)
    ism_int_u = np.zeros(6)
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
        while not stop_event.is_set():
            if visualizer is not None and not visualizer.is_running():
                print("[INFO] Visualization closed, stopping controller loop.")
                break

            t_now = time.perf_counter() - start_time

            q = np.array(rtde_r.getActualQ(), dtype=float)
            dq = np.array(rtde_r.getActualQd(), dtype=float)

            # [Fx, Fy, Fz, Tx, Ty, Tz]
            tcp_force = np.array(rtde_r.getActualTCPForce(), dtype=float)

            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            x_curr_pos = data.site_xpos[ee_site_id].copy()
            x_curr_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()
            
            # 默认目标速度为 0
            dx_des = np.zeros(6)

            if t_now <= STABILIZE_END:
                q_des = q.copy()
                dq_des = np.zeros(6)
                ddq_des = np.zeros(6)
                q_start = q.copy()
            elif t_now <= RESET_END:
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)
                # 平滑插值
                # 使用五次多项式或者简单的速度计算
                v_max = 1.0 # rad/s max arbitrary, we just linearly interpolate
                q_des = (1.0 - alpha) * q_start + alpha * q_target
                dq_des = np.zeros(6)
                ddq_des = np.zeros(6)
            else:
                if not traj_started:
                    traj_started = True
                    q_start_traj = q.copy()

                t_traj = t_now - RESET_END
                q_des, dq_des, ddq_des = get_joint_traj(t_traj, q_start_traj)

            # 获取动力学项
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            M6 = M[:6, :6]

            # 计算前馈力矩
            # 由于重力已经在 UR 内部补偿，我们需要补偿科氏力和惯性力
            tau_bias = data.qfrc_bias[:6] 
            tau_grav = data.qfrc_gravcomp[:6]
            # --- 1. 计算关节PD ---
            tau_fb = KP * (q_des - q) + KD * (dq_des - dq)
            
            # --- 2. 计算前馈补偿 ---
            tau_ff = M6 @ ddq_des + (tau_bias - tau_grav)

            tau_PD = tau_fb + tau_ff

            M_inv = np.linalg.inv(M6 + 1e-6 * np.eye(6))

            # u 是名义加速度： u = M^{-1} (tau_nom - C - G)
            # 因为我们的 tau_nom = tau_fb + M * ddq_des + C，经过代入得：
            u = ddq_des + M_inv @ tau_fb
            
            sigma = np.zeros(6)
            u_ISM = np.zeros(6)
            
            if traj_started:
                if not ism_init:
                    dq_prev = dq.copy()
                    ism_init = True
                # 1. 实际速度的变化量 (Actual velocity increment)
                delta_dq_actual = dq - dq_prev
                # 更新前一个速度值
                dq_prev = dq.copy()
                # 2. 名义加速度带来的期望速度变化量 (Nominal velocity increment)
                delta_dq_nom = u * CONTROL_DT
                # 3. 增量式更新滑模面
                sigma += (delta_dq_actual - delta_dq_nom)
                # sigma = (delta_dq_actual - delta_dq_nom)

                u_ISM = -UMAX_ISM * np.tanh(K_TANH * sigma)

                # u_ISM = -UMAX_ISM * np.sign(sigma)

                tau = tau_PD + M6 @ u_ISM
            else:
                tau = tau_PD

            tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

            if t_now > STABILIZE_END:
                ok = rtde_c.directTorque(tau.tolist(), False)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

            total_loop_count += 1
            error_q = q_des - q
            extra_data = {
                "dq": dq,
                "dq_des": dq_des,
                "error_q": error_q,
                "tau": tau,
                "tau_PD": tau_PD,
                "tau_ff": tau_ff if 'tau_ff' in locals() else np.zeros(6),
                "sigma": sigma,
                "u": u,
                "u_ISM": u_ISM,
                "tcp_force": tcp_force,
            }
            if t_now >= RESET_END:
               logger.update(total_loop_count, q_des, q, np.zeros(3), x_curr_pos, tau, extra=extra_data)

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(np.zeros(3))
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                # 依然传递 x_curr_pos 和零目标，但不依赖任务空间解耦了
                visualizer.update(q, trajectory_points, target_trajectory, np.zeros(3), np.eye(3))

            # 频率打印
            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                print(f"[INFO] Control Frequency: {freq_loop_count / elapsed:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            next_tick += CONTROL_DT
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Stopped by user")
    finally:
        print("[INFO] Cleaning up...")
        if visualizer is not None:
            visualizer.stop()
        logger.stop()
        try:
            rtde_c.stopScript()
            rtde_c.disconnect()
        except:
            pass
        print("[INFO] Script stopped")

if __name__ == "__main__":
    main()
