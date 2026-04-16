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
KP = np.array([1500.0, 1500.0, 1500.0, 150.0, 150.0, 10.0], dtype=float)
KD = np.array([40.0, 40.0, 40.0, 5.0, 5.0, 1.0], dtype=float)
TORQUE_LIMITS = np.array([50.0, 50.0, 50.0, 15.0, 15.0, 15.0], dtype=float)

# --- 任务空间解耦 PD 控制器参数 ---
# tau = J^T * [Kp_task * e6 - Kd_task * (J * dq)]
KP_TASK = np.array([5000.0, 5000.0, 5000.0, 30.0, 30.0, 30.0], dtype=float)
KD_TASK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

# --- ISM 参数 ---
# sigma = S1 * (q - q_t0 - ∫dq) + S2 * (dq - dq_t0 - ∫u)
S1_ISM = 1.0 * np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
UMAX_ISM = 0.0 * np.array([30.0, 15.0, 20.0, 50.0, 50.0, 5000.0], dtype=float)

K_TANH = 10.0 

# --- 全局停止事件 ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    stop_event.set()

# --- 轨迹参数 ---
CIRCLE_RADIUS = 0.06
CIRCLE_OMEGA = 1.2
CONTROL_DT = 0.002  # 500Hz
VIS_FREQ = 50.0

def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    return x_des_pos_init + np.array([
        - 2.0 * circle_radius * (np.cos(circle_omega * t) - 1.0),
        1.0 * circle_radius * np.sin(2.0 * circle_omega * t),
        - 0.5 * circle_radius * (np.cos(circle_omega * t) - 1.0),
    ])

def get_traj_vel(t: float, circle_radius: float, circle_omega: float) -> np.ndarray:
    """计算 figure-8 轨迹的期望速度 (dx_des_pos)"""
    return np.array([
        2.0 * circle_radius * circle_omega * np.sin(circle_omega * t),
        2.0 * circle_radius * circle_omega * np.cos(2.0 * circle_omega * t),
        0.5 * circle_radius * circle_omega * np.sin(circle_omega * t),
    ])

def get_traj_acc(t: float, circle_radius: float, circle_omega: float) -> np.ndarray:
    """计算 figure-8 轨迹的期望加速度 (ddx_des_pos)"""
    return np.array([
        2.0 * circle_radius * (circle_omega**2) * np.cos(circle_omega * t),
        -4.0 * circle_radius * (circle_omega**2) * np.sin(2.0 * circle_omega * t),
        0.5 * circle_radius * (circle_omega**2) * np.cos(circle_omega * t),
    ])

def get_target_ori(
    t: float,
    x_des_pos_init: np.ndarray,
    circle_radius: float,
    circle_omega: float,
    dt: float = 1e-3,
) -> np.ndarray:
    """构建与轨迹切线对齐的姿态"""
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

def compute_task_errors(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    current_pos: np.ndarray,
    current_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1. 计算平移位置误差
    pos_err = target_pos - current_pos
    
    # 2. 将当前旋转矩阵转为四元数
    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, current_mat.flatten())
    # 确保两个四元数在同一个半球内（若 qd·q < 0，则翻转当前四元数）
    if np.dot(target_quat, curr_quat) < 0:
        curr_quat = -curr_quat

    # 3. 计算姿态误差：Δq = qd ⊗ q^{-1}，并取轴角 e_rot = u * theta
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, curr_quat)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, target_quat, q_inv)
    rot_err = np.zeros(3)
    mujoco.mju_quat2Vel(rot_err, dq, 1.0)
    
    return pos_err, rot_err, curr_quat

def map_task_target_to_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    x_des_pos: np.ndarray,
    x_des_quat: np.ndarray,
    q_curr: np.ndarray,
) -> np.ndarray:
    """使用伪逆将期望的任务空间位姿误差投影到关节空间目标"""
    current_pos = data.site_xpos[ee_site_id].copy()
    current_mat = data.site_xmat[ee_site_id].reshape(3, 3)
    pos_err, rot_err, _ = compute_task_errors(x_des_pos, x_des_quat, current_pos, current_mat)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    J6 = np.vstack([jacp[:, :6], jacr[:, :6]])

    e6 = np.concatenate([pos_err, rot_err])
    dq_cmd = np.linalg.pinv(J6) @ e6
    return q_curr + dq_cmd, rot_err

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

            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            x_curr_pos = data.site_xpos[ee_site_id].copy()
            x_curr_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()
            
            # 默认目标速度为 0
            dx_des = np.zeros(6)

            if t_now <= STABILIZE_END:
                x_des_pos = x_curr_pos.copy()
                x_des_mat = x_curr_mat.copy()
                x_des_quat = np.zeros(4)
                mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
                q_des = q.copy()
                dq_des = np.zeros(6)
                q_start = q.copy()
                rot_err = np.zeros(3)
            elif t_now <= RESET_END:
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)

                # 重置阶段：保持在轨迹起点确定的 q 目标 (在此脚本中设为稳定后的初始位置)
                q_des = (1.0 - alpha) * q_start + alpha * q_target
                dq_des = np.zeros(6)
                rot_err = np.zeros(3)

                data_plan.qpos[:6] = q_des
                mujoco.mj_forward(model, data_plan)
                x_des_pos = data_plan.site_xpos[ee_site_id].copy()
                x_des_mat = data_plan.site_xmat[ee_site_id].reshape(3, 3).copy()
                x_des_quat = np.zeros(4)
                mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
            else:
                if not traj_started:
                    traj_started = True

                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, CIRCLE_RADIUS, CIRCLE_OMEGA)
                dx_des_pos = get_traj_vel(t_traj, CIRCLE_RADIUS, CIRCLE_OMEGA)
                x_des_mat = np.zeros(9)
                mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
                x_des_mat = x_des_mat.reshape(3, 3)
                
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            M6 = M[:6, :6]
            M_inv = np.linalg.inv(M6 + 1e-6 * np.eye(6))

            pos_err = np.zeros(3)
            rot_err = np.zeros(3)
            if traj_started:
                # 获取前馈速度和前馈加速度
                dx_des_pos = get_traj_vel(t_traj, CIRCLE_RADIUS, CIRCLE_OMEGA)
                ddx_des_pos = get_traj_acc(t_traj, CIRCLE_RADIUS, CIRCLE_OMEGA)
                
                # 为了简单，姿态先设为保持目标不变（期望角速度和角加速度为0）
                dx_des = np.concatenate([dx_des_pos, np.zeros(3)])
                ddx_des = np.concatenate([ddx_des_pos, np.zeros(3)])

                # --- 1. 计算误差与雅可比 ---
                pos_err, rot_err, _ = compute_task_errors(x_des_pos, x_des_quat, x_curr_pos, x_curr_mat)
                
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
                J6 = np.vstack([jacp[:, :6], jacr[:, :6]])
                
                # --- 2. 任务空间阻抗控制律 (加入前馈) ---
                e6_pos = np.concatenate([pos_err, rot_err])
                
                # 关键修复1：速度误差应该是 期望速度 - 实际速度
                v6_actual = J6 @ dq
                e6_vel = dx_des - v6_actual 
                
                # 设定任务空间的虚拟质量 
                M_TASK = 1.0
                
                # 关键修复2：完整的阻抗力矩 = 前馈力 + 刚度恢复力 + 阻尼力
                w_task = M_TASK * ddx_des + KP_TASK * e6_pos + KD_TASK * e6_vel
                
                u_pos = J6.T @ np.concatenate([w_task[:3], np.zeros(3)])
                u_ori = J6.T @ np.concatenate([np.zeros(3), w_task[3:]])
                tau_PD = u_pos + u_ori

            else:
                # --- 初始化阶段：关节 PD ---
                tau_PD = KP * (q_des - q) + KD * (dq_des - dq)
                u_pos = tau_PD.copy()
                u_ori = np.zeros(6)

            # --- 4. 积分滑模面 ISM 计算 ---
            u = M_inv @ tau_PD
            
            sigma = np.zeros(6)
            u_ISM = np.zeros(6)
            
            if traj_started:
                if not ism_init:
                    dq_prev = dq.copy()
                    ism_init = True
                # 1. 实际速度的变化量 (Actual velocity increment)
                delta_dq_actual = dq - dq_prev
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
                "u_pos": u_pos,
                "u_ori": u_ori,
                "pos_err": pos_err,
                "rot_err": rot_err,
                "sigma": sigma,
                "u": u,
                "u_ISM": u_ISM,
            }
            if t_now >= RESET_END:
               logger.update(total_loop_count, q_des, q, x_des_pos, x_curr_pos, tau, extra=extra_data)

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)

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
