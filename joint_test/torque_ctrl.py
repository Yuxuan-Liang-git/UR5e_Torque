#!/usr/bin/env python3
"""UR5e Joint 6 Identification Trajectory (Paper-based).
Stages:
0: Initialization & Stabilization
1: Friction Pre-identification (Region A-E CV Cycles)
2: Dynamic Parameter Identification (5th-Order Fourier Series)
"""

import argparse
import time
import sys
import numpy as np
import yaml
from pathlib import Path
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import threading
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Global visualization frequency (Hz)
VIS_FREQ = 50.0

class VisualizationWorker:
    def __init__(self, xml_path, render_hz=VIS_FREQ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.latest_q = None
        self.running = True
        self.render_period = 1.0 / max(1.0, float(render_hz))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25
                while self.running and viewer.is_running():
                    with self.lock:
                        q = self.latest_q.copy() if self.latest_q is not None else None
                    if q is not None:
                        with viewer.lock():
                            self.data.qpos[:6] = q
                            mujoco.mj_forward(self.model, self.data)
                            viewer.sync()
                    time.sleep(self.render_period)
        except Exception as e:
            print(f"[WARN] Visualization failed: {e}")
        finally:
            self.running = False

    def update(self, q):
        with self.lock:
            self.latest_q = np.array(q, copy=True)

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Joint6 Identification Traj")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="/home/amdt/ur_force_ws/my_ws/config/ctrl_config.yaml", help="Control config file")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        with open(args.config, 'r') as f:
            ctrl_cfg = yaml.safe_load(f)
        torque_limits = np.array(ctrl_cfg['safety']['torque_limits'])
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # --- MuJoCo 可视化路径修正 ---
    # 优先寻找包含完整依赖的 ur5e_gripper/scene.xml
    xml_path = Path("ur5e_gripper/scene.xml").absolute()
    if not xml_path.exists():
        xml_path = Path("universal_robots_ur5e/scene.xml").absolute()
    
    if not xml_path.exists():
        print(f"[ERROR] Could not find MuJoCo XML at {xml_path}")
        # 尝试一些常见的默认位置
        xml_path = Path("scene.xml").absolute()

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    q_init = np.array(rtde_r.getActualQ())
    
    # --- 1. 摩擦力预辨识参数 (Stage 1) ---
    # V_SCAN_STEPS = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0] # 扫描速度档位

    V_SCAN_STEPS = [0.5, 1.0, 1.5, 2.0] # 扫描速度档位
    SCAN_RANGE = 1.2 # rad, 扫描运动摆幅
    MAX_ACC = 0.2    # 规划加速度
    
    # --- 2. 惯量激励参数 (Stage 2: Fourier) ---
    T_FOURIER = 25.0
    F_BASE = 0.04
    W_BASE = 2 * np.pi * F_BASE
    # 5 阶傅里叶系数
    A_FOUR = np.array([0.20, 0.15, 0.10, 0.05, 0.02])
    B_FOUR = np.array([0.18, 0.12, 0.08, 0.04, 0.01])
    # 积分常数偏移，确保从 q=0 开始不突跳
    Q_FIX = 0.0
    for l in range(1, 6):
        Q_FIX += B_FOUR[l-1] / (l * W_BASE)

    # PID 增益 (1-5 轴锁死，6 轴激励)
    kp = np.array([1500.0, 1500.0, 1500.0, 300.0, 100.0, 30.0])
    kd = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 5.0])
    ki_j6 = 20.0

    # 载入已有模型参数 (可选前馈)
    param_file = Path("Data/identified_params.yaml")
    if param_file.exists():
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        strib = params.get("Stribeck", {})
        J_ID = params.get("J", 0.0)
        B_ID = strib.get("B", 0.0)
        FC_ID = strib.get("Fc", 0.0)
        FS_ID = strib.get("Fs", 0.0)
        VS_ID = strib.get("vs", 0.05)
        BIAS_ID = params.get("bias", 0.0)
    else:
        J_ID, B_ID, FC_ID, FS_ID, VS_ID, BIAS_ID = 0.0, 0.0, 0.0, 0.0, 0.05, 0.0

    # Live Plot Setup
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    line_pos_des, = axs[0].plot([], [], 'r--', label='Des Pos')
    line_pos_act, = axs[0].plot([], [], 'b-', label='Act Pos')
    line_vel_des, = axs[1].plot([], [], 'r--', label='Des Vel')
    line_vel_act, = axs[1].plot([], [], 'b-', label='Act Vel')
    line_acc_des, = axs[2].plot([], [], 'r--', label='Des Acc')
    line_acc_act, = axs[2].plot([], [], 'b-', label='Act Acc (Diff)')
    line_torque,  = axs[3].plot([], [], 'k-', label='Torque')
    
    for ax in axs: ax.grid(True); ax.legend(loc='upper right')
    axs[0].set_ylabel('Pos (rad)'); axs[1].set_ylabel('Vel (rad/s)')
    axs[2].set_ylabel('Acc (rad/s^2)'); axs[3].set_ylabel('Torque (Nm)')

    data_log = []
    start_time = time.perf_counter()
    visualizer = VisualizationWorker(xml_path)
    
    dt = 0.002
    plot_dt = 0.1
    next_tick = time.perf_counter()
    next_plot = time.perf_counter()
    
    # 阶段时间控制点
    T_INIT_MOVE = 5.0  # 移动到辨识起点的时间
    T_STABILIZE = T_INIT_MOVE + 2.0 # 稳定时间
    
    q_error_int_j6 = 0.0
    prev_dq = 0.0
    
    # 状态机
    v_idx = 0
    stage1_finished = False
    fourier_start_time = 0.0
    t_v_start = T_STABILIZE

    print("[INFO] Trajectory control started. Keeping Joints 1-5 locked.")
    try:
        while visualizer.is_running():
            t_curr = time.perf_counter() - start_time
            q = np.array(rtde_r.getActualQ())
            dq = np.array(rtde_r.getActualQd())
            
            acc_act = (dq[5] - prev_dq) / dt
            prev_dq = dq[5]
            
            q_ref, dq_ref, ddq_ref = 0.0, 0.0, 0.0
            stage = 0

            # --- 激励轨迹逻辑 ---
            if t_curr < T_INIT_MOVE:
                # 初始移动到辨识起点 (-SCAN_RANGE)
                stage = 0
                alpha = t_curr / T_INIT_MOVE
                target_start = -SCAN_RANGE
                q_ref = (1 - alpha) * q_init[5] + alpha * target_start
                dq_ref = (target_start - q_init[5]) / T_INIT_MOVE
                ddq_ref = 0.0
            elif t_curr < T_STABILIZE:
                # 在辨识起点稳定
                stage = 0
                q_ref = -SCAN_RANGE
                dq_ref = 0.0
                ddq_ref = 0.0
                t_v_start = t_curr
            elif v_idx < len(V_SCAN_STEPS):
                # --- Stage 1: 摩擦力预辨识循环 (CV Scan) ---
                stage = 1
                vk = V_SCAN_STEPS[v_idx]
                dt_v = t_curr - t_v_start
                
                # 设计阶梯扫描: 0 -> A -> -A -> 0
                # t1(acc), t2(cv), t3(dec), t4(acc_rev), t5(cv_rev), t6(dec_rev)
                ta = vk / MAX_ACC
                tc = (2 * SCAN_RANGE) / vk
                dur = 2 * (2 * ta + tc)
                
                t_in = dt_v % dur
                if t_in < ta: # 加速
                    dq_ref = (vk / ta) * t_in
                    q_ref = -SCAN_RANGE + 0.5 * (vk / ta) * t_in**2
                    ddq_ref = vk / ta
                elif t_in < ta + tc: # 正向匀速段 (Region B)
                    dq_ref = vk
                    q_ref = -SCAN_RANGE + 0.5 * vk * ta + vk * (t_in - ta)
                    ddq_ref = 0.0
                elif t_in < 2 * ta + tc: # 减速
                    dq_ref = vk - (vk/ta) * (t_in - (ta+tc))
                    q_ref = SCAN_RANGE - 0.5 * (vk/ta) * (2*ta + tc - t_in)**2
                    ddq_ref = -vk/ta
                elif t_in < 3 * ta + tc: # 反向加速
                    dq_ref = -(vk/ta) * (t_in - (2*ta+tc))
                    q_ref = SCAN_RANGE - 0.5 * (vk/ta) * (t_in - (2*ta+tc))**2
                    ddq_ref = -vk/ta
                elif t_in < 3 * ta + 2 * tc: # 反向匀速段 (Region D)
                    dq_ref = -vk
                    q_ref = SCAN_RANGE - 0.5 * vk * ta - vk * (t_in - (3*ta+tc))
                    ddq_ref = 0.0
                else: # 回到起点减速
                    dq_ref = -vk + (vk/ta) * (t_in - (3*ta+2*tc))
                    q_ref = -SCAN_RANGE + 0.5 * (vk/ta) * (dur - t_in)**2
                    ddq_ref = vk/ta

                if dt_v >= dur:
                    v_idx += 1
                    t_v_start = t_curr
                    print(f"[INFO] Stage 1: Velocity {vk:.2f} finished. Next: {V_SCAN_STEPS[v_idx] if v_idx < len(V_SCAN_STEPS) else 'Fourier'}")
                    if v_idx >= len(V_SCAN_STEPS):
                        fourier_start_time = t_curr
            
            elif t_curr - fourier_start_time < T_FOURIER:
                # --- Stage 2: 傅里叶信号激励 ---
                stage = 2
                tf = t_curr - fourier_start_time
                q_ref, dq_ref, ddq_ref = Q_FIX, 0.0, 0.0
                for l in range(1, 6):
                    lw = l * W_BASE
                    al, bl = A_FOUR[l-1], B_FOUR[l-1]
                    q_ref += (al/lw) * np.sin(lw * tf) - (bl/lw) * np.cos(lw * tf)
                    dq_ref += al * np.cos(lw * tf) + bl * np.sin(lw * tf)
                    ddq_ref += -al * lw * np.sin(lw * tf) + bl * lw * np.cos(lw * tf)
            else:
                print("[INFO] Trajectory complete.")
                break

            # 控制指令下达
            q_target = q_init.copy()
            q_target[5] = q_ref
            dq_target = np.zeros(6)
            dq_target[5] = dq_ref
            
            # --- 第 6 轴 PD+I 控制 ---
            err_pos = q_ref - q[5]
            q_error_int_j6 += err_pos * dt
            q_error_int_j6 = np.clip(q_error_int_j6, -0.5, 0.5) # limit integral
            
            tau = kp * (q_target - q) + kd * (dq_target - dq)
            tau[5] += ki_j6 * q_error_int_j6
            
            # 辨识过程中暂不进行模型补偿，除非用户显式要求在控制循环中验证
            # 仅在需要验证时打开：
            # tau_ff = J_ID * ddq_ref + B_ID * dq_ref + FC_ID * np.sign(dq_ref) + BIAS_ID
            # tau[5] += tau_ff
            
            tau = np.clip(tau, -torque_limits, torque_limits)
            if not rtde_c.directTorque(tau.tolist(), True): break
            
            # 记录数据
            data_log.append([t_curr, q_ref, q[5], dq_ref, dq[5], ddq_ref, acc_act, tau[5], stage])
            visualizer.update(q)

            # 实时绘图
            if time.perf_counter() > next_plot:
                recent = np.array(data_log[-2000:])
                if len(recent) > 0:
                    win_m = recent[:, 0] > (t_curr - 5.0)
                    win = recent[win_m]
                    line_pos_des.set_data(win[:, 0], win[:, 1])
                    line_pos_act.set_data(win[:, 0], win[:, 2])
                    line_vel_des.set_data(win[:, 0], win[:, 3])
                    line_vel_act.set_data(win[:, 0], win[:, 4])
                    line_acc_des.set_data(win[:, 0], win[:, 5])
                    line_acc_act.set_data(win[:, 0], win[:, 6])
                    line_torque.set_data(win[:, 0], win[:, 7])
                    for ax in axs: ax.relim(); ax.autoscale_view()
                    axs[-1].set_xlim(left=max(0, t_curr-5), right=max(5, t_curr))
                    fig.canvas.draw_idle(); fig.canvas.flush_events()
                next_plot = time.perf_counter() + plot_dt

            next_tick += dt
            wait = next_tick - time.perf_counter()
            if wait > 0: time.sleep(wait)

    except KeyboardInterrupt: print("\n[INFO] Stopped by user.")
    finally:
        rtde_c.stopScript(); rtde_c.disconnect(); rtde_r.disconnect(); visualizer.stop()
        if data_log:
            full = np.array(data_log)
            np.savetxt("Data/Joint6_data.csv", full, delimiter=",", 
                       header="time,q_des,q_act,dq_des,dq_act,ddq_des,ddq_act,torque,stage", 
                       comments="")
            plt.savefig("Data/Joint6_plot.png")
            print("[INFO] Data saved to Data/Joint6_data.csv. Close window to exit.")
            plt.show(block=True)

if __name__ == "__main__":
    main()
