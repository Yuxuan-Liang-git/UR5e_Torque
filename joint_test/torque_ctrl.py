#!/usr/bin/env python3
"""Joint 6 Sine Movement for UR5e.
Real-time plotting with 5s window for Position, Velocity, Acceleration, and Torque.
Saves data to Data/Joint6_data.csv and Data/Joint6_plot.png.
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
    parser = argparse.ArgumentParser(description="UR5e Joint6 Sine Movement")
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

    xml_path = Path("ur5e_gripper/scene.xml")
    if not xml_path.exists():
        xml_path = Path("universal_robots_ur5e/scene.xml")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    q_init = np.array(rtde_r.getActualQ())
    
    # 激励信号参数
    V_STEPS = np.arange(0.2, 2.0, 0.2)
    A_CONST = np.pi      # 匀速段幅值 (rad)
    N_CONST_LOOPS = 2    # 每阶匀速段循环次数
    
    T_CHIRP = 100.0       # Chirp 段持续时间 (s)
    A_CHIRP = 0.5        # Chirp 段幅值 (rad)
    F0_CHIRP = 0.1       # 起始频率 (Hz)
    F1_CHIRP = 1.0       # 终止频率 (Hz)

    # PID 增益设置 (Joints 1-6)
    kp = np.array([1500.0, 1500.0, 1500.0, 300.0, 100.0, 40.0])
    kd = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 10.0])
    ki_j6 = 5.0  # 第 6 轴 积分增益

    # 起动转矩辨识参数 (Stage 3)
    N_BREAKAWAY_REPS = 4
    BREAKAWAY_RAMP_RATE = 0.2    # N·m/s
    BREAKAWAY_MOVE_THRESH = 0.0001 # rad
    BREAKAWAY_COOLDOWN = 1.0     # s

    # 加载辨识出来的参数 (用于第 6 轴补偿)
    param_file = Path("Data/identified_params.yaml")
    if param_file.exists():
        print(f"[INFO] Loading parameters from {param_file} for Joint 6 compensation...")
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        J_ID = params.get("J", 0.0)
        B_ID = params.get("B", 0.0)
        FC_ID = params.get("Fc", 0.0)
        BIAS_ID = params.get("bias", 0.0)
    else:
        print("[WARN] No identified_params.yaml found. No compensation will be applied.")
        J_ID, B_ID, FC_ID, BIAS_ID = 0.0, 0.0, 0.0, 0.0

    # Live Plot Setup
    plt.ion()
    # 增加两个子图用于角速度和角加速度
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    line_pos_des, = axs[0].plot([], [], 'r--', label='Target Pos')
    line_pos_act, = axs[0].plot([], [], 'b-', label='Actual Pos')
    
    line_vel_des, = axs[1].plot([], [], 'r--', label='Target Vel')
    line_vel_act, = axs[1].plot([], [], 'b-', label='Actual Vel')
    
    line_acc_des, = axs[2].plot([], [], 'r--', label='Target Acc')
    line_acc_act, = axs[2].plot([], [], 'b-', label='Actual Acc')
    
    line_torque,  = axs[3].plot([], [], 'k-', label='Torque')

    axs[0].set_ylabel('Pos (rad)')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Joint 6 Live Tracking (5s Window)')
    
    axs[1].set_ylabel('Vel (rad/s)')
    axs[1].legend(loc='upper right')
    
    axs[2].set_ylabel('Acc (rad/s^2)')
    axs[2].legend(loc='upper right')
    
    axs[3].set_ylabel('Torque (Nm)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')

    data_log = []
    start_time = time.perf_counter()
    visualizer = VisualizationWorker(xml_path)
    
    dt = 0.002
    plot_dt = 0.05
    next_tick = time.perf_counter()
    next_plot = time.perf_counter()
    START_WAIT = 2.0      # 启动后静止等待 2 秒
    STABILIZE_DUR = 5.0   # 回零运动时长
    PAUSE_DUR = 2.0       # 回到零点后静止 2 秒
    
    T0 = START_WAIT
    T1 = T0 + STABILIZE_DUR
    T2 = T1 + PAUSE_DUR   # 启动探测阶段的时间 (START_BREAKAWAY)
    
    # 匀速段和扫频段的时间将根据探测结束时间动态计算
    durations_v = [N_CONST_LOOPS * (4 * A_CONST / v) for v in V_STEPS]
    cum_durations_v = np.cumsum([0.0] + durations_v)
    t_p1_total = cum_durations_v[-1]
    
    prev_dq = 0.0
    q_error_int_j6 = 0.0  # 第 6 轴误差积分
    
    # 起动转矩探测状态变量
    breakaway_rep = 0
    breakaway_dir = 1  # 1: 正向, -1: 反向
    breakaway_sub_state = "RAMP" # "RAMP" or "COOLDOWN"
    t_sub_start = T2
    q_start_break = 0.0
    tau_ramp_val = 0.0
    breakaway_finished = False
    t_p1_start = 0.0 # 待定

    print("[INFO] Control loop started. Press Ctrl+C to stop.")
    try:
        while visualizer.is_running():
            t_curr = time.perf_counter() - start_time
            q = np.array(rtde_r.getActualQ())
            dq = np.array(rtde_r.getActualQd())
            
            # 计算加速度 (有限差分)
            acc_act = (dq[5] - prev_dq) / dt
            prev_dq = dq[5]
            
            q_des = q_init.copy()
            dq_des = np.zeros(6)
            ddq_des = 0.0
            
            # --- 轨迹生成逻辑 ---
            if t_curr < T0:
                # 初始静止阶段: 保持在起始位置
                stage = 0
                q_ref = q_init[5]
                dq_ref = 0.0
                ddq_ref = 0.0
            elif t_curr < T1:
                # 平滑移动阶段: 从 q_init 回到 0
                stage = 0
                alpha = (t_curr - T0) / (T1 - T0)
                q_ref = (1 - alpha) * q_init[5] + alpha * 0.0
                dq_ref = (0.0 - q_init[5]) / (T1 - T0)
                ddq_ref = 0.0
            elif t_curr < T2:
                # 回零后静止阶段: 在 0 位停留
                stage = 0
                q_ref = 0.0
                dq_ref = 0.0
                ddq_ref = 0.0
                q_start_break = q[5] # 为 Stage 3 记录起始位置
                t_sub_start = T2
            elif not breakaway_finished:
                # --- 阶段 1: 起动转矩探测 (逐渐增加力矩) ---
                stage = 1
                t_in_sub = t_curr - t_sub_start
                
                if breakaway_sub_state == "RAMP":
                    tau_ramp_val = breakaway_dir * BREAKAWAY_RAMP_RATE * t_in_sub
                    q_ref = q_start_break 
                    dq_ref = 0.0
                    ddq_ref = 0.0
                    
                    # 检测到显著运动
                    if abs(q[5] - q_start_break) > BREAKAWAY_MOVE_THRESH:
                        print(f"[INFO] Breakaway detected! Rep: {breakaway_rep+1}, Dir: {breakaway_dir}, Tau: {tau_ramp_val:.4f}")
                        breakaway_sub_state = "COOLDOWN"
                        t_sub_start = t_curr
                
                elif breakaway_sub_state == "COOLDOWN":
                    tau_ramp_val = 0.0
                    q_ref = q[5] 
                    dq_ref = 0.0
                    ddq_ref = 0.0
                    
                    if t_in_sub > BREAKAWAY_COOLDOWN:
                        if breakaway_dir == 1:
                            breakaway_dir = -1
                        else:
                            breakaway_dir = 1
                            breakaway_rep += 1
                        
                        if breakaway_rep >= N_BREAKAWAY_REPS:
                            print("[INFO] Breakaway identification finished. Starting Ramp phase (Stage 2).")
                            breakaway_finished = True
                            t_p1_start = t_curr
                        else:
                            breakaway_sub_state = "RAMP"
                            t_sub_start = t_curr
                            q_start_break = q[5]

            else:
                # 已经是 breakaway_finished 为 True
                t_traj = t_curr - t_p1_start
                
                # --- 阶段 2: 阶梯匀速往复 ---
                if t_traj < t_p1_total:
                    stage = 2
                    idx = np.searchsorted(cum_durations_v, t_traj, side='right') - 1
                    v_curr = V_STEPS[idx]
                    t_in_block = t_traj - cum_durations_v[idx]
                    
                    t_cycle = 4 * A_CONST / v_curr
                    t_ref_in_cycle = t_in_block % t_cycle
                    t_ramp1 = A_CONST / v_curr
                    t_ramp2 = 3 * A_CONST / v_curr
                    
                    if t_ref_in_cycle < t_ramp1:
                        q_ref = v_curr * t_ref_in_cycle
                        dq_ref = v_curr
                        ddq_ref = 0.0
                    elif t_ref_in_cycle < t_ramp2:
                        q_ref = A_CONST - v_curr * (t_ref_in_cycle - t_ramp1)
                        dq_ref = -v_curr
                        ddq_ref = 0.0
                    else:
                        q_ref = -A_CONST + v_curr * (t_ref_in_cycle - t_ramp2)
                        dq_ref = v_curr
                        ddq_ref = 0.0
                else:
                    # --- 阶段 3: Chirp 信号 ---
                    stage = 3
                    t_chirp = t_traj - t_p1_total
                    if t_chirp < T_CHIRP:
                        k = (F1_CHIRP - F0_CHIRP) / T_CHIRP
                        phase = 2 * np.pi * (F0_CHIRP * t_chirp + 0.5 * k * t_chirp**2)
                        
                        q_ref = A_CHIRP * np.sin(phase)
                        d_phase_dt = 2 * np.pi * (F0_CHIRP + k * t_chirp)
                        dq_ref = A_CHIRP * np.cos(phase) * d_phase_dt
                        
                        d2_phase_dt2 = 2 * np.pi * k
                        ddq_ref = -A_CHIRP * np.sin(phase) * (d_phase_dt**2) + A_CHIRP * np.cos(phase) * d2_phase_dt2
                    else:
                        print("[INFO] Identification trajectory finished. Exiting...")
                        visualizer.stop()
                        break
            
            q_des[5] = q_ref # 使用绝对坐标 0
            dq_des[5] = dq_ref
            ddq_des = ddq_ref

            tau_pd = kp * (q_des - q) + kd * (dq_des - dq)
            
            # --- 第 6 轴增加积分项 (I) ---
            err_j6 = q_des[5] - q[5]
            q_error_int_j6 += err_j6 * dt
            # 抗饱和 (Anti-windup): 限制积分项最大力矩
            INT_MAX_TORQUE = 2.0
            q_error_int_j6 = np.clip(q_error_int_j6, -INT_MAX_TORQUE/ki_j6, INT_MAX_TORQUE/ki_j6)
            
            # --- 第 6 轴增加摩擦力与惯量补偿 ---
            # 补偿模型: tau_ff = J*ddq_des + B*dq_act + Fc*sign(dq_act) + bias
            VEL_THRESHOLD = 0.01
            if abs(dq[5]) > VEL_THRESHOLD:
                fric_sign = np.sign(dq[5])
            else:
                fric_sign = dq[5] / VEL_THRESHOLD
            
            # 计算第 6 轴的前馈力矩
            tau_ff_j6 = J_ID * ddq_des + B_ID * dq[5] + FC_ID * fric_sign + BIAS_ID
            
            # 合并反馈与前馈 (对第 6 轴)
            tau = tau_pd.copy()
            # 最终力矩 = PD项 + I项 + 前馈项
            if stage == 1:
                # 探测阶段禁用第 6 轴 PD，使用斜坡力矩
                tau[5] = tau_ramp_val
            else:
                tau[5] = tau_pd[5] + ki_j6 * q_error_int_j6 + tau_ff_j6
            
            tau = np.clip(tau, -torque_limits, torque_limits)

            if t_curr > 0.1:
                if not rtde_c.directTorque(tau.tolist(), True):
                    break
            
            # 记录数据: [时间, q_des, q_act, dq_des, dq_act, ddq_des, ddq_act, tau, stage]
            data_log.append([t_curr, q_des[5], q[5], dq_des[5], dq[5], ddq_des, acc_act, tau[5], stage])
            visualizer.update(q)

            if time.perf_counter() > next_plot:
                recent_data = np.array(data_log[-2500:]) 
                if len(recent_data) > 0:
                    mask = recent_data[:, 0] > (t_curr - 5.0)
                    win_data = recent_data[mask]
                    
                    line_pos_des.set_data(win_data[:, 0], win_data[:, 1])
                    line_pos_act.set_data(win_data[:, 0], win_data[:, 2])
                    
                    line_vel_des.set_data(win_data[:, 0], win_data[:, 3])
                    line_vel_act.set_data(win_data[:, 0], win_data[:, 4])
                    
                    line_acc_des.set_data(win_data[:, 0], win_data[:, 5])
                    line_acc_act.set_data(win_data[:, 0], win_data[:, 6])
                    
                    line_torque.set_data(win_data[:, 0], win_data[:, 7])
                    
                    for ax in axs:
                        ax.relim()
                        ax.autoscale_view()
                    
                    axs[-1].set_xlim(left=max(0, t_curr - 5.0), right=max(5.0, t_curr))
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                next_plot = time.perf_counter() + plot_dt

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Cleaning up...")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        visualizer.stop()
        plt.ioff()
        
        if data_log:
            full_data = np.array(data_log)
            save_path = Path("Data")
            save_path.mkdir(exist_ok=True)
            # 更新表头
            header = "time,q_des,q_act,dq_des,dq_act,ddq_des,ddq_act,torque,stage"
            np.savetxt(save_path / "Joint6_data.csv", full_data, delimiter=",", 
                       header=header, comments="")
            plt.savefig(save_path / "Joint6_plot.png")
            print(f"[INFO] Data saved. Closing plot window to exit.")
            plt.show(block=True) 

if __name__ == "__main__":
    main()
    sys.exit(0)
