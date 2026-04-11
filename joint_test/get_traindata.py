#!/usr/bin/env python3
"""
Generalized UR5e Joint Data Collection.
Identifies the joint specified by JOINT_ACT.
"""

import time
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from get_ref import generate_trajectory

# ==========================================
# 配置区
JOINT_ACT = 1  # 0-5 (0: Base, 5: Wrist 3)
ROBOT_IP  = "192.168.56.101"
# ==========================================

def main():
    joint_idx = JOINT_ACT # 0-5 索引
    print(f"[INFO] Initializing identification for Joint {JOINT_ACT}...")
    
    print(f"[INFO] Connecting to robot {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    
    # 1. 从 Ref.yaml 加载全局配置
    config_path = Path("Config/Ref.yaml")
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    
    q_init_yaml = np.array(full_cfg.get("q_init", [1.57, -1.57, -1.57, -1.57, 1.57, 0.0]))
    print(f"[INFO] Loaded Q_INIT from Ref.yaml: {q_init_yaml}")

    # 2. 预移动：平滑移动到辨识初始姿态
    q_actual = np.array(rtde_r.getActualQ())
    dist = np.linalg.norm(q_actual - q_init_yaml)
    if dist > 0.01:
        print(f"[INFO] Moving to reference q_init (Dist={dist:.4f})...")
        rtde_c.moveJ(q_init_yaml.tolist(), 0.5, 0.5) # 慢速移动确保安全
        time.sleep(1.0)
    
    # 3. 生成参考轨迹 (以此 q_init 为基准)
    print(f"[INFO] Generating trajectory for Joint {JOINT_ACT}...")
    t_vec, q_ref_vec, dq_ref_vec, ddq_ref_vec, stage_vec = generate_trajectory(
        joint_idx=joint_idx, 
        q_init_act=q_init_yaml[joint_idx]
    )
    
    # 2. 加载 Kt 校准值
    kt_file = Path("Config/Kt_calibration_results.yaml")
    if kt_file.exists():
        with open(kt_file, 'r') as f:
            kt_data = yaml.safe_load(f)
            KT_LIST = kt_data.get("Kt", [11.0, 11.0, 11.0, 8.4, 8.4, 8.4])
            print(f"[INFO] Loaded Kt: {KT_LIST}")
    else:
        KT_LIST = [11.0, 11.0, 11.0, 8.4, 8.4, 8.4]
        print("[WARN] No Kt config found. Using defaults.")

    # 3. 可视化设置
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    line_q_des, = axs[0].plot([], [], 'r--', label='Des Pos')
    line_q_act, = axs[0].plot([], [], 'b-', label='Act Pos')
    line_dq_act, = axs[1].plot([], [], 'b-', label='Act Vel')
    line_tau_act, = axs[2].plot([], [], 'k-', label='Torque (Nm)')
    line_stage, = axs[3].plot([], [], 'g-', label='Stage')
    
    for ax in axs: ax.grid(True); ax.legend(loc='upper right')
    axs[0].set_ylabel('Pos (rad)'); axs[1].set_ylabel('Vel (rad/s)')
    axs[2].set_ylabel('Torque (Nm)'); axs[3].set_ylabel('Stage')
    axs[3].set_xlabel('Time (s)')

    dt = 0.002
    plot_dt = 0.1
    next_plot_time = 0.0
    lookahead_time = 0.1
    gain = 300.0
    
    data_log = []
    
    print(f"[INFO] Starting servoJ tracking on Joint {JOINT_ACT} (500Hz)...")
    try:
        real_start_time = time.perf_counter()
        for i in range(len(t_vec)):
            loop_start = time.perf_counter()
            t_now = t_vec[i]
            
            # 只有 JOINT_ACT 轴跟随轨迹，其余固定在本预设姿态
            q_target = q_init_yaml.copy()
            q_target[joint_idx] = q_ref_vec[i]
            
            rtde_c.servoJ(q_target.tolist(), 0.0, 0.0, dt, lookahead_time, gain)
            
            q_act = rtde_r.getActualQ()
            dq_act = rtde_r.getActualQd()
            i_act = np.array(rtde_r.getActualCurrent())
            tau_act = i_act * np.array(KT_LIST)
            
            # 记录活跃轴的数据
            item = [
                t_now, 
                q_ref_vec[i], q_act[joint_idx], 
                dq_ref_vec[i], dq_act[joint_idx], 
                ddq_ref_vec[i], 
                tau_act[joint_idx], 
                stage_vec[i]
            ]
            data_log.append(item)
            
            # 实时绘图与进度条显示
            if t_now >= next_plot_time:
                # 绘制
                win_size = 5000
                recent_data = np.array(data_log[-win_size:])
                if len(recent_data) > 0:
                    t_win = recent_data[:, 0]
                    line_q_des.set_data(t_win, recent_data[:, 1])
                    line_q_act.set_data(t_win, recent_data[:, 2])
                    line_dq_act.set_data(t_win, recent_data[:, 4])
                    line_tau_act.set_data(t_win, recent_data[:, 6])
                    line_stage.set_data(t_win, recent_data[:, 7])
                    for ax in axs: 
                        ax.relim(); ax.autoscale_view()
                    axs[-1].set_xlim(left=max(0, t_now - 10), right=max(10, t_now))
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                
                # 打印进度条
                progress = (i + 1) / len(t_vec)
                bar_len = 30
                filled_len = int(bar_len * progress)
                bar = '█' * filled_len + '-' * (bar_len - filled_len)
                eta = (len(t_vec) - i) * dt
                print(f"\r[Progress] |{bar}| {progress*100:.1f}%  ETA: {eta:.1f}s", end="")
                
                next_plot_time += plot_dt
            
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("[INFO] Stopping servoJ and saving data...")
        rtde_c.servoStop()
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        
        if data_log:
            df = pd.DataFrame(data_log, columns=['time', 'q_des', 'q_act', 'dq_des', 'dq_act', 'ddq_des', 'torque', 'stage'])
            output_dir = Path("Data")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"Data_Joint_{JOINT_ACT}.csv"
            df.to_csv(output_path, index=False)
            print(f"[INFO] Data saved to {output_path}")

if __name__ == "__main__":
    main()
