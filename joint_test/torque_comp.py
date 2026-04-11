#!/usr/bin/env python3
"""UR5e Joint 6 Torque Control with Friction and Inertia Compensation.
Uses identified parameters: J, B, Fc, bias.
"""

import argparse
import time
import sys
import numpy as np
import yaml
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Joint6 Friction Compensation Control")
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

    # 1. 加载辨识出来的参数
    param_file = Path("Data/identified_params.yaml")
    if param_file.exists():
        print(f"[INFO] Loading parameters from {param_file}...")
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        J_ID = params.get("J", 0.025780)
        B_ID = params.get("B", 0.494677)
        FC_ID = params.get("Fc", 0.371542)
        BIAS_ID = params.get("bias", -0.022616)
    else:
        print(f"[WARN] {param_file} not found. Using default/hardcoded values.")
        J_ID = 0.025780
        B_ID = 0.494677
        FC_ID = 0.371542
        BIAS_ID = -0.022616

    print(f"辨识参数: J={J_ID:.6f}, B={B_ID:.6f}, Fc={FC_ID:.6f}, Bias={BIAS_ID:.6f}")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    q_init = np.array(rtde_r.getActualQ())
    
    # 锁死其他关节的 PD 参数 (Joints 1-5)
    KP_LOCK = [1500.0, 1500.0, 1500.0, 300.0, 100.0]
    KD_LOCK = [40.0, 40.0, 40.0, 10.0, 10.0]
    
    # 补偿策略参数 (Joint 6)
    VEL_THRESHOLD = 0.01
    
    dt = 0.002
    start_time = time.perf_counter()
    next_tick = start_time

    print("[INFO] Pure Compensation Mode started (Joints 1-5 Locked). Press Ctrl+C to stop.")
    try:
        while True:
            t_curr = time.perf_counter() - start_time
            q = np.array(rtde_r.getActualQ())
            dq = np.array(rtde_r.getActualQd())
            
            # --- 阶段 1-5: 锁死逻辑 (PD 控制回 q_init) ---
            tau_lock = []
            for i in range(5):
                t_i = KP_LOCK[i] * (q_init[i] - q[i]) + KD_LOCK[i] * (0.0 - dq[i])
                tau_lock.append(t_i)
            
            # --- 阶段 6: 纯前馈补偿逻辑 ---
            if abs(dq[5]) > VEL_THRESHOLD:
                fric_sign = np.sign(dq[5])
            else:
                fric_sign = dq[5] / VEL_THRESHOLD
                
            tau_fric = B_ID * dq[5] + FC_ID * fric_sign
            tau_bias = BIAS_ID
            tau_total_j6 = tau_fric + tau_bias
            
            # 组合力矩指令 (增加一个 0.9 的系数以提高稳定性)
            tau_cmd = tau_lock + [0.9 * tau_total_j6]
            tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)

            if t_curr > 0.1:
                if not rtde_c.directTorque(tau_cmd.tolist(), True):
                    break
            
            # 维持频率
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

if __name__ == "__main__":
    main()
