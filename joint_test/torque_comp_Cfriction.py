#!/usr/bin/env python3
"""
Generalized UR5e Multi-Joint Torque Compensation.
Supports simultaneous compensation for multiple joints while locking others at q_home.
"""

import time
import numpy as np
import yaml
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ==========================================
# 配置区
JOINT_LIST  = [0, 1, 2, 3, 4, 5]  # 需要释放并补偿摩擦力的关节列表 (0-5)
ROBOT_IP    = "192.168.56.101"
COMP_FACTOR = 0.8        # 安全补偿系数 (0-1)
# ==========================================

def main():
    print(f"[INFO] Initializing Compensation Mode for Joints: {JOINT_LIST}")
    
    print(f"[INFO] Connecting to robot {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    
    # 1. 加载 q_home (来自 Ref.yaml)
    ref_path = Path("Config/Ref.yaml")
    if not ref_path.exists():
        print(f"[ERROR] {ref_path} not found.")
        return
    with open(ref_path, 'r') as f:
        ref_cfg = yaml.safe_load(f)
    q_home = np.array(ref_cfg.get("q_home", [1.5, -1.5, -1.5, -1.5, 0.0, 1.5]))
    print(f"[INFO] Target Home Pose: {q_home}")

    # 2. 加载图表中的库伦摩擦力参数 (Table VI) 折算到关节侧
    # UR5e 各关节减速比为 101
    GEAR_RATIO = 101.0
    
    # 关节索引 0-5 对应 图表中的 Joint 1-6
    # 库伦摩擦力 T_c_joint = T_c_motor * GEAR_RATIO
    # 粘性摩擦系数 Kv_joint = Kv_motor * GEAR_RATIO^2
    FRICTION_PARAMS = {
        0: {'Kv': 0.475e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0938 * GEAR_RATIO, 'Fc_rev': 0.071 * GEAR_RATIO},
        1: {'Kv': 1.073e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0973 * GEAR_RATIO, 'Fc_rev': 0.113 * GEAR_RATIO},
        2: {'Kv': 0.382e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0750 * GEAR_RATIO, 'Fc_rev': 0.083 * GEAR_RATIO},
        3: {'Kv': 0.295e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0206 * GEAR_RATIO, 'Fc_rev': 0.008 * GEAR_RATIO},
        4: {'Kv': 0.114e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0201 * GEAR_RATIO, 'Fc_rev': 0.001 * GEAR_RATIO},
        5: {'Kv': 0.188e-3 * (GEAR_RATIO**2), 'Fc_dir': 0.0276 * GEAR_RATIO, 'Fc_rev': 0.005 * GEAR_RATIO},
    }
    
    joint_models = {}
    for j_idx in JOINT_LIST:
        model = FRICTION_PARAMS[j_idx]
        joint_models[j_idx] = model
        print(f"[INFO] Loaded Joint {j_idx} friction: Fc_dir={model['Fc_dir']}, Fc_rev={model['Fc_rev']}, Kv={model['Kv']}")

    # 3. 控制参数
    # 大关节 (0-2) 使用更强的锁定刚度，小关节 (3-5) 稍弱
    KP_LOCK = [1500.0, 1500.0, 1500.0, 500.0, 400.0, 300.0]
    KD_LOCK = [40.0, 40.0, 40.0, 15.0, 10.0, 8.0]
    VEL_THRESHOLD = 0.01 
    torque_limits = np.array([150, 150, 150, 40, 40, 40]) # 安全阈值
    dt = 0.002

    # --- 阶段 1: 线性插值初始化 (5s) ---
    print(f"[INFO] Starting 5s smooth initialization to q_home...")
    q_start = np.array(rtde_r.getActualQ())
    t_init = 5.0
    start_time = time.perf_counter()
    
    while True:
        t_loop = time.perf_counter() - start_time
        if t_loop >= t_init: break
        
        q_curr = np.array(rtde_r.getActualQ())
        dq_curr = np.array(rtde_r.getActualQd())
        
        # 线性插值计算当前时刻的目标位置
        alpha = t_loop / t_init
        q_target = q_start + (q_home - q_start) * alpha
        
        # 全轴 PD 控制锁定
        tau_cmd = []
        for i in range(6):
            tau = KP_LOCK[i] * (q_target[i] - q_curr[i]) + KD_LOCK[i] * (0.0 - dq_curr[i])
            tau_cmd.append(tau)
        
        tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)
        rtde_c.directTorque(tau_cmd.tolist())
        time.sleep(dt)

    # --- 阶段 2: 实时前馈补偿与 PD 锁死 ---
    print(f"\n[INFO] Multi-Joint Compensation Mode ON.")
    print(f"[INFO] Active Joints (Compensated): {JOINT_LIST}")
    print(f"[INFO] Locked Joints: {[i for i in range(6) if i not in JOINT_LIST]}")
    print("[INFO] Press Ctrl+C to exit.")
    
    try:
        while True:
            loop_start = time.perf_counter()
            q = np.array(rtde_r.getActualQ())
            dq = np.array(rtde_r.getActualQd())
            
            tau_cmd = [0.0] * 6
            
            for i in range(6):
                if i in JOINT_LIST:
                    # 获取该轴的模型与参数
                    m = joint_models[i]
                    v_i = dq[i]
                    
                    # 库伦摩擦力模型 (非对称) + 粘性摩擦力
                    # 正向运动与反向运动的摩擦系数不同
                    if v_i > VEL_THRESHOLD:
                        tau_fric = m['Fc_dir'] + m['Kv'] * v_i
                    elif v_i < -VEL_THRESHOLD:
                        tau_fric = -m['Fc_rev'] + m['Kv'] * v_i
                    else:
                        # 速度在死区(±VEL_THRESHOLD)内进行平滑缩放过渡，避免正反向切换时力矩发生阶跃震荡
                        if v_i > 0:
                            tau_fric = (m['Fc_dir'] / VEL_THRESHOLD) * v_i + m['Kv'] * v_i
                        else:
                            tau_fric = (m['Fc_rev'] / VEL_THRESHOLD) * v_i + m['Kv'] * v_i
                    
                    # 综合补偿
                    tau_cmd[i] = COMP_FACTOR * tau_fric
                else:
                    # 其余轴：PD 锁死在 q_home
                    tau_cmd[i] = KP_LOCK[i] * (q_home[i] - q[i]) + KD_LOCK[i] * (0.0 - dq[i])
            
            tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)
            
            if not rtde_c.directTorque(tau_cmd.tolist(), False):
                break
                
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        print("[INFO] Cleaning up...")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()

if __name__ == "__main__":
    main()
