#!/usr/bin/env python3
"""UR5e Joint 6 Impedance Control with Friction Compensation.
Allows setting virtual Inertia (M), Damping (B), and Stiffness (K).
Tau = Tau_friction + Tau_bias + M_virt*ddq + B_virt*dq + K_virt*(q-q0)
"""

import time
import numpy as np
import yaml
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def main():
    # 1. 加载辨识参数
    param_file = Path("Data/identified_params.yaml")
    if param_file.exists():
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        J_IDENT = params.get("J", 0.134)
        BIAS_IDENT = params.get("bias", 0.0)
        stribeck = params.get("Stribeck", {})
        FC = stribeck.get("Fc", 1.77)
        FS = stribeck.get("Fs", 1.38)
        B_FRIC = stribeck.get("B", 1.52)
        VS = stribeck.get("vs", 0.19)
    else:
        print("[ERROR] No identified params found. Run sys_iden.py first.")
        return

    # 2. 理想阻抗参数设置 (基于物理参数)
    M_VIRT = 0.05    # 虚拟惯量 (kg*m^2)
    K_VIRT = 0.0   # 虚拟刚度 (Nm/rad). 设为 0 表示透明拖动
    ZETA   = 2.0    # 阻尼比 (1.0 为临界阻尼, <1 为欠阻尼, >1 为过阻尼)
    
    # 自动计算虚拟阻尼 B_VIRT = 2 * zeta * sqrt(M * K)
    if K_VIRT > 0:
        B_VIRT = 2 * ZETA * np.sqrt(M_VIRT * K_VIRT)
    else:
        # 如果 K=0，则阻尼比失去定义，回退到纯阻尼模式
        B_VIRT = 3.0 
    
    ROBOT_IP = "192.168.56.101"
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    
    q_start = np.array(rtde_r.getActualQ())
    q_ref = q_start[5] # 阻抗平衡点
    
    # 补偿平滑参数
    VEL_THRESHOLD = 0.05
    
    dt = 0.002
    start_time = time.perf_counter()
    next_tick = start_time

    print(f"\n[INFO] Joint Impedance Mode ON (Stability: VEL_THRESHOLD)")
    print(f"-> Virtual Mass: {M_VIRT}, Damping: {B_VIRT}, Stiffness: {K_VIRT}")
    print(f"-> Friction Compensation: FC={FC:.2f}, FS={FS:.2f}")
    print("[Press Ctrl+C to exit]")

    try:
        while True:
            loop_start = time.perf_counter()
            
            q = np.array(rtde_r.getActualQ())
            dq = np.array(rtde_r.getActualQd())
            
            # --- 关节 1-5 锁死 ---
            kp_lock = [1500, 1500, 1500, 300, 200]
            kd_lock = [40, 40, 40, 10, 10]
            tau_lock = []
            for i in range(5):
                tau_lock.append(kp_lock[i]*(q_start[i]-q[i]) - kd_lock[i]*dq[i])
            
            # --- 阶段 6: 阻抗 + 补偿逻辑 ---
            v6 = dq[5]
            p6 = q[5]
            
            # 1. 摩擦力补偿 (Stribeck 模型)
            friction_mag = FC + (FS - FC) * np.exp(-(v6/VS)**2)
            
            # 改进：使用分段线性平滑处理 (对应 torque_comp 的逻辑)
            if abs(v6) > VEL_THRESHOLD:
                fric_sign = np.sign(v6)
            else:
                fric_sign = v6 / VEL_THRESHOLD
            
            tau_fric = friction_mag * fric_sign + B_FRIC * v6
            
            # 2. 理想阻抗力矩 (Active Impedance)
            tau_impedance = -B_VIRT * v6 - K_VIRT * (p6 - q_ref)
            
            # 3. 总力矩 = 摩擦补偿 + 固定偏置 + 阻抗控制分量
            tau_j6 = tau_fric + BIAS_IDENT + tau_impedance
            
            # 合成最终指令：[锁定轴 1-5] + [阻抗轴 6]
            tau_cmd = tau_lock + [tau_j6]
            
            # 4. 下发力矩 (限幅保护)
            tau_cmd = np.clip(tau_cmd, -40, 40).tolist()
            rtde_c.directTorque(tau_cmd, True)
            
            # 维持频率
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()

if __name__ == "__main__":
    main()
