#!/usr/bin/env python3
"""Sequentially verify the Torque Constant (Kt) for all 6 joints of UR5e.
Approaches each joint, applies a sine torque via directTorque, and regresses Kt.
Generates a plot for each joint.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.56.101"

def main():
    print(f"[INFO] Connecting to robot {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    # 1. 从 Ref.yaml 加载初始化姿态
    ref_path = Path("Config/Ref.yaml")
    if not ref_path.exists():
        print(f"[ERROR] {ref_path} not found.")
        return
    with open(ref_path, 'r') as f:
        ref_cfg = yaml.safe_load(f)
    q_init_yaml = np.array(ref_cfg.get("q_init", [1.5, -1.5, -1.5, -1.5, 1.5, 0.0]))
    print(f"[INFO] Loaded Q_INIT from Ref.yaml: {q_init_yaml}")

    # 2. 预移动到初始姿态
    q_actual = np.array(rtde_r.getActualQ())
    dist = np.linalg.norm(q_actual - q_init_yaml)
    if dist > 0.01:
        print(f"[INFO] Moving to reference q_init (Dist={dist:.4f})...")
        rtde_c.moveJ(q_init_yaml.tolist(), 0.4, 0.4) 
        time.sleep(1.0)

    # 全局参数
    DURATION_PER_JOINT = 10.0 # 每个关节测试 10 秒
    FREQ = 0.5               # 测试频率
    dt = 0.002
    
    # 激励幅值 (根据用户反馈调整)
    # 大关节建议至少 3.0-5.0Nm 以上才能观察到电流变化，1.5Nm 往往无法克服静摩擦
    AMPS = [3.0, 5.0, 5.0, 0.5, 0.5, 0.5]
    
    kt_results = []
    config_path = Path("Config")
    config_path.mkdir(exist_ok=True)

    print("\n" + "="*50)
    print("      UR5e Multi-Joint Kt Calibration")
    print("="*50)

    try:
        for j_idx in range(6):
            print(f"\n[Testing Joint {j_idx}] ...")
            amp = AMPS[j_idx]
            data_log = []
            
            # 锁定除当前测试轴以外的轴
            kp = np.array([1200.0, 1200.0, 1200.0, 300.0, 200.0, 100.0])
            kd = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 5.0])
            kp[j_idx] = 0.0
            kd[j_idx] = 0.0
            
            start_time = time.perf_counter()
            while True:
                t = time.perf_counter() - start_time
                if t > DURATION_PER_JOINT: break
                
                loop_start = time.perf_counter()
                
                q = np.array(rtde_r.getActualQ())
                dq = np.array(rtde_r.getActualQd())
                curr = np.array(rtde_r.getActualCurrent())
                
                # 当前轴下发正弦力矩
                tau_cmd = amp * np.sin(2 * np.pi * FREQ * t)
                
                # 控制混合 (锁死非测试轴在 q_init_yaml)
                tau = kp * (q_init_yaml - q) - kd * dq
                tau[j_idx] = tau_cmd
                
                rtde_c.directTorque(tau.tolist(), True)
                data_log.append([tau_cmd, curr[j_idx]])
                
                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            
            # 计算 Kt
            data = np.array(data_log)
            taus, currs = data[:, 0], data[:, 1]
            
            # 线性回归 (Tau = Kt * I + offset)
            Phi = np.column_stack([currs, np.ones_like(currs)])
            # 过滤过小电流点 (死区)
            mask = np.abs(currs) > 0.01
            if np.sum(mask) > 10:
                sol, _, _, _ = np.linalg.lstsq(Phi[mask], taus[mask], rcond=None)
                kt, offset = sol
            else:
                kt, offset = 0.0, 0.0
            
            kt_results.append(kt)
            print(f"-> Joint {j_idx} Kt: {kt:.4f} Nm/A (Offset={offset:.4f})")

            # 绘制当前轴的图
            plt.figure(figsize=(8, 6))
            plt.scatter(currs, taus, alpha=0.3, s=10, label='Measured Data')
            # 绘制拟合线
            i_grid = np.linspace(np.min(currs), np.max(currs), 100)
            plt.plot(i_grid, kt * i_grid + offset, 'r', lw=2, label=f'Fit: Kt={kt:.4f}')
            plt.xlabel("Joint Current [A]")
            plt.ylabel("Commanded Torque [Nm]")
            plt.title(f"Joint {j_idx} Torque-Current Correlation")
            plt.grid(True)
            plt.legend()
            save_name = config_path / f"Kt_Verification_Joint_{j_idx}.png"
            plt.savefig(save_name)
            print(f"[INFO] Plot saved to {save_name}")
            plt.close() # 关闭以防干扰下一个
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        
        if kt_results:
            print("\n" + "="*50)
            print("Final Calibration Results Table:")
            print("-" * 30)
            for i, val in enumerate(kt_results):
                print(f" Joint {i} |  {val:7.4f} Nm/A")
            print("-" * 30)
            
            with open(config_path / "Kt_calibration_results.yaml", "w") as f:
                yaml.dump({"Kt": [float(x) for x in kt_results]}, f)
                print(f"[INFO] Summary saved to {config_path / 'Kt_calibration_results.yaml'}")

if __name__ == "__main__":
    main()
