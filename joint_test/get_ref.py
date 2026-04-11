import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def generate_trajectory(joint_idx=5, q_init_act=0.0, config_path="Config/Ref.yaml"):
    """
    Generate trajectory for a specific joint.
    joint_idx: 0 to 5 (Base to Wrist 3)
    q_init_act: initial position of the active joint
    """
    # 1. 加载配置文件
    try:
        with open(config_path, 'r') as f:
            full_cfg = yaml.safe_load(f)
            # 使用 0-indexed 名称
            joint_key = f"joint_{joint_idx}"
            cfg = full_cfg['joint_params'][joint_key]
    except Exception as e:
        print(f"[ERROR] Failed to load config for {joint_idx}: {e}")
        # 默认回退到 joint_6 的参数
        cfg = {
            "v_min": 0.1, "v_max": 2.0, "num_scan_points": 10,
            "scan_range": 1.2, "max_acc": 1.0, "t_fourier": 20.0,
            "f_base": 0.3, "amps": [0.4, 0.2, 0.1], "freq_mults": [1.0, 2.0, 3.0]
        }

    # 2. 提取并映射参数
    V_MIN = cfg['v_min']
    V_MAX = cfg['v_max']
    N_SCAN = cfg['num_scan_points']
    SCAN_RANGE = cfg['scan_range']
    MAX_ACC = cfg['max_acc']
    T_FOURIER = cfg['t_fourier']
    F_BASE = cfg['f_base']
    AMPS = cfg['amps']
    FREQ_MULTS = cfg['freq_mults']

    # --- 自动生成速度扫描序列 (平方分布实现低速加密) ---
    if N_SCAN > 1:
        V_SCAN_STEPS = [V_MIN + (V_MAX - V_MIN) * (i / (N_SCAN - 1))**2 for i in range(N_SCAN)]
    else:
        V_SCAN_STEPS = [V_MIN]
    
    W_BASE = 2 * np.pi * F_BASE
    dt = 0.002
    
    # --- 动态计算总时长 ---
    T_INIT_MOVE = 5.0
    T_SCAN = 0.0
    for vk in V_SCAN_STEPS:
        ta = vk / MAX_ACC
        tc = (2 * SCAN_RANGE) / vk - ta
        if tc < 0:
            tc = 0
            ta = np.sqrt(2 * SCAN_RANGE / MAX_ACC)
        T_SCAN += 2 * (2 * ta + tc)
    T_MOVE_TO_ZERO = 3.0
    
    T_TOTAL = T_INIT_MOVE + T_SCAN + T_MOVE_TO_ZERO + T_FOURIER + 0.5 
    time_vec = np.arange(0, T_TOTAL, dt)
    
    q_out = []
    dq_out = []
    ddq_out = []
    stage_out = []
    
    # 状态机变量
    stage = 0
    v_idx = 0
    sub_stage = "INIT_MOVE"
    t_sub_start = 0.0
    q_curr_ref = q_init_act
    
    def get_smooth_move(t_sub, q_start, q_end, duration):
        if t_sub >= duration: return q_end, 0.0, 0.0
        alpha = t_sub / duration
        s = 10*alpha**3 - 15*alpha**4 + 6*alpha**5
        ds = (30*alpha**2 - 60*alpha**3 + 30*alpha**4) / duration
        dds = (60*alpha - 180*alpha**2 + 120*alpha**3) / (duration**2)
        q = q_start + (q_end - q_start) * s
        dq = (q_end - q_start) * ds
        ddq = (q_end - q_start) * dds
        return q, dq, ddq

    for t_curr in time_vec:
        q_ref, dq_ref, ddq_ref = q_curr_ref, 0.0, 0.0
        t_in_sub = t_curr - t_sub_start
        
        if sub_stage == "INIT_MOVE":
            stage = 0
            # 目标位置改为 q_init_act - SCAN_RANGE
            q_ref, dq_ref, ddq_ref = get_smooth_move(t_in_sub, q_init_act, q_init_act - SCAN_RANGE, 5.0)
            if t_in_sub >= 5.0:
                sub_stage = "SCAN_CYCLE"
                t_sub_start = t_curr
        
        elif sub_stage == "SCAN_CYCLE":
            stage = 1
            vk = V_SCAN_STEPS[v_idx]
            ta = vk / MAX_ACC
            tc = (2 * SCAN_RANGE) / vk - ta
            if tc < 0:
                tc = 0
                ta = np.sqrt(2 * SCAN_RANGE / MAX_ACC)
                vk = MAX_ACC * ta
                
            dur = 2 * (2 * ta + tc)
            t_in = t_in_sub
            
            # 在 [q_init_act - SCAN_RANGE, q_init_act + SCAN_RANGE] 之间摆动
            if t_in < ta: # 正向加速
                dq_ref = (vk / ta) * t_in
                q_ref = (q_init_act - SCAN_RANGE) + 0.5 * (vk / ta) * t_in**2
                ddq_ref = vk / ta
            elif t_in < ta + tc: # 正向匀速
                dq_ref = vk
                q_ref = (q_init_act - SCAN_RANGE + 0.5 * vk * ta) + vk * (t_in - ta)
                ddq_ref = 0.0
            elif t_in < 2 * ta + tc: # 正向减速
                t_dec = t_in - (ta + tc)
                dq_ref = vk - (vk / ta) * t_dec
                q_ref = (q_init_act + SCAN_RANGE - 0.5 * vk * ta) + vk * t_dec - 0.5 * (vk / ta) * t_dec**2
                ddq_ref = -vk / ta
            elif t_in < 3 * ta + tc: # 反向加速
                t_acc_rev = t_in - (2 * ta + tc)
                dq_ref = -(vk / ta) * t_acc_rev
                q_ref = (q_init_act + SCAN_RANGE) - 0.5 * (vk / ta) * t_acc_rev**2
                ddq_ref = -vk / ta
            elif t_in < 3 * ta + 2 * tc: # 反向匀速
                dq_ref = -vk
                q_ref = (q_init_act + SCAN_RANGE - 0.5 * vk * ta) - vk * (t_in - (3 * ta + tc))
                ddq_ref = 0.0
            elif t_in < dur: # 反向减速
                t_dec_rev = t_in - (3 * ta + 2 * tc)
                dq_ref = -vk + (vk / ta) * t_dec_rev
                q_ref = (q_init_act - SCAN_RANGE + 0.5 * vk * ta) - vk * t_dec_rev + 0.5 * (vk / ta) * t_dec_rev**2
                ddq_ref = vk / ta
            
            if t_in >= dur:
                v_idx += 1
                t_sub_start = t_curr
                if v_idx >= len(V_SCAN_STEPS):
                    sub_stage = "MOVE_TO_ZERO"
                else:
                    sub_stage = "SCAN_CYCLE"
        
        elif sub_stage == "MOVE_TO_ZERO":
            stage = 1.5
            # 回到 q_init_act
            q_ref, dq_ref, ddq_ref = get_smooth_move(t_in_sub, q_init_act - SCAN_RANGE, q_init_act, 3.0)
            if t_in_sub >= 3.0:
                sub_stage = "FOURIER"
                t_sub_start = t_curr
                
        elif sub_stage == "FOURIER":
            stage = 2
            tf = t_in_sub
            q_ref, dq_ref, ddq_ref = q_init_act, 0.0, 0.0 # 修改: 以 q_init_act 为偏置
            for i in range(3):
                wi = FREQ_MULTS[i] * W_BASE
                Ai = AMPS[i]
                q_ref += Ai * np.sin(wi * tf)
                dq_ref += Ai * wi * np.cos(wi * tf)
                ddq_ref += -Ai * (wi**2) * np.sin(wi * tf)
            
            if tf >= T_FOURIER:
                sub_stage = "FINISH"
                t_sub_start = t_curr
        else:
            stage = 3 # Finished
            q_ref, dq_ref, ddq_ref = q_init_act, 0.0, 0.0
            
        q_curr_ref = q_ref
        q_out.append(q_ref)
        dq_out.append(dq_ref)
        ddq_out.append(ddq_ref)
        stage_out.append(stage)
        
    return time_vec, np.array(q_out), np.array(dq_out), np.array(ddq_out), np.array(stage_out)

if __name__ == "__main__":
    # Test for Joint 5 (Wrist 3)
    t, q, dq, ddq, stg = generate_trajectory(joint_idx=5)
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1); plt.plot(t, q, 'b'); plt.ylabel("Pos (rad)"); plt.title("Joint 5 Identification Trajectory")
    plt.subplot(4, 1, 2); plt.plot(t, dq, 'g'); plt.ylabel("Vel (rad/s)")
    plt.subplot(4, 1, 3); plt.plot(t, ddq, 'r'); plt.ylabel("Acc (rad/s^2)")
    plt.subplot(4, 1, 4); plt.plot(t, stg, 'k'); plt.ylabel("Stage")
    plt.tight_layout()
    plt.savefig("Data/Trajectory_Verification_Joint5.png")
    print("[INFO] Plot saved to Data/Trajectory_Verification_Joint5.png")
