import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

JOINT_ACT = 0

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
    T_CV_TARGET = 0.5 # 期望稳速运行 1.0 秒
    T_SCAN = 0.0
    for vk_req in V_SCAN_STEPS:
        # 核心公式: 2*R = v^2/a + v * T_cv => R = 0.5*(v^2/a + v*T_cv)
        rk_calc = 0.5 * (vk_req**2 / MAX_ACC + vk_req * T_CV_TARGET)
        rk = min(rk_calc, SCAN_RANGE) # 不超过配置的最大行程
        
        vk = vk_req
        if (vk**2 / MAX_ACC) > rk:
            vk = np.sqrt(rk * MAX_ACC)
        
        ta = vk / MAX_ACC
        d_half = ta + rk / vk 
        d_full = ta + (2 * rk) / vk 
        T_SCAN += d_half + d_full + d_full + d_full + d_half
    
    T_TOTAL = T_SCAN + T_FOURIER + 0.5 
    time_vec = np.arange(0, T_TOTAL, dt)
    
    q_out, dq_out, ddq_out, stage_out, dq_step_out = [], [], [], [], []
    
    # 状态机变量
    stage = 0
    v_idx = 0
    # sub_cycle_idx: 0: 0->-R, 1: -R->+R, 2: +R->-R, 3: -R->+R, 4: +R->0
    sub_cycle_idx = 0 
    sub_stage = "SCAN_CYCLE"
    t_sub_start = 0.0
    q_curr_ref = q_init_act
    
    for t_curr in time_vec:
        q_ref, dq_ref, ddq_ref = q_curr_ref, 0.0, 0.0
        dq_step = 0.0
        t_in_sub = t_curr - t_sub_start
        
        if sub_stage == "SCAN_CYCLE":
            stage = 1
            vk_req = V_SCAN_STEPS[v_idx]
            # 核心公式: 2*R = v^2/a + v * T_cv => R = 0.5*(v^2/a + v*T_cv)
            rk_calc = 0.5 * (vk_req**2 / MAX_ACC + 1.0 * vk_req) # T_CV_TARGET = 1.0
            rk = min(rk_calc, SCAN_RANGE)
            
            vk = vk_req
            if (vk**2 / MAX_ACC) > rk:
                vk = np.sqrt(rk * MAX_ACC)
            
            dq_step = vk_req # 记录名义目标值
            ta = vk / MAX_ACC
            d_h = ta + rk / vk
            d_f = ta + (2 * rk) / vk
            dur_list = [d_h, d_f, d_f, d_f, d_h]
            dur = dur_list[sub_cycle_idx]

            if sub_cycle_idx == 0: # 0 -> -R
                tc = d_h - 2*ta
                if t_in_sub < ta:
                    dq_ref = -(vk/ta) * t_in_sub
                    q_ref = q_init_act - 0.5*(vk/ta)*t_in_sub**2
                    ddq_ref = -vk/ta
                elif t_in_sub < ta + tc:
                    dq_ref = -vk
                    q_ref = (q_init_act - 0.5*vk*ta) - vk*(t_in_sub - ta)
                    ddq_ref = 0.0
                else:
                    t_dec = t_in_sub - (ta + tc)
                    dq_ref = -vk + (vk/ta)*t_dec
                    q_ref = (q_init_act - rk + 0.5*vk*ta) - vk*t_dec + 0.5*(vk/ta)*t_dec**2
                    ddq_ref = vk/ta
            
            elif sub_cycle_idx in [1, 3]: # -R -> +R
                tc = d_f - 2*ta
                if t_in_sub < ta:
                    dq_ref = (vk/ta) * t_in_sub
                    q_ref = (q_init_act - rk) + 0.5*(vk/ta)*t_in_sub**2
                    ddq_ref = vk/ta
                elif t_in_sub < ta + tc:
                    dq_ref = vk
                    q_ref = (q_init_act - rk + 0.5*vk*ta) + vk*(t_in_sub - ta)
                    ddq_ref = 0.0
                else:
                    t_dec = t_in_sub - (ta + tc)
                    dq_ref = vk - (vk/ta)*t_dec
                    q_ref = (q_init_act + rk - 0.5*vk*ta) + vk*t_dec - 0.5*(vk/ta)*t_dec**2
                    ddq_ref = -vk/ta
            
            elif sub_cycle_idx == 2: # +R -> -R
                tc = d_f - 2*ta
                if t_in_sub < ta:
                    dq_ref = -(vk/ta) * t_in_sub
                    q_ref = (q_init_act + rk) - 0.5*(vk/ta)*t_in_sub**2
                    ddq_ref = -vk/ta
                elif t_in_sub < ta + tc:
                    dq_ref = -vk
                    q_ref = (q_init_act + rk - 0.5*vk*ta) - vk*(t_in_sub - ta)
                    ddq_ref = 0.0
                else:
                    t_dec = t_in_sub - (ta + tc)
                    dq_ref = -vk + (vk/ta)*t_dec
                    q_ref = (q_init_act - rk + 0.5*vk*ta) - vk*t_dec + 0.5*(vk/ta)*t_dec**2
                    ddq_ref = vk/ta
            
            else: # 4: +R -> 0
                tc = d_h - 2*ta
                if t_in_sub < ta:
                    dq_ref = -(vk/ta) * t_in_sub
                    q_ref = (q_init_act + rk) - 0.5*(vk/ta)*t_in_sub**2
                    ddq_ref = -vk/ta
                elif t_in_sub < ta + tc:
                    dq_ref = -vk
                    q_ref = (q_init_act + rk - 0.5*vk*ta) - vk*(t_in_sub - ta)
                    ddq_ref = 0.0
                else:
                    t_dec = t_in_sub - (ta + tc)
                    dq_ref = -vk + (vk/ta)*t_dec
                    q_ref = (q_init_act + 0.5*vk*ta) - vk*t_dec + 0.5*(vk/ta)*t_dec**2
                    ddq_ref = vk/ta

            if t_in_sub >= dur:
                sub_cycle_idx += 1
                t_sub_start = t_curr
                if sub_cycle_idx > 4:
                    sub_cycle_idx = 0
                    v_idx += 1
                    if v_idx >= len(V_SCAN_STEPS):
                        sub_stage = "FOURIER"
                    else:
                        sub_stage = "SCAN_CYCLE"
                
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
        dq_step_out.append(dq_step)
        
    return time_vec, np.array(q_out), np.array(dq_out), np.array(ddq_out), np.array(stage_out), np.array(dq_step_out)

if __name__ == "__main__":
    output_path = f"Data/Trajectory_Verification_{JOINT_ACT}.png"
    t, q, dq, ddq, stg, v_step = generate_trajectory(joint_idx=JOINT_ACT)
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1); plt.plot(t, q, 'b'); plt.ylabel("Pos (rad)"); plt.title(f"Joint {JOINT_ACT} Identification Trajectory")
    plt.subplot(4, 1, 2); plt.plot(t, dq, 'g'); plt.ylabel("Vel (rad/s)")
    plt.subplot(4, 1, 3); plt.plot(t, ddq, 'r'); plt.ylabel("Acc (rad/s^2)")
    plt.subplot(4, 1, 4); plt.plot(t, stg, 'k'); plt.ylabel("Stage")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Plot saved to {output_path}")
