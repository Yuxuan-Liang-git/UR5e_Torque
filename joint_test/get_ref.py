import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(q_init_5=0.0, num_scan_points=10):
    # --- 自动生成速度扫描序列 (平方分布实现低速加密) ---
    v_min, v_max = 0.1, 2.0
    if num_scan_points > 1:
        V_SCAN_STEPS = [v_min + (v_max - v_min) * (i / (num_scan_points - 1))**2 for i in range(num_scan_points)]
    else:
        V_SCAN_STEPS = [v_min]
    
    SCAN_RANGE = 1.2
    MAX_ACC = 1.0 # 提高加速
    
    T_FOURIER = 20.0
    F_BASE = 0.3
    W_BASE = 2 * np.pi * F_BASE
    A_FOUR = np.array([0.20, 0.15, 0.10, 0.05, 0.02])
    B_FOUR = np.array([0.18, 0.12, 0.08, 0.04, 0.01])
    Q_FIX = 0.0
    for l in range(1, 6):
        Q_FIX += B_FOUR[l-1] / (l * W_BASE)

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
    
    T_TOTAL = T_INIT_MOVE + T_SCAN + T_MOVE_TO_ZERO + T_FOURIER + 0.5 # 留 0.5s 余量
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
    q_curr_ref = q_init_5 # 记录当前指令位置以保持连续
    
    def get_smooth_move(t_sub, q_start, q_end, duration):
        if t_sub >= duration:
            return q_end, 0.0, 0.0
        alpha = t_sub / duration
        # 五次多项式平滑轨迹
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
            # 初始移动到起点
            stage = 0
            q_ref, dq_ref, ddq_ref = get_smooth_move(t_in_sub, q_init_5, -SCAN_RANGE, 5.0)
            if t_in_sub >= 5.0:
                sub_stage = "SCAN_CYCLE"
                t_sub_start = t_curr
        
        elif sub_stage == "SCAN_CYCLE":
            stage = 1
            vk = V_SCAN_STEPS[v_idx]
            ta = vk / MAX_ACC
            # 梯形位移公式: Total_S = 2*SCAN_RANGE = vk*ta + vk*tc
            # 因此 tc = (2*SCAN_RANGE)/vk - ta
            tc = (2 * SCAN_RANGE) / vk - ta
            if tc < 0: # 保护: 如果加速度太小导致无法达到目标速度
                tc = 0
                ta = np.sqrt(2 * SCAN_RANGE / MAX_ACC)
                vk = MAX_ACC * ta
                
            dur = 2 * (2 * ta + tc)
            t_in = t_in_sub
            
            if t_in < ta: # 正向加速
                dq_ref = (vk / ta) * t_in
                q_ref = -SCAN_RANGE + 0.5 * (vk / ta) * t_in**2
                ddq_ref = vk / ta
            elif t_in < ta + tc: # 正向匀速 (Region B)
                dq_ref = vk
                q_ref = (-SCAN_RANGE + 0.5 * vk * ta) + vk * (t_in - ta)
                ddq_ref = 0.0
            elif t_in < 2 * ta + tc: # 正向减速
                t_dec = t_in - (ta + tc)
                dq_ref = vk - (vk / ta) * t_dec
                q_ref = (SCAN_RANGE - 0.5 * vk * ta) + vk * t_dec - 0.5 * (vk / ta) * t_dec**2
                ddq_ref = -vk / ta
            elif t_in < 3 * ta + tc: # 反向加速
                t_acc_rev = t_in - (2 * ta + tc)
                dq_ref = -(vk / ta) * t_acc_rev
                q_ref = SCAN_RANGE - 0.5 * (vk / ta) * t_acc_rev**2
                ddq_ref = -vk / ta
            elif t_in < 3 * ta + 2 * tc: # 反向匀速 (Region D)
                dq_ref = -vk
                q_ref = (SCAN_RANGE - 0.5 * vk * ta) - vk * (t_in - (3 * ta + tc))
                ddq_ref = 0.0
            elif t_in < dur: # 反向减速
                t_dec_rev = t_in - (3 * ta + 2 * tc)
                dq_ref = -vk + (vk / ta) * t_dec_rev
                q_ref = (-SCAN_RANGE + 0.5 * vk * ta) - vk * t_dec_rev + 0.5 * (vk / ta) * t_dec_rev**2
                ddq_ref = vk / ta
            
            if t_in >= dur:
                v_idx += 1
                t_sub_start = t_curr
                if v_idx >= len(V_SCAN_STEPS):
                    sub_stage = "MOVE_TO_ZERO"
                else:
                    sub_stage = "SCAN_CYCLE" # 准备下一档速度
        
        elif sub_stage == "MOVE_TO_ZERO":
            stage = 1.5 # 过渡阶段
            q_ref, dq_ref, ddq_ref = get_smooth_move(t_in_sub, -SCAN_RANGE, 0.0, 3.0)
            if t_in_sub >= 3.0:
                sub_stage = "FOURIER"
                t_sub_start = t_curr
                
        elif sub_stage == "FOURIER":
            stage = 2
            tf = t_in_sub
            # 三个正弦信号叠加: q = sum(Ai * sin(wi * t))
            # 频率分别为 f, 2f, 3f
            AMPS = [0.4, 0.2, 0.1] # 幅值分布
            FREQ_MULTS = [1.0, 2.0, 3.0]
            
            q_ref, dq_ref, ddq_ref = 0.0, 0.0, 0.0
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
            stage = 3 # 已完成
            q_ref, dq_ref, ddq_ref = 0.0, 0.0, 0.0
            
        q_curr_ref = q_ref
        q_out.append(q_ref)
        dq_out.append(dq_ref)
        ddq_out.append(ddq_ref)
        stage_out.append(stage)
        
    return time_vec, np.array(q_out), np.array(dq_out), np.array(ddq_out), np.array(stage_out)

if __name__ == "__main__":
    t, q, dq, ddq, stg = generate_trajectory()
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1); plt.plot(t, q, 'b'); plt.ylabel("Pos (rad)"); plt.grid(True)
    plt.subplot(4, 1, 2); plt.plot(t, dq, 'g'); plt.ylabel("Vel (rad/s)"); plt.grid(True)
    plt.subplot(4, 1, 3); plt.plot(t, ddq, 'r'); plt.ylabel("Acc (rad/s^2)"); plt.grid(True)
    plt.subplot(4, 1, 4); plt.plot(t, stg, 'k'); plt.ylabel("Stage"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("Data/Trajectory_Verification_Fixed.png")
    plt.show()
