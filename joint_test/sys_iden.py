import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

# --- 滤波器工具函数 ---
def lowpass_filter(data, cutoff=5, fs=500, order=4):
    """
    零相位巴特沃斯低通滤波器
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

# 1. 加载数据
data_path = "Data/Joint6_data.csv"
if not Path(data_path).exists():
    print(f"[ERROR] 数据文件不存在: {data_path}")
    exit(1)

print(f"正在读取数据: {data_path}...")
df = pd.read_csv(data_path)

t = df['time'].values
q_raw = df['q_act'].values
dq_raw = df['dq_act'].values
ddq_raw = df['ddq_act'].values
tau_raw = df['torque'].values
stage = df['stage'].values if 'stage' in df.columns else np.zeros_like(t)

# --- 2. 信号预处理 (滤波) ---
print("[INFO] 正在执行数据滤波 (Cutoff=7Hz)...")
fs = 100
dq = lowpass_filter(dq_raw, cutoff=10, fs=fs)
ddq = lowpass_filter(ddq_raw, cutoff=7, fs=fs)
tau = lowpass_filter(tau_raw, cutoff=10, fs=fs)

# 3. 数据分段逻辑
mask_s1 = (stage == 1)
mask_s2 = (stage == 2)
mask_s3 = (stage == 3)

# --- 步骤 1: 辨识系统偏置 (Offset) 和对称起动转矩 Fs ---
if np.any(mask_s1):
    tau_s1 = tau[mask_s1]
    tp_avg = np.max(tau_s1[tau_s1 > 0]) if np.any(tau_s1 > 0) else 0.0
    tn_avg = np.min(tau_s1[tau_s1 < 0]) if np.any(tau_s1 < 0) else 0.0
    offset_id = (tp_avg + tn_avg) / 2.0
    Fs_id = (tp_avg - tn_avg) / 2.0
    print(f"\n[Stage 1] 偏置标定: Bias={offset_id:.4f}, Fs={Fs_id:.4f}")
else:
    offset_id, Fs_id = 0.0, 1.0

# --- 步骤 2: 辨识 Stribeck 与 库伦模型 ---
tau_clean = tau - offset_id

def stribeck_model(v, Fc, B, vs):
    # 使用 np.abs(v) 确保指数项始终为负，防止负转速时指数爆炸 (适用于各种幂次)
    return (Fc + (Fs_id - Fc) * np.exp(-np.abs(v/vs)**1.5)) * np.sign(v) + B * v

def coulomb_model(v, Fc, B):
    return Fc * np.sign(v) + B * v

dq_err = df['dq_des'].values - df['dq_act'].values
mask_s2_stable = mask_s2 & (np.abs(dq_err) < 0.1)

if np.any(mask_s2_stable):
    dq_s2 = dq[mask_s2_stable]
    tau_s2_clean = tau_clean[mask_s2_stable]
    
    # 拟合 Stribeck
    p_strib, _ = curve_fit(stribeck_model, dq_s2, tau_s2_clean, p0=[Fs_id*0.7, 0.1, 0.05], bounds=([0,0,0.005], [Fs_id, 5, 0.5]))
    Fc_strib, B_strib, vs_strib = p_strib
    
    # 拟合 库伦
    p_coul, _ = curve_fit(coulomb_model, dq_s2, tau_s2_clean, p0=[Fc_strib, B_strib])
    Fc_coul, B_coul = p_coul
    
    print(f"[Stage 2] Stribeck: Fc={Fc_strib:.4f}, B={B_strib:.4f}, vs={vs_strib:.4f}")
    print(f"[Stage 2] Coulomb:  Fc={Fc_coul:.4f}, B={B_coul:.4f}")
else:
    Fc_strib, B_strib, vs_strib = 0.5, 0.1, 0.05
    Fc_coul, B_coul = 0.5, 0.1

# --- 步骤 3: 辨识惯量 J ---
if np.any(mask_s3):
    dq_s3 = dq[mask_s3]
    ddq_s3 = ddq[mask_s3]
    tau_s3_clean = tau[mask_s3] - offset_id
    tau_net = tau_s3_clean - stribeck_model(dq_s3, Fc_strib, B_strib, vs_strib)
    J_id, _, _, _ = np.linalg.lstsq(ddq_s3[np.abs(ddq_s3)>0.5].reshape(-1, 1), tau_net[np.abs(ddq_s3)>0.5], rcond=None)
    J_id = float(J_id[0])
    print(f"[Stage 3] Inertia J: {J_id:.6f}")
else:
    J_id = 0.02

# --- 4. 总体验证与参数保存 ---
tau_total_pred = J_id * ddq + stribeck_model(dq, Fc_strib, B_strib, vs_strib) + offset_id
rmse = np.sqrt(np.mean((tau_raw - tau_total_pred)**2))

# 保存参数
params = {
    "J": J_id,
    "Stribeck": {
        "B": float(B_strib),
        "Fc": float(Fc_strib),
        "Fs": float(Fs_id),
        "vs": float(vs_strib),
    },
    "Coulomb": {
        "B": float(B_coul),
        "Fc": float(Fc_coul),
    },
    "bias": float(offset_id),
    "rmse": float(rmse),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open("Data/identified_params.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False)
print(f"辨识参数已更新至: Data/identified_params.yaml (包含双模型结果)")

# --- 7. 可视化界面 (按照用户要求重构) ---
plt.style.use('bmh') # 使用更美观的绘图风格
fig = plt.figure(figsize=(15, 12))

# 1. 滤波对比图 (展示原始 vs 滤波后的角加速度)
plt.subplot(3, 1, 1)
t_win = (t > 15) & (t < 20) # 截取一段展示细节
if not np.any(t_win): t_win = slice(0, 2000)
plt.plot(t[t_win], ddq_raw[t_win], color='gray', alpha=0.3, label='Raw Acceleration (Noise)')
plt.plot(t[t_win], ddq[t_win], color='red', linewidth=1.5, label='Filtered Acceleration (Clean)')
plt.title("Filtering Performance: Raw vs. Filtered Signal (Acceleration)", fontsize=14)
plt.xlabel("Time [s]")
plt.ylabel("Acc [rad/s^2]")
plt.legend(loc='upper right')

# 2. 摩擦力模型多图合一 (展示 滤波后数据 + Stribeck线 + 库伦线)
plt.subplot(3, 1, 2)
if np.any(mask_s2_stable):
    plt.scatter(dq_s2, tau_s2_clean, s=2, color='darkblue', alpha=0.15, label='Filtered Data Points (Stage 2)')

v_fine = np.linspace(-2.2, 2.2, 500)
plt.plot(v_fine, stribeck_model(v_fine, Fc_strib, B_strib, vs_strib), color='red', linewidth=2.5, label='Stribeck Model Fit')
plt.plot(v_fine, coulomb_model(v_fine, Fc_coul, B_coul), color='black', linestyle='--', linewidth=2, label='Coulomb Model Fit')
plt.axhline(y=Fs_id, color='green', linestyle=':', label=f'Static Friction Fs ({Fs_id:.3f})')
plt.axhline(y=-Fs_id, color='green', linestyle=':')

plt.title("Friction Model Comparison on Filtered Data", fontsize=14)
plt.xlabel("Velocity [rad/s]")
plt.ylabel("Torque (Bias Removed) [N·m]")
plt.legend(loc='lower right')
plt.ylim([-Fs_id*1.5, Fs_id*1.5])

# 3. 全段拟合验证
plt.subplot(3, 1, 3)
tau_pred = J_id * ddq + stribeck_model(dq, Fc_strib, B_strib, vs_strib) + offset_id
plt.plot(t, tau_raw, 'k', alpha=0.2, label='Measured Raw')
plt.plot(t, tau_pred, 'red', alpha=0.8, label='Final Model Prediction')
plt.title(f"Overall Model Validation (RMSE: {np.sqrt(np.mean((tau_raw-tau_pred)**2)):.4f} Nm)", fontsize=14)
plt.xlabel("Time [s]")
plt.ylabel("Torque [N·m]")
plt.legend()

plt.tight_layout()
plt.savefig("Data/Identification_Visual_Report.png")
print("\n[INFO] 增强版可视化报告已保存至: Data/Identification_Visual_Report.png")
plt.show()
