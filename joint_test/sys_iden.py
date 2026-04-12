import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

# --- 滤波器工具函数 ---
def lowpass_filter(data, cutoff=7, fs=500, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) <= 3 * max(len(a), len(b)): return data
    return filtfilt(b, a, data)

JOINT_ACT = 0  # 要辨识的轴编号 (0-5)
CONFIG_PATH = "Config/Ref.yaml"
# ==========================================

# 1. 加载数据
data_path = f"Data/Data_Joint_{JOINT_ACT}.csv"
if not Path(data_path).exists():
    print(f"[ERROR] 数据文件不存在: {data_path}")
    exit(1)

print(f"正在读取数据: {data_path}...")
df = pd.read_csv(data_path)

# 加载配置以获取 q_init
with open(CONFIG_PATH, 'r') as f:
    ref_cfg = yaml.safe_load(f)
    q_init_all = np.array(ref_cfg.get("q_init", [0, -1.57, 0, -1.57, 0, 0]))
    q_init_val = q_init_all[JOINT_ACT]

t = df['time'].values
q = df['q_act'].values
dq_raw = df['dq_act'].values
dq_ref_raw = df['dq_des'].values # 获取干净的参考转速
# 实测数据通常不直接提供 ddq_act，需通过数值微分计算
print("[INFO] 正在对实际速度进行数值微分以获取加速度...")
# 使用 np.gradient 计算中心差分，比 np.diff 长度对齐更简单
ddq_raw = np.gradient(dq_raw, t)
tau_raw = df['torque'].values
stage = df['stage'].values
dq_step_raw = df['dq_step'].values # 实测名义速度

# --- 2. 预处理 (滤波) ---
print("[INFO] 正在执行数据滤波 (针对数值微分产生的噪声进行强化处理)...")
fs = 500
dq = lowpass_filter(dq_raw, cutoff=12, fs=fs)
# 加速度微分后的噪声通常很大，滤波截止频率设得稍低一点（例如 5Hz）
ddq = lowpass_filter(ddq_raw, cutoff=5, fs=fs)
tau = lowpass_filter(tau_raw, cutoff=12, fs=fs)

# 阶段掩码
mask_s1 = (stage == 1)
mask_s2 = (stage == 2)

# --- 步骤 1: 摩擦力预辨识 (Stage 1: Position-Aligned Extraction) ---
if np.any(mask_s1):
    dq_s1 = dq[mask_s1]
    dq_ref_s1 = dq_ref_raw[mask_s1]
    tau_s1 = tau[mask_s1]
    q_s1 = q[mask_s1]
    ddq_s1 = ddq[mask_s1]
    dq_step_s1 = dq_step_raw[mask_s1]

    # 自适应计算核心探测区
    q_mid = (np.max(q_s1) + np.min(q_s1)) / 2.0
    q_range = np.max(q_s1) - np.min(q_s1)
    q_semi_width = q_range * 0.25 
    
    unique_vs = np.unique(np.abs(dq_step_s1))
    
    print(f"\n[Stage 1] 正在进行 Joint {JOINT_ACT} 摩擦力对冲提取 (基于相对位置 q_rel 镜像匹配)...")
    
    fric_data = [] 
    ACC_THRESHOLD = 2.0
    
    for v_target in unique_vs:
        # 在名义档位内筛选
        mask_v_nominal = (dq_step_s1 == v_target)
        
        # 提取当前档位的有效数据
        df_v = pd.DataFrame({
            'q_rel': q_s1[mask_v_nominal] - q_init_val,
            'dq': dq_s1[mask_v_nominal],
            'tau': tau_s1[mask_v_nominal],
            'ddq': ddq_s1[mask_v_nominal]
        })
        
        # 滤波：稳态转速 + 低加速度
        v_eps = 0.15
        df_pos = df_v[(df_v['dq'] > (v_target - v_eps)) & (df_v['dq'] < (v_target + v_eps)) & (np.abs(df_v['ddq']) < ACC_THRESHOLD)]
        df_neg = df_v[(df_v['dq'] > (-v_target - v_eps)) & (df_v['dq'] < (-v_target + v_eps)) & (np.abs(df_v['ddq']) < ACC_THRESHOLD)]
        
        def extract_friction_mirrored(df_sub):
            if len(df_sub) < 10: return None
            # 将 q_rel 划分为细小的正负对称区间进行对冲
            # 例如划分为 20 个 bin，寻找镜像存在的 pair
            rk_max = np.max(np.abs(df_sub['q_rel']))
            bins = np.linspace(0, rk_max, 20)
            obs = []
            for i in range(len(bins)-1):
                q_min, q_max = bins[i], bins[i+1]
                # 寻找正位置点和负位置点
                mask_p = (df_sub['q_rel'] >= q_min) & (df_sub['q_rel'] < q_max)
                mask_n = (df_sub['q_rel'] <= -q_min) & (df_sub['q_rel'] > -q_max)
                if np.any(mask_p) and np.any(mask_n):
                    tau_p = df_sub.loc[mask_p, 'tau'].mean()
                    tau_n = df_sub.loc[mask_n, 'tau'].mean()
                    # (tau_p + tau_n) / 2 = 摩擦力 (假设重力项大小相等方向相反)
                    obs.append((tau_p + tau_n) / 2.0)
            return np.mean(obs) if obs else None

        f_pos = extract_friction_mirrored(df_pos)
        f_neg = extract_friction_mirrored(df_neg)
        
        if f_pos is not None and f_neg is not None:
            # 最终摩擦力观测值：(f_pos - f_neg) / 2
            tf = (f_pos - f_neg) / 2.0
            fric_data.append([v_target, tf])
            print(f"  [V={v_target:.2f}] 镜像对冲观察值: {tf:7.4f} (基于 {len(df_pos)+len(df_neg)} 样本)")
        elif f_pos is not None:
             fric_data.append([v_target, f_pos])
             print(f"  [V={v_target:.2f}] 单向对冲观察值: {f_pos:7.4f} (只获取到正向数据)")
        elif f_neg is not None:
             fric_data.append([v_target, -f_neg])
             print(f"  [V={v_target:.2f}] 单向对冲观察值: {-f_neg:7.4f} (只获取到负向数据)")
            
    fric_points = np.array(fric_data)
    if len(fric_points) > 0:
        def stribeck_fit_func(v, Fc, B, vs, Fs_fit):
            return (Fc + (Fs_fit - Fc) * np.exp(-(v/vs)**2)) * np.sign(v) + B * v
        
        try:
            # 参考历史版本收紧边界: Fc/Fs 不应超过 2.0 (针对小关节)
            popt, _ = curve_fit(stribeck_fit_func, fric_points[:, 0], fric_points[:, 1], 
                                 p0=[0.5, 0.1, 0.05, 0.6], 
                                 bounds=([0,0,0.005,0], [40.0, 5.0, 3.0, 10.0]))
            Fc_id, B_id, vs_id, Fs_id = popt
            print(f"\n辨识完成 (Joint {JOINT_ACT}): Fs={Fs_id:.4f}, Fc={Fc_id:.4f}, B={B_id:.4f}, vs={vs_id:.4f}")
        except Exception as e:
            print(f"[ERROR] 拟合失败: {e}")
            Fs_id, Fc_id, B_id, vs_id = 1.0, 0.5, 0.1, 0.05
        except Exception as e:
            print(f"[ERROR] 拟合失败: {e}")
            Fs_id, Fc_id, B_id, vs_id = 1.0, 0.5, 0.1, 0.05
    else:
        print("[WARN] 未能在核心区提取到对称摩擦点，请检查摆幅配置。")
        Fs_id, Fc_id, B_id, vs_id = 1.0, 0.5, 0.1, 0.05
else:
    print("[WARN] 无 Stage 1 数据")
    Fs_id, Fc_id, B_id, vs_id = 1.0, 0.5, 0.1, 0.05

# --- 步骤 2: 惯量辨识 (Stage 2: Fourier Excitation) ---
if np.any(mask_s2):
    print(f"\n[Stage 2] 正在进行 Joint {JOINT_ACT} 惯量辨识 (基于傅里叶激励)...")
    dq_s2 = dq[mask_s2]
    ddq_s2 = ddq[mask_s2]
    tau_s2 = tau[mask_s2]
    
    # 扣除摩擦力矩 (使用 Stage 1 的模型)
    def friction_compensate(v):
        return (Fc_id + (Fs_id - Fc_id) * np.exp(-(np.abs(v)/vs_id)**2)) * np.sign(v) + B_id * v
    
    tau_fric = friction_compensate(dq_s2)
    tau_net = tau_s2 - tau_fric
    
    # 回归 J (和潜在的 offset)
    Phi = np.column_stack([ddq_s2, np.ones_like(ddq_s2)])
    theta, _, _, _ = np.linalg.lstsq(Phi, tau_net, rcond=None)
    J_id, bias_id = theta
    print(f"辨识完成 (Joint {JOINT_ACT}): J={J_id:.6f}, Bias={bias_id:.6f}")
else:
    J_id, bias_id = 0.02, 0.0

# 保存结果
save_dir = Path("Config")
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f"joint_{JOINT_ACT}_param.yaml"

params = {
    "joint_id": JOINT_ACT,
    "J": float(J_id),
    "Stribeck": {"B": float(B_id), "Fc": float(Fc_id), "Fs": float(Fs_id), "vs": float(vs_id)},
    "bias": float(bias_id),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(save_path, "w") as f:
    yaml.dump(params, f, default_flow_style=False)
print(f"[INFO] 辨识参数已保存至: {save_path}")

# 绘图
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
if 'fric_points' in locals() and len(fric_points) > 0:
    v_pts = fric_points[:, 0]
    f_pts = fric_points[:, 1]
    plt.scatter(v_pts, f_pts, color='blue', label='Extracted Points')
    plt.scatter(-v_pts, -f_pts, color='cyan', alpha=0.5, label='Mirrored Points')
    
    v_range = np.linspace(-np.max(unique_vs)-0.2, np.max(unique_vs)+0.2, 1000)
    tau_f_range = (Fc_id + (Fs_id - Fc_id) * np.exp(-(v_range/vs_id)**2)) * np.sign(v_range) + B_id * v_range
    plt.plot(v_range, tau_f_range, 'r', lw=2, label='Complete Stribeck Model')
    
plt.axhline(0, color='black', lw=1, ls='--')
plt.axvline(0, color='black', lw=1, ls='--')
plt.title(f"Joint {JOINT_ACT} Friction Identification")
plt.xlabel("Velocity [rad/s]"); plt.ylabel("Torque [Nm]"); plt.legend()

plt.subplot(2, 1, 2)
# 验证绘图
tau_pred = (Fc_id + (Fs_id - Fc_id) * np.exp(-(dq/vs_id)**2)) * np.sign(dq) + B_id * dq + bias_id + J_id * ddq
plt.plot(t, tau, 'k', alpha=0.3, label='Filtered Measured')
plt.plot(t, tau_pred, 'r--', label='Model Prediction')
plt.title(f"Joint {JOINT_ACT} Overall Validation (RMSE: {np.sqrt(np.mean((tau-tau_pred)**2)):.4f} Nm)")
plt.xlabel("Time [s]"); plt.ylabel("Torque [Nm]"); plt.legend()

plt.tight_layout()
report_path = f"Config/Identification_Report_Joint_{JOINT_ACT}.png"
plt.savefig(report_path)
print(f"[INFO] 辨识报告已保存至: {report_path}")
plt.show()
