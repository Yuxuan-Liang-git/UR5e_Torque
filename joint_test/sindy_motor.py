import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sindy_utils import pool_data_motor, sparsify_dynamics, get_library_labels
from scipy.integrate import solve_ivp

# 1. 配置文件路径
data_path = "Data/Joint6_data.csv"

# 2. 加载数据
print(f"正在读取数据: {data_path}...")
df = pd.read_csv(data_path)

# 提取关键列
t = df['time'].values
q = df['q_act'].values
dq = df['dq_act'].values
ddq = df['ddq_act'].values
tau = df['torque'].values

# 计算时间步长 (检查是否均匀)
dt_seq = np.diff(t)
dt_avg = np.mean(dt_seq)
print(f"数据点数量: {len(t)}")
print(f"平均时间步长: {dt_avg:.6f} s (约 {1/dt_avg:.1f} Hz)")

# 3. 准备 SINDy 输入
# 状态 X = [q, dq], 控制 U = [tau]
# 目标 dX/dt = [dq, ddq]
X = np.column_stack([q, dq])
U = tau.reshape(-1, 1)
dXdt = np.column_stack([dq, ddq])

# 候选函数库输入: [q, dq, tau]
yin = np.column_stack([q, dq, tau])

# 4. 执行辨识
# 根据测试结果调整阈值
poly_order = 1
lambdas = [0.01, 0.01] 
n_states = 2

# 从辨识结果中尝试读取 vs (Stribeck velocity)
param_file = Path("Data/identified_params.yaml")
vs_val = 0.05 # 默认值
if param_file.exists():
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
        vs_val = params.get("vs", 0.05)
        print(f"[INFO] Using identified Stribeck vs: {vs_val:.4f} for SINDy library.")

print("\n开始 SINDy 辨识...")
# 注意：此处的 pool_data_motor 内部已更新，包含特定 vs 的 Stribeck 项
Theta = pool_data_motor(yin, poly_order=poly_order, include_sign=True)
Xi = sparsify_dynamics(Theta, dXdt, lambdas, n_states)

# 打印辨识出的方程
lib_labels = get_library_labels(['q', 'dq', 'tau'], poly_order, include_sign=True)
state_names = ['q_dot', 'dq_dot']

print("\n辨识出的动力学方程:")
for i in range(n_states):
    eq = f"{state_names[i]} = "
    terms = []
    for j in range(len(lib_labels)):
        if abs(Xi[j, i]) > 1e-6:
            terms.append(f"({Xi[j, i]:.6f}) * {lib_labels[j]}")
    if not terms:
        eq += "0"
    else:
        eq += " + ".join(terms)
    print(eq)

# 5. 模型验证 (预测对比)
dXdt_pred = Theta @ Xi

# 计算 RMSE
rmse = np.sqrt(np.mean((dXdt - dXdt_pred)**2, axis=0))
print(f"\n预测 RMSE: q_dot={rmse[0]:.6f}, dq_dot={rmse[1]:.6f}")

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, dXdt[:, 0], 'k', label='Measured dq')
plt.plot(t, dXdt_pred[:, 0], 'r--', label='SINDy pred dq')
plt.title("Velocity Prediction (q_dot)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, dXdt[:, 1], 'k', label='Measured ddq')
plt.plot(t, dXdt_pred[:, 1], 'r--', label='SINDy pred ddq')
plt.title("Acceleration Prediction (dq_dot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Data/SINDy_Prediction_Comparison.png")
print("\n预测对比图已保存至: Data/SINDy_Prediction_Comparison.png")

# 6. 数值仿真验证 (前向积分)
print("\n开始数值仿真验证...")
def motor_dynamics(t_val, x_val, t_data, u_data, Xi, poly_order):
    # 插值获取当前时刻的控制量 tau
    tau_val = np.interp(t_val, t_data, u_data)
    
    # 构造库输入 [q, dq, tau]
    curr_yin = np.array([[x_val[0], x_val[1], tau_val]])
    
    # 生成库
    # 注意: pool_data_motor 期望 2D 数组
    theta_curr = pool_data_motor(curr_yin, poly_order=poly_order, include_sign=True)
    
    # 计算导数
    dxdt = theta_curr @ Xi
    return dxdt.flatten()

# 仅仿真一段数据，避免时间过长
sim_len = min(2000, len(t)) 
t_sim = t[:sim_len]
x0 = X[0, :]

sol = solve_ivp(motor_dynamics, [t_sim[0], t_sim[-1]], x0, 
                t_eval=t_sim, args=(t, tau, Xi, poly_order),
                method='RK45')

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_sim, X[:sim_len, 0], 'k', label='Measured q')
plt.plot(sol.t, sol.y[0, :], 'b--', label='Simulated q')
plt.title("Position Trajectory Comparison")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_sim, X[:sim_len, 1], 'k', label='Measured dq')
plt.plot(sol.t, sol.y[1, :], 'b--', label='Simulated dq')
plt.title("Velocity Trajectory Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Data/SINDy_Simulation_Validation.png")
print("仿真验证图已保存至: Data/SINDy_Simulation_Validation.png")

plt.show()
