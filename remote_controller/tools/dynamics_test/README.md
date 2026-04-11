# Mockway Robot - Inverse Dynamics Testing

基于 Pinocchio 库实现的 Mockway 机器人动力学逆解测试工程。

## 概述

本测试工程使用 Pinocchio（一个高效的刚体动力学库）来计算 Mockway 机器人的逆向动力学。

**逆向动力学（Inverse Dynamics）**：给定关节的位置、速度和加速度，计算所需的关节力矩。

数学表达式：
```
τ = M(q)·a + C(q,v)·v + g(q)
```

其中：
- `τ`: 关节力矩
- `M(q)`: 质量矩阵（惯性矩阵）
- `C(q,v)`: 科里奥利和离心力矩阵
- `g(q)`: 重力项
- `q`: 关节位置
- `v`: 关节速度
- `a`: 关节加速度

## 功能特性

本测试程序实现了以下功能：

### 离线仿真测试
1. **静态配置测试**：计算多个静态姿态下的重力补偿力矩
2. **轨迹跟踪测试**：沿正弦轨迹计算所需力矩并可视化
3. **动力学分解测试**：验证 M、C、g 各项的正确性
4. **可视化输出**：生成位置、速度、加速度和力矩的时间曲线图

### 实时力矩补偿控制（新增）
1. **实时重力补偿**：200Hz控制频率的重力力矩补偿
2. **完整动力学补偿**：基于加速度估计的全动力学补偿（惯性+科氏+离心+重力）
3. **加速度估计**：速度差分 + 低通滤波（截止频率可调）
4. **多模式切换**：支持运行时切换补偿模式
5. **配置系统**：支持YAML配置文件和命令行参数覆盖

## 安装依赖

### 方法 1：使用 pip 安装

```bash
cd tools/dynamics_test
pip install -r requirements.txt
```

### 方法 2：使用 conda 安装

```bash
# 创建新环境（推荐）
conda create -n mockway_dynamics python=3.8
conda activate mockway_dynamics

# 安装 Pinocchio
conda install pinocchio -c conda-forge

# 安装其他依赖
pip install numpy matplotlib
```

### 依赖说明

- **pin** (Pinocchio >=3.0.0): 刚体动力学计算库
- **numpy** (>=1.20.0): 数值计算
- **matplotlib** (>=3.3.0): 数据可视化

## 使用方法

### 运行动力学仿真测试

```bash
cd tools/dynamics_test
python inverse_dynamics_test.py
```

这将运行所有三个测试：
1. 静态配置测试
2. 动力学分解测试
3. 轨迹跟踪测试（并生成可视化图表）

### 运行实时力矩补偿控制（新增）

```bash
cd tools/dynamics_test
python realtime_torque_compensation.py
```

**前置要求**：
- 硬件连接：确保 USB-CAN 适配器已连接（默认COM9）
- 电机配置：
  - 关节1：DM-J4310-2EC 电机，CAN ID = 1
  - 关节2：DM4340 电机，CAN ID = 2
- 安全提示：首次运行前请确保机械臂处于安全位置

**功能模式**：
1. **重力补偿**：实时补偿重力力矩，机械臂可手动移动且感觉"失重"
2. **完整动力学补偿**：补偿重力、科里奥利力和离心力
3. **交互式控制**：可实时切换补偿模式和调整参数

### 输出示例

```
============================================================
Mockway Robot - Inverse Dynamics Testing with Pinocchio
============================================================

Model loaded: mockway_description
Number of joints: 2
Joint names: ['joint1', 'joint2']

============================================================
TEST 1: Static Configurations (Zero Velocity & Acceleration)
============================================================

Zero position:
  Joint positions: [0. 0.] deg
  Required torques: [0. 0.] Nm
  Gravity torques:  [0. 0.] Nm

Joint1 = 45°:
  Joint positions: [45. 0.] deg
  Required torques: [0.123 0.456] Nm
  Gravity torques:  [0.123 0.456] Nm

...
```

## 代码结构

```
dynamics_test/
├── inverse_dynamics_test.py         # 动力学仿真测试程序
├── realtime_torque_compensation.py  # 实时力矩补偿控制程序（新增）
├── requirements.txt                 # Python 依赖
├── README.md                        # 本文档
└── trajectory_results.png           # 生成的可视化结果（运行后生成）
```

## 核心 API

### MockwayDynamics 类

```python
class MockwayDynamics:
    def __init__(self, urdf_path)
    def compute_inverse_dynamics(self, q, v, a) -> tau
    def compute_mass_matrix(self, q) -> M
    def compute_coriolis(self, q, v) -> c
    def compute_gravity(self, q) -> g
    def get_end_effector_pose(self, q) -> (position, orientation)
```

### 使用示例

```python
from inverse_dynamics_test import MockwayDynamics
import numpy as np

# 初始化
dynamics = MockwayDynamics("path/to/mockway_description.urdf")

# 定义关节状态
q = np.array([0.5, 0.3])      # 位置 [rad]
v = np.array([0.1, -0.2])     # 速度 [rad/s]
a = np.array([1.0, 0.5])      # 加速度 [rad/s²]

# 计算所需力矩
tau = dynamics.compute_inverse_dynamics(q, v, a)
print(f"Required torques: {tau} Nm")

# 分别计算各项
M = dynamics.compute_mass_matrix(q)
c = dynamics.compute_coriolis(q, v)
g = dynamics.compute_gravity(q)
```

## 测试说明

### Test 1: 静态配置测试

测试在不同关节位置下的静态平衡（零速度、零加速度）。在此情况下，所需力矩应等于重力补偿力矩。

### Test 2: 动力学分解测试

验证逆动力学的正确性，通过对比：
- RNEA 算法（递归牛顿-欧拉算法）的结果
- 手动计算 M·a + C·v + g 的结果

两者应该一致。

### Test 3: 轨迹跟踪测试

沿正弦轨迹运动，计算每个时刻所需的关节力矩，并生成可视化图表：
- 关节位置曲线
- 关节速度曲线
- 关节加速度曲线
- 所需关节力矩曲线

## 参考资料

### Pinocchio 文档
- 官方网站: https://stack-of-tasks.github.io/pinocchio/
- GitHub: https://github.com/stack-of-tasks/pinocchio
- 教程: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/

### 逆向动力学相关
- [Rigid Body Dynamics Algorithms](https://link.springer.com/book/10.1007/978-1-4899-7560-7) by Roy Featherstone
- [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) by Kevin M. Lynch and Frank C. Park

## 常见问题

**Q: 为什么在静态配置下力矩不为零？**

A: 即使机器人静止，重力仍会产生力矩。逆动力学计算的是维持给定状态所需的力矩，包括重力补偿。

**Q: 如何修改测试轨迹？**

A: 在 `test_trajectory_tracking()` 函数中修改 `amplitude` 和 `frequency` 参数即可。

**Q: 可以用于其他机器人吗？**

A: 可以，只需将 URDF 文件路径改为你的机器人模型即可。确保 URDF 包含正确的惯性参数。

**Q: 完整动力学模式下力矩输出抖动怎么办？**

A: 这通常是由于速度测量噪声导致的加速度估计不准确。可以尝试：
   1. 降低滤波器截止频率（如从5Hz降到2-3Hz）
   2. 检查电机速度反馈是否稳定
   3. 确保控制频率足够高（推荐≥200Hz）

**Q: 如何选择合适的滤波器截止频率？**

A: 根据运动速度选择：
   - 慢速示教（< 30°/s）：2-3 Hz，力矩输出更平滑
   - 正常运动（30-90°/s）：5 Hz（默认），平衡响应和噪声
   - 快速运动（> 90°/s）：8-10 Hz，减少相位延迟

   原则：速度越快，截止频率越高；要求越平滑，截止频率越低。

**Q: 为什么重力补偿和完整动力学补偿效果差别不大？**

A: 在慢速运动时，惯性力、科氏力和离心力相对较小，重力占主导。只有在快速运动或频繁加减速时，完整动力学补偿的优势才会明显。

## 实时力矩补偿控制详解

### RealtimeTorqueController 类

**新增功能**：实时动力学力矩补偿控制器

```python
class RealtimeTorqueController:
    def __init__(self, config: DynamicsTestConfig = None)
    def setup()                           # 初始化CAN和电机
    def enable_motors()                   # 使能电机
    def disable_motors()                  # 失能电机
    def start_control(mode="gravity")     # 启动控制循环
    def stop_control()                    # 停止控制循环
    def get_state_snapshot()              # 获取当前状态
    def shutdown()                        # 关闭控制器

    # 内部方法
    def _compute_filter_alpha(cutoff_freq)     # 计算滤波器系数
    def _apply_lowpass_filter(new, old, alpha) # 应用低通滤波
    def compute_compensation_torque(q, v, mode) # 计算补偿力矩（含加速度估计）
```

### 补偿模式说明

1. **"gravity"** - 重力补偿模式
   - 仅补偿重力力矩：`tau = g(q)`
   - 适用场景：手动示教、拖动示教
   - 效果：机械臂感觉"失重"，易于手动移动

2. **"full_dynamics"** - 完整动力学补偿模式
   - 补偿所有动力学项：`tau = M(q)·a + C(q,v)·v + g(q)`
   - 加速度估计：通过速度差分计算 `a = (v - v_prev) / dt`
   - 低通滤波：对加速度进行滤波以减少噪声（截止频率 5Hz）
   - 适用场景：需要精确动力学建模的控制，快速运动时的补偿
   - 效果：补偿重力、科里奥利力、离心力和惯性力

3. **"none"** - 无补偿模式
   - 不提供任何补偿
   - 用于对比测试

### 控制参数

- **control_rate**: 控制频率（默认 200 Hz）
- **kp**: MIT控制模式位置增益（默认 0.0，纯力矩控制）
- **kd**: MIT控制模式阻尼增益（默认 1.0）
- **accel_filter_cutoff**: 加速度低通滤波器截止频率（默认 5.0 Hz）
  - 降低频率（2-3 Hz）：更平滑的滤波，适合慢速运动
  - 提高频率（8-10 Hz）：更快的响应，适合快速运动

### 使用示例

```python
from realtime_torque_compensation import RealtimeTorqueController
from config_loader import load_config
import time

# 方法1：使用默认配置
controller = RealtimeTorqueController()

# 方法2：加载配置文件
config = load_config('config/dynamics_test.yaml')
controller = RealtimeTorqueController(config)

# 方法3：运行时调整滤波器参数
controller = RealtimeTorqueController()
controller._accel_filter_cutoff = 8.0  # 提高截止频率到8Hz
controller._accel_filter_alpha = controller._compute_filter_alpha(8.0)

try:
    # 设置并使能
    controller.setup()
    controller.enable_motors()

    # 启动完整动力学补偿（使用加速度估计）
    controller.start_control(mode="full_dynamics")

    # 运行一段时间
    time.sleep(10.0)

    # 获取当前状态
    state = controller.get_state_snapshot()
    print(f"位置: {state['q']} rad")
    print(f"速度: {state['v']} rad/s")
    print(f"力矩: {state['tau']} Nm")

finally:
    # 关闭
    controller.shutdown()
```

### 安全提示

⚠️ **重要安全注意事项**：

1. **首次运行**：确保机械臂处于安全位置，周围无障碍物
2. **紧急停止**：可随时按 Ctrl+C 停止程序
3. **力矩限制**：电机有力矩限制（Joint1: 9Nm, Joint2: 28Nm）
4. **监控温度**：长时间运行注意监控电机温度
5. **CAN连接**：确保CAN通信稳定，避免通信丢失

### 技术特性

- **实时性**：200Hz 控制频率，满足实时控制需求
- **线程安全**：使用锁机制保护共享状态
- **模块化设计**：动力学计算与电机控制解耦
- **多模式支持**：可在运行时切换补偿模式
- **加速度估计**：基于速度差分的实时加速度估计
- **噪声抑制**：一阶低通滤波器减少测量噪声影响

### 加速度估计与滤波技术

在完整动力学补偿模式下，系统使用以下方法估计关节加速度：

1. **速度差分**
   ```
   a_raw[k] = (v[k] - v[k-1]) / dt
   ```
   其中 `dt` 是控制周期（默认 5ms @ 200Hz）

2. **一阶低通滤波**
   ```
   a_filtered[k] = α · a_raw[k] + (1-α) · a_filtered[k-1]
   ```
   滤波器系数：`α = dt / (RC + dt)`，其中 `RC = 1 / (2π · fc)`

   默认截止频率 `fc = 5 Hz` 对应：
   - 3dB 衰减点在 5Hz
   - 相位延迟约 18ms
   - 有效抑制高频噪声（>20Hz）

3. **参数调优建议**
   - **慢速运动**（< 30°/s）：使用 2-3 Hz 截止频率，获得更平滑的力矩输出
   - **中速运动**（30-90°/s）：使用 5 Hz 截止频率（默认），平衡响应速度和噪声
   - **快速运动**（> 90°/s）：使用 8-10 Hz 截止频率，减少相位延迟

### 动力学补偿效果对比

| 补偿模式 | 补偿内容 | 适用场景 | 计算复杂度 |
|---------|---------|---------|-----------|
| none | 无 | 对比测试 | 无 |
| gravity | 仅重力 g(q) | 手动示教、慢速运动 | 低 |
| full_dynamics | 惯性+科氏+离心+重力 | 快速运动、精确控制 | 中 |

## 许可

本测试工程遵循项目整体许可协议。
