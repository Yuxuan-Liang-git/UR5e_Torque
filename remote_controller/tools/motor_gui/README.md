# 达妙电机 DM-J4310-2EC 图形控制界面

## 简介

这是一个用于控制达妙电机 DM-J4310-2EC 的图形化控制界面，通过维特 USB-CAN 适配器进行 CAN 通信。

## 功能特性

- 通过串口连接维特 USB-CAN 适配器
- **支持多种电机型号**（DM-J4310-2EC、DM4340）
- 支持 MIT 控制模式
- 支持梯形加减速运动控制
- 实时显示电机状态（位置、速度、扭矩、温度等）
- 图形化参数配置界面

## 系统要求

- Python 3.7+
- 维特 USB-CAN 适配器
- 达妙电机（支持以下型号）：
  - DM-J4310-2EC（最大扭矩 10 Nm，最大速度 30 rad/s）
  - DM4340（最大扭矩 28 Nm，最大速度 10 rad/s）

## 支持的电机型号对比

| 参数 | DM-J4310-2EC | DM4340 |
|------|--------------|--------|
| 最大位置 (MIT模式) | ±12.5 rad | ±12.5 rad |
| **最大速度** | **30 rad/s** | **10 rad/s** |
| **最大扭矩** | **10 Nm** | **28 Nm** |
| 最大Kp | 500 | 500 |
| 最大Kd | 5 | 5 |
| 特性 | 高速、轻载 | 大扭矩、重载 |

**选择建议**：
- **DM-J4310-2EC**：高速度（30 rad/s）、中等扭矩（10 Nm），适合轻载、高速场景（如小型机械臂、云台、快速运动关节）
- **DM4340**：大扭矩（28 Nm）、低速度（10 rad/s），适合重载、大扭矩场景（如重载机械臂、移动机器人、需要大力矩的关节）

## 安装依赖

### 方式一：使用 pip（推荐用于已有 Python 环境）

在 `tools/motor_gui` 目录下运行：

```bash
pip install -r requirements.txt
```

### 方式二：使用 conda（推荐用于隔离环境）

如果您使用 Anaconda 或 Miniconda，可以创建独立的 conda 环境：

```bash
# 创建名为 motor_gui 的 conda 环境（Python 3.9）
conda create -n motor_gui python=3.9 -y

# 激活环境
conda activate motor_gui

# 安装 pyserial
conda install -c conda-forge pyserial -y

# 或者使用 pip 安装（在 conda 环境中）
pip install -r requirements.txt

# 运行程序
python motor_gui.py

# 使用完毕后退出环境
# conda deactivate
```

**注意**: conda 环境中的 Python 通常已经包含 tkinter，无需额外安装。

### 依赖说明

- **pyserial**: 用于串口通信，与 USB-CAN 适配器进行数据交互
- **tkinter**: Python 内置 GUI 库（某些 Linux 发行版需要单独安装）

### Linux 系统额外安装

在某些 Linux 发行版上，需要单独安装 tkinter：

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Fedora:**
```bash
sudo dnf install python3-tkinter
```

**Arch Linux:**
```bash
sudo pacman -S tk
```

## 快速开始

### 使用 conda 环境（推荐）

```bash
# 1. 进入项目目录
cd /path/to/mockway_robotics/tools/motor_gui

# 2. 创建并激活 conda 环境
conda create -n motor_gui python=3.9 -y
conda activate motor_gui

# 3. 安装依赖
conda install -c conda-forge pyserial -y

# 4. 启动图形界面
python motor_gui.py

# 5. 下次使用时，只需激活环境即可
# conda activate motor_gui
# python motor_gui.py
```

### 使用 pip 环境

```bash
# 1. 进入项目目录
cd /path/to/mockway_robotics/tools/motor_gui

# 2. （可选）创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动图形界面
python motor_gui.py
```

## 使用方法

### 启动图形界面

```bash
python motor_gui.py
```

### 配置参数

1. **连接配置**
   - COM口：设置串口端口号（Windows: COM9, Linux: /dev/ttyUSB0）
   - 串口波特率：默认 921600
   - **电机类型**：选择电机型号（DM-J4310-2EC 或 DM4340）
     - DM-J4310-2EC: 最大扭矩 10 Nm，最大速度 30 rad/s（高速、轻载）
     - DM4340: 最大扭矩 28 Nm，最大速度 10 rad/s（大扭矩、重载）
   - 点击"连接"按钮建立连接

2. **电机参数配置**
   - 电机ID：CAN 总线上的电机 ID（默认 2）
   - Master ID：主机 ID，用于接收反馈帧（默认 0）
   - Kp：位置比例系数（默认 40.0）
   - Kd：位置微分系数（默认 1.0）
   - 最大速度：梯形加减速的最大速度（默认 8.0 rad/s）
   - 最大加速度：梯形加减速的最大加速度（默认 15.0 rad/s²）

3. **电机控制**
   - 连接后点击"使能电机"启用电机
   - 使用"正转"和"反转"按钮控制电机运动
   - 按住按钮：电机加速到目标位置
   - 松开按钮：电机开始减速停止

### 串口设置说明

本程序通过串口与维特 USB-CAN 适配器通信：

- **Windows 平台**: 串口名称通常为 `COM3`, `COM4`, `COM9` 等
- **Linux 平台**: 串口名称通常为 `/dev/ttyUSB0`, `/dev/ttyUSB1` 等
- **macOS 平台**: 串口名称通常为 `/dev/tty.usbserial-XXXX` 或 `/dev/cu.usbserial-XXXX`

#### 查看可用串口

**Linux/macOS:**
```bash
ls /dev/tty*
# 或者
python -m serial.tools.list_ports
```

**Windows:**
在设备管理器中查看端口（COM 和 LPT）

#### Linux 串口权限设置

如果遇到权限问题，需要将当前用户添加到 dialout 组：

```bash
sudo usermod -a -G dialout $USER
```

然后注销并重新登录，或者运行：
```bash
newgrp dialout
```

## 文件说明

- `motor_gui.py`: 图形界面主程序
- `dm_motor_driver.py`: 电机驱动程序，包含 CAN 通信和电机控制逻辑
- `requirements.txt`: Python 依赖包列表
- `README.md`: 本说明文档

## 故障排除

### 无法打开串口

- 检查串口名称是否正确
- 检查 USB-CAN 适配器是否正确连接
- 检查串口权限（Linux 系统）
- 检查是否有其他程序正在占用串口

### 无法接收到电机反馈数据

- 检查 CAN 总线连接是否正常
- 检查电机 ID 和 Master ID 设置是否正确
- 检查电机电源是否正常
- 检查 CAN 波特率设置（默认 1000000）

### 电机不响应

- 确认电机已使能
- 检查电机错误状态
- 检查电机温度是否过高
- 检查电源电压是否正常

