# MY_JOINT — 单关节/单电机 SINDy 系统辨识

## 概述

本项目将 **SINDy (Sparse Identification of Nonlinear Dynamics)** 应用于 **单电机 / 单关节** 的辨识动力学。本模块最初为 `MY_MANIPULATOR` (多自由度机械臂)，现已简化和重命名为 `MY_JOINT`。

## 模型说明

### 状态空间

| 类别 | 变量 | 维度 | 说明 |
|------|------|------|------|
| 关节角度 | q | 1 | 电机旋转角度 |
| 关节角速度 | dq | 1 | 对应角度的时间导数 |
| **总状态** | **x** | **2** | `[q, dq]` |
| 控制输入 | τ | 1 | 关节力矩 [Nm] |

### 文件结构

```
MY_JOINT/
├── EX_JOINT_SI_SINDYc.m         # 主脚本 (入口)
├── JointSys.m                   # 单电机 / 关节真实动力学模型
├── getTrainingData.m            # 训练数据生成
├── trainSINDYc_Joint.m          # SINDy 辨识训练
├── VIZ_SI_Validation_Joint.m    # 结果可视化
└── README.md                    # 本文件
```

## 使用方法

### 快速开始

```matlab
% 在 MATLAB 中，进入 MY_JOINT 目录
cd MY_JOINT

% 运行主脚本
EX_JOINT_SI_SINDYc
```
