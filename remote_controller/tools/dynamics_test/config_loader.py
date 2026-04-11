#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025. Li Jianbin. All rights reserved.
# MIT License

"""
Mockway Robot - Configuration Loader for Dynamics Test

This module provides configuration loading and validation for the real-time
torque compensation control system.
"""

import sys
import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Add motor driver to path
sys.path.append(str(Path(__file__).parent.parent / "motor_gui"))
from DM_CAN import DM_Motor_Type as MotorType


# Motor type string to enum mapping
MOTOR_TYPE_MAP = {
    "DM_J4310_2EC": MotorType.DM4310,
    "DM4340": MotorType.DM4340,
}


@dataclass
class MotorConfig:
    """单个电机的配置"""
    motor_id: int
    motor_type: MotorType
    master_id: int
    description: str = ""
    direction: int = 1  # 电机旋转方向: 1=正向(与关节同向), -1=反向(与关节反向)


@dataclass
class DynamicsTestConfig:
    """完整的动力学测试配置"""
    can_port: str
    can_serial_baudrate: int
    can_baudrate: int
    motors: List[MotorConfig]
    control_rate: int
    compensation_mode: str
    kp: float
    kd: float
    log_interval: float
    verbose: bool


def motor_type_from_string(type_str: str) -> MotorType:
    """
    将字符串转换为 MotorType 枚举

    Args:
        type_str: 电机型号字符串

    Returns:
        MotorType 枚举值

    Raises:
        ValueError: 如果电机型号无效
    """
    if type_str not in MOTOR_TYPE_MAP:
        available_types = ", ".join(MOTOR_TYPE_MAP.keys())
        raise ValueError(
            f"无效的电机型号: '{type_str}'\n"
            f"可用的电机型号: {available_types}"
        )
    return MOTOR_TYPE_MAP[type_str]


def get_default_config() -> DynamicsTestConfig:
    """
    返回硬编码的默认配置（向后兼容）

    Returns:
        DynamicsTestConfig: 默认配置对象
    """
    # 默认电机配置（3个电机）
    default_motors = [
        MotorConfig(
            motor_id=1,
            motor_type=MotorType.DM4310,
            master_id=0,
            description="Joint 1 - Shoulder",
            direction=1
        ),
        MotorConfig(
            motor_id=2,
            motor_type=MotorType.DM4340,
            master_id=0,
            description="Joint 2 - Elbow",
            direction=1
        ),
        MotorConfig(
            motor_id=3,
            motor_type=MotorType.DM4340,
            master_id=0,
            description="Joint 3 - Forearm",
            direction=-1
        ),
    ]

    return DynamicsTestConfig(
        can_port="COM9",
        can_serial_baudrate=921600,
        can_baudrate=1000000,
        motors=default_motors,
        control_rate=200,
        compensation_mode="gravity",
        kp=0.0,
        kd=0.005,
        log_interval=0.5,
        verbose=False
    )


def validate_config(config: DynamicsTestConfig) -> List[str]:
    """
    验证配置有效性

    Args:
        config: 配置对象

    Returns:
        错误消息列表（如果为空则配置有效）
    """
    errors = []

    # 验证电机数量
    if len(config.motors) == 0:
        errors.append("至少需要配置一个电机")

    # 验证电机ID唯一性
    motor_ids = [m.motor_id for m in config.motors]
    if len(motor_ids) != len(set(motor_ids)):
        errors.append("电机ID必须唯一")

    # 验证控制频率
    if config.control_rate <= 0:
        errors.append(f"控制频率必须大于0 (当前: {config.control_rate})")

    # 验证补偿模式
    valid_modes = ["gravity", "full_dynamics", "none"]
    if config.compensation_mode not in valid_modes:
        errors.append(
            f"无效的补偿模式: '{config.compensation_mode}' "
            f"(可用: {', '.join(valid_modes)})"
        )

    # 验证控制参数
    if config.kp < 0:
        errors.append(f"kp必须非负 (当前: {config.kp})")

    if config.kd < 0:
        errors.append(f"kd必须非负 (当前: {config.kd})")

    # 验证日志间隔
    if config.log_interval <= 0:
        errors.append(f"日志间隔必须大于0 (当前: {config.log_interval})")

    return errors


def load_config(config_path: Optional[str] = None) -> DynamicsTestConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径（如果为None，使用默认路径）

    Returns:
        DynamicsTestConfig: 配置对象

    Raises:
        FileNotFoundError: 如果指定的配置文件不存在
        yaml.YAMLError: 如果YAML格式错误
        ValueError: 如果配置验证失败
    """
    # 如果没有指定路径，使用默认路径
    if config_path is None:
        config_path = Path(__file__).parent / "dynamics_test.yaml"
    else:
        config_path = Path(config_path)

    # 检查文件是否存在
    if not config_path.exists():
        print(f"警告: 配置文件不存在: {config_path}")
        print("使用默认配置")
        return get_default_config()

    # 读取并解析YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"错误: YAML格式错误")
        print(f"  文件: {config_path}")
        print(f"  错误: {e}")
        print("\n使用默认配置")
        return get_default_config()
    except Exception as e:
        print(f"错误: 无法读取配置文件: {e}")
        print("使用默认配置")
        return get_default_config()

    # 解析配置
    try:
        # CAN配置
        can_config = yaml_data.get('can', {})
        can_port = can_config.get('port', 'COM9')
        can_serial_baudrate = can_config.get('serial_baudrate', 921600)
        can_baudrate = can_config.get('can_baudrate', 1000000)

        # 电机配置
        motors_data = yaml_data.get('motors', [])
        motors = []
        for motor_data in motors_data:
            motor = MotorConfig(
                motor_id=motor_data.get('id'),
                motor_type=motor_type_from_string(motor_data.get('type')),
                master_id=motor_data.get('master_id', 0),
                description=motor_data.get('description', ''),
                direction=motor_data.get('direction', 1)  # 默认为1(正向)
            )
            motors.append(motor)

        # 控制配置
        control_config = yaml_data.get('control', {})
        control_rate = control_config.get('rate', 200)
        compensation_mode = control_config.get('compensation_mode', 'gravity')

        # MIT参数
        mit_params = control_config.get('mit_params', {})
        kp = mit_params.get('kp', 0.0)
        kd = mit_params.get('kd', 1.0)

        # 日志配置
        logging_config = yaml_data.get('logging', {})
        log_interval = logging_config.get('print_status_interval', 0.5)
        verbose = logging_config.get('verbose', False)

        # 创建配置对象
        config = DynamicsTestConfig(
            can_port=can_port,
            can_serial_baudrate=can_serial_baudrate,
            can_baudrate=can_baudrate,
            motors=motors,
            control_rate=control_rate,
            compensation_mode=compensation_mode,
            kp=kp,
            kd=kd,
            log_interval=log_interval,
            verbose=verbose
        )

        # 验证配置
        errors = validate_config(config)
        if errors:
            print(f"错误: 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            print("\n使用默认配置")
            return get_default_config()

        return config

    except ValueError as e:
        print(f"错误: 配置解析失败: {e}")
        print("使用默认配置")
        return get_default_config()
    except KeyError as e:
        print(f"错误: 缺少必需的配置字段: {e}")
        print("使用默认配置")
        return get_default_config()
    except Exception as e:
        print(f"错误: 加载配置时发生未知错误: {e}")
        print("使用默认配置")
        return get_default_config()


def print_config_summary(config: DynamicsTestConfig):
    """
    打印配置摘要

    Args:
        config: 配置对象
    """
    print("\n" + "="*60)
    print("配置摘要")
    print("="*60)
    print(f"CAN端口: {config.can_port}")
    print(f"CAN串口波特率: {config.can_serial_baudrate}")
    print(f"CAN总线波特率: {config.can_baudrate}")
    print(f"\n电机数量: {len(config.motors)}")
    for i, motor in enumerate(config.motors, 1):
        print(f"  电机{i}:")
        print(f"    CAN ID: {motor.motor_id}")
        print(f"    型号: {motor.motor_type.name}")
        print(f"    主机ID: {motor.master_id}")
        direction_str = "正向(与关节同向)" if motor.direction == 1 else "反向(与关节反向)"
        print(f"    旋转方向: {motor.direction} ({direction_str})")
        if motor.description:
            print(f"    描述: {motor.description}")
    print(f"\n控制频率: {config.control_rate} Hz")
    print(f"补偿模式: {config.compensation_mode}")
    print(f"MIT参数: kp={config.kp}, kd={config.kd}")
    print(f"日志间隔: {config.log_interval} 秒")
    print(f"详细输出: {config.verbose}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """测试配置加载"""
    print("测试配置加载模块\n")

    # 测试默认配置
    print("1. 测试默认配置:")
    default_config = get_default_config()
    print_config_summary(default_config)

    # 测试加载配置文件
    print("\n2. 测试加载配置文件:")
    config_path = Path(__file__).parent / "dynamics_test.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
        print_config_summary(config)
    else:
        print(f"配置文件不存在: {config_path}")

    print("\n配置加载模块测试完成")
