#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025. Li Jianbin. All rights reserved.
# MIT License

"""
Mockway Robot - Real-time Torque Compensation using MuJoCo

This program implements real-time dynamics-based torque compensation for the Mockway robot:
- Uses MuJoCo for gravity torque computation and visualization
- Controls multiple joints using DM motors via CAN (dynamically configured)
- Supports configurable number of motors with individual direction settings
- Provides gravity compensation
"""

import sys
import time
import numpy as np
from pathlib import Path
import threading
import argparse

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Add motor driver to path
sys.path.append(str(Path(__file__).parent.parent / "motor_gui"))
from DM_CAN import MotorControl, DM_Motor_Type as MotorType, Control_Type
from dm_motor_driver import DMMotor

# Import configuration loader
from config_loader import load_config, DynamicsTestConfig, get_default_config, print_config_summary

# Default configuration file paths
DEFAULT_SERIAL_CONFIG_PATH = Path(__file__).parent / "dynamics_test.yaml"


class MujocoInterface:
    """MuJoCo interface for visualization and gravity torque computation"""

    def __init__(self, model_path: str, num_joints: int):
        self.model_path = model_path
        self.num_joints = num_joints
        self.model = None
        self.data = None
        self.viewer = None
        self.enabled = False
        self._lock = threading.Lock()

    def setup(self):
        """Load MuJoCo model and launch passive viewer"""
        if not MUJOCO_AVAILABLE:
            print("警告: 未安装 mujoco Python 包，MuJoCo可视化与重力计算不可用")
            return False

        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)

            if self.model.nq < self.num_joints or self.model.nv < self.num_joints:
                print(f"警告: MuJoCo模型自由度不足 (nq={self.model.nq}, nv={self.model.nv}), 需要至少 {self.num_joints}")
                return False

            # 非阻塞可视化窗口
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.enabled = True
            print("MuJoCo 可视化已启动")
            return True
        except Exception as e:
            print(f"警告: MuJoCo 初始化失败: {e}")
            self.enabled = False
            return False

    def update_state(self, q, v=None):
        """Update MuJoCo state and refresh viewer"""
        if not self.enabled:
            return

        with self._lock:
            n = min(self.num_joints, self.model.nq, len(q))
            self.data.qpos[:n] = q[:n]

            if self.model.nv > 0:
                self.data.qvel[:] = 0.0
                if v is not None:
                    vn = min(self.num_joints, self.model.nv, len(v))
                    self.data.qvel[:vn] = v[:vn]

            mujoco.mj_forward(self.model, self.data)

            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def compute_gravity(self, q):
        """
        Compute gravity torques from MuJoCo bias forces at zero velocity.

        Returns:
            g: gravity torque vector in joint coordinates
        """
        if not self.enabled:
            raise RuntimeError("MuJoCo interface is not enabled")

        with self._lock:
            n = min(self.num_joints, self.model.nq, len(q))
            self.data.qpos[:n] = q[:n]
            if self.model.nv > 0:
                self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return self.data.qfrc_bias[:self.num_joints].copy()

    def close(self):
        """Close MuJoCo viewer"""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self.enabled = False


class RealtimeTorqueController:
    """
    Real-time torque compensation controller for Mockway robot

    Integrates MuJoCo gravity model with DM motor control
    """

    def __init__(self, config: DynamicsTestConfig = None):
        """
        Initialize the real-time controller

        Args:
            config: Configuration object (if None, use default config)
        """
        # Load configuration
        if config is None:
            config = get_default_config()

        self.config = config

        # Hardcoded MuJoCo scene file
        workspace_dir = Path(__file__).parent.parent.parent
        mjcf_path = workspace_dir / "mockway_description/urdf/scene.xml"

        if not mjcf_path.exists():
            raise FileNotFoundError(f"MuJoCo模型文件未找到: {mjcf_path}")

        self.mujoco_interface = MujocoInterface(str(mjcf_path), len(config.motors))

        # Initialize CAN adapter / Serial
        import serial
        # 串口路径 
        actual_port = config.can_port.split(' - ')[0].strip()
        print(f"\n解析串口路径: '{config.can_port}' -> '{actual_port}'")
        print(f"初始化串口: {actual_port}, 波特率: {config.can_serial_baudrate}")
        
        try:
            self.ser = serial.Serial(actual_port, config.can_serial_baudrate, timeout=0.1)
            if not self.ser.is_open:
                raise serial.SerialException(f"无法打开串口 {actual_port}")
            print(f"成功打开串口: {actual_port}")
        except Exception as e:
            print(f"\n\033[91m错误: 无法连接至串口 '{actual_port}'！\033[0m")
            print(f"\033[91m详情: {str(e)}\033[0m")
            print("\033[93m请确保：\n1. 电机控制器的 USB 已经插入\n2. 串口路径在 dynamics_test.yaml 中正确配置\n3. 开发板或 USB-CAN 模块已通电\033[0m")
            sys.exit(1)
            
        self.mc = MotorControl(self.ser)

        # Motor instances (will be initialized in setup)
        self.motors = []  # List[DMMotor] - dynamic list of motors
        self.num_motors = len(config.motors)

        # Motor directions (1=forward, -1=reverse)
        self.motor_directions = np.array([m.direction for m in config.motors])

        # Control parameters
        self.control_rate = config.control_rate  # Hz
        self.dt = 1.0 / self.control_rate

        # Control mode
        self.compensation_mode = config.compensation_mode  # "gravity", "none"

        # MIT control parameters
        self.kp = config.kp  # Position gain (set to 0 for pure torque control)
        self.kd = config.kd  # Damping gain

        # Logging parameters
        self.log_interval = config.log_interval
        self.verbose = config.verbose

        # Control thread
        self._control_thread = None
        self._running = False
        self._stop_event = threading.Event()

        # Current state (dynamic size based on number of motors)
        self._current_q = np.zeros(self.num_motors)
        self._current_v = np.zeros(self.num_motors)
        self._current_tau_cmd = np.zeros(self.num_motors)
        self._state_lock = threading.Lock()

        # Calibration offsets (motor position at joint zero configuration)
        self._joint_offsets = np.zeros(self.num_motors)


    def _unified_can_callback(self, frame_id: int, data: bytes, frame_type: int):
        """
        统一的CAN接收回调函数，将数据分发给所有电机

        Args:
            frame_id: CAN帧ID
            data: CAN数据
            frame_type: 帧类型
        """
        # 将CAN帧分发给所有电机处理
        for motor in self.motors:
            motor._on_can_frame(frame_id, data, frame_type)

    def setup(self):
        """Setup serial connection and motors"""
        print("\n" + "="*60)
        print("设置实时力矩控制系统")
        print("="*60)

        # Initialize motors dynamically from configuration
        print("\n初始化电机...")

        for motor_config in self.config.motors:
            motor = DMMotor(
                MotorType=motor_config.motor_type,
                SlaveID=motor_config.motor_id,
                MasterID=motor_config.master_id
            )
            self.motors.append(motor)
            self.mc.addMotor(motor)
            print(f"  电机 {motor_config.motor_id} ({motor_config.description}): {motor_config.motor_type.name}")

        time.sleep(0.5)
        # 使能所有电机并切换到 MIT 模式
        print("\n使能电机并切换到 MIT 模式...")
        for motor in self.motors:
            self.mc.switchControlMode(motor, Control_Type.MIT)
            self.mc.enable(motor)
            time.sleep(0.1)

        # 检查电机联通性 (等待反馈)
        print("\n验证电机反馈...")
        max_wait = 2.0
        start_time = time.time()
        connected_motors = []
        
        while time.time() - start_time < max_wait:
            for i, motor in enumerate(self.motors):
                if motor not in connected_motors:
                    state = motor.get_state()
                    if state["timestamp"] > 0:
                        connected_motors.append(motor)
            
            if len(connected_motors) == len(self.motors):
                break
            time.sleep(0.1)

        if len(connected_motors) < len(self.motors):
            print(f"\n\033[91m警告: 仅检测到 {len(connected_motors)}/{len(self.motors)} 个电机的反馈！\033[0m")
            for i, motor in enumerate(self.motors):
                if motor not in connected_motors:
                    print(f"  \033[91m- 电机 {motor.SlaveID} 未响应\033[0m")
            
            print("\033[93m请确保：\n1. 电机电源已打开\n2. CAN总线连接正常\n3. 电机ID/主站ID配置正确\033[0m")
            # 允许继续运行以便进行纯动力学仿真测试，或在此处选择 sys.exit
        else:
            print("\033[92m所有电机连接正常，已获取实时反馈数据\033[0m")

        print("\n系统就绪")
        print("\n已设置统一CAN接收回调")

        # 启动 MuJoCo 可视化与重力计算接口（重力补偿依赖）
        if not self._ensure_mujoco_ready():
            raise RuntimeError("MuJoCo 初始化失败，无法进行重力补偿控制")

        # 基于“当前机械臂姿态已与URDF零位对齐”的约定，自动完成零位标定
        self.calibrate_zero_position()

        time.sleep(0.2)
        print(f"\n电机初始化完成，共 {len(self.motors)} 个电机")

    def _ensure_mujoco_ready(self) -> bool:
        """Ensure MuJoCo interface is initialized and ready."""
        if self.mujoco_interface.enabled:
            return True

        print("\n正在加载 MuJoCo...")
        ok = self.mujoco_interface.setup()
        if ok:
            return True

        print("\n\033[91m错误: MuJoCo 未就绪，无法启动重力补偿。\033[0m")
        if not MUJOCO_AVAILABLE:
            print("\033[93m请先安装 mujoco Python 包，例如: pip install mujoco\033[0m")
        else:
            print("\033[93m请检查模型路径、图形环境(DISPLAY)和 MuJoCo 运行依赖\033[0m")
        return False

    def enable_motors(self):
        """Enable all motors"""
        print("\n使能电机...")
        for motor in self.motors:
            self.mc.enable(motor)
            time.sleep(0.1)
        time.sleep(0.2)

        # Wait for feedback
        max_wait = 2.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            all_ready = True
            for i, motor in enumerate(self.motors):
                state = motor.get_state()
                if state["timestamp"] <= 0:
                    all_ready = False
                    break

            if all_ready:
                for i, motor in enumerate(self.motors):
                    state = motor.get_state()
                    print(f"电机{i+1}位置: {state['position']:.4f} rad")
                break
            time.sleep(0.01)
        else:
            print("警告: 未能接收到电机反馈数据")

        print("电机已使能")

    def disable_motors(self):
        """Disable all motors"""
        print("\n失能电机...")
        for motor in self.motors:
            if motor:
                self.mc.disable(motor)
        time.sleep(0.1)
        print("电机已失能")

    def _get_current_joint_raw_state(self):
        """
        Get current joint state in JOINT coordinates without offset compensation.

        Returns:
            q_raw: Joint positions (num_motors,)
            v_raw: Joint velocities (num_motors,)
        """
        q_raw = np.zeros(self.num_motors)
        v_raw = np.zeros(self.num_motors)

        for i, motor in enumerate(self.motors):
            state = motor.get_state()
            q_raw[i] = state["position"] * self.motor_directions[i]
            v_raw[i] = state["velocity"] * self.motor_directions[i]

        return q_raw, v_raw

    def get_current_state(self):
        """
        Get current joint state from motors

        Returns:
            q: Joint positions (num_motors,) in JOINT coordinates
            v: Joint velocities (num_motors,) in JOINT coordinates
        """
        q_raw, v_raw = self._get_current_joint_raw_state()
        q = q_raw - self._joint_offsets
        return q, v_raw

    def compute_compensation_torque(self, q, v, mode="gravity"):
        """
        Compute compensation torque based on mode

        Args:
            q: Joint positions (num_motors,)
            v: Joint velocities (num_motors,)
            mode: Compensation mode ("gravity", "none")

        Returns:
            tau: Compensation torque (num_motors,)
        """
        _ = v  # Reserved for future use

        if mode == "none":
            return np.zeros(self.num_motors)
        if mode == "gravity":
            if self.mujoco_interface.enabled:
                return self.mujoco_interface.compute_gravity(q)
            raise RuntimeError("MuJoCo 未就绪，无法计算重力补偿力矩")

        raise ValueError(f"Unknown compensation mode: {mode}")

    def send_torque_command(self, tau):
        """
        Send torque commands to motors via MIT control mode

        Args:
            tau: Joint torques (num_motors,) in Nm (in joint coordinates)
        """
        # First collect all motor states to avoid potential reference issues
        states = []
        for motor in self.motors:
            states.append(motor.get_state())

        # Extract raw motor positions
        q_motor = np.zeros(self.num_motors)
        for i, state in enumerate(states):
            q_motor[i] = state["position"]  # Raw motor position (not converted)

        # Send MIT control commands to all motors
        # Using kp=0, kd>0 for damping, and t_ff for torque command
        for i, motor in enumerate(self.motors):
            # 下发重力补偿力矩（取反用于抵消）
            motor_torque = -tau[i] * self.motor_directions[i]
            
            # 调试打印
            if self.verbose and i == 1:
                 print(f"J2 TargetJointTau: {tau[i]:.4f} Nm, CommandMotorTau: {motor_torque:.4f} Nm")

            self.mc.controlMIT(
                DM_Motor=motor,
                kp=self.kp,
                kd=self.kd,
                q=q_motor[i],  # 使用当前电机位置作为期望位置
                dq=0.0,
                tau=motor_torque  # 下发抵消力矩
            )

    def calibrate_zero_position(self):
        """
        Calibrate joint zero positions based on current motor positions.
        Assume current physical pose is aligned with URDF initial configuration.
        """
        print("\n" + "="*60)
        print("关节零点确认 (Calibration)")
        print("="*60)
        print("按当前姿态标定为 URDF 初始姿态")
        
        # Wait for stable feedback
        time.sleep(1.0)
        
        # 使用未补偿的原始关节读数做标定，避免受历史offset影响
        q_joint_raw, _ = self._get_current_joint_raw_state()
        urdf_initial_q = np.zeros(self.num_motors)
        if self.num_motors >= 2:
            # 当前机械臂初始时，Joint2 相对于零位应为 +90°
            urdf_initial_q[1] = np.pi / 2.0
        self._joint_offsets = q_joint_raw - urdf_initial_q
        
        print("\n标定结果:")
        for i, offset in enumerate(self._joint_offsets):
            print(f"  关节 {i+1} offset: {offset:.4f} rad, 目标URDF初始角: {urdf_initial_q[i]:.4f} rad")
        print("="*60 + "\n")

    def _control_loop(self):
        """Main control loop running at specified rate"""
        print(f"\n控制循环启动 (频率: {self.control_rate} Hz)")

        loop_count = 0
        last_print_time = time.time()

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                # 获取经过偏移修正后的关节空间坐标 q, v
                q, v = self.get_current_state()

                # 更新 MuJoCo 可视化状态
                self.mujoco_interface.update_state(q, v)

                # 计算补偿力矩
                tau = self.compute_compensation_torque(q, v, self.compensation_mode)

                # 下发力矩命令
                self.send_torque_command(-tau)

                # 定期打印当前识别到的姿态
                if self.verbose and time.time() - last_print_time > self.log_interval:
                    print(f"当前关节角度 q: {np.round(np.rad2deg(q), 1)} deg")
                    last_print_time = time.time()

                # Update internal state
                with self._state_lock:
                    self._current_q = q
                    self._current_v = v
                    self._current_tau_cmd = tau

                # Print status periodically
                loop_count += 1
                if time.time() - last_print_time >= self.log_interval:
                    q_str = ', '.join([f"{qi:6.3f}" for qi in q])
                    v_str = ', '.join([f"{vi:6.3f}" for vi in v])
                    tau_str = ', '.join([f"{ti:6.3f}" for ti in tau])
                    print(f"\r位置: [{q_str}] rad  "
                          f"速度: [{v_str}] rad/s  "
                          f"力矩: [{tau_str}] Nm", end='')
                    last_print_time = time.time()

            except Exception as e:
                print(f"\n控制循环错误: {e}")
                break

            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -self.dt:
                print(f"\n警告: 控制循环超时 ({elapsed*1000:.1f} ms)")

        print("\n控制循环已停止")

    def start_control(self, mode="gravity"):
        """
        Start the control loop

        Args:
            mode: Compensation mode ("gravity", "none")
        """
        if self._running:
            print("控制循环已在运行")
            return

        self.compensation_mode = mode
        if mode == "gravity" and not self._ensure_mujoco_ready():
            print("未启动控制循环")
            return

        print(f"\n启动力矩补偿控制 (模式: {mode})")

        self._running = True
        self._stop_event.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def stop_control(self):
        """Stop the control loop"""
        if not self._running:
            return

        print("\n\n停止控制循环...")
        self._stop_event.set()

        if self._control_thread:
            self._control_thread.join(timeout=2.0)

        self._running = False

    def get_state_snapshot(self):
        """Get current state snapshot (thread-safe)"""
        with self._state_lock:
            return {
                'q': self._current_q.copy(),
                'v': self._current_v.copy(),
                'tau': self._current_tau_cmd.copy()
            }

    def shutdown(self):
        """Shutdown the controller"""
        print("\n" + "="*60)
        print("关闭控制器")
        print("="*60)

        # Stop control loop
        self.stop_control()

        # Disable motors
        self.disable_motors()

        # Close Serial connection
        time.sleep(0.2)
        if hasattr(self, 'ser'):
            self.ser.close()
        self.mujoco_interface.close()

        print("控制器已关闭")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Mockway Robot - Real-time Torque Compensation Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置文件
  python realtime_torque_compensation.py

  # 指定配置文件
  python realtime_torque_compensation.py --config /path/to/config.yaml

  # 覆盖CAN端口
  python realtime_torque_compensation.py --can-port COM3

  # 覆盖补偿模式
  python realtime_torque_compensation.py --mode gravity

  # 直接运行演示（跳过菜单）
  python realtime_torque_compensation.py --demo gravity
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help=f'串口/电机配置文件路径 (默认: {DEFAULT_SERIAL_CONFIG_PATH})'
    )

    parser.add_argument(
        '--can-port',
        type=str,
        default=None,
        help='CAN适配器串口 (覆盖配置文件)'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['gravity', 'none'],
        default=None,
        help='补偿模式 (覆盖配置文件)'
    )

    parser.add_argument(
        '--control-rate',
        type=int,
        default=None,
        help='控制频率 Hz (覆盖配置文件)'
    )

    parser.add_argument(
        '--demo',
        type=str,
        choices=['gravity', 'comparison', 'interactive'],
        default=None,
        help='直接运行指定演示'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )

    return parser.parse_args()


def demo_gravity_compensation(config=None):
    """Demonstrate gravity compensation"""
    if config is None:
        config = get_default_config()

    print("\n" + "="*60)
    print("演示: 重力补偿模式")
    print("="*60)
    print("机器人将保持当前位置，仅补偿重力")
    print("提示: 启动前请确保机械臂处于 URDF 定义的零位姿(Zero Configuration)")
    print("="*60)

    controller = RealtimeTorqueController(config)

    try:
        # Setup
        controller.setup()
        controller.enable_motors()

        # Start gravity compensation
        controller.start_control(mode="gravity")

        # Run for a period
        print("\n按 Ctrl+C 停止...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n用户中断")

    finally:
        controller.shutdown()


def demo_comparison(config=None):
    """Compare different compensation modes"""
    if config is None:
        config = get_default_config()

    print("\n" + "="*60)
    print("演示: 补偿模式对比")
    print("="*60)

    controller = RealtimeTorqueController(config)

    try:
        # Setup
        controller.setup()
        controller.enable_motors()

        modes = ["none", "gravity"]
        duration = 10.0  # seconds per mode

        for mode in modes:
            print(f"\n\n{'='*60}")
            print(f"测试模式: {mode}")
            print(f"持续时间: {duration} 秒")
            print(f"{'='*60}")

            controller.start_control(mode=mode)

            time.sleep(duration)

            controller.stop_control()

            # Get final state
            state = controller.get_state_snapshot()
            print(f"\n最终状态:")
            print(f"  位置: {state['q']} rad")
            print(f"  速度: {state['v']} rad/s")
            # print(f"  加速度: {state['a']} rad/s²")
            print(f"  力矩: {state['tau']} Nm")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        controller.shutdown()


def interactive_mode(config=None):
    """Interactive control mode"""
    if config is None:
        config = get_default_config()

    print("\n" + "="*60)
    print("Mockway机器人 - 实时力矩补偿控制")
    print("="*60)

    controller = RealtimeTorqueController(config)

    try:
        # Setup
        controller.setup()
        controller.enable_motors()

        while True:
            print("\n" + "="*60)
            print("请选择操作:")
            print("1. 启动重力补偿")
            print("2. 停止补偿")
            print("3. 查看当前状态")
            print("4. 调整控制参数")
            print("0. 退出")
            print("="*60)

            choice = input("\n请输入选择 (0-4): ").strip()

            if choice == '1':
                controller.start_control(mode="gravity")
                print("\n重力补偿已启动")
                print("按任意键返回菜单...")
                input()

            elif choice == '2':
                controller.stop_control()
                print("\n补偿已停止")

            elif choice == '3':
                state = controller.get_state_snapshot()
                print(f"\n当前状态:")
                print(f"  关节位置: {np.rad2deg(state['q'])} deg")
                print(f"  关节速度: {state['v']} rad/s")
                print(f"  补偿力矩: {state['tau']} Nm")
                print(f"\n控制参数:")
                print(f"  补偿模式: {controller.compensation_mode}")
                print(f"  控制频率: {controller.control_rate} Hz")
                print(f"  Kp: {controller.kp}, Kd: {controller.kd}")

            elif choice == '4':
                print(f"\n当前参数:")
                print(f"  Kp: {controller.kp}")
                print(f"  Kd: {controller.kd}")
                print(f"  控制频率: {controller.control_rate} Hz")

                try:
                    kp_new = float(input(f"输入新的Kp (当前: {controller.kp}, 回车跳过): ").strip() or controller.kp)
                    kd_new = float(input(f"输入新的Kd (当前: {controller.kd}, 回车跳过): ").strip() or controller.kd)

                    controller.kp = kp_new
                    controller.kd = kd_new
                    print("参数已更新")
                except ValueError:
                    print("输入无效，参数未改变")

            elif choice == '0':
                break

            else:
                print("无效选择")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        controller.shutdown()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration (priority: command line > config file > default)
    try:
        if args.config:
            print(f"加载配置文件: {args.config}")
            config = load_config(args.config)
        else:
            config_path = DEFAULT_SERIAL_CONFIG_PATH
            if config_path.exists():
                print(f"使用默认配置文件: {config_path}")
                config = load_config(str(config_path))
            else:
                print(f"配置文件不存在，使用内置默认配置")
                config = get_default_config()

        # Command line arguments override config file
        if args.can_port:
            config.can_port = args.can_port
            print(f"CAN端口被命令行参数覆盖: {args.can_port}")

        if args.mode:
            config.compensation_mode = args.mode
            print(f"补偿模式被命令行参数覆盖: {args.mode}")

        if args.control_rate:
            config.control_rate = args.control_rate
            print(f"控制频率被命令行参数覆盖: {args.control_rate} Hz")

        if args.verbose:
            config.verbose = True

        # Print configuration summary
        print_config_summary(config)

    except Exception as e:
        print(f"配置加载失败: {e}")
        print("使用内置默认配置")
        config = get_default_config()

    # Run demo or interactive mode based on arguments
    try:
        if args.demo:
            # Direct demo mode (skip menu)
            if args.demo == 'gravity':
                demo_gravity_compensation(config)
            elif args.demo == 'comparison':
                demo_comparison(config)
            elif args.demo == 'interactive':
                interactive_mode(config)
        else:
            # Interactive menu mode
            print("\n" + "="*60)
            print("Mockway机器人 - 实时力矩补偿系统")
            print("="*60)
            print("\n请选择演示模式:")
            print("1. 重力补偿演示")
            print("2. 补偿模式对比")
            print("3. 交互式控制")
            print("0. 退出")

            choice = input("\n请输入选择 (0-3): ").strip()

            if choice == '1':
                demo_gravity_compensation(config)
            elif choice == '2':
                demo_comparison(config)
            elif choice == '3':
                interactive_mode(config)
            elif choice == '0':
                print("退出")
            else:
                print("无效选择")

    except KeyboardInterrupt:
        print("\n\n退出")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
