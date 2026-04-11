#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025. Li Jianbin. All rights reserved.
# MIT License

"""
达妙电机 DM-J4310-2EC 驱动程序
通过维特USB-CAN适配器进行CAN通信

支持功能:
- MIT控制模式
- 位置-速度控制模式（支持梯形加减速）
- 速度控制模式
- 电机使能/禁用
- 状态反馈读取
"""

import serial
import struct
import time
import threading
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import IntEnum
import math


class MotorType(IntEnum):
    """电机型号类型"""
    DM_J4310_2EC = 0  # DM-J4310-2EC 电机
    DM4340 = 1        # DM4340 电机


class MotorMode(IntEnum):
    """电机控制模式"""
    MIT = 0           # MIT模式
    POSITION_SPEED = 1  # 位置-速度模式（梯形加减速）
    SPEED = 2         # 速度模式


class MotorError(IntEnum):
    """电机错误类型"""
    NONE = 0
    ENABLED = 0x1          # 使能状态（不是错误）
    OVERVOLTAGE = 0x8      # 过压
    UNDERVOLTAGE = 0x9     # 欠压
    OVERCURRENT = 0xA      # 过流
    MOS_OVERTEMP = 0xB     # MOS过温
    COIL_OVERTEMP = 0xC    # 线圈过温
    COMM_LOSS = 0xD        # 通信丢失
    OVERLOAD = 0xE         # 过载


@dataclass
class MotorState:
    """电机状态数据"""
    motor_id: int = 0
    position: float = 0.0      # 位置 (rad)
    velocity: float = 0.0      # 速度 (rad/s)
    torque: float = 0.0        # 扭矩 (Nm)
    temperature_mos: int = 0   # MOS温度 (℃)
    temperature_rotor: int = 0 # 转子温度 (℃)
    enabled: bool = False      # 使能状态（从反馈获取）
    error: MotorError = MotorError.NONE
    timestamp: float = 0.0     # 时间戳


class WitMotionUSBCAN:
    """
    维特USB-CAN适配器驱动类
    
    实现AT指令模式下的CAN数据收发
    """
    
    # 帧类型定义
    FRAME_TYPE_STD_DATA = 0x00    # 标准数据帧
    FRAME_TYPE_STD_REMOTE = 0x01  # 标准远程帧
    FRAME_TYPE_EXT_DATA = 0x02    # 扩展数据帧
    FRAME_TYPE_EXT_REMOTE = 0x03  # 扩展远程帧
    
    def __init__(self, port: str, baudrate: int = 921600, can_baudrate: int = 1000000):
        """
        初始化USB-CAN适配器
        
        Args:
            port: 串口端口号（如 'COM3' 或 'COM9'）
            baudrate: 串口波特率，默认9600
            can_baudrate: CAN波特率，默认1000000 (1Mbps)
        """
        self.port = port
        self.baudrate = baudrate
        self.can_baudrate = can_baudrate
        self.serial: Optional[serial.Serial] = None
        self._rx_buffer = bytearray()
        self._rx_callback: Optional[Callable] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
    def open(self) -> bool:
        """打开串口连接并配置CAN适配器"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            time.sleep(0.1)
            
            # # 进入配置模式并设置CAN波特率
            # if not self._configure():
            #     print("警告: CAN配置可能未成功，继续运行...")
            
            # 进入AT指令模式
            if not self._enter_at_mode():
                print("警告: 进入AT指令模式可能未成功")
                
            # 启动接收线程
            self._running = True
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            
            print(f"USB-CAN适配器已连接: {self.port}")
            return True
            
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()
        print("USB-CAN适配器已断开")
    
    def _send_at_command(self, cmd: str, wait_response: bool = True) -> str:
        """发送AT指令并等待响应"""
        if not self.serial:
            return ""
        
        with self._lock:
            # 清空接收缓冲区
            self.serial.reset_input_buffer()
            
            # 发送命令
            full_cmd = f"{cmd}\r\n"
            self.serial.write(full_cmd.encode('ascii'))
            
            if not wait_response:
                return ""
            
            # 等待响应
            time.sleep(0.1)
            response = ""
            timeout = time.time() + 0.5
            while time.time() < timeout:
                if self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    response += data.decode('ascii', errors='ignore')
                    if 'OK' in response or 'ERROR' in response:
                        break
                time.sleep(0.01)
            
            return response.strip()
    
    def _configure(self) -> bool:
        """配置CAN适配器"""
        # 进入配置模式
        resp = self._send_at_command("AT+CG")
        if "OK" not in resp:
            return False
        
        # 设置CAN波特率
        resp = self._send_at_command(f"AT+CAN_BAUD={self.can_baudrate}")
        if "OK" not in resp:
            return False
        
        # 配置滤波器（允许所有帧通过）
        resp = self._send_at_command("AT+CAN_FILTER0=1,0,0,0")
        
        return True
    
    def _enter_at_mode(self) -> bool:
        """进入AT指令模式"""
        resp = self._send_at_command("AT+AT")
        print(f"进入AT指令模式响应: {resp}")
        return "OK" in resp
    
    def set_rx_callback(self, callback: Callable):
        """设置接收回调函数"""
        self._rx_callback = callback
    
    def _rx_loop(self):
        """接收数据循环"""
        while self._running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting:
                    data = self.serial.read(self.serial.in_waiting)
                    self._rx_buffer.extend(data)
                    self._process_rx_buffer()
                else:
                    time.sleep(0.001)
            except Exception as e:
                if self._running:
                    print(f"接收错误: {e}")
                break
    
    def _process_rx_buffer(self):
        """处理接收缓冲区数据"""
        # 查找完整的AT指令响应帧
        # 格式: AT(2) + ID和类型混合(4) + 长度(1) + 数据(0-8) + \r\n(2)
        while len(self._rx_buffer) >= 9:  # 最小帧长度: AT(2) + ID_TYPE(4) + 长度(1) + \r\n(2) = 9
            # 查找 "AT" 开头
            try:
                at_idx = self._rx_buffer.index(b'AT'[0])
                if at_idx > 0:
                    self._rx_buffer = self._rx_buffer[at_idx:]
            except ValueError:
                self._rx_buffer.clear()
                return

            if len(self._rx_buffer) < 2:
                return

            if self._rx_buffer[0:2] != b'AT':
                self._rx_buffer = self._rx_buffer[1:]
                continue

            if len(self._rx_buffer) < 7:  # AT(2) + ID_TYPE(4) + 长度(1)
                return

            # 解析帧长度（现在在索引6的位置）
            data_len = self._rx_buffer[6]
            if data_len > 8:
                self._rx_buffer = self._rx_buffer[1:]
                continue

            frame_len = 2 + 4 + 1 + data_len + 2  # AT + ID_TYPE + 长度 + 数据 + \r\n
            if len(self._rx_buffer) < frame_len:
                return

            # 提取帧数据
            frame = bytes(self._rx_buffer[:frame_len])
            self._rx_buffer = self._rx_buffer[frame_len:]
            
            # 验证帧结尾
            if frame[-2:] == b'\r\n':
                self._parse_can_frame(frame[:-2])  # 去掉 \r\n
    
    def _parse_can_frame(self, frame: bytes):
        """解析CAN帧"""
        if len(frame) < 7:
            return
        # print(f"接收CAN帧: {frame.hex()}")

        # 跳过 "AT" 前缀，解析ID和类型混合字节（4字节，大端序）
        id_type_bytes = frame[2:6]
        raw_id = struct.unpack('>I', id_type_bytes)[0]

        # 根据bit1的值判断帧类型（标准帧或扩展帧）
        # bit1=0: 标准帧, bit1=1: 扩展帧
        is_extended = (raw_id & 0x04) != 0

        if is_extended:
            # 扩展帧，ID在高29位
            frame_id = (raw_id >> 3) & 0x1FFFFFFF
            frame_type = self.FRAME_TYPE_EXT_DATA
        else:
            # 标准帧，ID在高11位
            frame_id = (raw_id >> 21) & 0x7FF
            frame_type = self.FRAME_TYPE_STD_DATA

        data_len = frame[6]
        data = frame[7:7+data_len] if data_len > 0 else b''
        
        # 调用回调函数
        if self._rx_callback:
            self._rx_callback(frame_id, data, frame_type)
    
    def send_can_frame(self, frame_id: int, data: bytes, 
                       frame_type: int = FRAME_TYPE_STD_DATA) -> bool:
        """
        发送CAN帧
        
        Args:
            frame_id: CAN帧ID
            data: 数据（0-8字节）
            frame_type: 帧类型
            
        Returns:
            是否发送成功
        """
        if not self.serial or not self.serial.is_open:
            return False
        
        if len(data) > 8:
            print("错误: CAN数据长度不能超过8字节")
            return False
        
        # 构建帧ID字节（根据帧类型）
        if frame_type in [self.FRAME_TYPE_STD_DATA, self.FRAME_TYPE_STD_REMOTE]:
            # 标准帧，ID放在高11位
            raw_id = (frame_id & 0x7FF) << 21 | (0x00 << 1)
        else:
            # 扩展帧，ID放在高29位
            raw_id = (frame_id & 0x1FFFFFFF) << 3 | (0x02 << 1)
        
        id_bytes = struct.pack('>I', raw_id)

        # 构建AT指令帧
        # 格式: "AT"(2) + ID和类型混合(4) + 数据长度(1) + 数据(0-8) + "\r\n"(2)
        frame = bytearray()
        frame.extend(b'AT')
        frame.extend(id_bytes)
        frame.append(len(data))
        frame.extend(data)
        frame.extend(b'\r\n')
        
        # print(f"发送CAN帧: {frame.hex()}")
        
        try:
            with self._lock:
                self.serial.write(bytes(frame))
            return True
        except Exception as e:
            print(f"发送CAN帧失败: {e}")
            return False


class DMMotor:
    """
    达妙电机驱动类（支持多种型号）

    支持型号:
    - DM-J4310-2EC
    - DM4340

    支持功能:
    - MIT控制模式
    - 位置-速度控制模式（带梯形加减速）
    - 速度控制模式
    """

    # 电机参数配置（根据型号不同）
    MOTOR_PARAMS = {
        MotorType.DM_J4310_2EC: {
            'P_MAX': 12.5,      # 最大位置 (rad) - MIT模式
            'V_MAX': 30.0,      # 最大速度 (rad/s)
            'T_MAX': 10.0,      # 最大扭矩 (Nm)
            'KP_MAX': 500.0,    # 最大Kp
            'KD_MAX': 5.0,      # 最大Kd
            'name': 'DM-J4310-2EC'
        },
        MotorType.DM4340: {
            'P_MAX': 12.5,      # 最大位置 (rad) - MIT模式
            'V_MAX': 8.0,       # 最大速度 (rad/s) - DM4340速度较低
            'T_MAX': 28.0,      # 最大扭矩 (Nm) - DM4340扭矩更大
            'KP_MAX': 500.0,    # 最大Kp
            'KD_MAX': 5.0,      # 最大Kd
            'name': 'DM4340'
        }
    }

    # 特殊命令
    CMD_ENABLE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
    CMD_DISABLE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
    CMD_SET_ZERO = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])
    CMD_CLEAR_ERROR = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB])

    def __init__(self, can_adapter: WitMotionUSBCAN, motor_id: int = 1, master_id: int = 0,
                 motor_type: MotorType = MotorType.DM_J4310_2EC):
        """
        初始化电机

        Args:
            can_adapter: USB-CAN适配器实例
            motor_id: 电机CAN ID (1-127)
            master_id: 主机ID，用于接收反馈帧
            motor_type: 电机型号类型
        """
        self.can = can_adapter
        self.motor_id = motor_id
        self.master_id = master_id
        self.motor_type = motor_type
        self.state = MotorState(motor_id=motor_id)
        self._state_lock = threading.Lock()
        self._enabled = False

        # 根据电机型号设置参数
        params = self.MOTOR_PARAMS[motor_type]
        self.P_MAX = params['P_MAX']
        self.V_MAX = params['V_MAX']
        self.T_MAX = params['T_MAX']
        self.KP_MAX = params['KP_MAX']
        self.KD_MAX = params['KD_MAX']
        self.motor_name = params['name']

        # 梯形加减速参数
        self.max_acceleration = 10.0   # 最大加速度 (rad/s²)
        self.max_velocity = 10.0       # 最大速度 (rad/s)

        # 注册接收回调
        self.can.set_rx_callback(self._on_can_frame)

        print(f"初始化电机: {self.motor_name}, ID={motor_id}, T_MAX={self.T_MAX} Nm")

    def float_to_uint(self, x: float, x_min: float, x_max: float, bits):
        x = max(x_min, min(x, x_max))  # 限幅到 [x_min, x_max] 范围
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def uint_to_float(self, x, min: float, max: float, bits):
        span = max - min
        data_norm = float(x) / ((1 << bits) - 1)
        temp = data_norm * span + min
        return float(temp)

    def _on_can_frame(self, frame_id: int, data: bytes, frame_type: int):
        """CAN帧接收回调"""
        # 检查是否是本电机的反馈帧
        if frame_id != self.master_id:
            # print(f"收到非本电机反馈帧，ID: {frame_id:X}, 期望ID: {self.master_id:X}")
            return

        if len(data) < 8:
            # print(f"收到无效反馈帧，ID: {frame_id:X}, 数据长度: {len(data)}")
            return

        # 解析反馈数据
        # D[0]: ID | (ERR << 4)  -- ID在低4位，ERR在高4位
        # D[1]: POS[15:8]
        # D[2]: POS[7:0]
        # D[3]: VEL[11:4]
        # D[4]: (VEL[3:0] << 4) | T[11:8]
        # D[5]: T[7:0]
        # D[6]: T_MOS
        # D[7]: T_Rotor

        motor_id = data[0] & 0x0F  # 低4位是ID
        if motor_id != (self.motor_id & 0x0F):
            return

        error_code = (data[0] >> 4) & 0x0F  # 高4位是ERR

        # 解析位置 (16位有符号)
        pos_raw = (data[1] << 8) | data[2]

        # 解析速度 (12位有符号)
        vel_raw = (data[3] << 4) | ((data[4] >> 4) & 0x0F)

        # 解析扭矩 (12位有符号)
        torque_raw = ((data[4] & 0x0F) << 8) | data[5]

        # 转换为实际值（线性映射）
        position = self.uint_to_float(pos_raw, -self.P_MAX, self.P_MAX, 16)
        velocity = self.uint_to_float(vel_raw, -self.V_MAX, self.V_MAX, 12)
        torque = self.uint_to_float(torque_raw, -self.T_MAX, self.T_MAX, 12)
        # print(f"位置: {position:.3f} rad, 速度: {velocity:.3f} rad/s, 扭矩: {torque:.3f} Nm, 错误码: {error_code}")

        # 温度
        temp_mos = data[6]
        temp_rotor = data[7]

        # 解析使能状态和错误状态
        # ERR=1表示使能状态，ERR=0表示失能状态，ERR>=8表示错误
        is_enabled = (error_code == 0x1)
        actual_error = MotorError.NONE

        if error_code >= 0x8:
            # 实际错误
            actual_error = MotorError(error_code) if error_code in MotorError._value2member_map_ else MotorError.NONE

        # 更新状态
        with self._state_lock:
            self.state.position = position
            self.state.velocity = velocity
            self.state.torque = torque
            self.state.temperature_mos = temp_mos
            self.state.temperature_rotor = temp_rotor
            self.state.enabled = is_enabled
            self.state.error = actual_error
            self.state.timestamp = time.time()

            # 更新内部使能状态标志
            self._enabled = is_enabled
    
    def enable(self) -> bool:
        """使能电机"""
        result = self.can.send_can_frame(self.motor_id, self.CMD_ENABLE)
        if result:
            print(f"电机 {self.motor_id} 使能命令已发送")
            # 注意：实际使能状态由电机反馈决定，会在_on_can_frame中更新
        return result

    def disable(self) -> bool:
        """禁用电机"""
        result = self.can.send_can_frame(self.motor_id, self.CMD_DISABLE)
        if result:
            print(f"电机 {self.motor_id} 失能命令已发送")
            # 注意：实际使能状态由电机反馈决定，会在_on_can_frame中更新
        return result

    def is_enabled(self) -> bool:
        """获取电机使能状态（从反馈获取）"""
        return self._enabled
    
    def set_zero(self) -> bool:
        """设置当前位置为零点（注意：双编码器电机此命令无效）"""
        result = self.can.send_can_frame(self.motor_id, self.CMD_SET_ZERO)
        if result:
            print(f"电机 {self.motor_id} 零点已设置")
        return result

    def clear_error(self) -> bool:
        """清除错误状态"""
        result = self.can.send_can_frame(self.motor_id, self.CMD_CLEAR_ERROR)
        if result:
            print(f"电机 {self.motor_id} 清除错误命令已发送")
        return result

    def get_state(self) -> MotorState:
        """获取电机状态"""
        with self._state_lock:
            return MotorState(
                motor_id=self.state.motor_id,
                position=self.state.position,
                velocity=self.state.velocity,
                torque=self.state.torque,
                temperature_mos=self.state.temperature_mos,
                temperature_rotor=self.state.temperature_rotor,
                enabled=self.state.enabled,
                error=self.state.error,
                timestamp=self.state.timestamp
            )
    
    def control_mit(self, p_des: float, v_des: float, kp: float, kd: float, t_ff: float) -> bool:
        """
        MIT控制模式
        
        torque = kp * (p_des - p) + kd * (v_des - v) + t_ff
        
        Args:
            p_des: 目标位置 (rad)
            v_des: 目标速度 (rad/s)
            kp: 位置比例系数 [0, 500]
            kd: 位置微分系数 [0, 5]
            t_ff: 前馈扭矩 (Nm)
            
        Returns:
            是否发送成功
        """
        # 限幅
        p_des = max(-self.P_MAX, min(self.P_MAX, p_des))
        v_des = max(-self.V_MAX, min(self.V_MAX, v_des))
        kp = max(0, min(self.KP_MAX, kp))
        kd = max(0, min(self.KD_MAX, kd))
        # 转换为整数（线性映射）
        t_ff = max(-self.T_MAX, min(self.T_MAX, t_ff))
        p_int = self.float_to_uint(p_des, -self.P_MAX, self.P_MAX, 16)
        v_int = self.float_to_uint(v_des, -self.V_MAX, self.V_MAX, 12)
        kp_int = self.float_to_uint(kp, 0, 500, 12)
        kd_int = self.float_to_uint(kd, 0, 5, 12)
        t_int = self.float_to_uint(t_ff, -self.T_MAX, self.T_MAX, 12)
        
        # 打包数据
        # D[0]: p_des[15:8]
        # D[1]: p_des[7:0]
        # D[2]: v_des[11:4]
        # D[3]: (v_des[3:0] << 4) | kp[11:8]
        # D[4]: kp[7:0]
        # D[5]: kd[11:4]
        # D[6]: (kd[3:0] << 4) | t_ff[11:8]
        # D[7]: t_ff[7:0]
        data = bytes([
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F),
            kp_int & 0xFF,
            (kd_int >> 4) & 0xFF,
            ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
            t_int & 0xFF
        ])
        
        return self.can.send_can_frame(self.motor_id, data)
    
    def control_position_speed(self, position: float, velocity: float) -> bool:
        """
        位置-速度控制模式（梯形加减速）
        
        电机会以梯形加减速曲线运动到目标位置，velocity是匀速段的最大速度
        
        Args:
            position: 目标位置 (rad)
            velocity: 最大速度 (rad/s)，即梯形加减速的匀速段速度
            
        Returns:
            是否发送成功
        """
        # 打包为float，小端序
        data = struct.pack('<ff', position, abs(velocity))
        
        # 帧ID = 0x100 + motor_id
        frame_id = 0x100 + self.motor_id
        
        return self.can.send_can_frame(frame_id, data)
    
    def control_speed(self, velocity: float) -> bool:
        """
        速度控制模式
        
        Args:
            velocity: 目标速度 (rad/s)
            
        Returns:
            是否发送成功
        """
        # 打包为float，小端序
        data = struct.pack('<f', velocity)
        
        # 帧ID = 0x200 + motor_id
        frame_id = 0x200 + self.motor_id
        
        return self.can.send_can_frame(frame_id, data)


class TrapezoidalMotionController:
    """
    梯形加减速运动控制器
    
    实现软件层面的梯形加减速位置控制
    可用于MIT模式下的平滑位置控制
    """
    
    def __init__(self, motor: DMMotor):
        """
        初始化运动控制器

        Args:
            motor: 电机实例
        """
        self.motor = motor
        
        # 运动参数
        self.max_velocity = 10.0       # 最大速度 (rad/s)
        self.max_acceleration = 20.0   # 最大加速度 (rad/s²)
        
        # MIT控制参数
        self.kp = 50.0    # 位置比例系数
        self.kd = 2.0     # 位置微分系数
        
        # 运动状态
        self._target_position = 0.0
        self._current_cmd_position = 0.0
        self._current_cmd_velocity = 0.0
        self._motion_active = False
        self._control_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 控制频率
        self.control_rate = 200  # Hz
    
    def set_motion_params(self, max_velocity: float, max_acceleration: float):
        """
        设置运动参数
        
        Args:
            max_velocity: 最大速度 (rad/s)
            max_acceleration: 最大加速度 (rad/s²)
        """
        self.max_velocity = abs(max_velocity)
        self.max_acceleration = abs(max_acceleration)
    
    def set_control_params(self, kp: float, kd: float):
        """
        设置MIT控制参数
        
        Args:
            kp: 位置比例系数
            kd: 位置微分系数
        """
        self.kp = kp
        self.kd = kd
    
    def move_to_position(self, target_position: float, blocking: bool = False) -> bool:
        """
        移动到目标位置（梯形加减速）
        
        Args:
            target_position: 目标位置 (rad)
            blocking: 是否阻塞等待运动完成
            
        Returns:
            是否启动成功
        """
        if self._motion_active:
            print("警告: 上一个运动还未完成")
            self.stop()
        
        self._target_position = target_position
        self._stop_event.clear()
        self._motion_active = True
        
        # 获取当前位置作为起始点
        state = self.motor.get_state()
        self._current_cmd_position = state.position
        self._current_cmd_velocity = 0.0
        
        # 启动控制线程
        self._control_thread = threading.Thread(
            target=self._trapezoidal_control_loop,
            daemon=True
        )
        self._control_thread.start()
        
        if blocking:
            self._control_thread.join()
        
        return True
    
    def _trapezoidal_control_loop(self):
        """梯形加减速控制循环"""
        dt = 1.0 / self.control_rate
        
        while not self._stop_event.is_set():
            start_time = time.time()
            
            # 计算位置误差
            position_error = self._target_position - self._current_cmd_position
            
            # 检查是否到达目标
            if abs(position_error) < 0.001 and abs(self._current_cmd_velocity) < 0.01:
                # 到达目标位置
                self._current_cmd_velocity = 0.0
                self._motion_active = False
                print(f"到达目标位置: {self._target_position:.4f} rad")
                break
            
            # 计算减速距离
            decel_distance = (self._current_cmd_velocity ** 2) / (2 * self.max_acceleration)
            
            # 判断运动方向
            direction = 1 if position_error > 0 else -1
            
            # 判断是否需要减速
            if abs(position_error) <= decel_distance + 0.01:
                # 减速阶段
                if abs(self._current_cmd_velocity) > 0.01:
                    self._current_cmd_velocity -= direction * self.max_acceleration * dt
                    # 速度方向修正
                    if direction > 0 and self._current_cmd_velocity < 0:
                        self._current_cmd_velocity = 0
                    elif direction < 0 and self._current_cmd_velocity > 0:
                        self._current_cmd_velocity = 0
            else:
                # 加速或匀速阶段
                if abs(self._current_cmd_velocity) < self.max_velocity:
                    self._current_cmd_velocity += direction * self.max_acceleration * dt
                    # 限制最大速度
                    if abs(self._current_cmd_velocity) > self.max_velocity:
                        self._current_cmd_velocity = direction * self.max_velocity
            
            # 更新命令位置
            self._current_cmd_position += self._current_cmd_velocity * dt
            
            # 发送MIT控制命令
            self.motor.control_mit(
                p_des=self._current_cmd_position,
                v_des=self._current_cmd_velocity,
                kp=self.kp,
                kd=self.kd,
                t_ff=0.0
            )
            
            # 控制周期
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop(self):
        """停止运动"""
        self._stop_event.set()
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        self._motion_active = False
        self._current_cmd_velocity = 0.0
        
        # 发送停止命令
        state = self.motor.get_state()
        self.motor.control_mit(
            p_des=state.position,
            v_des=0.0,
            kp=self.kp,
            kd=self.kd,
            t_ff=0.0
        )
    
    def is_motion_active(self) -> bool:
        """检查是否正在运动"""
        return self._motion_active
    
    def get_command_position(self) -> float:
        """获取当前命令位置"""
        return self._current_cmd_position
    
    def get_command_velocity(self) -> float:
        """获取当前命令速度"""
        return self._current_cmd_velocity


# =============================================================================
# 示例和测试代码
# =============================================================================

def demo_position_control():
    """位置控制演示"""
    print("=" * 60)
    print("达妙电机 DM-J4310-2EC 位置控制演示")
    print("=" * 60)
    
    # 创建USB-CAN适配器
    # 请根据实际情况修改端口号
    can_adapter = WitMotionUSBCAN(
        port='COM9',  # Linux下，Windows下改为 'COM3' 等
        baudrate=921600,
        can_baudrate=1000000
    )
    
    try:
        # 打开连接
        if not can_adapter.open():
            print("无法打开USB-CAN适配器")
            return
        
        # 创建电机实例
        motor = DMMotor(
            can_adapter=can_adapter,
            motor_id=2,      # 电机CAN ID
            master_id=0   # 反馈帧ID（建议设置为 motor_id * 0x11）
        )
        
        # 等待连接稳定
        time.sleep(0.5)
        
        # 使能电机
        motor.enable()
        time.sleep(0.2)
        
        print("\n--- 使用内置位置-速度模式（硬件梯形加减速）---")
        
        # 方式1：使用电机内置的位置-速度模式（推荐）
        # 电机会自动执行梯形加减速
        target_pos = 3.14  # 目标位置 (rad)，约180度
        max_vel = 5.0      # 最大速度 (rad/s)
        
        print(f"移动到位置: {target_pos:.2f} rad, 最大速度: {max_vel:.1f} rad/s")
        motor.control_position_speed(target_pos, max_vel)
        
        # 等待运动完成并读取状态
        for _ in range(50):
            time.sleep(0.1)
            state = motor.get_state()
            print(f"位置: {state.position:7.4f} rad, "
                  f"速度: {state.velocity:7.4f} rad/s, "
                  f"扭矩: {state.torque:6.3f} Nm, "
                  f"温度: {state.temperature_mos}°C")
        
        print("\n--- 使用软件梯形加减速（MIT模式）---")
        
        # 方式2：使用软件梯形加减速控制器（MIT模式）
        controller = TrapezoidalMotionController(motor)
        controller.set_motion_params(
            max_velocity=8.0,      # 最大速度 8 rad/s
            max_acceleration=15.0  # 最大加速度 15 rad/s²
        )
        controller.set_control_params(kp=40.0, kd=1.0)
        
        # 移动到另一个位置
        target_pos2 = 0.0
        print(f"移动到位置: {target_pos2:.2f} rad")
        controller.move_to_position(target_pos2, blocking=True)
        
        # 再移动一次
        target_pos3 = 6.28  # 约360度
        print(f"移动到位置: {target_pos3:.2f} rad")
        controller.move_to_position(target_pos3, blocking=True)
        
        # 禁用电机
        motor.disable()
        
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 关闭连接
        can_adapter.close()


def demo_speed_control():
    """速度控制演示"""
    print("=" * 60)
    print("达妙电机 DM-J4310-2EC 速度控制演示")
    print("=" * 60)
    
    can_adapter = WitMotionUSBCAN(
        port='COM9',
        baudrate=921600,
        can_baudrate=1000000
    )
    
    try:
        if not can_adapter.open():
            print("无法打开USB-CAN适配器")
            return
        
        motor = DMMotor(can_adapter, motor_id=2, master_id=0)
        time.sleep(0.5)
        
        motor.enable()
        time.sleep(0.2)
        
        # 速度控制测试
        print("设置速度: 5 rad/s")
        motor.control_speed(5.0)
        
        for _ in range(30):
            time.sleep(0.1)
            state = motor.get_state()
            print(f"位置: {state.position:7.4f} rad, "
                  f"速度: {state.velocity:7.4f} rad/s")
        
        print("设置速度: -5 rad/s")
        motor.control_speed(-5.0)
        
        for _ in range(30):
            time.sleep(0.1)
            state = motor.get_state()
            print(f"位置: {state.position:7.4f} rad, "
                  f"速度: {state.velocity:7.4f} rad/s")
        
        print("停止")
        motor.control_speed(0.0)
        time.sleep(1.0)
        
        motor.disable()
        
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        can_adapter.close()


def demo_mit_control():
    """MIT控制模式演示"""
    print("=" * 60)
    print("达妙电机 DM-J4310-2EC MIT控制模式演示")
    print("=" * 60)

    can_adapter = WitMotionUSBCAN(
        port='COM9',
        baudrate=921600,
        can_baudrate=1000000
    )

    try:
        if not can_adapter.open():
            print("无法打开USB-CAN适配器")
            return

        motor = DMMotor(can_adapter, motor_id=2, master_id=0)
        time.sleep(0.5)

        motor.enable()

        # 等待接收到位置数据
        print("等待接收位置数据...")
        max_wait_time = 2.0  # 最多等待2秒
        start_wait = time.time()
        while motor.get_state().timestamp == 0.0:
            time.sleep(0.01)
            if time.time() - start_wait > max_wait_time:
                print("错误: 未能接收到电机反馈数据")
                return

        # 获取当前位置作为初始位置
        initial_state = motor.get_state()
        initial_position = initial_state.position
        print(f"当前位置: {initial_position:.4f} rad")

        # 目标位置
        target_position = initial_position + math.pi  # 从当前位置移动π rad
        print(f"目标位置: {target_position:.4f} rad")

        # MIT模式梯形加减速位置控制
        max_velocity = 8.0       # 最大速度 (rad/s)
        max_acceleration = 15.0  # 最大加速度 (rad/s²)
        kp = 40.0
        kd = 1.0

        # 控制参数
        control_rate = 200  # Hz
        dt = 1.0 / control_rate

        current_cmd_position = initial_position
        current_cmd_velocity = 0.0

        print("\nMIT模式: 使用梯形加减速移动")

        while True:
            start_time = time.time()

            # 计算位置误差
            position_error = target_position - current_cmd_position

            # 检查是否到达目标
            if abs(position_error) < 0.001 and abs(current_cmd_velocity) < 0.01:
                current_cmd_velocity = 0.0
                print(f"到达目标位置: {target_position:.4f} rad")
                break

            # 计算减速距离
            decel_distance = (current_cmd_velocity ** 2) / (2 * max_acceleration)

            # 判断运动方向
            direction = 1 if position_error > 0 else -1

            # 判断是否需要减速
            if abs(position_error) <= decel_distance + 0.01:
                # 减速阶段
                if abs(current_cmd_velocity) > 0.01:
                    current_cmd_velocity -= direction * max_acceleration * dt
                    # 速度方向修正
                    if direction > 0 and current_cmd_velocity < 0:
                        current_cmd_velocity = 0
                    elif direction < 0 and current_cmd_velocity > 0:
                        current_cmd_velocity = 0
            else:
                # 加速或匀速阶段
                if abs(current_cmd_velocity) < max_velocity:
                    current_cmd_velocity += direction * max_acceleration * dt
                    # 限制最大速度
                    if abs(current_cmd_velocity) > max_velocity:
                        current_cmd_velocity = direction * max_velocity

            # 更新命令位置
            current_cmd_position += current_cmd_velocity * dt

            # 发送MIT控制命令
            motor.control_mit(
                p_des=current_cmd_position,
                v_des=current_cmd_velocity,
                kp=kp,
                kd=kd,
                t_ff=0.0
            )

            # 每隔一段时间打印状态
            if int(time.time() * 20) % 10 == 0:  # 约每0.5秒打印一次
                state = motor.get_state()
                print(f"命令位置: {current_cmd_position:6.3f} rad, "
                      f"实际位置: {state.position:6.3f} rad, "
                      f"命令速度: {current_cmd_velocity:6.3f} rad/s")

            # 控制周期
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        time.sleep(1.0)
        motor.disable()

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        can_adapter.close()


def demo_enable_disable():
    """电机使能/失能演示"""
    print("=" * 60)
    print("达妙电机 DM-J4310-2EC 使能/失能控制")
    print("=" * 60)

    can_adapter = WitMotionUSBCAN(
        port='COM9',
        baudrate=921600,
        can_baudrate=1000000
    )

    try:
        if not can_adapter.open():
            print("无法打开USB-CAN适配器")
            return

        motor = DMMotor(can_adapter, motor_id=2, master_id=0)
        time.sleep(0.5)

        while True:
            print("\n请选择操作:")
            print("1. 使能电机")
            print("2. 失能电机")
            print("3. 查看电机状态")
            print("0. 返回主菜单")

            choice = input("\n请输入选择 (0-3): ").strip()

            if choice == '1':
                motor.enable()
                time.sleep(0.1)
                state = motor.get_state()
                print(f"电机已使能 - 当前位置: {state.position:.4f} rad, "
                      f"速度: {state.velocity:.4f} rad/s, "
                      f"温度: {state.temperature_mos}°C")
            elif choice == '2':
                motor.disable()
                print("电机已失能")
            elif choice == '3':
                state = motor.get_state()
                print(f"\n--- 电机状态 ---")
                print(f"ID: {state.motor_id}")
                print(f"位置: {state.position:.4f} rad ({state.position * 180 / 3.14159:.2f}°)")
                print(f"速度: {state.velocity:.4f} rad/s")
                print(f"扭矩: {state.torque:.4f} Nm")
                print(f"MOS温度: {state.temperature_mos}°C")
                print(f"转子温度: {state.temperature_rotor}°C")
                print(f"错误状态: {state.error.name}")
                print(f"时间戳: {state.timestamp:.3f}")
            elif choice == '0':
                break
            else:
                print("无效选择")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        can_adapter.close()


def main():
    """主函数"""
    print("\n请选择演示模式:")
    print("1. 位置控制（梯形加减速）")
    print("2. 速度控制")
    print("3. MIT控制")
    print("4. 使能/失能控制")
    print("0. 退出")

    try:
        choice = input("\n请输入选择 (0-4): ").strip()

        if choice == '1':
            demo_position_control()
        elif choice == '2':
            demo_speed_control()
        elif choice == '3':
            demo_mit_control()
        elif choice == '4':
            demo_enable_disable()
        elif choice == '0':
            print("退出")
        else:
            print("无效选择")

    except KeyboardInterrupt:
        print("\n退出")


# 向后兼容别名（保留旧的类名）
DM_J4310_Motor = DMMotor


if __name__ == "__main__":
    main()
