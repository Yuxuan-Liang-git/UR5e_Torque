#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025. Li Jianbin. All rights reserved.
# MIT License

"""
达妙电机 DM-J4310-2EC 驱动程序
本文件已改写为基于 DM_CAN.py 进行封装。
"""

import math
import time
import threading
from typing import Optional
from DM_CAN import Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable

class DMMotor(Motor):
    """
    达妙电机封装类，继承自 DM_CAN.Motor。
    增加了更易读的状态获取接口。
    """
    def __init__(self, MotorType, SlaveID, MasterID):
        super().__init__(MotorType, SlaveID, MasterID)
    
    def get_state(self):
        """返回当前状态的字典"""
        return {
            "position": self.state_q,
            "velocity": self.state_dq,
            "torque": self.state_tau,
            "timestamp": time.time()
        }

class TrapezoidalMotionController:
    """
    梯形加减速运动控制器
    使用 DM_CAN 的 MIT 模式发送指令
    """
    def __init__(self, motor_control: MotorControl, motor: DMMotor):
        self.mc = motor_control
        self.motor = motor
        
        # 运动参数
        self.max_velocity = 8.0
        self.max_acceleration = 15.0
        
        # MIT 控制参数
        self.kp = 40.0
        self.kd = 1.0
        
        self._target_position = 0.0
        self._current_cmd_position = 0.0
        self._current_cmd_velocity = 0.0
        self._motion_active = False
        self._stop_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        self.control_rate = 200 # Hz

    def set_motion_params(self, max_velocity: float, max_acceleration: float):
        self.max_velocity = abs(max_velocity)
        self.max_acceleration = abs(max_acceleration)

    def set_control_params(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def is_motion_active(self):
        """返回当前运动是否处于激活状态"""
        return self._motion_active

    def move_to_position(self, target_position: float, blocking: bool = False):
        if self._motion_active:
            self.stop()
        
        self._target_position = target_position
        self._stop_event.clear()
        self._motion_active = True
        
        # 起始点为当前反馈位置
        self._current_cmd_position = self.motor.state_q
        self._current_cmd_velocity = 0.0
        
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        
        if blocking:
            self._control_thread.join()

    def _control_loop(self):
        dt = 1.0 / self.control_rate
        while not self._stop_event.is_set():
            start_time = time.time()
            pos_err = self._target_position - self._current_cmd_position
            
            # 到达判断
            if abs(pos_err) < 0.01 and abs(self._current_cmd_velocity) < 0.01:
                self._current_cmd_velocity = 0.0
                self._motion_active = False
                break
            
            # 计算减速距离
            decel_dist = (self._current_cmd_velocity ** 2) / (2 * self.max_acceleration)
            direction = 1 if pos_err > 0 else -1
            
            # 判断是否需要减速
            if abs(pos_err) <= decel_dist + 0.01:
                if abs(self._current_cmd_velocity) > 0.01:
                    self._current_cmd_velocity -= direction * self.max_acceleration * dt
                    if (direction > 0 and self._current_cmd_velocity < 0) or (direction < 0 and self._current_cmd_velocity > 0):
                        self._current_cmd_velocity = 0
            else:
                if abs(self._current_cmd_velocity) < self.max_velocity:
                    self._current_cmd_velocity += direction * self.max_acceleration * dt
                    if abs(self._current_cmd_velocity) > self.max_velocity:
                        self._current_cmd_velocity = direction * self.max_velocity
            
            self._current_cmd_position += self._current_cmd_velocity * dt
            
            # 使用 DM_CAN 的 MIT 控制接口
            self.mc.controlMIT(
                self.motor, 
                kp=self.kp, 
                kd=self.kd, 
                q=self._current_cmd_position, 
                dq=self._current_cmd_velocity, 
                tau=0.0
            )
            
            # 每 0.5 秒打印一次状态用于调试
            if int(time.time() * 20) % 10 == 0:
                st = self.motor.get_state()
                print(f"Target: {self._target_position:.3f}, Cmd: {self._current_cmd_position:.3f}, Actual: {st['position']:.3f}")

            elapsed = time.time() - start_time
            if dt > elapsed:
                time.sleep(dt - elapsed)

    def stop(self):
        self._stop_event.set()
        if self._control_thread:
            self._control_thread.join(timeout=0.1)
        self._motion_active = False

def main():
    import serial
    ser = serial.Serial('/dev/ttyACM0', 921600, timeout=0.1)
    mc = MotorControl(ser)
    
    motor = DMMotor(DM_Motor_Type.DM4310, 0x01, 0x11)
    mc.addMotor(motor)
    
    # 切换到 MIT 模式并使能
    print("切换 MIT 模式并使能电机...")
    mc.switchControlMode(motor, Control_Type.MIT)
    mc.enable(motor)
    time.sleep(0.5)
    
    controller = TrapezoidalMotionController(mc, motor)
    
    try:
        print("\n移动到 3.14 rad...")
        controller.move_to_position(3.14, blocking=True)
        
        print("\n等待 1 秒...")
        time.sleep(1)
        
        print("\n回到 0.0 rad...")
        controller.move_to_position(0.0, blocking=True)
        
    except KeyboardInterrupt:
        print("\n用户中断")
        controller.stop()
    finally:
        mc.disable(motor)
        ser.close()

if __name__ == "__main__":
    main()
