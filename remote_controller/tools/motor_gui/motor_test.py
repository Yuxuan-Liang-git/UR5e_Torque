#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
from dm_motor_driver import WitMotionUSBCAN, DMMotor, MotorType

def main():
    # 串口配置（根据实际情况修改端口）
    # Linux 通常是 /dev/ttyUSB0 或 /dev/ttyACM0
    # Windows 通常是 COMx
    port = '/dev/ttyACM0' 
    baudrate = 921600

    print(f"正在连接到适配器: {port}...")
    can_adapter = WitMotionUSBCAN(port=port, baudrate=baudrate)

    if not can_adapter.open():
        print("无法打开串口，请检查权限或端口号。")
        return

    try:
        # 创建电机 1 (ID=1, 反馈 MasterID=0x11，对应你的示例)
        # 注意：在 DMMotor 中 master_id 用于过滤反馈帧
        motor1 = DMMotor(can_adapter, motor_id=0x01, master_id=0x11, motor_type=MotorType.DM_J4310_2EC)
        
        # 创建电机 2 (ID=6, 反馈 MasterID=0x15)
        motor2 = DMMotor(can_adapter, motor_id=0x06, master_id=0x15, motor_type=MotorType.DM_J4310_2EC)

        print("等待电机连接...")
        time.sleep(0.5)

        # 使能电机
        print("使能电机...")
        motor1.enable()
        motor2.enable()
        time.sleep(0.2)

        print("开始正弦波位置/速度控制测试...")
        start_time = time.time()
        
        for i in range(1000):
            t = time.time() - start_time
            q = math.sin(t)
            
            # 使用位置-速度控制模式 (类似于示例中的 control_Pos_Vel)
            motor1.control_position_speed(position=q*8, velocity=30.0)
            
            # 使用速度控制模式 (类似于示例中的 control_Vel)
            motor2.control_speed(velocity=8*q)

            # 打印状态 (从反馈获取)
            state1 = motor1.get_state()
            state2 = motor2.get_state()
            
            if i % 100 == 0:
                print(f"Time: {t:.2f}s")
                print(f"  Motor1 - Pos: {state1.position:.3f}, Vel: {state1.velocity:.3f}, Tau: {state1.torque:.3f}")
                print(f"  Motor2 - Pos: {state2.position:.3f}, Vel: {state2.velocity:.3f}, Tau: {state2.torque:.3f}")

            time.sleep(0.005) # 200Hz

        # 关闭测试
        print("停止电机并失能...")
        motor1.control_speed(0)
        motor2.control_speed(0)
        time.sleep(0.1)
        motor1.disable()
        motor2.disable()

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试发生错误: {e}")
    finally:
        can_adapter.close()
        print("测试结束。")

if __name__ == "__main__":
    main()
