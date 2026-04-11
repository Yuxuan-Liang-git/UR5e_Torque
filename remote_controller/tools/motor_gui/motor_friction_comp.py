#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import serial
import math
import sys
import select
import tty
import termios
from pathlib import Path
from DM_CAN import MotorControl, DM_Motor_Type, Control_Type
from dm_motor_driver import DMMotor

def get_key():
    """Non-blocking key read."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def get_params_path(motor_id):
    return Path(__file__).resolve().parent / f"motor_params_id_0x{motor_id:02X}.json"


def load_params(motor_id, filename=None):
    if filename is None:
        filename = get_params_path(motor_id)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            params = json.load(f)
            file_motor_id = params.get("motor_id")
            if file_motor_id is not None and int(file_motor_id) != int(motor_id):
                print(
                    f"Warning: {filename} 中的 motor_id={file_motor_id} "
                    f"与当前 motor_id={motor_id} 不一致。"
                )
            return params
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default parameters.")
        return {"tc": 0.1, "bias": 0.0, "b": 0.001}

def main():
    # --- 配置 ---
    port = "/dev/ttyACM0"
    baudrate = 921600
    motor_id = 0x03
    master_id = 0x00
    motor_type = DM_Motor_Type.DM4340


    
    # 加载辨识结果
    params = load_params(motor_id)
    Tc = params["tc"]
    bias = params["bias"]
    B = max(0.0, params["b"])
    if params["b"] < 0.0:
        print(f"Warning: 读取到负阻尼 B={params['b']:.6f}，已强制钳位为 0.0")

    # Dither/PWM 配置: 在低速区叠加方波力矩，帮助克服静摩擦
    dither_enabled = True
    dither_amp = 0.8*Tc
    dither_freq = 200.0
    dither_vel_threshold = 0.2
    
    print(f"加载参数文件: {get_params_path(motor_id)}")
    print(f"加载参数: Tc={Tc:.4f}, B={B:.4f}, bias={bias:.4f}")

    # 串口与控制初始化
    ser = serial.Serial(port, baudrate, timeout=0.1)
    mc = MotorControl(ser)
    motor = DMMotor(motor_type, motor_id, master_id)
    mc.addMotor(motor)

    if not mc.switchControlMode(motor, Control_Type.MIT):
        raise RuntimeError("切换 MIT 模式失败。")

    print("使能电机...")
    mc.enable(motor)
    time.sleep(0.5)

    print("\n--- 摩擦力补偿测试 ---")
    print("模式 1: 不补偿 (tau = 0) - 按键盘 '1' 切换")
    print("模式 2: 摩擦力补偿 (tau = tau_fric) - 按键盘 '2' 切换")
    print("模式 3: 摩擦力补偿 + Dither (tau = tau_fric + dither_tau) - 按键盘 '3' 切换")
    print("按 'q' 键退出测试...")
    print("请手动转动电机感受阻力差异...")
    print(
        f"Dither: {'on' if dither_enabled else 'off'}, "
        f"amp={dither_amp:.4f} Nm, freq={dither_freq:.1f} Hz, "
        f"|v|<{dither_vel_threshold:.2f} rad/s 时启用"
    )

    mode = 1
    start_time = time.time()
    
    try:
        while True:
            # 读取键盘输入
            key = get_key()
            if key == '1':
                mode = 1
                print(f"\n模式已切换: 模式1 (关闭补偿)")
            elif key == '2':
                mode = 2
                print(f"\n模式已切换: 模式2 (开启补偿)")
            elif key == '3':
                mode = 3
                print(f"\n模式已切换: 模式3 (开启补偿 + Dither)")
            elif key == 'q':
                print("\n收到退出指令...")
                break

            # 获取状态
            st = motor.get_state()
            v = float(st["velocity"])
            
            if mode == 2:
                # 只有摩擦力补偿
                tau_fric = Tc * math.tanh(v * 10.0) + B * v + bias
                tau_cmd = tau_fric
            elif mode == 3:
                # 摩擦力补偿 + Dither
                tau_fric = Tc * math.tanh(v * 10.0) + B * v + bias
                if dither_enabled and abs(v) < dither_vel_threshold:
                    phase = 2.0 * math.pi * dither_freq * (time.time() - start_time)
                    dither_tau = dither_amp if math.sin(phase) >= 0.0 else -dither_amp
                else:
                    dither_tau = 0.0

                tau_fric += dither_tau
                tau_cmd = tau_fric
            else:
                tau_cmd = 0.0

            # 发送控制指令
            mc.controlMIT(
                DM_Motor=motor,
                kp=0.0,
                kd=0.0,
                q=float(st["position"]),
                dq=0.0,
                tau=tau_cmd,
            )
            
            time.sleep(0.0025) # 500Hz

    except KeyboardInterrupt:
        print("\n停止测试。")
    finally:
        mc.disable(motor)
        ser.close()
        print("串口已关闭。")

if __name__ == "__main__":
    main()
