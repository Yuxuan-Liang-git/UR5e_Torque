import math
from DM_CAN import *
import serial
import time
import socket
import json

# 配置 UDP 发送 (PlotJuggler JSON Mode)
UDP_IP = "127.0.0.1"
UDP_PORT = 9870
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_plotjuggler(motor1_target, motor1_actual, motor2_target, motor2_actual):
    """
    发送数据到 PlotJuggler
    格式: JSON 字典
    """
    data = {
        "timestamp": float(time.time()),
        "m1_target_vel": float(motor1_target),
        "m1_actual_vel": float(motor1_actual),
        "m2_target_vel": float(motor2_target),
        "m2_actual_vel": float(motor2_actual)
    }
    msg = json.dumps(data).encode('utf-8')
    sock.sendto(msg, (UDP_IP, UDP_PORT))

Motor1=Motor(DM_Motor_Type.DM4310,0x01,0x11)
Motor2=Motor(DM_Motor_Type.DM4310,0x06,0x15)
serial_device = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
MotorControl1=MotorControl(serial_device)
MotorControl1.addMotor(Motor1)
MotorControl1.addMotor(Motor2)

if MotorControl1.switchControlMode(Motor1,Control_Type.POS_VEL):
    print("switch POS_VEL success")
if MotorControl1.switchControlMode(Motor2,Control_Type.VEL):
    print("switch VEL success")
print("sub_ver:",MotorControl1.read_motor_param(Motor1,DM_variable.sub_ver))
print("Gr:",MotorControl1.read_motor_param(Motor1,DM_variable.Gr))

# if MotorControl1.change_motor_param(Motor1,DM_variable.KP_APR,54):
#     print("write success")
print("PMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.PMAX))
print("MST_ID:",MotorControl1.read_motor_param(Motor1,DM_variable.MST_ID))
print("VMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.VMAX))
print("TMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.TMAX))
print("Motor2:")
print("PMAX:",MotorControl1.read_motor_param(Motor2,DM_variable.PMAX))
print("MST_ID:",MotorControl1.read_motor_param(Motor2,DM_variable.MST_ID))
print("VMAX:",MotorControl1.read_motor_param(Motor2,DM_variable.VMAX))
print("TMAX:",MotorControl1.read_motor_param(Motor2,DM_variable.TMAX))
# MotorControl1.enable(Motor3)
MotorControl1.save_motor_param(Motor1)
MotorControl1.save_motor_param(Motor2)
MotorControl1.enable(Motor1)
MotorControl1.enable(Motor2)

while True:
    q=math.sin(time.time())
    target_vel1 = q*8 
    target_vel2 = 8*q

    # MotorControl1.control_pos_force(Motor1, 10, 1000,100)
    # MotorControl1.control_Vel(Motor1, q*5)
    MotorControl1.control_Pos_Vel(Motor1,target_vel1,30)
    # print("Motor1:","POS:",Motor1.getPosition(),"VEL:",Motor1.getVelocity(),"TORQUE:",Motor1.getTorque())
    # MotorControl1.controlMIT(Motor2, 35, 0.1, 8*q, 0, 0)

    MotorControl1.control_Vel(Motor2, target_vel2)
    
    # 实时发送到 PlotJuggler
    send_to_plotjuggler(target_vel1, Motor1.getVelocity(), target_vel2, Motor2.getVelocity())

    # print("Motor2:","POS:",Motor2.getPosition(),"VEL:",Motor2.getVelocity(),"TORQUE:",Motor2.getTorque())
    # print(Motor1.getTorque())
    # print(Motor2.getTorque())
    time.sleep(0.001)
    # MotorControl1.control(Motor3, 50, 0.3, q, 0, 0)

# 脚本结束关闭套接字
sock.close()
#语句结束关闭串口
serial_device.close()