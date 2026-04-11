# Copyright (c) 2025. Li Jianbin. All rights reserved.
# MIT License

"""
达妙电机 DM-J4310-2EC 图形控制界面
基于重构后的 dm_motor_driver.py (DM_CAN 封装版)
完整保留原版 GUI 布局与功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import serial.tools.list_ports
from dm_motor_driver import DMMotor, TrapezoidalMotionController
from DM_CAN import MotorControl, DM_Motor_Type, Control_Type

class MotorControlGUI:
    """电机控制图形界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("达妙电机控制界面 (适配版)")
        self.root.geometry("1200x850")

        # 变量
        self.ser = None
        self.mc = None
        self.motor = None
        self.controller = None
        self.connected = False
        self.enabled = False
        self.update_running = True

        # 运动控制变量
        self.motion_active = False
        self.target_position = 0.0
        self.control_lock = threading.Lock()

        # 力矩控制变量
        self.torque_control_active = False
        self.target_torque = 0.0
        self.torque_control_lock = threading.Lock()

        # 保存的位置
        self.saved_position = None

        # 创建界面
        self.create_widgets()

        # 启动状态更新线程
        self.update_thread = threading.Thread(target=self.update_status_loop, daemon=True)
        self.update_thread.start()

    def get_available_ports(self):
        ports = serial.tools.list_ports.comports()
        return [f"{p.device} - {p.description}" if p.description else p.device for p in ports]

    def refresh_ports(self):
        available_ports = self.get_available_ports()
        self.port_combo['values'] = available_ports
        if available_ports:
            self.port_var.set(available_ports[0])
        else:
            self.port_var.set("")

    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # ===== 连接配置区 =====
        connection_frame = ttk.LabelFrame(self.root, text="连接配置", padding=10)
        connection_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        ttk.Label(connection_frame, text="串口:").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(connection_frame, textvariable=self.port_var, state="readonly", width=35)
        self.port_combo.grid(row=0, column=1, padx=5)
        self.refresh_btn = ttk.Button(connection_frame, text="刷新", command=self.refresh_ports, width=6)
        self.refresh_btn.grid(row=0, column=2, padx=5)

        ttk.Label(connection_frame, text="波特率:").grid(row=0, column=3, sticky="w", padx=(20, 0))
        self.baudrate_var = tk.StringVar(value="921600")
        ttk.Entry(connection_frame, textvariable=self.baudrate_var, width=10).grid(row=0, column=4, padx=5)

        ttk.Label(connection_frame, text="电机类型:").grid(row=1, column=0, pady=(10, 0))
        self.motor_type_var = tk.StringVar(value="DM-J4310-2EC")
        ttk.Combobox(connection_frame, textvariable=self.motor_type_var, values=["DM-J4310-2EC", "DM4340"], state="readonly", width=15).grid(row=1, column=1, pady=(10, 0), sticky="w")

        self.connect_btn = ttk.Button(connection_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.grid(row=1, column=3, pady=(10, 0))

        # ===== 电机参数配置区 =====
        param_frame = ttk.LabelFrame(self.root, text="电机参数配置", padding=10)
        param_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        ttk.Label(param_frame, text="电机ID:").grid(row=0, column=0)
        self.motor_id_var = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=self.motor_id_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Master ID:").grid(row=0, column=2, padx=(20, 0))
        self.master_id_var = tk.StringVar(value="0")
        ttk.Entry(param_frame, textvariable=self.master_id_var, width=8).grid(row=0, column=3, padx=5)

        # ===== 控制区 =====
        control_frame = ttk.LabelFrame(self.root, text="基本控制", padding=10)
        control_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.enable_btn = ttk.Button(control_frame, text="使能电机", command=self.toggle_enable, state="disabled")
        self.enable_btn.grid(row=0, column=0, padx=5)
        self.status_label = ttk.Label(control_frame, text="状态: 未连接", foreground="gray")
        self.status_label.grid(row=0, column=1, padx=20)

        utility_frame = ttk.Frame(control_frame)
        utility_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        self.set_zero_btn = ttk.Button(utility_frame, text="设置零点", command=self.set_zero, state="disabled")
        self.set_zero_btn.grid(row=0, column=0, padx=5)
        self.clear_error_btn = ttk.Button(utility_frame, text="清除错误", command=self.clear_error, state="disabled")
        self.clear_error_btn.grid(row=0, column=1, padx=5)
        self.save_pos_btn = ttk.Button(utility_frame, text="保存当前位置", command=self.save_position, state="disabled")
        self.save_pos_btn.grid(row=0, column=2, padx=5)
        self.goto_saved_btn = ttk.Button(utility_frame, text="回到保存位置", command=self.goto_saved_position, state="disabled")
        self.goto_saved_btn.grid(row=0, column=3, padx=5)

        # ===== 位置控制区 (MIT 模式) =====
        pos_frame = ttk.LabelFrame(self.root, text="位置控制 (MIT 模式 - 梯形加减速)", padding=10)
        pos_frame.grid(row=3, column=0, padx=(10, 5), pady=5, sticky="nsew")

        params_subframe = ttk.Frame(pos_frame)
        params_subframe.grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(params_subframe, text="Kp:").grid(row=0, column=0)
        self.kp_var = tk.StringVar(value="40.0")
        ttk.Entry(params_subframe, textvariable=self.kp_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(params_subframe, text="Kd:").grid(row=0, column=2, padx=10)
        self.kd_var = tk.StringVar(value="1.0")
        ttk.Entry(params_subframe, textvariable=self.kd_var, width=8).grid(row=0, column=3, padx=5)
        ttk.Label(params_subframe, text="Max Vel:").grid(row=1, column=0, pady=5)
        self.max_v_var = tk.StringVar(value="8.0")
        ttk.Entry(params_subframe, textvariable=self.max_v_var, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(params_subframe, text="Max Accel:").grid(row=1, column=2, padx=10)
        self.max_a_var = tk.StringVar(value="15.0")
        ttk.Entry(params_subframe, textvariable=self.max_a_var, width=8).grid(row=1, column=3, padx=5)

        ttk.Separator(pos_frame, orient="horizontal").grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(pos_frame, text="目标位置:").grid(row=2, column=0)
        self.target_pos_label = ttk.Label(pos_frame, text="-- rad", font=("Arial", 10, "bold"))
        self.target_pos_label.grid(row=2, column=1)

        btn_frame = ttk.Frame(pos_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        self.move_neg_btn = tk.Button(btn_frame, text="反转 (-12 rad)", bg="#ffcccc", width=15, height=2, state="disabled")
        self.move_neg_btn.pack(side="left", padx=5)
        self.move_neg_btn.bind("<ButtonPress-1>", lambda e: self.on_move_press(-12.0))
        self.move_neg_btn.bind("<ButtonRelease-1>", lambda e: self.on_move_release())
        self.move_pos_btn = tk.Button(btn_frame, text="正转 (+12 rad)", bg="#ccffcc", width=15, height=2, state="disabled")
        self.move_pos_btn.pack(side="left", padx=5)
        self.move_pos_btn.bind("<ButtonPress-1>", lambda e: self.on_move_press(12.0))
        self.move_pos_btn.bind("<ButtonRelease-1>", lambda e: self.on_move_release())

        self.motion_st_label = ttk.Label(pos_frame, text="运动状态: 静止", foreground="gray")
        self.motion_st_label.grid(row=4, column=0, columnspan=2)

        # ===== 力矩控制区 =====
        tau_frame = ttk.LabelFrame(self.root, text="力矩控制 (MIT 模式)", padding=10)
        tau_frame.grid(row=3, column=1, padx=(5, 10), pady=5, sticky="nsew")
        self.torque_var = tk.StringVar(value="0.0")
        ttk.Label(tau_frame, text="目标力矩 (Nm):").grid(row=0, column=0)
        ttk.Entry(tau_frame, textvariable=self.torque_var, width=10).grid(row=0, column=1)
        self.torque_scale = tk.Scale(tau_frame, from_=-10.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, length=300, command=self.on_torque_scale)
        self.torque_scale.grid(row=1, column=0, columnspan=2, pady=10)
        self.start_tau_btn = ttk.Button(tau_frame, text="启动力矩控制", command=self.start_torque, state="disabled")
        self.start_tau_btn.grid(row=2, column=0)
        self.stop_tau_btn = ttk.Button(tau_frame, text="停止力矩控制", command=self.stop_torque, state="disabled")
        self.stop_tau_btn.grid(row=2, column=1)
        self.tau_st_label = ttk.Label(tau_frame, text="力矩控制: 未激活", foreground="gray")
        self.tau_st_label.grid(row=3, column=0, columnspan=2, pady=5)

        # ===== 状态显示区 =====
        disp_frame = ttk.LabelFrame(self.root, text="电机实时反馈", padding=10)
        disp_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.fb_pos = ttk.Label(disp_frame, text="位置: -- rad", font=("Arial", 10))
        self.fb_pos.grid(row=0, column=0, padx=20)
        self.fb_vel = ttk.Label(disp_frame, text="速度: -- rad/s", font=("Arial", 10))
        self.fb_vel.grid(row=0, column=1, padx=20)
        self.fb_tau = ttk.Label(disp_frame, text="力矩: -- Nm", font=("Arial", 10))
        self.fb_tau.grid(row=0, column=2, padx=20)
        self.fb_err = ttk.Label(disp_frame, text="错误: --", font=("Arial", 10))
        self.fb_err.grid(row=0, column=3, padx=20)

        self.refresh_ports()

    def toggle_connection(self):
        if not self.connected:
            try:
                import serial
                port_full = self.port_var.get()
                port = port_full.split(" - ")[0] if " - " in port_full else port_full
                baud = int(self.baudrate_var.get())
                mid = int(self.motor_id_var.get())
                mst = int(self.master_id_var.get())

                self.ser = serial.Serial(port, baud, timeout=0.1)
                self.mc = MotorControl(self.ser)
                self.motor = DMMotor(DM_Motor_Type.DM4310, mid, mst)
                self.mc.addMotor(self.motor)
                self.controller = TrapezoidalMotionController(self.mc, self.motor)

                self.connected = True
                self.connect_btn.config(text="断开")
                self.enable_btn.config(state="normal")
                self.status_label.config(text="状态: 已连接", foreground="green")
                self.set_zero_btn.config(state="normal")
                self.clear_error_btn.config(state="normal")
                messagebox.showinfo("成功", "已连接到电机")
            except Exception as e:
                messagebox.showerror("错误", str(e))
        else:
            self.stop_torque()
            if self.enabled: self.toggle_enable()
            self.connected = False
            if self.ser: self.ser.close()
            self.connect_btn.config(text="连接")
            self.enable_btn.config(state="disabled")
            self.status_label.config(text="状态: 未连接", foreground="gray")

    def toggle_enable(self):
        if not self.motor: return
        if not self.enabled:
            self.mc.switchControlMode(self.motor, Control_Type.MIT)
            self.mc.enable(self.motor)
            self.enabled = True
            self.enable_btn.config(text="失能电机")
            self.move_neg_btn.config(state="normal")
            self.move_pos_btn.config(state="normal")
            self.save_pos_btn.config(state="normal")
            self.start_tau_btn.config(state="normal")
        else:
            self.controller.stop()
            self.stop_torque()
            self.mc.disable(self.motor)
            self.enabled = False
            self.enable_btn.config(text="使能电机")
            self.move_neg_btn.config(state="disabled")
            self.move_pos_btn.config(state="disabled")
            self.start_tau_btn.config(state="disabled")

    def set_zero(self):
        if self.mc: self.mc.set_zero_position(self.motor)

    def clear_error(self):
        # DM_CAN.py 中虽然没有显式 clear_error，但 enable 指令通常带清除
        self.mc.enable(self.motor)

    def save_position(self):
        if self.motor:
            self.saved_position = self.motor.state_q
            self.goto_saved_btn.config(state="normal")
            messagebox.showinfo("位置已保存", f"当前位置 {self.saved_position:.3f} 已保存")

    def goto_saved_position(self):
        if self.saved_position is not None:
            self.on_move_press(self.saved_position)

    def on_move_press(self, target):
        if not self.enabled: return
        self.target_position = target
        self.target_pos_label.config(text=f"{target:.2f} rad")
        self.controller.set_motion_params(float(self.max_v_var.get()), float(self.max_a_var.get()))
        self.controller.set_control_params(float(self.kp_var.get()), float(self.kd_var.get()))
        threading.Thread(target=lambda: self.controller.move_to_position(target, blocking=False), daemon=True).start()

    def on_move_release(self):
        # 原版逻辑：松开时计算减速距离并设为新目标
        if not self.enabled: return
        v = self.motor.state_dq
        a = float(self.max_a_var.get())
        decel_dist = (v ** 2) / (2 * a)
        stop_pos = self.motor.state_q + (decel_dist if v > 0 else -decel_dist)
        self.controller.move_to_position(stop_pos, blocking=False)
        self.target_pos_label.config(text=f"{stop_pos:.2f} rad [停止]")

    def on_torque_scale(self, val):
        self.target_torque = float(val)
        self.torque_var.set(f"{self.target_torque:.1f}")

    def start_torque(self):
        self.controller.stop()
        self.torque_control_active = True
        self.start_tau_btn.config(state="disabled")
        self.stop_tau_btn.config(state="normal")
        self.tau_st_label.config(text="力矩控制: 激活", foreground="green")
        # 启动力矩维持线程
        threading.Thread(target=self.torque_loop, daemon=True).start()

    def stop_torque(self):
        self.torque_control_active = False
        self.start_tau_btn.config(state="normal")
        self.stop_tau_btn.config(state="disabled")
        self.tau_st_label.config(text="力矩控制: 未激活", foreground="gray")

    def torque_loop(self):
        while self.torque_control_active and self.enabled:
            # MIT 模式下 Kp=0, Kd=0 即为纯力矩控制 (Tau 为前馈)
            self.mc.controlMIT(self.motor, 0, 0, 0, 0, self.target_torque)
            time.sleep(0.01)

    def update_status_loop(self):
        while self.update_running:
            if self.connected and self.motor:
                try:
                    # 在使能状态下，如果没有任何指令发送，则发送读取状态包以触发电机反馈
                    # 达妙电机的反馈是基于“问答式”的，即发送一个控制包，电机回传一个状态包
                    if self.enabled and not self.controller.is_motion_active() and not self.torque_control_active:
                        # 调用 DM_CAN 中的 refresh_motor_status
                        # 它通过发送 0xCC 指令强制触发反馈，且不带力矩/位置控制量，不影响控制器运行
                        self.mc.refresh_motor_status(self.motor)
                    else:
                        # 如果正在运动，控制包本身就会触发反馈，仅需接收
                        self.mc.recv()
                    
                    st = self.motor.get_state()
                    self.fb_pos.config(text=f"位置: {st['position']:.3f} rad")
                    self.fb_vel.config(text=f"速度: {st['velocity']:.3f} rad/s")
                    self.fb_tau.config(text=f"力矩: {st['tau']:.3f} Nm" if 'tau' in st else f"力矩: {st['torque']:.3f} Nm")
                    self.motion_st_label.config(text="运动状态: 运行中" if self.controller.is_motion_active() else "运动状态: 静止")
                except Exception as e:
                    print(f"Update loop error: {e}")
            time.sleep(0.04) # 约 25Hz 刷新率

    def on_closing(self):
        self.update_running = False
        self.on_move_release()
        if self.connected: self.toggle_connection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotorControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
