#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import json
import numpy as np
import serial
from pathlib import Path

from DM_CAN import MotorControl, DM_Motor_Type, Control_Type
from dm_motor_driver import DMMotor
def send_mit_torque(mc, motor, tau_cmd):
    """Send pure torque command while holding current position."""
    st = motor.get_state()
    mc.controlMIT(
        DM_Motor=motor,
        kp=0.0,
        kd=0.0,
        q=float(st["position"]),
        dq=0.0,
        tau=float(tau_cmd),
    )
    return motor.get_state()


def run_ramped_sine_until_motion(
    mc,
    motor,
    sample_hz,
    duration,
    amp_start,
    amp_end,
    freq_hz,
    vel_threshold,
):
    """
    Apply a sine torque with linearly increasing amplitude and
    record multiple +/- motion onsets. After the first onset is detected,
    the sine amplitude is slightly increased to collect repeated start events.
    """
    dt = 1.0 / sample_hz
    pos_trigger_taus = []
    neg_trigger_taus = []
    pos_trigger_vels = []
    neg_trigger_vels = []
    extra_amp_scale = 1.2
    required_repeats = 3
    motion_latched = False
    current_sign = 0

    print("\n阶段1: 渐增正弦力矩辨识库仑摩擦")
    print(
        f"amp={amp_start:.3f}->{amp_end:.3f} Nm, "
        f"freq={freq_hz:.3f} Hz, vel_threshold={vel_threshold:.3f} rad/s"
    )

    start = time.time()
    while True:
        loop_start = time.time()
        t = loop_start - start
        if t >= duration:
            break

        amp = amp_start + (amp_end - amp_start) * (t / duration)
        if motion_latched:
            amp = min(amp_end, amp * extra_amp_scale)
        tau_cmd = amp * math.sin(2.0 * math.pi * freq_hz * t)
        st = send_mit_torque(mc, motor, tau_cmd)
        vel = float(st["velocity"])
        tau_meas = float(st["torque"])

        if abs(vel) < 0.5 * vel_threshold:
            current_sign = 0

        if vel > vel_threshold and current_sign != 1:
            current_sign = 1
            motion_latched = True
            pos_trigger_taus.append(tau_meas)
            pos_trigger_vels.append(vel)
            print(
                f"  检测到正向起转[{len(pos_trigger_taus)}]: "
                f"tau={tau_meas:.6f}, vel={vel:.6f}, t={t:.3f}s"
            )

        if vel < -vel_threshold and current_sign != -1:
            current_sign = -1
            motion_latched = True
            neg_trigger_taus.append(tau_meas)
            neg_trigger_vels.append(vel)
            print(
                f"  检测到反向起转[{len(neg_trigger_taus)}]: "
                f"tau={tau_meas:.6f}, vel={vel:.6f}, t={t:.3f}s"
            )

        if len(pos_trigger_taus) >= required_repeats and len(neg_trigger_taus) >= required_repeats:
            break

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    if not pos_trigger_taus or not neg_trigger_taus:
        raise RuntimeError("库仑摩擦辨识失败：未能同时检测到正反向起转。")

    tau_pos = float(np.mean(pos_trigger_taus))
    tau_neg = float(np.mean(neg_trigger_taus))
    tc_est = 0.5 * (tau_pos - tau_neg)
    tau_bias = 0.5 * (tau_pos + tau_neg)

    print(f"  正向起转均值   = {tau_pos:.6f} (n={len(pos_trigger_taus)})")
    print(f"  反向起转均值   = {tau_neg:.6f} (n={len(neg_trigger_taus)})")
    print(f"  Tc             = {tc_est:.6f}")
    print(f"  bias           = {tau_bias:.6f}")
    print(
        f"  起转速度均值   = +{float(np.mean(pos_trigger_vels)):.6f} / "
        f"{float(np.mean(neg_trigger_vels)):.6f}"
    )
    return tc_est, tau_bias


def run_sine_sweep_excitation(
    mc,
    motor,
    sample_hz,
    duration,
    torque_amp,
    freq_start,
    freq_end,
    label,
):
    """Run a sinusoidal torque sweep in MIT mode and log dynamic data."""
    dt = 1.0 / sample_hz
    t_log = []
    w_log = []
    tau_log = []

    print(f"\n{label}")
    print(
        f"正弦扫频力矩: amp={torque_amp:.3f} Nm, "
        f"freq={freq_start:.3f}->{freq_end:.3f} Hz, duration={duration:.1f}s"
    )

    start = time.time()
    while True:
        loop_start = time.time()
        t = loop_start - start
        if t >= duration:
            break

        freq = freq_start + (freq_end - freq_start) * (t / duration)
        tau_cmd = torque_amp * math.sin(2.0 * math.pi * freq * t)
        st = send_mit_torque(mc, motor, tau_cmd)

        t_log.append(t)
        w_log.append(float(st["velocity"]))
        tau_log.append(float(st["torque"]))

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return {
        "t": np.asarray(t_log, dtype=float),
        "w": np.asarray(w_log, dtype=float),
        "tau": np.asarray(tau_log, dtype=float),
    }


def identify_inertia_damping_from_sine_sweep(dataset, tc_est, tau_bias):
    """
    Identify J and B from sinusoidal sweep data using least squares:
        tau - Tc*sign(w) - bias = J*alpha + B*w
    """
    t = dataset["t"]
    w = dataset["w"]
    tau = dataset["tau"]
    alpha = np.gradient(w, t)

    mask = (np.abs(alpha) > 0.5) | (np.abs(w) > 0.5)
    if np.sum(mask) < 50:
        raise RuntimeError("惯量/阻尼辨识失败：扫频动态样本不足。")

    tau_eff = tau[mask] - tc_est * np.sign(w[mask]) - tau_bias
    A = np.column_stack((alpha[mask], w[mask]))
    x, _, _, _ = np.linalg.lstsq(A, tau_eff, rcond=None)
    j_est, b_est = x
    b_est = max(0.0, float(b_est))
    j_est = float(j_est)

    tau_fit = A @ np.array([j_est, b_est])
    residual = tau_eff - tau_fit
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    print("\n阶段2: 正弦扫频最小二乘辨识惯量和阻尼")
    print(f"  J    = {j_est:.6f}")
    print(f"  B    = {b_est:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  有效样本 = {int(np.sum(mask))}")
    return j_est, b_est, rmse, int(np.sum(mask))
def main():
    port = "/dev/ttyACM0"
    baudrate = 921600
    motor_id = 0x03
    master_id = 0x00
    motor_type = DM_Motor_Type.DM4340
    sample_hz = 200.0
    output_path = Path(__file__).resolve().parent / f"motor_params_id_0x{motor_id:02X}.json"

    print(f"打开串口: {port} @ {baudrate}")
    ser = serial.Serial(port, baudrate, timeout=0.1)
    mc = MotorControl(ser)

    motor = DMMotor(motor_type, motor_id, master_id)
    mc.addMotor(motor)

    print(f"配置电机 CAN ID=0x{motor_id:02X} 为 MIT 模式...")
    ok = mc.switchControlMode(motor, Control_Type.MIT)
    if not ok:
        raise RuntimeError("切换 MIT 模式失败。")

    print("使能电机...")
    mc.enable(motor)
    time.sleep(0.5)

    try:
        tc_est, tau_bias = run_ramped_sine_until_motion(
            mc=mc,
            motor=motor,
            sample_hz=sample_hz,
            duration=20.0,
            amp_start=0.4,
            amp_end=1.0,
            freq_hz=0.5,
            vel_threshold=0.3,
        )

        sweep_data = run_sine_sweep_excitation(
            mc=mc,
            motor=motor,
            sample_hz=sample_hz,
            duration=20.0,
            torque_amp=1.2,
            freq_start=0.1,
            freq_end=0.5,
            label="阶段2: 正弦扫频最小二乘回归辨识惯量和阻尼",
        )

        j_est, b_est, rmse, n_valid = identify_inertia_damping_from_sine_sweep(
            sweep_data,
            tc_est=tc_est,
            tau_bias=tau_bias,
        )

        print("\n辨识汇总:")
        print(f"  Tc (库仑摩擦) = {tc_est:.6f}")
        print(f"  bias          = {tau_bias:.6f}")
        print(f"  B (阻尼系数)  = {b_est:.6f}")
        print(f"  J (转子惯量)  = {j_est:.6f}")
        print(f"  RMSE          = {rmse:.6f}")
        print(f"  有效样本数    = {n_valid}")

        params = {
            "motor_id": motor_id,
            "motor_id_hex": f"0x{motor_id:02X}",
            "motor_type": motor_type.name,
            "tc": tc_est,
            "bias": tau_bias,
            "j": j_est,
            "b": b_est,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        print(f"\n参数已保存至 {output_path}")

    except KeyboardInterrupt:
        print("\n用户中断，停止测试。")
    finally:
        print("停止电机并失能...")
        try:
            st = motor.get_state()
            mc.controlMIT(
                DM_Motor=motor,
                kp=0.0,
                kd=0.0,
                q=float(st["position"]),
                dq=0.0,
                tau=0.0,
            )
            time.sleep(0.1)
            mc.disable(motor)
        finally:
            ser.close()
            print("测试结束。")


if __name__ == "__main__":
    main()
