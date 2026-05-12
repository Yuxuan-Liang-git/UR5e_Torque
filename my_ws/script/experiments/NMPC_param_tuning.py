#!/usr/bin/env python3
"""
NMPC 参数调试实机脚本。

脚本流程：
  1. 读取 config/home_pos.txt，将机械臂用关节 PD 拉到 home 姿态。
  2. 以 home 姿态为基准，对 JOINT_SELECT 中的关节叠加傅里叶级数参考。
  3. 先使用 NMPC + 局部 PD 稳定器进行 10s 跟踪测试。
  4. 回到 home 姿态后，使用关节 PD 对同一条正弦参考重复一次实验。
  5. 退出时先清零力矩并停止 UR 脚本，再保存 raw CSV、metrics 和每个测试轴的对比 PNG。

注意：
  - JOINT_SELECT 使用 Python 0-based 关节编号，例如 4/5 表示第 5/6 轴。
  - 当前傅里叶参考最大幅值会超过单项 FOURIER_ALPHA，实机前请先确认关节范围安全。
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Controller import FrictionCompensator, NMPCController, PDJointController  # noqa: E402

# ======================== 手动调参入口 ========================
# 0-based 关节编号：4/5 表示第 5/6 轴。
JOINT_SELECT = [4, 5]
# 每个关节都可以单独设置傅里叶级数参数；未选中关节即使配置了也不会运动。
FOURIER_FREQ = np.array(
    [
        [0.4, 0.7, 1.0],
        [0.4, 0.7, 1.0],
        [0.4, 0.7, 1.0],
        [0.4, 0.7, 1.0],
        [0.4, 0.7, 1.0],
        [0.4, 1.0, 1.5],
    ],
    dtype=float,
)
FOURIER_ALPHA = np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.7], dtype=float)
FOURIER_PHASE_DEG = np.array(
    [
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0],
    ],
    dtype=float,
)
RUN_SECONDS = 10.0
# ==============================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数，保留少量运行时可调项。"""

    parser = argparse.ArgumentParser(description="UR5e NMPC sine tracking parameter tuning.")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--home-pos", default="config/home_pos.txt", help="Home joint position file")
    parser.add_argument("--alpha", type=float, default=0.1, help="局部 PD 稳定器权重")
    parser.add_argument("--reset-seconds", type=float, default=3.0, help="实验前拉回 home 姿态的时间")
    parser.add_argument("--duration", type=float, default=RUN_SECONDS, help="NMPC 正弦测试时长 [s]")
    parser.add_argument("--output-dir", default="Data/NMPC_param_tuning", help="实验输出目录")
    parser.add_argument("--trial", default=None, help="可选文件名后缀；默认使用时间戳")
    parser.add_argument("--skip-pd", action="store_true", help="只运行 NMPC，不运行 PD 对比")
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    """将相对路径解析到仓库根目录下。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_config_path(config_path: Path, path_value: str | Path) -> Path:
    """解析 yaml 中的相对路径，兼容相对仓库根目录和相对 config 目录。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (
        REPO_ROOT / path,
        config_path.resolve().parent / path,
        config_path.resolve().parent.parent / path,
    ):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def load_yaml(config_path: Path) -> dict:
    """读取 yaml 配置文件。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_joint_position(path: Path) -> np.ndarray:
    """读取 6 维关节姿态文件。"""

    if not path.exists():
        raise FileNotFoundError(f"Joint position file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip().replace(",", " ")
    q = np.asarray([float(x) for x in line.split()], dtype=float)
    if q.shape != (6,):
        raise ValueError(f"Expected 6 joint values in {path}, got {q.shape[0]}")
    return q


def validate_sine_params() -> None:
    """检查手动设置的傅里叶参考参数是否一致。"""

    if np.asarray(FOURIER_FREQ).shape != (6, 3):
        raise ValueError("FOURIER_FREQ must have shape (6, 3)")
    if np.asarray(FOURIER_ALPHA).shape != (6,):
        raise ValueError("FOURIER_ALPHA must contain 6 values")
    if np.asarray(FOURIER_PHASE_DEG).shape != (6, 3):
        raise ValueError("FOURIER_PHASE_DEG must have shape (6, 3)")
    if np.any(FOURIER_FREQ < 0.0):
        raise ValueError("FOURIER_FREQ values must be >= 0")
    if np.any(FOURIER_ALPHA < 0.0):
        raise ValueError("FOURIER_ALPHA values must be >= 0")
    for joint in JOINT_SELECT:
        if not 0 <= int(joint) < 6:
            raise ValueError(f"JOINT_SELECT contains invalid joint index: {joint}")


def init_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    """创建 raw、figures、metrics 输出目录。"""

    raw_dir = output_dir / "raw"
    fig_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    for directory in (raw_dir, fig_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return raw_dir, fig_dir, metrics_dir


def axes_tag() -> str:
    """生成文件名中的测试轴标签。"""

    return "_".join(str(int(j)) for j in JOINT_SELECT)


def sine_reference(q_home: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """生成当前时刻的 6 维傅里叶级数参考位置和速度。"""

    q_ref = q_home.copy()
    dq_ref = np.zeros(6)
    for joint in JOINT_SELECT:
        j = int(joint)
        phases = np.deg2rad(FOURIER_PHASE_DEG[j])
        signal = 0.0
        signal_dot = 0.0
        for harmonic, phase in enumerate(phases, start=1):
            coeff = FOURIER_ALPHA[j] / float(harmonic**2)
            omega = 2.0 * np.pi * FOURIER_FREQ[j, harmonic - 1]
            angle = omega * t + phase
            signal += coeff * np.sin(angle)
            signal_dot += coeff * omega * np.cos(angle)
        q_ref[j] = q_home[j] + signal
        dq_ref[j] = signal_dot
    return q_ref, dq_ref


def build_ref_batch(q_home: np.ndarray, elapsed: float, horizon_steps: int, dt: float) -> np.ndarray:
    """为 NMPC 构造 N+1 个未来正弦参考点。"""

    ref_q = np.zeros((6, horizon_steps + 1))
    for k in range(horizon_steps + 1):
        qk, _ = sine_reference(q_home, elapsed + k * dt)
        ref_q[:, k] = qk
    return ref_q


def warm_start_nmpc(nmpc: NMPCController, q: np.ndarray, dq: np.ndarray) -> None:
    """用当前状态和重力力矩暖启动 NMPC。"""

    x0 = np.concatenate([q, dq])
    gravity = nmpc.compute_gravity_torque(q)
    for k in range(nmpc.N + 1):
        nmpc.solver.set(k, "x", x0)
    for k in range(nmpc.N):
        nmpc.solver.set(k, "u", gravity)


def stop_robot_safely(rtde_c: RTDEControlInterface | None) -> None:
    """退出 directTorque 时先清零，再停止 UR 脚本。"""

    if rtde_c is None:
        return
    try:
        zero_tau = [0.0] * 6
        for _ in range(3):
            rtde_c.directTorque(zero_tau, False)
            time.sleep(0.01)
        rtde_c.stopScript()
        print("[INFO] Robot torque script stopped.")
    except Exception as exc:
        print(f"[WARN] Failed to stop robot torque script cleanly: {exc}")


def send_zero_torque(rtde_c: RTDEControlInterface | None) -> None:
    """两段实验切换时短暂清零力矩，但不停止 UR directTorque 脚本。"""

    if rtde_c is None:
        return
    try:
        zero_tau = [0.0] * 6
        for _ in range(3):
            rtde_c.directTorque(zero_tau, False)
            time.sleep(0.01)
    except Exception as exc:
        print(f"[WARN] Failed to send zero torque: {exc}")


def hold_joint_position(
    rtde_c: RTDEControlInterface,
    rtde_r: RTDEReceiveInterface,
    pd_controller: PDJointController,
    friction: FrictionCompensator,
    q_hold: np.ndarray,
    torque_limits: np.ndarray,
    dt: float,
    seconds: float,
) -> None:
    """用关节 PD 平滑拉回并保持在 home 姿态。"""

    if seconds <= 0.0:
        return

    print(f"[INFO] Resetting/holding home posture for {seconds:.1f}s")
    start = time.perf_counter()
    next_tick = start
    q_begin = np.asarray(rtde_r.getActualQ(), dtype=float)
    while time.perf_counter() - start < seconds:
        elapsed = time.perf_counter() - start
        alpha = np.clip(elapsed / max(seconds, 1e-6), 0.0, 1.0)
        q_des = (1.0 - alpha) * q_begin + alpha * q_hold
        q = np.asarray(rtde_r.getActualQ(), dtype=float)
        dq = np.asarray(rtde_r.getActualQd(), dtype=float)
        tau_pd = pd_controller.compute_torque(q_des, q, np.zeros(6), dq)
        tau_fric = friction.compute_torque(dq)
        tau = np.clip(tau_pd + 0.0 * tau_fric, -torque_limits, torque_limits)
        ok = rtde_c.directTorque(tau.tolist(), False)
        if not ok:
            raise RuntimeError("directTorque failed during home reset phase")

        next_tick += dt
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()


def make_fieldnames() -> list[str]:
    """生成 raw CSV 字段名。"""

    fields = ["t"]
    for prefix in ("q_ref", "q", "dq", "tau", "tau_nmpc", "tau_nmpc_pd", "tau_pd", "q_nmpc", "dq_nmpc"):
        fields.extend([f"{prefix}_{j}" for j in range(6)])
    fields.extend(["solve_time_ms", "control_dt_ms"])
    return fields


def row_from_sample(
    t: float,
    q_ref: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    tau: np.ndarray,
    tau_nmpc: np.ndarray,
    tau_nmpc_pd: np.ndarray,
    tau_pd: np.ndarray,
    q_nmpc: np.ndarray,
    dq_nmpc: np.ndarray,
    solve_time_ms: float,
    control_dt_ms: float,
) -> dict[str, float]:
    """将一个控制周期样本展开成 CSV 行。"""

    row: dict[str, float] = {"t": float(t)}
    for prefix, values in (
        ("q_ref", q_ref),
        ("q", q),
        ("dq", dq),
        ("tau", tau),
        ("tau_nmpc", tau_nmpc),
        ("tau_nmpc_pd", tau_nmpc_pd),
        ("tau_pd", tau_pd),
        ("q_nmpc", q_nmpc),
        ("dq_nmpc", dq_nmpc),
    ):
        for j in range(6):
            row[f"{prefix}_{j}"] = float(values[j])
    row["solve_time_ms"] = float(solve_time_ms)
    row["control_dt_ms"] = float(control_dt_ms)
    return row


def load_log(path: Path) -> dict[str, np.ndarray]:
    """读取 raw CSV。"""

    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names}


def compute_joint_metrics(log: dict[str, np.ndarray], joint: int, controller: str) -> dict[str, float]:
    """计算单个测试关节的跟踪误差和力矩指标。"""

    t = log["t"]
    error = log[f"q_ref_{joint}"] - log[f"q_{joint}"]
    tau = log[f"tau_{joint}"]
    dtau = np.diff(tau, prepend=tau[0])
    control_dt = log["control_dt_ms"]
    solve_time = log["solve_time_ms"]

    valid_control_dt = control_dt[1:] if len(control_dt) > 1 else control_dt
    valid_control_dt = valid_control_dt[valid_control_dt > 0]
    mean_freq = float(1000.0 / np.mean(valid_control_dt)) if valid_control_dt.size else 0.0
    min_freq = float(1000.0 / np.max(valid_control_dt)) if valid_control_dt.size else 0.0

    return {
        "controller": controller,
        "joint": float(joint),
        "RMSE_q": float(np.sqrt(np.mean(error**2))),
        "RMS_error": float(np.sqrt(np.mean(error**2))),
        "Max_abs_error": float(np.max(np.abs(error))),
        "Tau_RMS": float(np.sqrt(np.mean(tau**2))),
        "Tau_peak": float(np.max(np.abs(tau))),
        "dTau_RMS": float(np.sqrt(np.mean(dtau**2))),
        "Mean_solve_time_ms": float(np.mean(solve_time)),
        "Max_solve_time_ms": float(np.max(solve_time)),
        "Mean_control_freq": mean_freq,
        "Min_control_freq": min_freq,
    }


def params_text(cfg: dict, alpha: float, joint: int) -> str:
    """生成图标题中的 NMPC 参数文本。"""

    nmpc_cfg = cfg.get("nmpc_controller", {})
    q_weight = nmpc_cfg.get("weight_q", [])
    v_weight = nmpc_cfg.get("weight_vel", [])
    r_weight = nmpc_cfg.get("weight_tau", [])
    return (
        f"Q={q_weight}\n"
        f"V={v_weight}\n"
        f"R={r_weight}, alpha={alpha:.3g}\n"
        f"joint={joint}, Fourier f={FOURIER_FREQ[joint].tolist()}Hz, "
        f"a={FOURIER_ALPHA[joint]:.3g}rad, phi={FOURIER_PHASE_DEG[joint].tolist()}deg"
    )


def save_joint_plot(
    fig_dir: Path,
    run_id: str,
    nmpc_log: dict[str, np.ndarray],
    nmpc_metrics: dict[str, float],
    pd_log: dict[str, np.ndarray] | None,
    pd_metrics: dict[str, float] | None,
    cfg: dict,
    alpha: float,
    joint: int,
) -> Path:
    """保存单个测试关节的 NMPC/PD 跟踪、误差和力矩对比图。"""

    t_nmpc = nmpc_log["t"]
    q_ref_nmpc = nmpc_log[f"q_ref_{joint}"]
    q_nmpc_actual = nmpc_log[f"q_{joint}"]
    error_nmpc = q_ref_nmpc - q_nmpc_actual
    tau_nmpc_total = nmpc_log[f"tau_{joint}"]
    tau_nmpc_component = nmpc_log[f"tau_nmpc_{joint}"]
    tau_nmpc_pd_component = nmpc_log[f"tau_nmpc_pd_{joint}"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"NMPC sine tracking joint {joint}\n{params_text(cfg, alpha, joint)}", fontsize=10)

    axes[0].plot(t_nmpc, q_ref_nmpc, "k--", label="q_ref")
    axes[0].plot(t_nmpc, q_nmpc_actual, label="NMPC q")
    if pd_log is not None:
        axes[0].plot(pd_log["t"], pd_log[f"q_{joint}"], label="PD q")
    axes[0].set_ylabel("q [rad]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_nmpc, error_nmpc, label="NMPC error")
    if pd_log is not None:
        pd_error = pd_log[f"q_ref_{joint}"] - pd_log[f"q_{joint}"]
        axes[1].plot(pd_log["t"], pd_error, label="PD error")
    axes[1].set_ylabel("error [rad]")
    if pd_metrics is None:
        axes[1].set_title(
            f"NMPC RMSE={nmpc_metrics['RMSE_q']:.4g} rad, "
            f"Max={nmpc_metrics['Max_abs_error']:.4g} rad"
        )
    else:
        axes[1].set_title(
            f"NMPC RMSE={nmpc_metrics['RMSE_q']:.4g} rad, "
            f"PD RMSE={pd_metrics['RMSE_q']:.4g} rad"
        )
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_nmpc, tau_nmpc_total, label="NMPC tau")
    if pd_log is not None:
        axes[2].plot(pd_log["t"], pd_log[f"tau_{joint}"], label="PD tau")
    axes[2].set_ylabel("tau [Nm]")
    if pd_metrics is None:
        axes[2].set_title(
            f"NMPC Tau_RMS={nmpc_metrics['Tau_RMS']:.4g} Nm, "
            f"Tau_peak={nmpc_metrics['Tau_peak']:.4g} Nm"
        )
    else:
        axes[2].set_title(
            f"NMPC Tau_RMS={nmpc_metrics['Tau_RMS']:.4g} Nm, "
            f"PD Tau_RMS={pd_metrics['Tau_RMS']:.4g} Nm"
        )
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t_nmpc, tau_nmpc_component, label="NMPC tau_nmpc")
    axes[3].plot(t_nmpc, tau_nmpc_pd_component, label="NMPC tau_nmpc_pd")
    if pd_log is not None:
        axes[3].plot(pd_log["t"], pd_log[f"tau_pd_{joint}"], label="PD tau_pd", alpha=0.8)
    axes[3].set_xlabel("t [s]")
    axes[3].set_ylabel("component tau [Nm]")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    out = fig_dir / f"{run_id}_joint{joint}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_metrics(metrics_dir: Path, run_id: str, metrics: list[dict[str, float]]) -> Path:
    """保存每个测试关节的指标。"""

    out = metrics_dir / f"{run_id}_metrics.csv"
    keys = [
        "controller",
        "joint",
        "RMSE_q",
        "RMS_error",
        "Max_abs_error",
        "Tau_RMS",
        "Tau_peak",
        "dTau_RMS",
        "Mean_solve_time_ms",
        "Max_solve_time_ms",
        "Mean_control_freq",
        "Min_control_freq",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row[key] for key in keys})
    return out


def analyze_and_plot(
    nmpc_raw_path: Path,
    pd_raw_path: Path | None,
    fig_dir: Path,
    metrics_dir: Path,
    run_id: str,
    cfg: dict,
    alpha: float,
) -> None:
    """读取 raw CSV，生成每个测试关节的 NMPC/PD 对比图和汇总指标。"""

    nmpc_log = load_log(nmpc_raw_path)
    pd_log = load_log(pd_raw_path) if pd_raw_path is not None else None
    all_metrics = []
    for joint in JOINT_SELECT:
        j = int(joint)
        nmpc_metrics = compute_joint_metrics(nmpc_log, j, "NMPC")
        all_metrics.append(nmpc_metrics)
        if pd_log is not None:
            pd_metrics = compute_joint_metrics(pd_log, j, "PD")
            all_metrics.append(pd_metrics)
        else:
            pd_metrics = None
        fig_path = save_joint_plot(fig_dir, run_id, nmpc_log, nmpc_metrics, pd_log, pd_metrics, cfg, alpha, j)
        print(f"[INFO] Figure: {fig_path}")

    metrics_path = save_metrics(metrics_dir, run_id, all_metrics)
    print(f"[INFO] Metrics: {metrics_path}")
    print("\ncontroller,joint,RMSE_q,Max_abs_error,Tau_RMS,Tau_peak,dTau_RMS,Mean_solve_time_ms,Mean_control_freq")
    for row in all_metrics:
        print(
            f"{row['controller']},"
            f"{int(row['joint'])},"
            f"{row['RMSE_q']:.6g},"
            f"{row['Max_abs_error']:.6g},"
            f"{row['Tau_RMS']:.6g},"
            f"{row['Tau_peak']:.6g},"
            f"{row['dTau_RMS']:.6g},"
            f"{row['Mean_solve_time_ms']:.6g},"
            f"{row['Mean_control_freq']:.6g}"
        )


def run_controller_trial(
    controller: str,
    rtde_c: RTDEControlInterface,
    rtde_r: RTDEReceiveInterface,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nmpc_controller: NMPCController,
    pd_controller: PDJointController,
    q_home: np.ndarray,
    torque_limits: np.ndarray,
    dt: float,
    duration: float,
    alpha: float,
    raw_path: Path,
    keep_running: list[bool],
) -> None:
    """运行一次 NMPC 或 PD 正弦跟踪实验，并保存 raw CSV。"""

    controller_upper = controller.upper()
    print(f"[INFO] Running {controller_upper} sine tuning for {duration:.2f}s, joints={JOINT_SELECT}")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=make_fieldnames())
        writer.writeheader()

        start = time.perf_counter()
        next_tick = start
        last_tick = start

        while keep_running[0]:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed > duration:
                break

            q = np.asarray(rtde_r.getActualQ(), dtype=float)
            dq = np.asarray(rtde_r.getActualQd(), dtype=float)
            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)

            q_ref, dq_ref = sine_reference(q_home, elapsed)
            solve_time_ms = 0.0

            if controller == "nmpc":
                ref_q_batch = build_ref_batch(
                    q_home=q_home,
                    elapsed=elapsed,
                    horizon_steps=nmpc_controller.N,
                    dt=nmpc_controller.dt,
                )
                solve_start = time.perf_counter()
                tau_nmpc, q_nmpc, dq_nmpc = nmpc_controller.compute_torque(data, q, dq, ref_q_batch)
                solve_time_ms = (time.perf_counter() - solve_start) * 1000.0
                tau_nmpc_pd = alpha * pd_controller.compute_torque(q_nmpc, q, dq_nmpc, dq)
                tau_pd = np.zeros(6)
                tau = tau_nmpc + tau_nmpc_pd
            else:
                q_nmpc = np.zeros(6)
                dq_nmpc = np.zeros(6)
                tau_nmpc = np.zeros(6)
                tau_nmpc_pd = np.zeros(6)
                # tau_pd = pd_controller.compute_torque(q_ref, q, dq_ref, dq)
                tau_pd = pd_controller.compute_torque(q_ref, q, np.zeros(6), dq)

                tau = tau_pd

            tau = np.clip(tau, -torque_limits, torque_limits)
            ok = rtde_c.directTorque(tau.tolist(), False)
            if not ok:
                print(f"[ERROR] directTorque failed during {controller_upper}")
                break

            control_dt_ms = (now - last_tick) * 1000.0
            last_tick = now
            writer.writerow(
                row_from_sample(
                    t=elapsed,
                    q_ref=q_ref,
                    q=q,
                    dq=dq,
                    tau=tau,
                    tau_nmpc=tau_nmpc,
                    tau_nmpc_pd=tau_nmpc_pd,
                    tau_pd=tau_pd,
                    q_nmpc=q_nmpc,
                    dq_nmpc=dq_nmpc,
                    solve_time_ms=solve_time_ms,
                    control_dt_ms=control_dt_ms,
                )
            )

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()


def run() -> None:
    """主流程：初始化 home 姿态，运行 NMPC 正弦测试，安全退出后保存分析结果。"""

    validate_sine_params()
    args = parse_args()
    config_path = resolve_path(args.config)
    home_pos_path = resolve_path(args.home_pos)
    output_dir = resolve_path(args.output_dir)
    raw_dir, fig_dir, metrics_dir = init_output_dirs(output_dir)

    cfg = load_yaml(config_path)
    dt = float(cfg.get("trajectory", {}).get("control_dt", 0.01))
    if dt <= 0.0:
        raise ValueError("trajectory.control_dt must be > 0")

    torque_limits = np.asarray(
        cfg.get("safety", {}).get("torque_limits", [20, 20, 20, 20, 20, 20]),
        dtype=float,
    )
    pd_cfg = cfg.get("pdjoint_controller", {})
    kp = np.asarray(pd_cfg.get("kp", [0] * 6), dtype=float)
    kd = np.asarray(pd_cfg.get("kd", [0] * 6), dtype=float)
    q_home = load_joint_position(home_pos_path)
    print(f"[INFO] Loaded home posture from {home_pos_path}: {np.array2string(q_home, precision=4)}")

    sim_mjcf_path = cfg.get("simulation", {}).get("mjcf_path")
    if sim_mjcf_path is None:
        raise ValueError("simulation.mjcf_path must be set in config")
    xml_path = resolve_config_path(config_path, sim_mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    pd_controller = PDJointController(model=model, kp=kp, kd=kd, torque_limits=torque_limits)
    nmpc_controller = NMPCController(
        model=model,
        config_path=str(config_path),
        rebuild=bool(cfg.get("nmpc_controller", {}).get("rebuild", False)),
    )

    fric_cfg = cfg.get("friction_compensation", {})
    friction = FrictionCompensator(
        param_path=resolve_config_path(config_path, fric_cfg.get("param_file", "config/joint_fric_WLS.yaml")),
        enabled=bool(fric_cfg.get("enabled", True)),
        comp_factor=float(fric_cfg.get("comp_factor", 1.0)),
        vel_threshold=float(fric_cfg.get("vel_threshold", 0.01)),
    )

    timestamp = args.trial or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"nmpc_sine_{timestamp}_axes_{axes_tag()}"
    nmpc_raw_path = raw_dir / f"{run_id}_nmpc.csv"
    pd_raw_path = None if args.skip_pd else raw_dir / f"{run_id}_pd.csv"

    keep_running = [True]

    def signal_handler(_sig, _frame):
        keep_running[0] = False

    signal.signal(signal.SIGINT, signal_handler)

    rtde_c: RTDEControlInterface | None = None
    rtde_r: RTDEReceiveInterface | None = None

    try:
        print(f"[INFO] Connecting to robot {args.robot_ip}")
        rtde_c = RTDEControlInterface(args.robot_ip)
        rtde_r = RTDEReceiveInterface(args.robot_ip)

        hold_joint_position(
            rtde_c=rtde_c,
            rtde_r=rtde_r,
            pd_controller=pd_controller,
            friction=friction,
            q_hold=q_home,
            torque_limits=torque_limits,
            dt=dt,
            seconds=args.reset_seconds,
        )

        q_warm = np.asarray(rtde_r.getActualQ(), dtype=float)
        dq_warm = np.asarray(rtde_r.getActualQd(), dtype=float)
        warm_start_nmpc(nmpc_controller, q_warm, dq_warm)

        run_controller_trial(
            controller="nmpc",
            rtde_c=rtde_c,
            rtde_r=rtde_r,
            model=model,
            data=data,
            nmpc_controller=nmpc_controller,
            pd_controller=pd_controller,
            q_home=q_home,
            torque_limits=torque_limits,
            dt=dt,
            duration=args.duration,
            alpha=args.alpha,
            raw_path=nmpc_raw_path,
            keep_running=keep_running,
        )

        if keep_running[0] and pd_raw_path is not None:
            send_zero_torque(rtde_c)
            hold_joint_position(
                rtde_c=rtde_c,
                rtde_r=rtde_r,
                pd_controller=pd_controller,
                friction=friction,
                q_hold=q_home,
                torque_limits=torque_limits,
                dt=dt,
                seconds=args.reset_seconds,
            )
            run_controller_trial(
                controller="pd",
                rtde_c=rtde_c,
                rtde_r=rtde_r,
                model=model,
                data=data,
                nmpc_controller=nmpc_controller,
                pd_controller=pd_controller,
                q_home=q_home,
                torque_limits=torque_limits,
                dt=dt,
                duration=args.duration,
                alpha=args.alpha,
                raw_path=pd_raw_path,
                keep_running=keep_running,
            )

    finally:
        stop_robot_safely(rtde_c)
        if rtde_c is not None:
            try:
                rtde_c.disconnect()
            except Exception:
                pass

    print(f"[INFO] Saved NMPC raw log: {nmpc_raw_path}")
    if pd_raw_path is not None and pd_raw_path.exists():
        print(f"[INFO] Saved PD raw log: {pd_raw_path}")
    else:
        pd_raw_path = None
    analyze_and_plot(nmpc_raw_path, pd_raw_path, fig_dir, metrics_dir, run_id, cfg, args.alpha)


if __name__ == "__main__":
    run()
