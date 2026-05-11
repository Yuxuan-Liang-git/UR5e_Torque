#!/usr/bin/env python3
"""
基于本地 VLA 输出 CSV 的单关节最小实机测试脚本。

典型用法：
  python3 script/experiments/min_vla_single_joint.py --joint 2
  python3 script/experiments/min_vla_single_joint.py --controller both --joint 2
  python3 script/experiments/min_vla_single_joint.py --controller pd --joint 2
  python3 script/experiments/min_vla_single_joint.py --controller nmpc --joint 2
  python3 script/experiments/min_vla_single_joint.py --analyze-only --joint 2

脚本目标：
  1. 从 final_exec_160704.csv 中读取一个关节的 VLA 输出角度轨迹。
  2. 默认先跑 PD，再回到同一初始姿态，然后跑 NMPC。
  3. 自动保存 raw CSV、计算 RMSE/jerk/力矩变化等指标，并生成对比图。

安全默认设置：
  默认使用 offset 模式，即把 CSV 轨迹的“相对变化量”叠加到 init_pos.txt 姿态上，
  避免机械臂突然跳到 CSV 中记录的绝对姿态。
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 避免 matplotlib 尝试写入 ~/.config/matplotlib 导致权限告警。
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

# 复用现有控制器实现，避免在实验脚本中复制控制律。
from Controller import FrictionCompensator, NMPCController, PDJointController  # noqa: E402


@dataclass
class Trajectory:
    """统一保存重采样后的单关节参考轨迹。"""

    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray


def parse_args() -> argparse.Namespace:
    """解析命令行参数；默认配置尽量偏向安全的最小实机测试。"""

    parser = argparse.ArgumentParser(
        description="基于 VLA CSV 的单关节 PD/NMPC 最小实机对比测试。"
    )
    parser.add_argument("--controller", choices=["pd", "nmpc", "both"], default="both")
    parser.add_argument("--analyze-only", action="store_true", help="只分析已有的 PD/NMPC CSV 日志，不连接机械臂。")
    parser.add_argument("--robot-ip", default="192.168.56.101")
    parser.add_argument("--config", default="config/ctrl_config.yaml")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="固定初始姿态文件，默认读取 config/init_pos.txt。")
    parser.add_argument("--trajectory", default="config/trajectory/final_exec_160704.csv")
    parser.add_argument("--joint", type=int, default=2, help="被测关节下标，0-based；例如 2 对应 CSV 中 real_q3。")
    parser.add_argument(
        "--reference-mode",
        choices=["offset", "absolute"],
        default="offset",
        help="offset: init_pos姿态+CSV相对变化；absolute: 直接使用CSV绝对角度。",
    )
    parser.add_argument("--duration", type=float, default=None, help="运行时长，单位秒；默认使用完整 CSV 时长。")
    parser.add_argument("--max-delta", type=float, default=0.15, help="offset 模式下的最大相对角度限幅，单位 rad。")
    parser.add_argument("--alpha", type=float, default=0.0, help="NMPC 模式下 PD 稳定器权重。")
    parser.add_argument("--reset-seconds", type=float, default=3.0, help="both 模式下两次实验之间回到初始姿态的时间。")
    parser.add_argument("--trial", default="trial01")
    parser.add_argument(
        "--output-dir",
        default="Data/experiments/min_vla_single_joint",
        help="实验输出目录，内部会包含 raw/、figures/、metrics/。",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    """将相对路径统一解析到仓库根目录下。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_config_path(config_path: Path, path_value: str | Path) -> Path:
    """解析 yaml 中的路径字段，兼容相对仓库根目录和相对配置文件目录两种写法。"""

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


def load_config(config_path: Path) -> dict:
    """读取控制参数 yaml。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_init_q(init_pos_path: Path) -> np.ndarray:
    """从 init_pos.txt 读取固定初始关节姿态。"""

    if not init_pos_path.exists():
        raise FileNotFoundError(f"Init position file not found: {init_pos_path}")
    with open(init_pos_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Init position file is empty: {init_pos_path}")
    q_init = np.asarray([float(x) for x in line.split()], dtype=float)
    if q_init.shape != (6,):
        raise ValueError(f"Init position must contain 6 joint values, got {q_init.shape[0]}: {init_pos_path}")
    return q_init


def load_vla_joint_trajectory(csv_path: Path, joint: int, control_dt: float, duration: float | None) -> Trajectory:
    """读取 CSV 中指定关节的 VLA 输出轨迹，并重采样到控制周期。"""

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    time_col = "timestamp"
    joint_col = f"real_q{joint + 1}"
    if time_col not in data.dtype.names:
        raise ValueError(f"Missing '{time_col}' column in {csv_path}")
    if joint_col not in data.dtype.names:
        raise ValueError(f"Missing '{joint_col}' column in {csv_path}")

    # CSV 里时间戳是绝对时间，这里转成从 0 开始的相对时间。
    t_raw = np.asarray(data[time_col], dtype=float)
    q_raw = np.asarray(data[joint_col], dtype=float)
    t_raw = t_raw - t_raw[0]

    if duration is None:
        duration = float(t_raw[-1])
    duration = min(float(duration), float(t_raw[-1]))
    if duration <= 0.0:
        raise ValueError("Trajectory duration must be positive")

    # 将 VLA 轨迹重采样到控制周期，保证 PD 和 NMPC 使用完全一致的参考。
    t = np.arange(0.0, duration + 0.5 * control_dt, control_dt)
    q = np.interp(t, t_raw, q_raw)
    dq = np.gradient(q, control_dt)
    return Trajectory(t=t, q=q, dq=dq)


def make_reference_at(
    traj: Trajectory,
    elapsed: float,
    q_start: np.ndarray,
    joint: int,
    reference_mode: str,
    max_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """根据当前运行时间生成 6 维参考，只让被测关节跟随 CSV。"""

    q_ref = q_start.copy()
    dq_ref = np.zeros(6)

    q_csv = np.interp(elapsed, traj.t, traj.q)
    dq_csv = np.interp(elapsed, traj.t, traj.dq)
    if reference_mode == "offset":
        # 安全模式：只取 CSV 的相对变化量，叠加到当前实机初始姿态上。
        # max_delta 用来防止 CSV 轨迹幅度过大时突然给出危险参考。
        delta = np.clip(q_csv - traj.q[0], -max_delta, max_delta)
        q_ref[joint] = q_start[joint] + delta
    else:
        # 绝对模式：直接使用 CSV 中记录的关节角。实机使用前需要确认起始姿态安全。
        q_ref[joint] = q_csv

    dq_ref[joint] = dq_csv
    return q_ref, dq_ref


def build_ref_batch(
    traj: Trajectory,
    elapsed: float,
    q_start: np.ndarray,
    joint: int,
    reference_mode: str,
    max_delta: float,
    horizon_steps: int,
    dt: float,
) -> np.ndarray:
    """为 NMPC 构造 N+1 个预测窗口参考点。"""

    ref_q = np.zeros((6, horizon_steps + 1))
    for k in range(horizon_steps + 1):
        qk, _ = make_reference_at(
            traj=traj,
            elapsed=min(elapsed + k * dt, traj.t[-1]),
            q_start=q_start,
            joint=joint,
            reference_mode=reference_mode,
            max_delta=max_delta,
        )
        ref_q[:, k] = qk
    return ref_q


def init_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    """创建 raw、figures、metrics 三类实验输出目录。"""

    raw_dir = output_dir / "raw"
    fig_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    for d in (raw_dir, fig_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)
    return raw_dir, fig_dir, metrics_dir


def warm_start_nmpc(nmpc: NMPCController, q: np.ndarray, dq: np.ndarray) -> None:
    """用当前机器人状态和重力力矩暖启动 NMPC，减少第一次求解突变。"""

    x0 = np.concatenate([q, dq])
    gravity = nmpc.compute_gravity_torque(q)
    for k in range(nmpc.N + 1):
        nmpc.solver.set(k, "x", x0)
    for k in range(nmpc.N):
        nmpc.solver.set(k, "u", gravity)


def clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    """复制命令行参数对象，并覆盖指定字段。"""

    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


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
    """用关节 PD 平滑拉回并保持在指定姿态，主要用于实验开始前和两组实验之间复位。"""

    if seconds <= 0.0:
        return

    print(f"[INFO] Resetting/holding initial posture for {seconds:.1f}s")
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
            raise RuntimeError("directTorque failed during reset/hold phase")

        next_tick += dt
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()


def run_experiment(args: argparse.Namespace, q_start_override: np.ndarray | None = None) -> tuple[Path, np.ndarray]:
    """连接实机并运行一次 PD 或 NMPC 单关节跟踪实验。"""

    # 解析配置、轨迹和输出路径。
    config_path = resolve_path(args.config)
    init_pos_path = resolve_path(args.init_pos)
    traj_path = resolve_path(args.trajectory)
    output_dir = resolve_path(args.output_dir)
    raw_dir, _, _ = init_output_dirs(output_dir)

    # 读取控制周期、力矩限幅和 PD 增益。
    cfg = load_config(config_path)
    dt = float(cfg.get("trajectory", {}).get("control_dt", 0.01))
    torque_limits = np.asarray(
        cfg.get("safety", {}).get("torque_limits", [20, 20, 20, 20, 20, 20]),
        dtype=float,
    )
    pd_cfg = cfg.get("pdjoint_controller", {})
    kp = np.asarray(pd_cfg.get("kp", [0] * 6), dtype=float)
    kd = np.asarray(pd_cfg.get("kd", [0] * 6), dtype=float)
    q_init = load_init_q(init_pos_path)
    print(f"[INFO] Loaded init posture from {init_pos_path}: {np.array2string(q_init, precision=4)}")

    # MuJoCo 模型用于控制器计算，不在这里做可视化。
    sim_mjcf_path = cfg.get("simulation", {}).get("mjcf_path")
    if sim_mjcf_path is None:
        raise ValueError("simulation.mjcf_path must be set in config")
    xml_path = resolve_config_path(config_path, sim_mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # PD 控制器始终需要：PD 模式直接使用，NMPC 模式下作为局部稳定器使用。
    pd_controller = PDJointController(model=model, kp=kp, kd=kd, torque_limits=torque_limits)
    nmpc_controller = None
    if args.controller == "nmpc":
        nmpc_controller = NMPCController(
            model=model,
            config_path=str(config_path),
            rebuild=bool(cfg.get("nmpc_controller", {}).get("rebuild", False)),
        )

    # 使用现有摩擦补偿设置，保证测试与当前实机部署方式一致。
    fric_cfg = cfg.get("friction_compensation", {})
    friction = FrictionCompensator(
        param_path=resolve_config_path(config_path, fric_cfg.get("param_file", "config/joint_fric_WLS.yaml")),
        enabled=bool(fric_cfg.get("enabled", True)),
        comp_factor=float(fric_cfg.get("comp_factor", 1.0)),
        vel_threshold=float(fric_cfg.get("vel_threshold", 0.01)),
    )

    # 读取指定关节的 VLA 输出轨迹，并决定本次运行时长。
    traj = load_vla_joint_trajectory(traj_path, args.joint, dt, args.duration)
    run_seconds = float(traj.t[-1])
    out_csv = raw_dir / f"{args.controller.upper()}_final_exec_joint{args.joint}_{args.trial}.csv"

    # 用列表封装布尔值，便于 signal_handler 修改外层变量。
    keep_running = [True]

    def signal_handler(_sig, _frame):
        keep_running[0] = False

    signal.signal(signal.SIGINT, signal_handler)

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # raw 日志只记录最小分析所需字段，方便后处理。
    fieldnames = [
        "t",
        "q_ref_j",
        "q_j",
        "dq_j",
        "tau_j",
        "control_dt_ms",
        "solve_time_ms",
        "solver_status",
        "q_nmpc_j",
        "dq_nmpc_j",
        "tau_nmpc_j",
        "tau_pd_j",
        "tau_fric_j",
    ]

    try:
        # 固定使用 init_pos.txt 中的姿态作为参考基准。
        # both 模式下 PD 和 NMPC 都会回到同一个 q_start，保证两组参考一致。
        q_start = q_init.copy() if q_start_override is None else q_start_override.copy()
        hold_joint_position(
            rtde_c=rtde_c,
            rtde_r=rtde_r,
            pd_controller=pd_controller,
            friction=friction,
            q_hold=q_start,
            torque_limits=torque_limits,
            dt=dt,
            seconds=args.reset_seconds,
        )
        q_warm = np.asarray(rtde_r.getActualQ(), dtype=float)
        dq_start = np.asarray(rtde_r.getActualQd(), dtype=float)

        if nmpc_controller is not None:
            print("[INFO] Warm-starting NMPC solver")
            warm_start_nmpc(nmpc_controller, q_warm, dq_start)

        print(
            f"[INFO] Running {args.controller.upper()} joint={args.joint}, "
            f"duration={run_seconds:.2f}s, mode={args.reference_mode}"
        )
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            start = time.perf_counter()
            next_tick = start
            last_tick = start

            while keep_running[0]:
                now = time.perf_counter()
                elapsed = now - start
                if elapsed > run_seconds:
                    break

                # 读取实机状态，并同步到 MuJoCo data，供 NMPC/动力学相关计算使用。
                q = np.asarray(rtde_r.getActualQ(), dtype=float)
                dq = np.asarray(rtde_r.getActualQd(), dtype=float)
                data.qpos[:6] = q
                data.qvel[:6] = dq
                mujoco.mj_forward(model, data)

                # 当前时刻的 6 维参考：只有被测关节随 CSV 变化，其余关节保持 q_start。
                q_ref, dq_ref = make_reference_at(
                    traj=traj,
                    elapsed=elapsed,
                    q_start=q_start,
                    joint=args.joint,
                    reference_mode=args.reference_mode,
                    max_delta=args.max_delta,
                )

                tau_nmpc = np.zeros(6)
                q_nmpc = np.zeros(6)
                dq_nmpc = np.zeros(6)
                solve_time_ms = 0.0
                solver_status = 0

                if args.controller == "nmpc":
                    assert nmpc_controller is not None
                    # NMPC 需要未来 N+1 个参考点，因此从同一条 CSV 轨迹中向前取预测窗口。
                    ref_q_batch = build_ref_batch(
                        traj=traj,
                        elapsed=elapsed,
                        q_start=q_start,
                        joint=args.joint,
                        reference_mode=args.reference_mode,
                        max_delta=args.max_delta,
                        horizon_steps=nmpc_controller.N,
                        dt=nmpc_controller.dt,
                    )
                    solve_start = time.perf_counter()
                    tau_nmpc, q_nmpc, dq_nmpc = nmpc_controller.compute_torque(data, q, dq, ref_q_batch)
                    solve_time_ms = (time.perf_counter() - solve_start) * 1000.0
                    # NMPC 输出前馈力矩和下一步局部目标，PD 只负责稳定跟踪该局部目标。
                    tau_pd = pd_controller.compute_torque(q_nmpc, q, dq_nmpc, dq)
                    tau_fric = friction.compute_torque(dq)
                    tau = tau_nmpc + args.alpha * tau_pd + 0.0 * tau_fric
                else:
                    # PD 基线：直接跟踪同一条单关节 VLA 参考轨迹。
                    tau_pd = pd_controller.compute_torque(q_ref, q, dq_ref, dq)
                    tau_fric = friction.compute_torque(dq)
                    tau = tau_pd + 0.0 * tau_fric

                # 最终力矩限幅后发送到机械臂。
                tau = np.clip(tau, -torque_limits, torque_limits)
                ok = rtde_c.directTorque(tau.tolist(), False)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

                control_dt_ms = (now - last_tick) * 1000.0
                last_tick = now

                # 只保存被测关节的数据，降低后续分析复杂度。
                j = args.joint
                writer.writerow(
                    {
                        "t": elapsed,
                        "q_ref_j": q_ref[j],
                        "q_j": q[j],
                        "dq_j": dq[j],
                        "tau_j": tau[j],
                        "control_dt_ms": control_dt_ms,
                        "solve_time_ms": solve_time_ms,
                        "solver_status": solver_status,
                        "q_nmpc_j": q_nmpc[j],
                        "dq_nmpc_j": dq_nmpc[j],
                        "tau_nmpc_j": tau_nmpc[j],
                        "tau_pd_j": tau_pd[j],
                        "tau_fric_j": tau_fric[j],
                    }
                )

                # 简单定周期循环：若求解超时，则重置 next_tick，避免累计延迟越来越大。
                next_tick += dt
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

    finally:
        rtde_c.stopScript()
        print(f"[INFO] Saved raw log: {out_csv}")
        print("[INFO] Robot control script stopped")

    return out_csv, q_start


def load_log(path: Path) -> dict[str, np.ndarray]:
    """读取单次实验 raw CSV，并按列名返回 numpy 数组。"""

    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names}


def compute_metrics(log: dict[str, np.ndarray]) -> dict[str, float]:
    """根据 raw 日志计算跟踪误差、加速度、jerk 和力矩变化指标。"""

    t = log["t"]
    dt = np.gradient(t)
    # 实机控制周期可能有轻微抖动，这里用中位数作为差分时间步长，避免个别异常周期放大噪声。
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.01
    e = log["q_ref_j"] - log["q_j"]
    dq = log["dq_j"]
    tau = log["tau_j"]
    # 由关节速度差分得到加速度，再由加速度差分得到 jerk。
    ddq = np.gradient(dq, dt_med)
    jerk = np.gradient(ddq, dt_med)
    # delta_tau 衡量相邻控制周期力矩变化，越小通常表示控制输入越平滑。
    delta_tau = np.diff(tau, prepend=tau[0])
    control_dt = log.get("control_dt_ms", np.zeros_like(t))
    solve_time = log.get("solve_time_ms", np.zeros_like(t))
    solver_status = log.get("solver_status", np.zeros_like(t))

    return {
        "RMSE_q": float(np.sqrt(np.mean(e**2))),
        "Max_Error_q": float(np.max(np.abs(e))),
        "RMS_ddq": float(np.sqrt(np.mean(ddq**2))),
        "RMS_jerk": float(np.sqrt(np.mean(jerk**2))),
        "Jerk_integral": float(np.sum(jerk**2) * dt_med),
        "dTau_RMS": float(np.sqrt(np.mean(delta_tau**2))),
        "Tau_peak": float(np.max(np.abs(tau))),
        "Control_frequency_mean": float(1000.0 / np.mean(control_dt[1:])) if len(control_dt) > 1 else 0.0,
        "Control_frequency_min": float(1000.0 / np.max(control_dt[1:])) if len(control_dt) > 1 else 0.0,
        "Mean_solve_time_ms": float(np.mean(solve_time)),
        "Max_solve_time_ms": float(np.max(solve_time)),
        "Solver_failure_count": float(np.count_nonzero(solver_status)),
    }


def save_metrics(metrics_dir: Path, joint: int, pd_metrics: dict[str, float], nmpc_metrics: dict[str, float]) -> Path:
    """保存 PD/NMPC 两组指标到 metrics CSV。"""

    out = metrics_dir / f"metrics_joint{joint}.csv"
    keys = list(pd_metrics.keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", *keys])
        writer.writerow(["PD", *[pd_metrics[k] for k in keys]])
        writer.writerow(["NMPC", *[nmpc_metrics[k] for k in keys]])
    return out


def plot_results(fig_dir: Path, joint: int, pd_log: dict[str, np.ndarray], nmpc_log: dict[str, np.ndarray]) -> list[Path]:
    """生成三类最小对比图：跟踪、加速度/jerk、力矩/力矩变化。"""

    outputs: list[Path] = []

    pd_t = pd_log["t"]
    nmpc_t = nmpc_log["t"]
    dt_pd = float(np.median(np.diff(pd_t))) if len(pd_t) > 1 else 0.01
    dt_nmpc = float(np.median(np.diff(nmpc_t))) if len(nmpc_t) > 1 else 0.01
    pd_ddq = np.gradient(pd_log["dq_j"], dt_pd)
    nmpc_ddq = np.gradient(nmpc_log["dq_j"], dt_nmpc)
    pd_jerk = np.gradient(pd_ddq, dt_pd)
    nmpc_jerk = np.gradient(nmpc_ddq, dt_nmpc)
    pd_dtau = np.diff(pd_log["tau_j"], prepend=pd_log["tau_j"][0])
    nmpc_dtau = np.diff(nmpc_log["tau_j"], prepend=nmpc_log["tau_j"][0])

    # 图 1：参考角度与 PD/NMPC 实际跟踪角度。
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd_t, pd_log["q_ref_j"], "k--", label="q_ref")
    ax.plot(pd_t, pd_log["q_j"], label="PD q")
    ax.plot(nmpc_t, nmpc_log["q_j"], label="NMPC q")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("joint position [rad]")
    ax.set_title(f"Joint {joint} tracking")
    ax.grid(True)
    ax.legend()
    out = fig_dir / f"joint{joint}_tracking.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    outputs.append(out)

    # 图 2：加速度和 jerk，用于观察轨迹执行是否平滑。
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(pd_t, pd_ddq, label="PD ddq")
    axes[0].plot(nmpc_t, nmpc_ddq, label="NMPC ddq")
    axes[0].set_ylabel("ddq [rad/s^2]")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(pd_t, pd_jerk, label="PD jerk")
    axes[1].plot(nmpc_t, nmpc_jerk, label="NMPC jerk")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("jerk [rad/s^3]")
    axes[1].grid(True)
    axes[1].legend()
    fig.suptitle(f"Joint {joint} acceleration and jerk")
    out = fig_dir / f"joint{joint}_acc_jerk.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    outputs.append(out)

    # 图 3：力矩和相邻周期力矩变化，用于观察控制输入是否平滑。
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(pd_t, pd_log["tau_j"], label="PD tau")
    axes[0].plot(nmpc_t, nmpc_log["tau_j"], label="NMPC tau")
    axes[0].set_ylabel("tau [Nm]")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(pd_t, pd_dtau, label="PD delta tau")
    axes[1].plot(nmpc_t, nmpc_dtau, label="NMPC delta tau")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("delta tau [Nm/sample]")
    axes[1].grid(True)
    axes[1].legend()
    fig.suptitle(f"Joint {joint} torque")
    out = fig_dir / f"joint{joint}_torque.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    outputs.append(out)

    return outputs


def find_latest(raw_dir: Path, prefix: str, joint: int) -> Path:
    """查找指定关节最新一份 PD 或 NMPC raw 日志。"""

    matches = sorted(raw_dir.glob(f"{prefix}_final_exec_joint{joint}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {prefix} log found in {raw_dir} for joint {joint}")
    return matches[-1]


def analyze(args: argparse.Namespace) -> None:
    """读取最新 PD/NMPC 日志，计算指标并生成图表。"""

    output_dir = resolve_path(args.output_dir)
    raw_dir, fig_dir, metrics_dir = init_output_dirs(output_dir)
    pd_path = find_latest(raw_dir, "PD", args.joint)
    nmpc_path = find_latest(raw_dir, "NMPC", args.joint)

    pd_log = load_log(pd_path)
    nmpc_log = load_log(nmpc_path)
    pd_metrics = compute_metrics(pd_log)
    nmpc_metrics = compute_metrics(nmpc_log)
    metrics_path = save_metrics(metrics_dir, args.joint, pd_metrics, nmpc_metrics)
    fig_paths = plot_results(fig_dir, args.joint, pd_log, nmpc_log)

    print(f"[INFO] PD log: {pd_path}")
    print(f"[INFO] NMPC log: {nmpc_path}")
    print(f"[INFO] Metrics: {metrics_path}")
    for path in fig_paths:
        print(f"[INFO] Figure: {path}")

    print("\ncontroller,RMSE_q,RMS_ddq,RMS_jerk,Jerk_integral,dTau_RMS,Tau_peak")
    print(
        "PD,"
        f"{pd_metrics['RMSE_q']:.6g},{pd_metrics['RMS_ddq']:.6g},"
        f"{pd_metrics['RMS_jerk']:.6g},{pd_metrics['Jerk_integral']:.6g},"
        f"{pd_metrics['dTau_RMS']:.6g},{pd_metrics['Tau_peak']:.6g}"
    )
    print(
        "NMPC,"
        f"{nmpc_metrics['RMSE_q']:.6g},{nmpc_metrics['RMS_ddq']:.6g},"
        f"{nmpc_metrics['RMS_jerk']:.6g},{nmpc_metrics['Jerk_integral']:.6g},"
        f"{nmpc_metrics['dTau_RMS']:.6g},{nmpc_metrics['Tau_peak']:.6g}"
    )


def main() -> None:
    """脚本入口：根据参数选择实机运行或离线分析。"""

    args = parse_args()
    if not 0 <= args.joint < 6:
        raise ValueError("--joint must be in [0, 5]")

    if args.analyze_only:
        analyze(args)
        return

    if args.controller == "both":
        # 完整最小流程：先 PD，记录初始姿态；再把机械臂拉回同一初始姿态，跑 NMPC；最后分析。
        print("[INFO] Running full sequence: PD -> reset -> NMPC -> analysis")
        pd_args = clone_args(args, controller="pd")
        _, q_start = run_experiment(pd_args)

        nmpc_args = clone_args(args, controller="nmpc")
        run_experiment(nmpc_args, q_start_override=q_start)
        analyze(args)
        return

    run_experiment(args)
    try:
        # 如果另一组数据已经存在，实验结束后自动分析；否则提示用户先跑完另一组。
        analyze(args)
    except FileNotFoundError as exc:
        print(f"[INFO] Analysis skipped: {exc}")
        print("[INFO] Run both PD and NMPC logs, then call --analyze-only.")


if __name__ == "__main__":
    main()
