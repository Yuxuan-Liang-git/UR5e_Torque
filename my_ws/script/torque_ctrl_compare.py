#!/usr/bin/env python3
"""Task-space tracking with PDJoint (actual) vs NMPC (comparison only)."""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mujoco
import numpy as np
import yaml
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from Controller import PDJointController
from nmpc_controller_ur5e import UR5eNMPC, mj2pin_pos, mj2pin_rot
from visualization import UDPLogger, VisualizationWorker


VIS_FREQ = 50.0
_SCRIPT_DIR = Path(__file__).parent


def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    return x_des_pos_init + np.array([
        2.0 * circle_radius * np.sin(circle_omega * t),
        circle_radius * np.sin(2.0 * circle_omega * t),
        0.0,
    ])


def get_target_ori(
    t: float,
    x_des_pos_init: np.ndarray,
    circle_radius: float,
    circle_omega: float,
    dt: float = 1e-3,
) -> np.ndarray:
    """Build orientation aligned with trajectory tangent."""
    p0 = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    p1 = get_traj_pos(t + dt, x_des_pos_init, circle_radius, circle_omega)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return np.eye(3)
    tangent /= norm
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, tangent)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm if y_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    z_axis = np.cross(tangent, y_axis)
    return np.column_stack([tangent, y_axis, z_axis])


def get_traj(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> tuple[np.ndarray, np.ndarray]:
    x_des_pos = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_mat = get_target_ori(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_quat = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
    if x_des_quat[0] < 0:
        x_des_quat *= -1.0
    return x_des_pos, x_des_quat


def slerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        res = q1 + alpha * (q2 - q1)
        return res / np.linalg.norm(res)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * alpha
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    return (s0 * q1) + (s1 * q2)


def compute_task_errors(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    current_pos: np.ndarray,
    current_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_err = target_pos - current_pos
    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, current_mat.flatten())
    quat_inv = np.zeros(4)
    mujoco.mju_negQuat(quat_inv, curr_quat)
    quat_err = np.zeros(4)
    mujoco.mju_mulQuat(quat_err, target_quat, quat_inv)
    if quat_err[0] < 0:
        quat_err *= -1.0
    rot_err = np.zeros(3)
    mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
    return pos_err, rot_err, curr_quat


def map_task_target_to_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    x_des_pos: np.ndarray,
    x_des_quat: np.ndarray,
    q_curr: np.ndarray,
) -> np.ndarray:
    """Project desired task-space pose error to joint-space target using pseudoinverse."""
    current_pos = data.site_xpos[ee_site_id].copy()
    current_mat = data.site_xmat[ee_site_id].reshape(3, 3)
    pos_err, rot_err, _ = compute_task_errors(x_des_pos, x_des_quat, current_pos, current_mat)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    J6 = np.vstack([jacp[:, :6], jacr[:, :6]])

    e6 = np.concatenate([pos_err, rot_err])
    dq_cmd = np.linalg.pinv(J6) @ e6
    return q_curr + dq_cmd


class ReferenceQueue:
    def __init__(self, N: int, dt: float, get_pos, get_ori):
        self.N = N
        self.dt = dt
        self.get_pos = get_pos
        self.get_ori = get_ori
        self._pos = np.zeros((3, N + 1))
        self._rot = np.zeros((9, N + 1))
        self._head = 0

    def init(self, p0: np.ndarray, R0: np.ndarray):
        pos_pin = mj2pin_pos(p0)
        rot_pin = mj2pin_rot(R0).flatten(order="F")
        for k in range(self.N + 1):
            self._pos[:, k] = pos_pin
            self._rot[:, k] = rot_pin
        self._head = 0

    def step(self, t_traj: float):
        t_new = t_traj + self.N * self.dt
        p_new = self.get_pos(t_new)
        R_new = self.get_ori(t_new)
        self._pos[:, self._head] = mj2pin_pos(p_new)
        self._rot[:, self._head] = mj2pin_rot(R_new).flatten(order="F")
        self._head = (self._head + 1) % (self.N + 1)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(self._head, self._head + self.N + 1) % (self.N + 1)
        return self._pos[:, idx], self._rot[:, idx]


class TorquePlotter:
    def __init__(self, maxlen: int = 1000):
        self.lock = threading.Lock()
        self.running = False
        self.t_data = deque(maxlen=maxlen)
        self.tau_actual = [deque(maxlen=maxlen) for _ in range(6)]
        self.tau_nmpc = [deque(maxlen=maxlen) for _ in range(6)]
        self.anim = None

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def update(self, t: float, tau_actual: np.ndarray, tau_nmpc: np.ndarray):
        with self.lock:
            self.t_data.append(float(t))
            for i in range(6):
                self.tau_actual[i].append(float(tau_actual[i]))
                self.tau_nmpc[i].append(float(tau_nmpc[i]))

    def _loop(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        fig.suptitle("Control Torque Comparison: PDJoint (Blue) vs NMPC (Red)")
        axs = axs.flatten()

        lines_actual = []
        lines_nmpc = []
        for i in range(6):
            axs[i].set_title(f"Joint {i + 1}")
            line_actual, = axs[i].plot([], [], "b", label="PDJoint", lw=1.5)
            line_nmpc, = axs[i].plot([], [], "r", label="NMPC", lw=1.5, alpha=0.7)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Torque (Nm)")
            axs[i].legend(loc="upper right", fontsize=8)
            axs[i].grid(True)
            lines_actual.append(line_actual)
            lines_nmpc.append(line_nmpc)

        def animate(_):
            if not self.running:
                return []
            with self.lock:
                if not self.t_data:
                    return []
                t = list(self.t_data)
                ta = [list(d) for d in self.tau_actual]
                tn = [list(d) for d in self.tau_nmpc]

            cur_t = t[-1]
            x_min = max(0.0, cur_t - 5.0)
            x_max = max(5.0, cur_t)

            for i in range(6):
                lines_actual[i].set_data(t, ta[i])
                lines_nmpc[i].set_data(t, tn[i])
                axs[i].set_xlim(x_min, x_max + 0.5)

                all_y = ta[i] + tn[i]
                if all_y:
                    min_y, max_y = min(all_y), max(all_y)
                    span = max(max_y - min_y, 1.0)
                    axs[i].set_ylim(min_y - 0.2 * span, max_y + 0.2 * span)

            return lines_actual + lines_nmpc

        self.anim = FuncAnimation(fig, animate, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        self.running = False


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e control torque comparison")

    ws_dir = _SCRIPT_DIR.parent
    default_config = ws_dir / "config" / "ctrl_config.yaml"
    default_init_pos = ws_dir / "config" / "init_pos.txt"

    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default=str(default_config), help="Control config file")
    parser.add_argument("--init-pos", default=str(default_init_pos), help="Initial joint positions file")
    parser.add_argument("--no-vis", action="store_true", help="Disable MuJoCo visualization thread")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="UDP destination IP for PlotJuggler")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP destination port for PlotJuggler")
    parser.add_argument("--udp-div", type=int, default=2, help="Send one packet every N control loops")
    parser.add_argument("--no-plot", action="store_true", help="Disable torque comparison plot")
    return parser.parse_args()


def load_init_q(init_pos_path: Path):
    if not init_pos_path.exists():
        return None
    try:
        with open(init_pos_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if not line:
            return None
        q_init = np.array([float(x) for x in line.split()], dtype=float)
        if q_init.shape[0] != 6:
            return None
        return q_init
    except Exception as e:
        print(f"[WARN] Failed to read init_pos file: {e}")
        return None


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        ctrl_cfg = yaml.safe_load(f)

    dt = float(ctrl_cfg.get("trajectory", {}).get("control_dt", 0.01))
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    safety_cfg = ctrl_cfg.get("safety", {})
    torque_limits = np.array(safety_cfg.get("torque_limits", [20, 20, 20, 10, 10, 10]), dtype=float)

    traj_cfg = ctrl_cfg.get("trajectory", {})
    circle_radius = float(traj_cfg.get("circle_radius", 0.08))
    circle_omega = float(traj_cfg.get("circle_omega", 0.8))

    pdjoint_cfg = ctrl_cfg.get("pdjoint_controller")
    if pdjoint_cfg is None:
        raise ValueError("Missing 'pdjoint_controller' section in ctrl_config.yaml")
    if "kp" not in pdjoint_cfg or "kd" not in pdjoint_cfg:
        raise ValueError("'pdjoint_controller' must contain both 'kp' and 'kd'")

    kp = np.array(pdjoint_cfg["kp"], dtype=float)
    kd = np.array(pdjoint_cfg["kd"], dtype=float)
    if kp.shape[0] != 6 or kd.shape[0] != 6:
        raise ValueError("pdjoint_controller.kp and pdjoint_controller.kd must both have 6 elements")

    nmpc_cfg = ctrl_cfg.get("nmpc_controller", {})
    nmpc_N = int(nmpc_cfg.get("horizon_steps", 30))
    nmpc_Tf = float(nmpc_cfg.get("horizon_time", 0.3))
    nmpc_dt = nmpc_Tf / max(1, nmpc_N)
    nmpc_rebuild = bool(nmpc_cfg.get("rebuild", False))

    xml_path = _SCRIPT_DIR / "ur5e_gripper" / "scene.xml"
    if not xml_path.exists():
        xml_path = _SCRIPT_DIR / "universal_robots_ur5e" / "scene.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    joint_controller = PDJointController(model=model, kp=kp, kd=kd, torque_limits=torque_limits)

    nmpc_controller = None
    try:
        print("[INFO] Building/Loading NMPC controller first before moving robot...")
        nmpc_controller = UR5eNMPC(N=nmpc_N, Tf=nmpc_Tf, rebuild=nmpc_rebuild)
        print("[INFO] NMPC controller loaded for comparison")
    except Exception as e:
        print(f"[WARN] NMPC controller unavailable, fallback to zeros: {e}")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    visualizer = None
    if not args.no_vis:
        visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)

    logger = UDPLogger(args.udp_ip, args.udp_port, send_every_n=args.udp_div)

    init_pos_path = Path("ur_client_library/init_pos.txt")
    if not init_pos_path.exists():
        init_pos_path = Path(args.init_pos)

    q_init = load_init_q(init_pos_path)
    q_actual0 = np.array(rtde_r.getActualQ(), dtype=float)

    if q_init is not None:
        q_target = q_init.copy()
        print(f"[INFO] Loaded init joint position from {init_pos_path}")
    else:
        q_target = q_actual0.copy()
        print("[WARN] init_pos not found/invalid, holding current joints")

    data.qpos[:6] = q_target
    mujoco.mj_forward(model, data)

    x_des_pos_init = data.site_xpos[ee_site_id].copy()
    x_des_mat_init = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    ref_queue = ReferenceQueue(
        nmpc_N,
        nmpc_dt,
        lambda t: get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega),
        lambda t: get_target_ori(t, x_des_pos_init, circle_radius, circle_omega),
    )
    ref_queue.init(x_des_pos_init, x_des_mat_init)

    if nmpc_controller is not None:
        x0_warm = np.concatenate([q_actual0, np.zeros(6)])
        for k in range(nmpc_N + 1):
            nmpc_controller.solver.set(k, "x", x0_warm)
        for k in range(nmpc_N):
            nmpc_controller.solver.set(k, "u", np.zeros(6))

    plotter = None
    if not args.no_plot:
        plotter = TorquePlotter(maxlen=1500)
        plotter.start()

    STABILIZE_END = 1.0
    RESET_END = 5.0
    RESET_BUFFER = 1.0
    q_start = None

    traj_started = False
    trajectory_points = []
    target_trajectory = []
    max_points = 200
    traj_sample_period = 0.05
    next_traj_sample = 0.0

    start_time = time.perf_counter()
    next_tick = time.perf_counter()
    total_loop_count = 0
    freq_start_time = time.perf_counter()
    freq_loop_count = 0

    try:
        while True:
            if visualizer is not None and not visualizer.is_running():
                print("[INFO] Visualization closed, stopping controller loop.")
                break

            t_now = time.perf_counter() - start_time

            q = np.array(rtde_r.getActualQ(), dtype=float)
            dq = np.array(rtde_r.getActualQd(), dtype=float)

            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            x_curr_pos = data.site_xpos[ee_site_id].copy()
            x_curr_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()

            if t_now <= STABILIZE_END:
                x_des_pos_start = data.site_xpos[ee_site_id].copy()
                x_des_mat_start = data.site_xmat[ee_site_id].reshape(3, 3)
                x_des_quat_start = np.zeros(4)
                mujoco.mju_mat2Quat(x_des_quat_start, x_des_mat_start.flatten())

                x_des_pos = x_curr_pos.copy()
                x_des_mat = x_curr_mat.copy()
                q_des = q.copy()
                dq_des = np.zeros(6)
                q_start = q.copy()
            elif t_now <= RESET_END:
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)

                # Reset phase: directly track q_target in joint space.
                q_des = (1.0 - alpha) * q_start + alpha * q_target
                dq_des = np.zeros(6)

                # Compute desired EE pose only for visualization/logging.
                data_plan.qpos[:6] = q_des
                mujoco.mj_forward(model, data_plan)
                x_des_pos = data_plan.site_xpos[ee_site_id].copy()
                x_des_mat = data_plan.site_xmat[ee_site_id].reshape(3, 3).copy()
            else:
                if not traj_started:
                    traj_started = True

                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, circle_radius, circle_omega)
                x_des_mat = np.zeros(9)
                mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
                x_des_mat = x_des_mat.reshape(3, 3)
                q_des = map_task_target_to_joint(model, data, ee_site_id, x_des_pos, x_des_quat, q)
                dq_des = np.zeros(6)

            tau_actual = joint_controller.compute_torque(q_des, q, dq_des, dq)

            if nmpc_controller is not None:
                x0 = np.concatenate([q, dq])
                if t_now <= RESET_END:
                    ref_pos = np.repeat(mj2pin_pos(np.asarray(x_des_pos, dtype=float))[:, None], nmpc_N + 1, axis=1)
                    ref_rot = np.repeat(mj2pin_rot(x_des_mat).reshape(9, 1, order="F"), nmpc_N + 1, axis=1)
                else:
                    t_traj = t_now - RESET_END
                    ref_queue.step(t_traj)
                    ref_pos, ref_rot = ref_queue.get()
                tau_nmpc = nmpc_controller.solve(x0, ref_pos, ref_rot)
                tau_nmpc = np.clip(tau_nmpc, -torque_limits, torque_limits)
            else:
                tau_nmpc = np.zeros(6)

            if t_now > STABILIZE_END:
                # Only PDJoint torque is sent to real robot.
                ok = rtde_c.directTorque(tau_actual.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

            total_loop_count += 1
            logger.update(total_loop_count, q_des, q, x_des_pos, x_curr_pos, tau_actual)
            if plotter is not None:
                plotter.update(t_now, tau_actual, tau_nmpc)

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)

            # Frequency printing
            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                print(f"[INFO] Control Frequency: {freq_loop_count / elapsed:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        if visualizer is not None:
            visualizer.stop()
        if plotter is not None:
            plotter.stop()
        logger.stop()
        rtde_c.stopScript()
        print("[INFO] Script stopped")


if __name__ == "__main__":
    main()
