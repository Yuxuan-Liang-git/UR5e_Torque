#!/usr/bin/env python3
"""Task space control comparison for UR5e.

Actual robot torque is always sent by ImpedanceController.
NMPC torque is computed in parallel and visualized for comparison.
"""

import argparse
import math
import json
import queue
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from Controller import ImpedanceController

_APP_MPC_DIR = Path("/home/amdt/app_ws/my_mjctrl/MPC_task_space")
if str(_APP_MPC_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_MPC_DIR))

from nmpc_controller_ur5e import UR5eNMPC, mj2pin_pos, mj2pin_rot  # type: ignore[import-not-found]

VIS_FREQ = 50.0


def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    x_des_pos = x_des_pos_init + np.array([
        2.0 * circle_radius * np.sin(circle_omega * t),
        1.0 * circle_radius * np.sin(2.0 * circle_omega * t),
        0.0,
    ])
    return x_des_pos


def get_target_ori(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float, dt: float = 1e-3) -> np.ndarray:
    """Tangent-aligned rotation matrix from trajectory finite difference."""
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


def slerp(q1, q2, alpha):
    """Quaternion spherical linear interpolation."""
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


class UDPLogger:
    def __init__(self, ip="127.0.0.1", port=9870, send_every_n=5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)
        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.send_every_n = max(1, int(send_every_n))
        self.latest = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                self.queue.get(timeout=0.1)

                with self.lock:
                    if self.latest is None:
                        continue
                    loop_id = self.latest[0]
                    if loop_id % self.send_every_n != 0:
                        continue
                    x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau, q, dq, f_task = self.latest[1:]

                data_dict = {
                    "pos_curr": {"x": float(x_curr_pos[0]), "y": float(x_curr_pos[1]), "z": float(x_curr_pos[2])},
                    "pos_des": {"x": float(x_des_pos[0]), "y": float(x_des_pos[1]), "z": float(x_des_pos[2])},
                    "quat_curr": x_curr_quat.tolist(),
                    "quat_des": x_des_quat.tolist(),
                    "target_torques": tau.tolist(),
                    "q": q.tolist(),
                    "dq": dq.tolist(),
                    "f_task": f_task.tolist(),
                }
                self.sock.sendto(json.dumps(data_dict).encode(), self.addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

    def update(self, loop_id, x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau, q, dq, f_task):
        try:
            with self.lock:
                self.latest = (
                    int(loop_id),
                    np.array(x_curr_pos, copy=True),
                    np.array(x_des_pos, copy=True),
                    np.array(x_curr_quat, copy=True),
                    np.array(x_des_quat, copy=True),
                    np.array(tau, copy=True),
                    np.array(q, copy=True),
                    np.array(dq, copy=True),
                    np.array(f_task, copy=True),
                )

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put_nowait(1)
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class TorquePlotter:
    def __init__(self, maxlen: int = 1000):
        self.lock = threading.Lock()
        self.running = False
        self.t_data = deque(maxlen=maxlen)
        self.tau_imp = [deque(maxlen=maxlen) for _ in range(6)]
        self.tau_nmpc = [deque(maxlen=maxlen) for _ in range(6)]
        self.anim = None

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def update(self, t, tau_i, tau_n):
        with self.lock:
            self.t_data.append(float(t))
            for i in range(6):
                self.tau_imp[i].append(float(tau_i[i]))
                self.tau_nmpc[i].append(float(tau_n[i]))

    def _loop(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        fig.suptitle("Control Torque Comparison: Impedance (Blue) vs NMPC (Red)")
        axs = axs.flatten()

        lines_imp = []
        lines_nmpc = []
        for i in range(6):
            axs[i].set_title(f"Joint {i + 1}")
            l_imp, = axs[i].plot([], [], "b", label="Impedance", lw=1.5)
            l_nmpc, = axs[i].plot([], [], "r", label="NMPC", lw=1.5, alpha=0.7)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Torque (Nm)")
            axs[i].legend(loc="upper right", fontsize=8)
            axs[i].grid(True)
            lines_imp.append(l_imp)
            lines_nmpc.append(l_nmpc)

        def animate(_):
            if not self.running:
                return []
            with self.lock:
                if not self.t_data:
                    return []
                t = list(self.t_data)
                ti = [list(d) for d in self.tau_imp]
                tn = [list(d) for d in self.tau_nmpc]

            cur_t = t[-1]
            x_min = max(0.0, cur_t - 5.0)
            x_max = max(5.0, cur_t)

            for i in range(6):
                lines_imp[i].set_data(t, ti[i])
                lines_nmpc[i].set_data(t, tn[i])
                axs[i].set_xlim(x_min, x_max + 0.5)

                all_y = ti[i] + tn[i]
                if all_y:
                    min_y, max_y = min(all_y), max(all_y)
                    span = max(max_y - min_y, 1.0)
                    axs[i].set_ylim(min_y - 0.2 * span, max_y + 0.2 * span)

            return lines_imp + lines_nmpc

        self.anim = FuncAnimation(fig, animate, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        self.running = False


class VisualizationWorker:
    def __init__(self, xml_path, render_hz=VIS_FREQ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.target_mocap_id = self.model.body("target").mocapid[0]
        self.latest_q = None
        self.latest_x_des_pos = None
        self.latest_x_des_mat = None
        self.trajectory_points = []
        self.target_trajectory = []
        self.running = True
        self.render_period = 1.0 / max(1.0, float(render_hz))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                viewer.opt.sitegroup[:] = True

                next_render = time.perf_counter()
                while self.running and viewer.is_running():
                    now = time.perf_counter()
                    if now < next_render:
                        time.sleep(min(0.002, next_render - now))
                        continue

                    with self.lock:
                        q = None if self.latest_q is None else self.latest_q.copy()
                        x_des_pos = None if self.latest_x_des_pos is None else self.latest_x_des_pos.copy()
                        x_des_mat = None if self.latest_x_des_mat is None else self.latest_x_des_mat.copy()
                        traj = list(self.trajectory_points)
                        target_traj = list(self.target_trajectory)

                    with viewer.lock():
                        if q is not None:
                            self.data.qpos[:6] = q
                            if x_des_pos is not None:
                                self.data.mocap_pos[self.target_mocap_id] = x_des_pos
                            if x_des_mat is not None:
                                x_des_quat = np.zeros(4)
                                mujoco.mju_mat2Quat(
                                    x_des_quat,
                                    np.asarray(x_des_mat, dtype=np.float64).reshape(3, 3).flatten(),
                                )
                                if x_des_quat[0] < 0:
                                    x_des_quat *= -1.0
                                self.data.mocap_quat[self.target_mocap_id] = x_des_quat
                            mujoco.mj_forward(self.model, self.data)
                            self._draw_trajectory(viewer, target_traj, color=[0, 1, 0, 1], clear=True, width=0.004)
                            self._draw_trajectory(viewer, traj, color=[1, 0, 0, 1], width=0.004)
                        else:
                            viewer.user_scn.ngeom = 0
                        viewer.sync()

                    next_render += self.render_period
                    if next_render < now:
                        next_render = now + self.render_period
        except Exception as e:
            print(f"[WARN] Visualization thread stopped: {e}")
        finally:
            self.running = False

    def update(self, q, trajectory_points, target_trajectory, x_des_pos=None, x_des_mat=None):
        with self.lock:
            self.latest_q = np.array(q, copy=True)
            self.latest_x_des_pos = None if x_des_pos is None else np.array(x_des_pos, copy=True)
            self.latest_x_des_mat = None if x_des_mat is None else np.array(x_des_mat, copy=True)
            self.trajectory_points = [np.array(p, copy=True) for p in trajectory_points]
            self.target_trajectory = [np.array(p, copy=True) for p in target_trajectory]

    def _draw_trajectory(self, viewer, positions, color=[0, 1, 0, 1], width=0.002, clear=False):
        if clear:
            viewer.user_scn.ngeom = 0

        for i in range(len(positions) - 1):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            from_pos = np.array(positions[i][:3], dtype=np.float64)
            to_pos = np.array(positions[i + 1][:3], dtype=np.float64)

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                [width, 0, 0],
                from_pos,
                np.eye(3).flatten(),
                color,
            )
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                from_pos,
                to_pos,
            )
            viewer.user_scn.ngeom += 1

    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e task-space control comparison")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--sys-config", default="config/sys_config.yaml", help="System config file")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="Initial joint positions file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        ctrl_cfg = yaml.safe_load(f)
    with open(args.sys_config, "r") as f:
        sys_cfg = yaml.safe_load(f)

    stiffness = np.array(ctrl_cfg["impedance_controller"]["stiffness"], dtype=float)
    damping_ratio = np.array(ctrl_cfg["impedance_controller"]["damping_ratio"], dtype=float)
    damping = damping_ratio * 2.0 * np.sqrt(stiffness)
    target_inertia = np.array(ctrl_cfg["impedance_controller"].get("target_inertia", [1.0] * 6), dtype=float)

    circle_radius = float(ctrl_cfg["trajectory"]["circle_radius"])
    circle_omega = float(ctrl_cfg["trajectory"]["circle_omega"])
    dt = float(ctrl_cfg["trajectory"]["control_dt"])
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    torque_limits = np.array(ctrl_cfg["safety"]["torque_limits"], dtype=float)
    vel_limits = np.array(ctrl_cfg["safety"].get("vel_limits"), dtype=float)

    nmpc_cfg = ctrl_cfg.get("nmpc_controller", {})
    nmpc_N = int(nmpc_cfg.get("horizon_steps", 30))
    nmpc_Tf = float(nmpc_cfg.get("horizon_time", 0.3))
    nmpc_dt = nmpc_Tf / max(1, nmpc_N)
    nmpc_rebuild = bool(nmpc_cfg.get("rebuild", False))

    xml_path = Path("script/ur5e_gripper/scene.xml")
    if not xml_path.exists():
        xml_path = Path("script/universal_robots_ur5e/scene.xml")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    q_init = None
    init_pos_path = Path("ur_client_library/init_pos.txt")
    if not init_pos_path.exists():
        init_pos_path = Path(args.init_pos)
    if init_pos_path.exists():
        try:
            with open(init_pos_path, "r") as f:
                line = f.readline().strip()
                if line:
                    q_init = np.array([float(x) for x in line.split()])
                    if len(q_init) == 6:
                        print(f"[INFO] Loaded initial joint position from {init_pos_path}")
                    else:
                        q_init = None
        except Exception as e:
            print(f"[WARN] Failed to read init_pos file: {e}")

    if q_init is not None:
        q = q_init
    else:
        q = rtde_r.getActualQ()

    data.qpos[:6] = q
    mujoco.mj_forward(model, data)

    x_des_pos_init = data.site_xpos[ee_site_id].copy()
    x_des_mat_init = data.site_xmat[ee_site_id].reshape(3, 3).copy()
    x_des_quat_init = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat_init, data.site_xmat[ee_site_id].flatten())

    trajectory_points = []
    target_trajectory = []
    max_points = 200
    start_time = time.perf_counter()

    logger = UDPLogger("127.0.0.1", 9870, send_every_n=5)
    visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)
    plotter = TorquePlotter()
    plotter.start()

    imp_controller = ImpedanceController(model, stiffness, damping, target_inertia, vel_limits)

    nmpc_controller = None
    try:
        nmpc_controller = UR5eNMPC(N=nmpc_N, Tf=nmpc_Tf, rebuild=nmpc_rebuild)
        print("[INFO] NMPC controller loaded for comparison")
    except Exception as e:
        print(f"[WARN] NMPC controller unavailable, fallback to zeros: {e}")

    ref_queue = ReferenceQueue(nmpc_N, nmpc_dt,
                               lambda t: get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega),
                               lambda t: get_target_ori(t, x_des_pos_init, circle_radius, circle_omega))
    ref_queue.init(x_des_pos_init, x_des_mat_init)

    if nmpc_controller is not None:
        x0_warm = np.concatenate([q, np.zeros(len(joint_names))])
        for k in range(nmpc_N + 1):
            nmpc_controller.solver.set(k, "x", x0_warm)
        for k in range(nmpc_N):
            nmpc_controller.solver.set(k, "u", np.zeros(len(joint_names)))

    total_loop_count = 0
    freq_loop_count = 0
    freq_start_time = time.perf_counter()
    next_tick = time.perf_counter()
    vis_period = 1.0 / max(1.0, float(VIS_FREQ))
    next_vis_update = time.perf_counter()
    traj_sample_period = 0.05
    next_traj_sample = 0.0
    traj_started = False

    STABILIZE_END = 1.0
    RESET_END = 5.0
    RESET_BUFFER = 1.0

    try:
        while True:
            if not visualizer.is_running():
                print("[INFO] Visualization closed, stopping controller loop.")
                break

            t_now = time.perf_counter() - start_time

            q = rtde_r.getActualQ()
            dq = rtde_r.getActualQd()
            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])

            v_ee = J @ dq
            x_curr_pos = data.site_xpos[ee_site_id]
            x_curr_mat = data.site_xmat[ee_site_id].reshape(3, 3)
            x_curr_quat = np.zeros(4)
            mujoco.mju_mat2Quat(x_curr_quat, data.site_xmat[ee_site_id].flatten())

            if t_now <= STABILIZE_END:
                x_des_pos_start = x_curr_pos.copy()
                x_des_quat_start = x_curr_quat.copy()
                x_des_pos = x_curr_pos.copy()
                x_des_quat = x_curr_quat.copy()
                v_des = np.zeros(6)
            elif t_now <= RESET_END:
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)
                x_des_pos = (1.0 - alpha) * x_des_pos_start + alpha * x_des_pos_init
                x_des_quat = slerp(x_des_quat_start, x_des_quat_init, alpha)
                v_des = np.zeros(6)
            else:
                if not traj_started:
                    traj_started = True
                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, circle_radius, circle_omega)
                v_des = np.zeros(6)

            x_des_mat = np.zeros(9)
            mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
            x_des_mat = x_des_mat.reshape(3, 3)

            tau_imp = imp_controller.compute_torque(data, ee_site_id, x_des_pos, x_des_quat, v_ee, v_des)
            tau_imp = np.clip(tau_imp, -torque_limits, torque_limits)

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
                tau_nmpc = np.clip(tau_nmpc, -nmpc_controller.torque_limits, nmpc_controller.torque_limits)
            else:
                tau_nmpc = np.zeros(6)

            if t_now > STABILIZE_END:
                ok = rtde_c.directTorque(tau_imp.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            total_loop_count += 1
            logger.update(total_loop_count, x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau_imp, q, dq, imp_controller.F_task)
            plotter.update(t_now, tau_imp, tau_nmpc)

            now_for_vis = time.perf_counter()
            if now_for_vis >= next_vis_update:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)
                next_vis_update += vis_period
                if next_vis_update < now_for_vis:
                    next_vis_update = now_for_vis + vis_period

            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                frequency = freq_loop_count / elapsed
                print(f"[INFO] Control Frequency: {frequency:.2f} Hz")
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
        visualizer.stop()
        plotter.stop()
        logger.stop()
        rtde_c.stopScript()
        print("[INFO] Script stopped")


if __name__ == "__main__":
    main()
