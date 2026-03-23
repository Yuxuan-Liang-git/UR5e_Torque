#!/usr/bin/env python3
"""Joint-space PD torque control for UR5e with figure-8 EE tracking.

Control law:
    tau = Kp * (q_des - q) + Kd * (dq_des - dq)

Joint references are generated from end-effector figure-8 trajectory using
Jacobian-based resolved-rate mapping, then tracked by joint-space PD.
"""

import argparse
import time
import threading
import socket
import json
import queue
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


VIS_FREQ = 50.0


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


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Joint-Space PD Torque Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="Initial joint positions file")
    parser.add_argument("--no-vis", action="store_true", help="Disable MuJoCo visualization thread")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="UDP destination IP for PlotJuggler")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP destination port for PlotJuggler")
    parser.add_argument("--udp-div", type=int, default=2, help="Send one packet every N control loops")
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


class VisualizationWorker:
    def __init__(self, xml_path: Path, render_hz: float = VIS_FREQ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        try:
            self.target_mocap_id = self.model.body("target").mocapid[0]
        except Exception:
            self.target_mocap_id = -1
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
                            if self.target_mocap_id >= 0 and x_des_pos is not None:
                                self.data.mocap_pos[self.target_mocap_id] = x_des_pos
                            if self.target_mocap_id >= 0 and x_des_mat is not None:
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


class UDPLogger:
    def __init__(self, ip="127.0.0.1", port=9870, send_every_n=2):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, int(port))
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
                    q_des, q, x_des_pos, x_curr_pos, tau = self.latest[1:]

                payload = {
                    "q_des": q_des.tolist(),
                    "q": q.tolist(),
                    "x_des_pos": x_des_pos.tolist(),
                    "x_curr_pos": x_curr_pos.tolist(),
                    "tau": tau.tolist(),
                }
                self.sock.sendto(json.dumps(payload).encode(), self.addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

    def update(self, loop_id, q_des, q, x_des_pos, x_curr_pos, tau):
        try:
            with self.lock:
                self.latest = (
                    int(loop_id),
                    np.array(q_des, copy=True),
                    np.array(q, copy=True),
                    np.array(x_des_pos, copy=True),
                    np.array(x_curr_pos, copy=True),
                    np.array(tau, copy=True),
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


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dt = float(cfg.get("trajectory", {}).get("control_dt", 0.01))
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    safety_cfg = cfg.get("safety", {})
    torque_limits = np.array(safety_cfg.get("torque_limits", [20, 20, 20, 10, 10, 10]), dtype=float)

    traj_cfg = cfg.get("trajectory", {})
    circle_radius = float(traj_cfg.get("circle_radius", 0.08))
    circle_omega = float(traj_cfg.get("circle_omega", 0.8))

    pd_cfg = cfg.get("joint_pd", {})
    kp = np.array(pd_cfg.get("kp", [3000, 3000, 3000, 1000, 200, 15]), dtype=float)
    kd = np.array(pd_cfg.get("kd", [100, 100, 100, 20, 5, 1]), dtype=float)
    if kp.shape[0] != 6 or kd.shape[0] != 6:
        raise ValueError("joint_pd.kp and joint_pd.kd must both have 6 elements")

    xml_path = Path("script/ur5e_gripper/scene.xml")
    if not xml_path.exists():
        xml_path = Path("script/universal_robots_ur5e/scene.xml")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

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
    x_des_quat_init = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat_init, data.site_xmat[ee_site_id].flatten())

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

            tau = kp * (q_des - q) + kd * (dq_des - dq)
            tau = np.clip(tau, -torque_limits, torque_limits)

            if t_now > STABILIZE_END:
                ok = rtde_c.directTorque(tau.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break

            total_loop_count += 1
            logger.update(total_loop_count, q_des, q, x_des_pos, x_curr_pos, tau)

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)

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
        logger.stop()
        rtde_c.stopScript()
        print("[INFO] Script stopped")


if __name__ == "__main__":
    main()
