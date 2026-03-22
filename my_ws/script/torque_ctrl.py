#!/usr/bin/env python3
"""Task space impedance control for UR5e using ur-rtde and MuJoCo for kinematics.

Formula: tau = J^T * (K_p * (x_des - x) + K_d * (v_des - v)) + gravity_comp
Note: UR direct torque mode expects torques AFTER internal gravity compensation.
So we only send J^T * F_task if we want the robot to handle gravity.
"""

import argparse
import time
import socket
import json
import threading
import queue
import numpy as np
import yaml
from pathlib import Path
import mujoco
import mujoco.viewer
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# 导入自定义控制器
from Controller import ImpedanceController

# Global visualization frequency (Hz), shared by control and visualization threads.
VIS_FREQ = 50.0


def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    # x_des_pos = np.array(x_des_pos_init, copy=True)
    x_des_pos = x_des_pos_init + np.array([
        2.0 * circle_radius * np.sin(circle_omega * t),
        1.0 * circle_radius * np.sin(2 * circle_omega * t),
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
    """Return desired end-effector position and quaternion for time t."""
    x_des_pos = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_mat = get_target_ori(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_quat = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
    if x_des_quat[0] < 0:
        x_des_quat *= -1.0
    return x_des_pos, x_des_quat


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
                # Wait for latest state update from control loop.
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
                message = json.dumps(data_dict)
                self.sock.sendto(message.encode(), self.addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

    def update(self, loop_id, x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau, q, dq, f_task):
        try:
            # Keep only the latest state and notify sender thread.
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

                    # MuJoCo passive viewer requires locking when mutating user_scn
                    # from a non-viewer thread context.
                    with viewer.lock():
                        if q is not None:
                            self.data.qpos[:6] = q
                            if x_des_pos is not None:
                                self.data.mocap_pos[self.target_mocap_id] = x_des_pos
                            if x_des_mat is not None:
                                x_des_quat = np.zeros(4)
                                mujoco.mju_mat2Quat(x_des_quat, np.asarray(x_des_mat, dtype=np.float64).reshape(3, 3).flatten())
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
        """Render trajectory segments in MuJoCo viewer."""
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
                color
            )
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                from_pos,
                to_pos
            )
            viewer.user_scn.ngeom += 1


    def is_running(self):
        return self.running

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Task Space Impedance Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--sys-config", default="config/sys_config.yaml", help="System config file")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="Initial joint positions file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        ctrl_cfg = yaml.safe_load(f)
    with open(args.sys_config, 'r') as f:
        sys_cfg = yaml.safe_load(f)

    stiffness = np.array(ctrl_cfg['impedance_controller']['stiffness'])
    damping_ratio = np.array(ctrl_cfg['impedance_controller']['damping_ratio'])
    damping = damping_ratio * 2 * np.sqrt(stiffness) 

    circle_radius = ctrl_cfg['trajectory']['circle_radius']
    circle_omega = ctrl_cfg['trajectory']['circle_omega']
    dt = float(ctrl_cfg['trajectory']['control_dt'])
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    torque_limits = np.array(ctrl_cfg['safety']['torque_limits'])
    vel_limits = np.array(ctrl_cfg['safety'].get('vel_limits'))
    target_inertia = np.array(ctrl_cfg['impedance_controller'].get('target_inertia', [1.0]*6))

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
            with open(init_pos_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    q_init = np.array([float(x) for x in line.split()])
                    if len(q_init) == 6:
                        print(f"[INFO] Loaded initial joint position from {args.init_pos}")
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
    x_des_quat_init = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat_init, data.site_xmat[ee_site_id].flatten())

    trajectory_points = []
    target_trajectory = []
    max_points = 200
    start_time = time.perf_counter()

    logger = UDPLogger("127.0.0.1", 9870, send_every_n=5)
    visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)
    
    # 初始化封装后的控制器对象，在此处传入参数
    controller = ImpedanceController(model, stiffness, damping, target_inertia, vel_limits)
    
    start_time = time.perf_counter()

    total_loop_count = 0
    freq_loop_count = 0
    freq_start_time = time.perf_counter()
    next_tick = time.perf_counter()
    vis_period = 1.0 / max(1.0, float(VIS_FREQ))
    next_vis_update = time.perf_counter()
    traj_sample_period = 0.05  # 20 Hz trajectory point sampling
    next_traj_sample = 0.0
    traj_started = False
    
    # Timing logic:
    # 0-2s: Stabilization (no torque)
    # 2-5s: Move to init_pos.txt pose
    # 5s+: Track trajectory
    STABILIZE_END = 2.0
    RESET_END = 5.0

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

            # Define desired state based on time phases
            if t_now <= STABILIZE_END:
                # Phase 1: Stabilization - no control torque will be sent
                x_des_pos = x_des_pos_init.copy()
                x_des_quat = x_des_quat_init.copy()
                v_des = np.zeros(6)
            elif t_now <= RESET_END:
                # Phase 2: Reset to init_pos.txt. 
                # x_des_pos_init and x_des_quat_init were computed from loading init_pos.txt earlier.
                x_des_pos = x_des_pos_init.copy()
                x_des_quat = x_des_quat_init.copy()
                v_des = np.zeros(6)
            else:
                # Phase 3: Trajectory tracking
                if not traj_started:
                    traj_started = True
                    # We no longer re-sync to actual pose here to honor init_pos.txt theoretical target.
                    # x_des_pos_init remains the pose calculated from file at startup.

                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, circle_radius, circle_omega)
                v_des = np.zeros(6)

            x_des_mat = np.zeros(9)
            mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
            x_des_mat = x_des_mat.reshape(3, 3)

            # 使用封装后的控制器计算关节力矩
            tau = controller.compute_torque(
                data, ee_site_id, 
                x_des_pos, x_des_quat, v_ee, v_des
            )
            tau = np.clip(tau, -torque_limits, torque_limits)

            # Send torque only after stabilization phase
            if t_now > STABILIZE_END:
                ok = rtde_c.directTorque(tau.tolist(), True)
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
            logger.update(total_loop_count, x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau, q, dq, controller.F_task)

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
        logger.stop()
        rtde_c.stopScript()
        print("[INFO] Script stopped")

if __name__ == "__main__":
    main()
