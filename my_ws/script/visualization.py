import threading
import socket
import json
import queue
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


VIS_FREQ = 50.0

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
            with mujoco.viewer.launch_passive(self.model, self.data,
                show_left_ui=False, show_right_ui=False
            ) as viewer:
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
                    loop_id, q_des, q, x_des_pos, x_curr_pos, tau, extra = self.latest

                payload = {
                    "q_des": q_des.tolist(),
                    "q": q.tolist(),
                    "x_des_pos": x_des_pos.tolist(),
                    "x_curr_pos": x_curr_pos.tolist(),
                    "tau": tau.tolist(),
                }
                if extra:
                    for k, v in extra.items():
                        if isinstance(v, np.ndarray):
                            payload[k] = v.tolist()
                        else:
                            payload[k] = v
                self.sock.sendto(json.dumps(payload).encode(), self.addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] UDP send failed: {e}")

    def update(self, loop_id, q_des, q, x_des_pos, x_curr_pos, tau, extra=None):
        try:
            with self.lock:
                self.latest = (
                    int(loop_id),
                    np.array(q_des, copy=True),
                    np.array(q, copy=True),
                    np.array(x_des_pos, copy=True),
                    np.array(x_curr_pos, copy=True),
                    np.array(tau, copy=True),
                    extra
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