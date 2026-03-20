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
                    x_curr_pos, x_des_pos, tau, q, dq, f_task = self.latest[1:]

                data_dict = {
                    "pos_curr": {"x": float(x_curr_pos[0]), "y": float(x_curr_pos[1]), "z": float(x_curr_pos[2])},
                    "pos_des": {"x": float(x_des_pos[0]), "y": float(x_des_pos[1]), "z": float(x_des_pos[2])},
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

    def update(self, loop_id, x_curr_pos, x_des_pos, tau, q, dq, f_task):
        try:
            # Keep only the latest state and notify sender thread.
            with self.lock:
                self.latest = (
                    int(loop_id),
                    np.array(x_curr_pos, copy=True),
                    np.array(x_des_pos, copy=True),
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

def draw_trajectory(viewer, positions, color=[0, 1, 0, 1], width=0.002, clear=False):
    """Render trajectory segments in MuJoCo viewer."""
    if clear:
        viewer.user_scn.ngeom = 0

    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        from_pos = np.array(positions[i][:3], dtype=np.float64)
        to_pos = np.array(positions[i + 1][:3], dtype=np.float64)
        
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            from_pos,
            to_pos
        )
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[0:4] = color
        viewer.user_scn.ngeom += 1


class VisualizationWorker:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.running = True
        self.latest = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25

                while self.running and viewer.is_running():
                    try:
                        self.queue.get(timeout=0.05)
                    except queue.Empty:
                        pass

                    with self.lock:
                        if self.latest is None:
                            continue
                        q, trajectory_points, target_trajectory = self.latest

                    self.data.qpos[:6] = q
                    mujoco.mj_forward(self.model, self.data)

                    draw_trajectory(viewer, target_trajectory, color=[0, 1, 0, 1], clear=True, width=0.003)
                    draw_trajectory(viewer, trajectory_points, color=[1, 0, 0, 1], width=0.003)
                    viewer.sync()
        except Exception as e:
            print(f"[WARN] Visualization thread stopped: {e}")

    def update(self, q, trajectory_points, target_trajectory):
        try:
            with self.lock:
                self.latest = (
                    np.array(q, copy=True),
                    [np.array(p, copy=True) for p in trajectory_points],
                    [np.array(p, copy=True) for p in target_trajectory],
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

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Task Space Impedance Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--sys-config", default="config/sys_config.yaml", help="System config file")
    parser.add_argument("--init-pos", default="config/init_pos.txt", help="Initial joint positions file")
    return parser.parse_args()

def rotation_error(R_d, R):
    """Compute the orientation error between two rotation matrices using quaternions."""
    # Convert rotation matrices to quaternions
    q_d = np.zeros(4)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q_d, R_d.flatten())
    mujoco.mju_mat2Quat(q, R.flatten())
    
    # Compute conjugate of current quaternion
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q)
    
    # Compute error quaternion: q_err = q_d * q_inv
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_d, q_inv)
    
    # Convert error quaternion to 3D velocity/error vector
    res = np.zeros(3)
    mujoco.mju_quat2Vel(res, q_err, 1.0)
    return res

def main():
    args = parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        ctrl_cfg = yaml.safe_load(f)
    with open(args.sys_config, 'r') as f:
        sys_cfg = yaml.safe_load(f)

    # Impedance parameters
    stiffness = np.array(ctrl_cfg['impedance_controller']['stiffness'])
    damping_ratio = np.array(ctrl_cfg['impedance_controller']['damping_ratio'])
    damping = damping_ratio * 2 * np.sqrt(stiffness) 

    # Trajectory parameters
    circle_radius = ctrl_cfg['trajectory']['circle_radius']
    circle_omega = ctrl_cfg['trajectory']['circle_omega']
    dt = float(ctrl_cfg['trajectory']['control_dt'])
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    torque_limits = np.array(ctrl_cfg['safety']['torque_limits'])

    # Load MuJoCo model for kinematics/Jacobian
    xml_path = Path("script/universal_robots_ur5e/scene_torque.xml")
    if not xml_path.exists():
        # Fallback to scene.xml if unique torque one doesn't exist
        xml_path = Path("script/universal_robots_ur5e/scene.xml")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        # try another common name
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    # Connect to robot
    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # Initial state - try to read from init_pos file
    q_init = None
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
                        print(f"[WARN] Invalid init_pos format (expected 6 values, got {len(q_init)}), using current position")
        except Exception as e:
            print(f"[WARN] Failed to read init_pos file: {e}, using current position")
    else:
        print(f"[INFO] init_pos file not found at {args.init_pos}, using current robot position")

    # Get initial position: either from file or current robot pose
    if q_init is not None:
        q = q_init
        print(f"[INFO] Using init_pos as trajectory center: {q}")
    else:
        q = rtde_r.getActualQ()
        print(f"[INFO] Using current robot position as trajectory center: {q}")

    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    
    # Set circular trajectory center as the EE pose at initial joint position
    x_des_pos_init = data.site_xpos[ee_site_id].copy()
    x_des_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    # Trajectory visualization containers
    trajectory_points = []  # Actual (red)
    target_trajectory = []  # Target (green)
    max_points = 200
    start_time = time.perf_counter()

    print(f"[INFO] Initial EE pos: {x_des_pos_init}")
    print(f"[INFO] Target control frequency: {1.0 / dt:.2f} Hz (dt={dt:.6f}s)")
    print("[INFO] Starting impedance control with circular trajectory. Press Ctrl+C to stop.")

    # UDP Logger
    logger = UDPLogger("127.0.0.1", 9870, send_every_n=5)
    visualizer = VisualizationWorker(xml_path)

    # Frequency calculation variables
    total_loop_count = 0
    freq_loop_count = 0
    freq_start_time = time.perf_counter()
    next_tick = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter() - start_time
            
            # 1. Generate circular trajectory in XY plane
            # x = x0 + R * cos(wt), y = y0 + R * sin(wt)
            x_des_pos = x_des_pos_init.copy()
            x_des_pos[0] += circle_radius * (np.cos(circle_omega * t_now) - 1.0)
            x_des_pos[1] += circle_radius * np.sin(circle_omega * t_now)
            
            v_des_pos = np.array([
                -circle_radius * circle_omega * np.sin(circle_omega * t_now),
                 circle_radius * circle_omega * np.cos(circle_omega * t_now),
                 0.0
            ])

            # 2. Get current state
            q = rtde_r.getActualQ()
            dq = rtde_r.getActualQd()
            
            # 3. Update MuJoCo model
            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            
            # 4. Compute Jacobian
            # jacp: 3xNV, jacr: 3xNV
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]]) # 6x6 for UR5e
            
            # 5. Compute EE velocity
            v_ee = J @ dq
            
            # 6. Compute pose error
            x_curr_pos = data.site_xpos[ee_site_id]
            x_curr_rot = data.site_xmat[ee_site_id].reshape(3, 3)
            
            pos_err = x_des_pos - x_curr_pos
            rot_err = rotation_error(x_des_rot, x_curr_rot)
            err = np.concatenate([pos_err, rot_err])
            
            # Extended velocity error (assuming no orientation velocity for now)
            v_err = np.concatenate([v_des_pos, np.zeros(3)]) - v_ee

            # 7. Task space force
            # F = K * err + D * v_err
            F_task = stiffness * err + damping * v_err
            
            # 8. Map to joint torques
            tau = J.T @ F_task
            
            # 9. Limit torques for safety
            tau = np.clip(tau, -torque_limits, torque_limits)
            
            # 10. Send to robot (Robot handles gravity compensation)
            ok = rtde_c.directTorque(tau.tolist(), True)
            
            if not ok:
                print("[ERROR] directTorque failed")
                break
                    
            # 11. Update trajectory buffers
            if not trajectory_points or (t_now % 0.05 < 0.002):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)

            total_loop_count += 1

            # Hand over latest control state to UDP thread.
            logger.update(total_loop_count, x_curr_pos, x_des_pos, tau, q, dq, F_task)

            # Hand over visualization data to visualization thread.
            visualizer.update(q, trajectory_points, target_trajectory)

            # Control Frequency Calculation and Printing
            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                frequency = freq_loop_count / elapsed
                print(f"[INFO] Control Frequency: {frequency:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            # Drive loop timing directly from configured dt.
            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                # If overrun occurs, realign to current time to avoid drift.
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
