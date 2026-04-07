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
from Controller import ImpedanceController, PDController

# Global visualization frequency (Hz), shared by control and visualization threads.
VIS_FREQ = 50.0


def get_traj_pos(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float) -> np.ndarray:
    """计算 8 字轨迹期望位置"""
    x_des_pos = x_des_pos_init + np.array([
        2.0 * circle_radius * np.sin(circle_omega * t),
        circle_radius * np.sin(2 * circle_omega * t),
        0.0,
    ])
    return x_des_pos


def get_target_ori(t: float, x_des_pos_init: np.ndarray, circle_radius: float, circle_omega: float, dt: float = 1e-3) -> np.ndarray:
    """根据轨迹切线方向计算期望姿态矩阵（有限差分法）"""
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
    """获取 t 时刻的期望位置和四元数姿态"""
    x_des_pos = get_traj_pos(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_mat = get_target_ori(t, x_des_pos_init, circle_radius, circle_omega)
    x_des_quat = np.zeros(4)
    mujoco.mju_mat2Quat(x_des_quat, x_des_mat.flatten())
    if x_des_quat[0] < 0:
        x_des_quat *= -1.0
    return x_des_pos, x_des_quat


def slerp(q1, q2, alpha):
    """四元数球面线性插值"""
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
    """通过 UDP 发送数据，用于 PlotJuggler 实时监控"""
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
                # 等待控制循环更新最新状态
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
            # 仅保留最新状态并通知后台线程发送
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
    """MuJoCo 被动可视化后台线程"""
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

                    # 更新 MuJoCo 模型状态并同步视图
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
        """在 MuJoCo 窗口中绘制轨迹线段"""
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
    parser.add_argument("--config", default="/home/amdt/ur_force_ws/my_ws/config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--sys-config", default="/home/amdt/ur_force_ws/my_ws/config/sys_config.yaml", help="System config file")
    parser.add_argument("--init-pos", default="/home/amdt/ur_force_ws/my_ws/config/init_pos.txt", help="Initial joint positions file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        ctrl_cfg = yaml.safe_load(f)
    with open(args.sys_config, 'r') as f:
        sys_cfg = yaml.safe_load(f)

    # --- 控制器选择逻辑 ---
    ctrl_mode = ctrl_cfg.get("control_mode", "PD")
    
    torque_limits = np.array(ctrl_cfg['safety']['torque_limits'])
    vel_limits = np.array(ctrl_cfg['safety'].get('vel_limits'))

    # 加载 MuJoCo 模型
    xml_path = Path("script/ur5e_gripper/scene.xml")
    if not xml_path.exists():
        xml_path = Path("script/universal_robots_ur5e/scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    if ctrl_mode == "PD":
        pd_cfg = ctrl_cfg.get("pd_controller", {})
        stiffness = np.diag(pd_cfg.get("stiffness", [100.0]*6))
        damping_ratio = np.diag(pd_cfg.get('damping_ratio', [1.0]*6))
        damping = damping_ratio * 2.0 * np.sqrt(stiffness)
        active_controller = PDController(model, stiffness, damping, vel_limits)
        print(f"[INFO] Using PD Controller")
    else:
        imp_cfg = ctrl_cfg.get("impedance_controller", {})
        # 支持多种配置格式
        if 'stiffness' in imp_cfg and isinstance(imp_cfg['stiffness'], list):
            stiffness = np.diag(imp_cfg['stiffness'])
            damping_ratio = np.diag(imp_cfg.get('damping_ratio', [1.0]*6))
            target_inertia = np.diag(imp_cfg.get('inertia', [1.0]*6))
            # 临界阻尼: D = 2 * zeta * sqrt(K * M)
            damping = damping_ratio * 2.0 * np.sqrt(stiffness @ target_inertia)
        else:
            target_inertia = np.array(imp_cfg.get('target_inertia', [1.0]*6))
            stiffness = np.array(imp_cfg['stiffness'])
            damping_ratio = np.array(imp_cfg['damping_ratio'])
            damping = damping_ratio * 2 * np.sqrt(stiffness * target_inertia)
        
        active_controller = ImpedanceController(model, stiffness, damping, target_inertia, vel_limits)
        print(f"[INFO] Using Impedance Controller")

    circle_radius = ctrl_cfg['trajectory']['circle_radius']
    circle_omega = ctrl_cfg['trajectory']['circle_omega']
    dt = float(ctrl_cfg['trajectory']['control_dt'])
    if dt <= 0.0:
        raise ValueError("control_dt must be > 0")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # 加载初始关节位置
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
    
    total_loop_count = 0
    freq_loop_count = 0
    freq_start_time = time.perf_counter()
    next_tick = time.perf_counter()
    vis_period = 1.0 / max(1.0, float(VIS_FREQ))
    next_vis_update = time.perf_counter()
    traj_sample_period = 0.05  # 20 Hz 轨迹采样
    next_traj_sample = 0.0
    traj_started = False
    
    # 时间规划逻辑:
    # 0-1s: 稳定阶段 (不发送力矩)
    # 1-5s: 平滑移动到初始位置
    # 5s+:  执行 8 字轨迹轨迹
    STABILIZE_END = 1.0
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

            # 根据阶段定义期望状态
            if t_now <= STABILIZE_END:
                # 阶段 1: 稳定阶段 - 记录初始位置，不发送控制力矩
                x_des_pos_start = x_curr_pos.copy()
                x_des_quat_start = x_curr_quat.copy()
                
                x_des_pos = x_curr_pos.copy()
                x_des_quat = x_curr_quat.copy()
                v_des = np.zeros(6)
            elif t_now <= RESET_END:
                # 阶段 2: 平滑线性插值到目标位置
                RESET_BUFFER = 1.0
                alpha = (t_now - STABILIZE_END) / (RESET_END - STABILIZE_END - RESET_BUFFER)
                alpha = np.clip(alpha, 0.0, 1.0)
                
                # 位置线性插值
                x_des_pos = (1.0 - alpha) * x_des_pos_start + alpha * x_des_pos_init
                # 姿态球面线性插值 (SLERP)
                x_des_quat = slerp(x_des_quat_start, x_des_quat_init, alpha)
                v_des = np.zeros(6)
            else:
                # 阶段 3: 执行 8 字轨迹
                if not traj_started:
                    traj_started = True
                
                t_traj = t_now - RESET_END
                x_des_pos, x_des_quat = get_traj(t_traj, x_des_pos_init, circle_radius, circle_omega)
                v_des = np.zeros(6)

            x_des_mat = np.zeros(9)
            mujoco.mju_quat2Mat(x_des_mat, x_des_quat)
            x_des_mat = x_des_mat.reshape(3, 3)

            # 使用封装后的控制器计算关节力矩
            tau = active_controller.compute_torque(
                data, ee_site_id, 
                x_des_pos, x_des_quat, v_ee, v_des
            )
            tau = np.clip(tau, -torque_limits, torque_limits)

            # 仅在稳定阶段结束后发送力矩
            if t_now > STABILIZE_END:
                ok = rtde_c.directTorque(tau.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break
                        
            # 记录轨迹点
            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(x_des_pos.copy())
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            total_loop_count += 1
            logger.update(total_loop_count, x_curr_pos, x_des_pos, x_curr_quat, x_des_quat, tau, q, dq, active_controller.F_task)

            # 更新可视化
            now_for_vis = time.perf_counter()
            if now_for_vis >= next_vis_update:
                visualizer.update(q, trajectory_points, target_trajectory, x_des_pos, x_des_mat)
                next_vis_update += vis_period
                if next_vis_update < now_for_vis:
                    next_vis_update = now_for_vis + vis_period

            # 打印控制频率
            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                frequency = freq_loop_count / elapsed
                print(f"[INFO] Control Frequency: {frequency:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            # 循环计时，确保 dt 间隔
            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        print("[INFO] Cleaning up...")
        if visualizer is not None:
            visualizer.stop()
        logger.stop()
        try:
            rtde_c.stopScript()
            rtde_c.disconnect()
        except:
            pass
        print("[INFO] Script stopped")

if __name__ == "__main__":
    main()
