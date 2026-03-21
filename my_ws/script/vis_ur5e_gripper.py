# 最简单的阻抗控制，直接把任务空间PD设定的力矩乘上质量阵再投影到关节空间
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端（避免 Qt 兼容性问题）
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Cartesian impedance control gains.
# 刚度得调得很高
impedance_pos = np.asarray([5000.0, 5000.0, 5000.0])  # [N/m]
impedance_ori = np.asarray([200000.0, 200000.0, 200000.0])  # [Nm/rad]

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# 绘图窗口宽度（秒）
window_width = 5

# IK 参数
ik_max_iters = 30
ik_tol = 1e-4
ik_damping = 1e-3  # 阻尼因子，避免雅可比奇异
ori_weight = 1.0   # 姿态误差权重，可调小以减缓旋转


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    dof_ids: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """阻尼最小二乘IK，跟踪位置+姿态，求受控关节角度。"""
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = data.qpos
    data_ik.qvel[:] = data.qvel
    mujoco.mj_forward(model, data_ik)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    drot = np.zeros(3)
    jac_pos = np.zeros((3, model.nv))
    jac_ori = np.zeros((3, model.nv))
    for _ in range(ik_max_iters):
        mujoco.mj_forward(model, data_ik)
        cur_pos = data_ik.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(site_quat, data_ik.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, target_quat, site_quat_conj)
        if error_quat[0] < 0:
            error_quat = -error_quat
        mujoco.mju_quat2Vel(drot, error_quat, 1.0)
        err6 = np.concatenate([target_pos - cur_pos, ori_weight * drot])
        if np.linalg.norm(err6) < ik_tol:
            break
        mujoco.mj_jacSite(model, data_ik, jac_pos, jac_ori, site_id)
        J_full = np.vstack([jac_pos[:, dof_ids], jac_ori[:, dof_ids]])
        JJ = J_full @ J_full.T + ik_damping * np.eye(6)
        dq = J_full.T @ np.linalg.solve(JJ, err6)
        data_ik.qpos[dof_ids] += dq
        mujoco.mj_normalizeQuat(model, data_ik.qpos)
    return data_ik.qpos[dof_ids].copy()


def make_pause_toggle(paused_state: dict):
    """构造键盘回调，切换暂停标志。"""

    def key_callback(keycode):
        if keycode == 32:  # 空格键
            paused_state["value"] = not paused_state["value"]
            status = "暂停" if paused_state["value"] else "继续"
            print(f"\n>>> 仿真{status} <<<\n")

    return key_callback


def draw_trajectory(viewer, positions, color=[0, 1, 0, 1], width=0.002, clear=False):
    """
    绘制轨迹线段。

    参数：
    - viewer: Mujoco 查看器对象。
    - positions: 存储位置的列表。
    - color: 线段颜色，RGBA 格式列表。
    - width: 线条宽度，默认为 0.002。
    - clear: 是否清除之前的几何体，默认为 False。
    """
    # 清除之前的可视化几何体（仅第一次调用时）
    if clear:
        viewer.user_scn.ngeom = 0

    # 使用 mjv_connector 绘制路径线段
    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        # 获取起点和终点的 3D 坐标
        from_pos = np.array(positions[i][:3], dtype=np.float64)
        to_pos = np.array(positions[i + 1][:3], dtype=np.float64)
        
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            from_pos,
            to_pos
        )
        # 设置几何体的颜色（逐个元素赋值）
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[0] = color[0]
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[1] = color[1]
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[2] = color[2]
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[3] = color[3]
        viewer.user_scn.ngeom += 1


class RealtimePlotter:
    """实时绘图类，在独立线程中运行matplotlib"""
    def __init__(self, maxlen=50000):
        self.maxlen = maxlen
        self.lock = threading.Lock()
        self.running = False
        
        # 数据存储
        self.time_data = deque(maxlen=maxlen)
        self.pos_error = [deque(maxlen=maxlen) for _ in range(3)]
        self.ori_error = [deque(maxlen=maxlen) for _ in range(3)]
        self.torque = [deque(maxlen=maxlen) for _ in range(6)]
        self.q_current = [deque(maxlen=maxlen) for _ in range(6)]
        self.q_target = [deque(maxlen=maxlen) for _ in range(6)]
        
        # 夹爪数据
        self.gripper_current = deque(maxlen=maxlen)
        self.gripper_target = deque(maxlen=maxlen)
        
        self.lines_err_pos = []
        self.lines_err_ori = []
        self.lines_tau = []
        self.lines_q_curr = []
        self.lines_q_targ = []
        self.line_gripper_curr = None
        self.line_gripper_targ = None

    def start(self):
        """在独立线程中启动绘图"""
        self.running = True
        plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        plot_thread.start()
        
    def stop(self):
        """停止绘图线程"""
        self.running = False
        
    def update_data(self, t, pos_err, ori_err, torques, q_curr, q_des, g_curr=None, g_targ=None):
        """更新数据，由主仿真线程调用"""
        with self.lock:
            self.time_data.append(t)
            for i in range(3):
                self.pos_error[i].append(pos_err[i])
                self.ori_error[i].append(ori_err[i])
            for i in range(6):
                self.torque[i].append(torques[i])
                self.q_current[i].append(q_curr[i])
                self.q_target[i].append(q_des[i])
            if g_curr is not None:
                self.gripper_current.append(g_curr)
            if g_targ is not None:
                self.gripper_target.append(g_targ)
    
    def _plot_loop(self):
        """绘图线程的主循环"""
        # Figure 1: 关节力矩 (3行2列)
        self.fig1, self.axes1 = plt.subplots(3, 2, figsize=(8, 4))
        self.fig1.subplots_adjust(hspace=0.5, wspace=0.3)
        self.fig1.suptitle('Joint Torques')
        
        axes_tau = self.axes1.flatten()
        for i in range(6):
            ax = axes_tau[i]
            ax.set_title(f'Joint {i+1} Torque (Nm)')
            ax.grid(True)
            self.lines_tau.append(ax.plot([], [], label='Torque')[0])

        # Figure 2: 控制情况 (4行2列) - 前3行显示关节角度，最后一行显示误差
        self.fig2, self.axes2 = plt.subplots(4, 2, figsize=(8, 6))
        self.fig2.subplots_adjust(hspace=0.6, wspace=0.3)
        self.fig2.suptitle('Control Performance')
        
        # 1. 关节角度 (前6个子图)
        axes_q = self.axes2.flatten()[:6]
        for i in range(6):
            ax = axes_q[i]
            ax.set_title(f'Joint {i+1} Angle (rad)')
            ax.grid(True)
            self.lines_q_curr.append(ax.plot([], [], '-', color='b', label='Current')[0])
            self.lines_q_targ.append(ax.plot([], [], '--', color='r', label='Target')[0])
            ax.legend(loc='upper right')
            
        # 2. 位置误差 (第4行第1列)
        ax_pos = self.axes2[3, 0]
        ax_pos.set_title('Position Error (m)')
        ax_pos.grid(True)
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        self.lines_err_pos = [ax_pos.plot([], [], color=colors[i], label=labels[i])[0] for i in range(3)]
        ax_pos.legend(loc='upper right')
        
        # 3. 姿态误差 (第4行第2列)
        ax_ori = self.axes2[3, 1]
        ax_ori.set_title('Orientation Error (rad)')
        ax_ori.grid(True)
        labels_ori = ['RX', 'RY', 'RZ']
        self.lines_err_ori = [ax_ori.plot([], [], color=colors[i], label=labels_ori[i])[0] for i in range(3)]
        ax_ori.legend(loc='upper right')
        
        # Figure 3: 夹爪开合 (1行1列)
        self.fig3, self.ax3 = plt.subplots(1, 1, figsize=(6, 4))
        self.ax3.set_title('Gripper Position Control')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Angle (rad)')
        self.ax3.grid(True)
        self.line_gripper_curr = self.ax3.plot([], [], '-', color='g', label='Current Position')[0]
        self.line_gripper_targ = self.ax3.plot([], [], '--', color='orange', label='Target Instruction')[0]
        self.ax3.legend(loc='upper right')
        
        def animate(frame):
            if not self.running: return []
            with self.lock:
                if not self.time_data: return []
                t = list(self.time_data)
                p_err = [list(d) for d in self.pos_error]
                o_err = [list(d) for d in self.ori_error]
                tau = [list(d) for d in self.torque]
                q_c = [list(d) for d in self.q_current]
                q_t = [list(d) for d in self.q_target]
                g_c = list(self.gripper_current)
                g_t = list(self.gripper_target)
            
            # Update Figure 1 (Torques)
            for i in range(6):
                self.lines_tau[i].set_data(t, tau[i])
            
            # Update Figure 2 (Angles)
            for i in range(6):
                self.lines_q_curr[i].set_data(t, q_c[i])
                self.lines_q_targ[i].set_data(t, q_t[i])
                
            # Update Figure 2 (Errors)
            for i in range(3):
                self.lines_err_pos[i].set_data(t, p_err[i])
                self.lines_err_ori[i].set_data(t, o_err[i])

            # Update Figure 3 (Gripper)
            if len(g_c) > 0:
                self.line_gripper_curr.set_data(t, g_c)
            if len(g_t) > 0:
                self.line_gripper_targ.set_data(t, g_t)
            
            # Auto-scale logic
            if len(t) > 0:
                current_time = t[-1]
                xmin = max(0, current_time - window_width)
                xmax = current_time + 0.5
                
                # Find indices in the visible window
                start_idx = 0
                for i, time_val in enumerate(t):
                    if time_val >= xmin:
                        start_idx = i
                        break
                
                # Helper to set ylim
                def adjust_ylim(ax, lines):
                    y_min, y_max = float('inf'), float('-inf')
                    has_data = False
                    for line in lines:
                        y_data = line.get_ydata()
                        if len(y_data) > start_idx:
                            visible_y = y_data[start_idx:]
                            if len(visible_y) > 0:
                                y_min = min(y_min, np.min(visible_y))
                                y_max = max(y_max, np.max(visible_y))
                                has_data = True
                    
                    if has_data:
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                        if margin == 0: margin = 0.1
                        ax.set_ylim(y_min - margin, y_max + margin)

                # Scale Figure 1 axes (Torques)
                axes_tau = self.axes1.flatten()
                for i in range(6):
                    ax = axes_tau[i]
                    ax.set_xlim(xmin, xmax)
                    adjust_ylim(ax, [self.lines_tau[i]])

                # Scale Figure 2 axes (Angles)
                axes_q = self.axes2.flatten()[:6]
                for i in range(6):
                    ax = axes_q[i]
                    ax.set_xlim(xmin, xmax)
                    adjust_ylim(ax, [self.lines_q_curr[i], self.lines_q_targ[i]])

                # Scale Figure 2 axes (Errors)
                # Position Error
                ax_pos = self.axes2[3, 0]
                ax_pos.set_xlim(xmin, xmax)
                adjust_ylim(ax_pos, self.lines_err_pos)
                
                # Orientation Error
                ax_ori = self.axes2[3, 1]
                ax_ori.set_xlim(xmin, xmax)
                adjust_ylim(ax_ori, self.lines_err_ori)

                # Scale Figure 3 (Gripper)
                self.ax3.set_xlim(xmin, xmax)
                adjust_ylim(self.ax3, [self.line_gripper_curr, self.line_gripper_targ])
            
            return self.lines_tau + self.lines_q_curr + self.lines_q_targ + self.lines_err_pos + self.lines_err_ori + [self.line_gripper_curr, self.line_gripper_targ]
        
        # 降低刷新频率到 200ms
        anim1 = FuncAnimation(self.fig1, animate, interval=200, blit=False)
        anim2 = FuncAnimation(self.fig2, animate, interval=200, blit=False)
        anim3 = FuncAnimation(self.fig3, animate, interval=200, blit=False)
        
        plt.show()
        
        self.running = False


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data (使用力矩控制的模型)
    # model = mujoco.MjModel.from_xml_path("my_ws/script/universal_robots_ur5e/scene_torque.xml")

    model = mujoco.MjModel.from_xml_path("my_ws/script/ur5e_gripper/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)

    # End-effector site we wish to control.
    site_id = model.site("attachment_site").id

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    
    # 夹爪致动器 ID
    try:
        gripper_actuator_id = model.actuator("robotiq_85_left_knuckle_joint").id
    except KeyError:
        gripper_actuator_id = None

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]

    # 获取夹爪位置数据（用于绘图）
    gripper_joint_id = None
    try:
        gripper_joint_id = model.joint("robotiq_85_left_knuckle_joint").id
    except KeyError:
        pass

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    # ========== 暂停/继续控制标志 ==========
    paused_state = {"value": False}
    key_callback = make_pause_toggle(paused_state)

    # ========== 创建实时绘图器（matplotlib，独立线程）==========
    plot_length = int(window_width / dt)
    plotter = RealtimePlotter(maxlen=plot_length)
    plotter.start()  # 启动绘图线程

    # Trajectory visualization
    trajectory_points = []  # 存储末端执行器实际轨迹点（红色）
    reference_trajectory = []  # 存储参考轨迹点（绿色）
    max_trajectory_points = 50  # 最大轨迹点数

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 调整相机视角
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        # 初始化目标位置和姿态
        initial_target_pos = np.array([0.5, 0.0, 0.3])
        data.mocap_pos[mocap_id] = initial_target_pos
        
        # 使用末端执行器的当前姿态作为初始目标
        mujoco.mj_forward(model, data)
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        while viewer.is_running():
            step_start = time.time()

            if not paused_state["value"]:
                # ========== 1. 更新目标轨迹 (8字形) ==========
                t = data.time
                # 中心位置
                center_pos = np.array([0.0, 0.4, 0.4])
                # 8字形轨迹: y = A*sin(w*t), z = B*sin(2*w*t)
                # 周期 T = 10s -> w = 2*pi/10 = 0.2*pi
                w = 0.8 * np.pi
                amp_x = 0.2
                amp_y = 0.1
                amp_z = 0.1
                
                target_pos = center_pos.copy()
                target_pos[0] += amp_x * np.sin(w * t)
                target_pos[1] += amp_y * np.sin(2 * w * t)
                target_pos[2] += amp_z * np.sin(w * t)
                
                # 更新 mocap 位置
                data.mocap_pos[mocap_id] = target_pos

                # ========== 计算切线方向并更新姿态 ==========
                # 速度向量 (切线方向)
                vx = amp_x * w * np.cos(w * t)
                vy = amp_y * 2 * w * np.cos(2 * w * t)
                vz = amp_z * w * np.cos(w * t)
                tangent = np.array([vx, vy, vz])
                norm = np.linalg.norm(tangent)
                
                if norm > 1e-6:
                    tangent /= norm
                    # 构建旋转矩阵: X轴沿切线方向, Z轴保持竖直向下 [0, 0, -1]
                    z_axis = np.array([0, 0, -1])
                    y_axis = np.cross(z_axis, tangent)
                    # 重新计算 z_axis 确保正交 (虽然这里已经是正交的)
                    # z_axis = np.cross(tangent, y_axis)
                    
                    # 构造 3x3 旋转矩阵 (列向量为基向量)
                    # R = [xaxis, yaxis, zaxis]
                    mat = np.zeros(9)
                    mat[0] = tangent[0]; mat[1] = y_axis[0]; mat[2] = z_axis[0]
                    mat[3] = tangent[1]; mat[4] = y_axis[1]; mat[5] = z_axis[1]
                    mat[6] = tangent[2]; mat[7] = y_axis[2]; mat[8] = z_axis[2]
                    
                    # 转换为四元数并更新 mocap
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, mat)
                    data.mocap_quat[mocap_id] = quat

                # ========== 从mocap目标通过IK解算关节期望（位置+姿态）但没有用在控制中 ==========
                target_pos = data.mocap_pos[mocap_id].copy()
                target_quat = data.mocap_quat[mocap_id].copy()
                q_des = solve_ik(model, data, site_id, dof_ids, target_pos, target_quat)

                # Position error (m).
                dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
                
                # Orientation error (rad, axis-angle).
                mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
                # 选择最短路径
                if error_quat[0] < 0:
                    error_quat *= -1
                drot = np.zeros(3)
                mujoco.mju_quat2Vel(drot, error_quat, 1.0)

                # Jacobian.
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
                # 提取对应于受控关节的列 (6, n_dofs)
                jac_reduced = jac[:, dof_ids]
                
                # 末端速度
                x_dot = jac_reduced @ data.qvel[dof_ids]

                # Compute the task-space inertia matrix.
                # 从关节空间的质量矩阵M计算任务空间的质量矩阵Mx
                mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
                Mx_inv = jac_reduced @ M_inv[dof_ids][:, dof_ids] @ jac_reduced.T
                if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                    Mx = np.linalg.inv(Mx_inv)
                else:
                    Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

                # 夹爪随轨迹开闭: q_gripper = 0.4 * (1 + sin(w*t)) -> 0到0.8弧度循环
                if gripper_actuator_id is not None:
                    pos_min = 0.096
                    pos_max = 0.64
                    ratio = 0.5 * np.sin(5 * w * t) + 0.5
                    gripper_ctrl = pos_min + (pos_max - pos_min) * ratio
                    data.ctrl[gripper_actuator_id] = gripper_ctrl
                else:
                    gripper_ctrl = None

                # 笛卡尔空间阻抗控制力: F = Kp*error - Kd*velocity
                F_des = np.zeros(6)
                F_des[:3] = Kp[:3] * dx - Kd[:3] * x_dot[:3]
                F_des[3:] = Kp[3:] * drot - Kd[3:] * x_dot[3:]
                
                # 动力学一致性映射到关节力矩
                tau = jac_reduced.T @ Mx @ F_des

                # Add gravity compensation.
                if gravity_compensation:
                    # 获取完整的重力项并提取受控关节对应的部分
                    tau += data.qfrc_bias[dof_ids]

                # Set the control signal and step the simulation.
                # 裁剪力矩
                tau_clipped = np.clip(tau, model.actuator_ctrlrange[actuator_ids, 0], model.actuator_ctrlrange[actuator_ids, 1])
                data.ctrl[actuator_ids] = tau_clipped

                # ========== 更新 matplotlib 绘图数据 ==========
                pos_error = data.mocap_pos[mocap_id] - data.site(site_id).xpos
                
                # 姿态误差转换为旋转向量表示
                ori_error_vec = np.zeros(3)
                mujoco.mju_quat2Vel(ori_error_vec, error_quat, 1.0)
                
                # 更新绘图器数据（每个仿真步都更新，线程安全）
                g_curr = data.qpos[gripper_joint_id] if gripper_joint_id is not None else None
                plotter.update_data(data.time, pos_error, ori_error_vec, tau_clipped, data.qpos[dof_ids], q_des, g_curr, gripper_ctrl)

                # 打印信息（每0.2秒）
                if data.time % 0.2 < dt:
                    pos_error_norm = np.linalg.norm(pos_error)
                    ori_error_norm = np.linalg.norm(ori_error_vec)

                    print("\n" + "=" * 85)
                    print(f"时间: {data.time:.4f}s")
                    print("-" * 85)
                    print("末端误差:")
                    print(
                        f"  位置误差: {pos_error_norm:>8.5f} m   "
                        f"[{pos_error[0]:>7.4f}, {pos_error[1]:>7.4f}, {pos_error[2]:>7.4f}]"
                    )
                    print(
                        f"  姿态误差: {ori_error_norm:>8.5f} rad "
                        f"[{ori_error_vec[0]:>7.4f}, {ori_error_vec[1]:>7.4f}, {ori_error_vec[2]:>7.4f}]"
                    )
                    print("-" * 85)
                    print("关节位置:")
                    for i, name in enumerate(joint_names):
                        print(f"{name:<12} q={data.qpos[i]:>8.4f}, qvel={data.qvel[i]:>8.4f}")
                    print("-" * 85)
                    print("关节力矩:")
                    for i, name in enumerate(joint_names):
                        print(f"{name:<12} τ={tau[actuator_ids[i]]:>11.4f} Nm")
                    print("=" * 85)

                # Step the simulation.
                mujoco.mj_step(model, data)

                # 记录轨迹点
                if len(trajectory_points) == 0 or data.time - trajectory_points[-1][-1] > 0.1:
                    current_pos = data.site(site_id).xpos.copy()
                    trajectory_points.append(np.append(current_pos, data.time))
                    if len(trajectory_points) > max_trajectory_points:
                        trajectory_points.pop(0)
                
                if len(reference_trajectory) == 0 or data.time - reference_trajectory[-1][-1] > 0.1:
                    target_pos_curr = data.mocap_pos[mocap_id].copy()
                    reference_trajectory.append(np.append(target_pos_curr, data.time))
                    if len(reference_trajectory) > max_trajectory_points:
                        reference_trajectory.pop(0)

            # 绘制两条轨迹
            draw_trajectory(viewer, reference_trajectory, color=[0, 1, 0, 1], clear=True)  # 绿色参考轨迹
            draw_trajectory(viewer, trajectory_points, color=[1, 0, 0, 1])  # 红色实际轨迹

            # 不管暂停与否都要同步viewer和camlight
            mujoco.mj_camlight(model, data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # 仿真结束，停止绘图线程
        plotter.stop()


if __name__ == "__main__":
    main()