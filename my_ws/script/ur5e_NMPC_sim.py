# 滑模控制 (Sliding Mode Control) for UR5e with MINK IK solver
from pathlib import Path
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
# Mujoco的IK库
import mink

_HERE = Path(__file__).parent
_XML = _HERE / "ur5e_gripper" / "scene.xml"

# ========== 简单PD控制参数 ==========
# 控制律: τ = M*q̈_des + C + g，其中 q̈_des = Kp*e + Kd*ė
# 这样可以提供动力学补偿，改进6轴跟踪性能
# Kp = np.diag([300.0, 300.0, 300.0, 300.0, 300.0, 10.0])   # 位置增益（增加6轴）
# Kd = np.diag([20.0, 20.0, 20.0, 10.0, 10.0, 0.1])     # 速度增益（增加6轴）

Kp = 0.5 * np.diag([2000.0, 2000.0, 2000.0, 20000.0, 20000.0, 50000.0])   # 位置增益（增加6轴）
Kd = np.diag([20.0, 20.0, 20.0, 50.0, 50.0, 100.0])     # 速度增益（增加6轴）

# 期望关节位置（初始为home位置，可动态修改）
q_desired = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5])  # 单位: 弧度
qd_desired = np.zeros(6)  # 期望速度
qdd_desired = np.zeros(6)  # 期望加速度
gravity_compensation: bool = True

# 绘图窗口宽度（秒）
window_width = 5

def make_pause_toggle(paused_state: dict):
    """构造键盘回调，切换暂停标志。"""

    def key_callback(keycode):
        if keycode == 32:  # 空格键
            paused_state["value"] = not paused_state["value"]
            status = "暂停" if paused_state["value"] else "继续"
            print(f"\n>>> 仿真{status} <<<\n")

    return key_callback


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
        
        self.lines_err_pos = []
        self.lines_err_ori = []
        self.lines_tau = []
        self.lines_q_curr = []
        self.lines_q_targ = []

    def start(self):
        """在独立线程中启动绘图"""
        self.running = True
        plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
        plot_thread.start()
        
    def stop(self):
        """停止绘图线程"""
        self.running = False
        
    def update_data(self, t, pos_err, ori_err, torques, q_curr, q_des):
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
            # ax.legend(loc='upper right')

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
            
            return self.lines_tau + self.lines_q_curr + self.lines_q_targ + self.lines_err_pos + self.lines_err_ori
        
        # 降低刷新频率到 200ms
        anim1 = FuncAnimation(self.fig1, animate, interval=200, blit=False)
        anim2 = FuncAnimation(self.fig2, animate, interval=200, blit=False)
        
        plt.show()
        
        self.running = False


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


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load simulation model/data (用于力矩控制仿真)
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # 创建独立的 MINT Configuration（与仿真数据解耦，仅用于计算期望）
    ik_model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(ik_model)
    
    # 定义 MINT tasks（基于独立的 IK 模型）
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=5.0,
            lm_damping=1e-4,
        ),
        posture_task := mink.PostureTask(configuration.model, cost=5e-4),
    ]

    # 配置限制（简化以提高求解器稳定性）
    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    # 获取 mocap target id（仿真模型）
    mid = model.body("target").mocapid[0]
    solver = "daqp"
    
    # Override the simulation timestep.
    dt = 0.002
    model.opt.timestep = dt

    # 末端site
    site_id = model.site("attachment_site").id

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id
    
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

    # Pre-allocate numpy arrays.
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

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
    
    plot_counter = 0  # 绘图计数器

    M = np.zeros((model.nv, model.nv))  # 关节空间质量矩阵

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,  # 注册键盘回调
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        
        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        configuration.q[:] = data.qpos  # IK 模型与仿真模型同步到同一姿态
        posture_task.set_target(configuration.q)

        # 设置初始目标位置
        initial_target_pos = np.array([0.5, 0.0, 0.3])
        data.mocap_pos[mid] = initial_target_pos
        configuration.data.mocap_pos[mid] = initial_target_pos

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 调整相机视角
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30

        # 启用site坐标系显示
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

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
                data.mocap_pos[mid] = target_pos

                # ========== 计算切线方向并更新姿态 ==========
                # 速度向量 (切线方向)
                vx = amp_x * w * np.cos(w * t)
                vy = amp_y * 2 * w * np.cos(2 * w * t)
                vz = amp_z * w * np.cos(w * t)
                tangent = np.array([vx, vy, vz])
                norm = np.linalg.norm(tangent)
                
                if norm > 1e-6:
                    # 归一化切线向量
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
                    data.mocap_quat[mid] = quat
                
                # ========== 2. 施加随机扰动 (最大5N) ==========
                # 获取末端 body id
                ee_body_id = model.site_bodyid[site_id]
                # 生成随机力 [-5, 5] N
                disturbance = np.random.uniform(-10, 10, size=3)
                # 应用外力 (前3个分量是力，后3个是力矩)
                data.xfrc_applied[ee_body_id, :3] = disturbance
                
                # ========== 使用 MINT 求解器计算 IK ==========
                # 首先同步 configuration 到当前实际状态 
                configuration.q[:] = data.qpos
                
                # Update task target from mocap
                T_wt = mink.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)

                # Compute velocity and integrate into the next configuration.
                # 注意：这里只是计算期望的关节角度，不会影响实际仿真
                try:
                    vel = mink.solve_ik(configuration, tasks, dt, solver, limits=limits, damping=1e-4)
                    configuration.integrate_inplace(vel, dt)
                except mink.exceptions.NoSolutionFound:
                    # 如果求解器失败，保持当前配置
                    print(f"Warning: IK solver failed at time {data.time:.3f}s, keeping previous configuration")
                
                # 获取 IK 求解后的关节位置作为期望值（这只是目标，不是实际值）
                q_des = configuration.q[dof_ids]

                # ========== 获取当前关节状态 ==========
                q = data.qpos[dof_ids]
                qd = data.qvel[dof_ids]
                
                # ========== PD控制算法 ==========
                # 位置误差
                pos_error = q_des - q
                pos_error = np.arctan2(np.sin(pos_error), np.cos(pos_error))  # 归一化到[-π, π]
                
                # 速度误差
                vel_error = qd_desired - qd
                
                # 获取质量矩阵和重力补偿
                mujoco.mj_fullM(model, M, data.qM)
                M_ctrl = M[np.ix_(dof_ids, dof_ids)]
                gravity_comp = data.qfrc_bias[dof_ids]

                # ========== 改进的PD控制：包含动力学补偿 ==========
                # 计算期望加速度：q̈_des = Kp*e + Kd*ė
                u0 = M_ctrl @ (Kp @ pos_error + Kd @ vel_error)
                
                # 完整控制律：τ = M*q̈_des + C + g
                # tau_ctrl = M_ctrl @ u0 + gravity_comp
                tau_ctrl = u0 + 0.5 * gravity_comp

                
                # Set the control signal and step the simulation.
                np.clip(tau_ctrl, *model.actuator_ctrlrange[actuator_ids].T, out=tau_ctrl)
                data.ctrl[actuator_ids] = tau_ctrl 
                
                # 用于绘图的变量
                joint_error = pos_error
                
                # ========== 更新 matplotlib 绘图数据 ==========
                pos_error = data.mocap_pos[mid] - data.site(site_id).xpos
                
                # 姿态误差转换为旋转向量表示
                mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, data.mocap_quat[mid], site_quat_conj)
                # 选择最短路径
                if error_quat[0] < 0:
                    error_quat *= -1
                ori_error_vec = np.zeros(3)
                mujoco.mju_quat2Vel(ori_error_vec, error_quat, 1.0)
                
                # 降低绘图数据更新频率 (每10步更新一次，即20ms)
                plot_counter += 1
                if plot_counter % 10 == 0:
                    plotter.update_data(data.time, pos_error, ori_error_vec, tau_ctrl, q, q_des)

                if data.time % 0.2 < dt:
                    error_norm = np.linalg.norm(joint_error)
                    tau_norm = np.linalg.norm(tau_ctrl)
                    print("\n" + "=" * 85)
                    print(f"时间: {data.time:.4f}s")
                    print("-" * 85)
                    print(f"关节误差范数: {error_norm:>8.5f} rad  | 力矩范数: {tau_norm:>8.5f} Nm")
                    print("-" * 85)
                    print("关节位置误差 & 实际位置:")
                    for i, name in enumerate(joint_names):
                        print(f"{name:<12} e={joint_error[i]:>8.4f} rad, q={q[i]:>8.4f}, q_d={q_des[i]:>8.4f}")
                    print("-" * 85)
                    print("关节力矩:")
                    for i, name in enumerate(joint_names):
                        print(f"{name:<12} τ={tau_ctrl[i]:>11.4f} Nm")
                    print("=" * 85)
                # 使用 MuJoCo step 执行力矩控制
                # 这里才是真正改变机器人状态的地方
                mujoco.mj_step(model, data)

                # 记录轨迹点
                if len(trajectory_points) == 0 or data.time - trajectory_points[-1][-1] > 0.1:
                    current_pos = data.site(site_id).xpos.copy()
                    trajectory_points.append(np.append(current_pos, data.time))
                    if len(trajectory_points) > max_trajectory_points:
                        trajectory_points.pop(0)
                
                if len(reference_trajectory) == 0 or data.time - reference_trajectory[-1][-1] > 0.1:
                    target_pos_curr = data.mocap_pos[mid].copy()
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