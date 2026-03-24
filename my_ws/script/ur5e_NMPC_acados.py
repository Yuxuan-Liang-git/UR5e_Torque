import sys
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import math
import yaml
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from nmpc_controller_ur5e import UR5eNMPC, mj2pin_pos, mj2pin_rot

_HERE = Path(__file__).parent.parent
_CONFIG_PATH = _HERE / "config" / "ctrl_config.yaml"

# 加载配置文件
with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

nmpc_cfg = config["nmpc_controller"]

# _XML  = _HERE /"script" / "universal_robots_ur5e" / "scene_torque.xml"
_XML  = _HERE / nmpc_cfg["mjcf_path"]

# ═══════════════════════════════════════════════════════════════════════════════
# NMPC parameters
# ═══════════════════════════════════════════════════════════════════════════════
N   = nmpc_cfg["horizon_steps"]   # prediction horizon steps
Tf  = nmpc_cfg["horizon_time"]    # prediction horizon (s)
dt  = nmpc_cfg.get("control_dt", Tf / N) # control / simulation step (s)
REBUILD_SOLVER = nmpc_cfg["rebuild"] # True: 重新生成C代码; False: 直接加载已有solver
SAVE_TRAJ = False

# Trajectory (圆形轨迹)
_TRAJ_CFG = config["trajectory"]
CIRCLE_RADIUS = _TRAJ_CFG["circle_radius"]
CIRCLE_SPEED  = _TRAJ_CFG["circle_omega"]
CIRCLE_CENTER = np.array([0.0, 0.5, 0.4])

# Plot window
ENABLE_PLOT  = True
window_width = 4.0   # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory helpers (MuJoCo world frame)
# ═══════════════════════════════════════════════════════════════════════════════
def get_circle_target(t: float) -> np.ndarray:
    w = CIRCLE_SPEED
    return np.array([
        CIRCLE_CENTER[0] + 2 * CIRCLE_RADIUS * math.sin(w * t),
        CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(2 * w * t),
        CIRCLE_CENTER[2] + 1.5 * CIRCLE_RADIUS * math.cos(w * t),
    ])


def get_target_orientation(t: float) -> np.ndarray:
    """基于有限差分计算切线方向的姿态阵."""
    p0 = get_circle_target(t)
    p1 = get_circle_target(t + dt)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return np.eye(3)
    tangent /= norm
    
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, tangent)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm if y_norm > 1e-6 else np.array([0., 1., 0.])
    z_axis = np.cross(tangent, y_axis)
    return np.column_stack([tangent, y_axis, z_axis])


class ReferenceQueue:
    """维护 N+1 个参考点的循环缓冲区."""
    def __init__(self, N: int, dt: float, get_pos, get_ori):
        self.N       = N
        self.dt      = dt
        self.get_pos = get_pos
        self.get_ori = get_ori
        self._pos    = np.zeros((3, N + 1))
        self._rot    = np.zeros((9, N + 1))
        self._head   = 0

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


def make_pause_toggle(state: dict):
    def cb(keycode):
        if keycode == 32:
            state["value"] = not state["value"]
            print("\n>>> 仿真", "暂停" if state["value"] else "继续", "<<<\n")
    return cb


class RealtimePlotter:
    def __init__(self, maxlen: int = 100000):
        self.lock    = threading.Lock()
        self.running = False
        self.t_data  = deque(maxlen=maxlen)
        self.pos_err = [deque(maxlen=maxlen) for _ in range(3)]
        self.ori_err = [deque(maxlen=maxlen) for _ in range(3)]
        self.torques = [deque(maxlen=maxlen) for _ in range(6)]
        self.jerk    = deque(maxlen=maxlen)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def update(self, t, pos_err_3, ori_err_3, torques_6, jerk_norm: float):
        with self.lock:
            self.t_data.append(t)
            for i in range(3):
                self.pos_err[i].append(pos_err_3[i])
                self.ori_err[i].append(ori_err_3[i])
            for i in range(6):
                self.torques[i].append(torques_6[i])
            self.jerk.append(jerk_norm)

    def _loop(self):
        fig, (ax_p, ax_o, ax_t, ax_j) = plt.subplots(4, 1, figsize=(11, 12))
        fig.suptitle("UR5e Acados NMPC — Real-time")

        ax_p.set_title("EE Position Error (m)")
        ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Error (m)"); ax_p.grid(True)
        ax_o.set_title("EE Orientation Error (rad, axis-angle)")
        ax_o.set_xlabel("Time (s)"); ax_o.set_ylabel("Error (rad)"); ax_o.grid(True)
        ax_t.set_title("Joint Torques (Nm)")
        ax_t.set_xlabel("Time (s)"); ax_t.set_ylabel("Torque (Nm)"); ax_t.grid(True)
        ax_j.set_title("EE Jerk Magnitude (m/s³)")
        ax_j.set_xlabel("Time (s)"); ax_j.set_ylabel("|jerk| (m/s³)"); ax_j.grid(True)

        colors_xyz = ["r", "g", "b"]
        lines_pos = [ax_p.plot([], [], c, label=l, lw=1.5)[0] for c, l in zip(colors_xyz, ["X", "Y", "Z"])]
        lines_ori = [ax_o.plot([], [], c, label=l, lw=1.5)[0] for c, l in zip(colors_xyz, ["rx", "ry", "rz"])]
        
        torque_labels = ["sh_pan", "sh_lift", "elbow", "w1", "w2", "w3"]
        torque_colors = ["r", "g", "b", "c", "m", "y"]
        lines_tau = [ax_t.plot([], [], torque_colors[i], label=torque_labels[i], lw=1.2)[0] for i in range(6)]
        
        line_jerk, = ax_j.plot([], [], "k", lw=1.5, label="|jerk|")
        ax_p.legend(ncol=3, fontsize=8); ax_o.legend(ncol=3, fontsize=8)
        ax_t.legend(ncol=3, fontsize=8); ax_j.legend(fontsize=8)

        def _set_ylim(ax, datasets, margin=0.1, min_span=1e-6):
            if not datasets or not datasets[0][0]: return
            x0, x1 = ax.get_xlim()
            vals = []
            for xs, ys in datasets:
                # 优化选择: 只看窗口内的数据
                xs_arr, ys_arr = np.array(xs), np.array(ys)
                mask = (xs_arr >= x0) & (xs_arr <= x1)
                if np.any(mask):
                    vals.extend(ys_arr[mask].tolist())
            if not vals: return
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, min_span)
            ax.set_ylim(lo - margin * span, hi + margin * span)

        def animate(_):
            if not self.running: return all_lines
            with self.lock:
                if not self.t_data: return all_lines
                t = list(self.t_data)
                pe = [list(d) for d in self.pos_err]
                oe = [list(d) for d in self.ori_err]
                tau = [list(d) for d in self.torques]
                jk = list(self.jerk)

            cur = t[-1]
            x0 = max(0, cur - window_width)
            for ax in (ax_p, ax_o, ax_t, ax_j): ax.set_xlim(x0, cur + 0.5)
            
            for i, ln in enumerate(lines_pos): ln.set_data(t, pe[i])
            for i, ln in enumerate(lines_ori): ln.set_data(t, oe[i])
            for i, ln in enumerate(lines_tau): ln.set_data(t, tau[i])
            line_jerk.set_data(t, jk)

            _set_ylim(ax_p, [(t, pe[i]) for i in range(3)])
            _set_ylim(ax_o, [(t, oe[i]) for i in range(3)])
            _set_ylim(ax_t, [(t, tau[i]) for i in range(6)])
            _set_ylim(ax_j, [(t, jk)])
            return all_lines

        all_lines = lines_pos + lines_ori + lines_tau + [line_jerk]
        anim = FuncAnimation(fig, animate, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        self.running = False


def draw_trajectory(viewer, positions, color=(0, 1, 0, 1), width=0.002, clear=False):
    if clear: viewer.user_scn.ngeom = 0
    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.array(positions[i][:3]), np.array(positions[i+1][:3]))
        g.rgba[:] = color
        viewer.user_scn.ngeom += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── MuJoCo ────────────────────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(str(_XML))
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    joint_names = ["shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"]
    actuator_ids = np.array([m.actuator(n).id for n in joint_names])
    site_id = m.site("attachment_site").id
    mocap_id = m.body("target").mocapid[0]
    nq = len(joint_names)

    # 夹爪致动器 ID
    try:
        gripper_actuator_id = m.actuator("robotiq_85_left_knuckle_joint").id
        gripper_joint_id = m.joint("robotiq_85_left_knuckle_joint").id
    except KeyError:
        gripper_actuator_id = None
        gripper_joint_id = None

    # 初始关节角从 keyframe 读取
    key_name = "home"
    key = m.key(key_name)
    key_id = key.id
    q0 = key.qpos.copy()
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    # ── Build acados solver ───────────────────────────────────────────────────
    print("Building acados solver …")
    ctrl = UR5eNMPC(N=N, Tf=Tf, rebuild=REBUILD_SOLVER)
    print("Solver ready.\n")

    # ── 轨迹函数定义 ──────────────────────────────────────────────────────────
    _get_pos = get_circle_target
    _get_ori = get_target_orientation

    # ── Reference queue ───────────────────────────────────────────────────────
    ref_queue = ReferenceQueue(N, dt, _get_pos, _get_ori)
    ref_queue.init(_get_pos(0.0), _get_ori(0.0))

    # ── 暖启动 ────────────────────────────────────────────────────────────────
    x0_warm = np.concatenate([q0[:nq], np.zeros(nq)])
    for k in range(N + 1): ctrl.solver.set(k, "x", x0_warm)
    for k in range(N): ctrl.solver.set(k, "u", np.zeros(nq))

    # ── Plotter ───────────────────────────────────────────────────────────────
    plotter = RealtimePlotter() if ENABLE_PLOT else None
    if plotter: plotter.start()

    # ── State ─────────────────────────────────────────────────────────────────
    paused = {"value": False}
    traj_start = time.time()
    traj_pts, ref_pts = [], []
    max_pts = 60
    _ee_vel_buf = deque(maxlen=3)
    data_log = []
    _solve_times = deque(maxlen=200)

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model=m, data=d, show_left_ui=False, show_right_ui=False,
        key_callback=make_pause_toggle(paused),
    ) as viewer:
        mujoco.mj_resetDataKeyframe(m, d, key_id)
        mujoco.mj_forward(m, d)
        print("=== UR5e Acados NMPC (Circle Tracking) ===")
        print("SPACE to pause/resume\n")
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            t_wall = time.time()
            if not paused["value"]:
                t_traj = t_wall - traj_start
                # Map MuJoCo arm joints to NMPC state
                q_cur = d.qpos[actuator_ids].copy()
                v_cur = d.qvel[actuator_ids].copy()
                x0 = np.concatenate([q_cur, v_cur])

                ref_queue.step(t_traj)
                ref_pos, ref_rot = ref_queue.get()

                u_opt = ctrl.solve(x0, ref_pos, ref_rot)
                _solve_times.append(ctrl.time_tot)
                
                # 限幅从配置文件读取
                torque_lim = np.array(nmpc_cfg.get("torque_limits", [150, 150, 150, 28, 28, 28]))
                u_opt = np.clip(u_opt, -torque_lim, torque_lim)
                d.ctrl[actuator_ids] = u_opt

                # ── 夹爪 PD 控制 ───────────────────────────────────────────────
                if gripper_actuator_id is not None:
                    pos_min = 0.096
                    pos_max = 0.64
                    w = 0.8 * np.pi
                    ratio = 0.5 * np.sin(5 * w * t_traj) + 0.5
                    gripper_ctrl = pos_min + (pos_max - pos_min) * ratio
                    # 映射到致动器力矩
                    d.ctrl[gripper_actuator_id] = gripper_ctrl

                # Mocap Visualisation
                t0_pos, t0_rot = _get_pos(t_traj), _get_ori(t_traj)
                d.mocap_pos[mocap_id], q_mj = t0_pos, np.zeros(4)
                mujoco.mju_mat2Quat(q_mj, t0_rot.flatten())
                d.mocap_quat[mocap_id] = q_mj

                # Error Calc
                ee_pin = ctrl.f_fk_pos(q_cur).full().flatten()
                pos_err = ee_pin - ref_pos[:, 0]
                ee_rot_pin = ctrl.f_fk_rot(q_cur).full()
                ref_rot_mat = ref_rot[:, 0].reshape(3, 3, order='F')
                R_err = ref_rot_mat.T @ ee_rot_pin
                tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1 + 1e-6, 1 - 1e-6)
                theta = np.arccos(tr)
                coeff = theta / (2.0 * np.sin(theta) + 1e-9)
                ori_err = coeff * np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])

                # Jerk
                nv_full = m.nv
                jacp = np.zeros((3, nv_full))
                mujoco.mj_jacSite(m, d, jacp, None, site_id)
                # Map relevant columns of Jacobian to arm joints
                ee_vel = jacp[:, actuator_ids] @ v_cur
                _ee_vel_buf.append((t_traj, ee_vel))
                if len(_ee_vel_buf) == 3:
                    t0, v0 = _ee_vel_buf[0]; t1, v1 = _ee_vel_buf[1]; t2, v2 = _ee_vel_buf[2]
                    acc0, acc1 = (v1-v0)/max(t1-t0,1e-6), (v2-v1)/max(t2-t1,1e-6)
                    jerk_norm = float(np.linalg.norm((acc1-acc0)/max(0.5*(t1-t0+t2-t1),1e-6)))
                else: jerk_norm = 0.0

                if plotter: plotter.update(t_traj, pos_err, ori_err, u_opt, jerk_norm)
                _pos_norm, _ori_norm = float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err))
                data_log.append((t_traj, _pos_norm, _ori_norm, jerk_norm))

                # Viz points
                now_ee = d.site(site_id).xpos.copy()
                if not traj_pts or t_traj - traj_pts[-1][3] > 0.05:
                    traj_pts.append(np.append(now_ee, t_traj))
                    if len(traj_pts) > max_pts: traj_pts.pop(0)
                if not ref_pts or t_traj - ref_pts[-1][3] > 0.05:
                    ref_pts.append(np.append(t0_pos, t_traj))
                    if len(ref_pts) > max_pts: ref_pts.pop(0)

                mujoco.mj_step(m, d)

            draw_trajectory(viewer, ref_pts, color=(0,1,0,1), clear=True)
            draw_trajectory(viewer, traj_pts, color=(1,0,0,1))
            viewer.sync()
            time.sleep(max(0, dt - (time.time() - t_wall)))

    if plotter: plotter.stop()

    if data_log and SAVE_TRAJ:
        out_name = f"{datetime.now().strftime('%Y%m%d')}_circle_NMPC.csv"
        data_dir = _HERE / "Data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(data_dir / out_name, np.array(data_log), delimiter=",",
                   header="Time_s,Pos_err_norm_m,Ori_err_norm_rad,Jerk_norm_m_s3", comments="")
        print(f"[data] 已保存记录 → {data_dir / out_name}")


if __name__ == "__main__":
    main()