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
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from nmpc_controller_ur5e import UR5eNMPC, mj2pin_pos, mj2pin_rot, _Rz90

_SCRIPT_DIR = Path(__file__).parent
_XML  = _SCRIPT_DIR / "ur5e_gripper" / "scene.xml"

# ── TrajLoader (从 Ref_traj 目录导入) ──────────────────────────────────────────
_REF_TRAJ_DIR = Path("/home/amdt/app_ws/my_mjctrl/Ref_traj")
sys.path.insert(0, str(_REF_TRAJ_DIR))
from ref_traj_loader import TrajLoader
# FILE_NAME = "20250224_pose.csv"
# FILE_NAME =  "20250224_pose_Smoothed.csv"
FILE_NAME = "20250224_pose_RealTimeSmoothed.csv"
_CSV = _REF_TRAJ_DIR / FILE_NAME

# ═══════════════════════════════════════════════════════════════════════════════
# NMPC parameters
# ═══════════════════════════════════════════════════════════════════════════════
N   = 30         # prediction horizon steps
Tf  = 0.3         # prediction horizon (s)  →  dt_mpc = 10 ms
dt  = Tf / N      # control / simulation step  (s)
REBUILD_SOLVER = True # True: 重新生成C代码（首次/修改模型后用）; False: 直接加载已有solver
USE_CSV_TRAJ   = False # True: 跟踪 CSV 轨迹; False: 跟踪圆形轨迹
# True: 保存轨迹日志到 Data 目录; False: 不保存
SAVE_TRAJ = True

# Trajectory (圆形轨迹)
CIRCLE_RADIUS = 0.1
CIRCLE_SPEED  = 0.4 * math.pi          # rad/s
CIRCLE_CENTER = np.array([0.0, 0.5, 0.4])

# Plot window
ENABLE_PLOT  = True
window_width = 4.0   # seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory helpers  (MuJoCo world frame, 保留备用，当前未使用)
# ═══════════════════════════════════════════════════════════════════════════════
def get_circle_target(t: float) -> np.ndarray:
    w = CIRCLE_SPEED
    return np.array([
        CIRCLE_CENTER[0] + 2 * CIRCLE_RADIUS * math.sin(w * t),
        CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(2 * w * t),
        CIRCLE_CENTER[2] + 1.5 * CIRCLE_RADIUS * math.cos(w * t),
    ])


def get_target_orientation(t: float) -> np.ndarray:
    """Tangent-aligned rotation matrix in MuJoCo frame."""
    # compute tangent direction by finite-difference of the position
    p0 = get_circle_target(t)
    p1 = get_circle_target(t + dt)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return np.eye(3)
    tangent /= norm
    # build orientation with X-axis along tangent, Z downwards
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, tangent)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm if y_norm > 1e-6 else np.array([0., 1., 0.])
    z_axis = np.cross(tangent, y_axis)
    return np.column_stack([tangent, y_axis, z_axis])


class ReferenceQueue:
    """
    Circular buffer that maintains a rolling horizon of N+1 reference points.
    Each control step: overwrite the stale head slot with the new horizon-end
    point (t_traj + N*dt), then advance the head pointer.
    Cost: O(1) per step — only 1 new trajectory point is computed.
    """
    def __init__(self, N: int, dt: float, get_pos, get_ori):
        """
        get_pos(t) -> (3,)   MuJoCo frame
        get_ori(t) -> (3,3)  MuJoCo frame
        可以传入 TrajLoader 的方法，也可以传入任意可调用对象。
        """
        self.N       = N
        self.dt      = dt
        self.get_pos = get_pos
        self.get_ori = get_ori
        self._pos    = np.zeros((3, N + 1))   # [3 x (N+1)]
        self._rot = np.zeros((9, N + 1))   # [9 x (N+1)], col-major
        self._head = 0                     # oldest slot = reference for current step

    def init(self, p0: np.ndarray, R0: np.ndarray):
        """
        Fill the entire buffer with the same initial pose (p0, R0).
        p0 : (3,)  position in MuJoCo frame
        R0 : (3,3) rotation matrix in MuJoCo frame
        """
        pos_pin = mj2pin_pos(p0)
        rot_pin = mj2pin_rot(R0).flatten(order="F")
        for k in range(self.N + 1):
            self._pos[:, k] = pos_pin
            self._rot[:, k] = rot_pin
        self._head = 0

    def step(self, t_traj: float):
        """
        Consume the current head (now stale) and append the new horizon-end
        point at t_traj + N*dt.  Call once per control step BEFORE get().
        """
        t_new = t_traj + self.N * self.dt
        p_new = self.get_pos(t_new)   # (3,)  MuJoCo frame
        R_new = self.get_ori(t_new)   # (3,3) MuJoCo frame
        self._pos[:, self._head] = mj2pin_pos(p_new)
        self._rot[:, self._head] = mj2pin_rot(R_new).flatten(order="F")
        self._head = (self._head + 1) % (self.N + 1)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return contiguous (3, N+1) and (9, N+1) arrays in correct time order
        [current_step, current+1, ..., current+N].
        """
        idx = np.arange(self._head, self._head + self.N + 1) % (self.N + 1)
        return self._pos[:, idx], self._rot[:, idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Pause callback
# ═══════════════════════════════════════════════════════════════════════════════
def make_pause_toggle(state: dict):
    def cb(keycode):
        if keycode == 32:
            state["value"] = not state["value"]
            print("\n>>> 仿真", "暂停" if state["value"] else "继续", "<<<\n")
    return cb


# ═══════════════════════════════════════════════════════════════════════════════
# Real-time plotter  (identical structure to SMC version)
# ═══════════════════════════════════════════════════════════════════════════════
class RealtimePlotter:
    def __init__(self, maxlen: int = 50000):
        self.lock    = threading.Lock()
        self.running = False
        self.t_data  = deque(maxlen=maxlen)
        self.pos_err = [deque(maxlen=maxlen) for _ in range(3)]   # x,y,z
        self.ori_err = [deque(maxlen=maxlen) for _ in range(3)]   # rx,ry,rz (rad)
        self.jerk    = deque(maxlen=maxlen)                        # ||jerk|| (m/s³)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def update(self, t, pos_err_3, ori_err_3, jerk_norm: float):
        with self.lock:
            self.t_data.append(t)
            for i in range(3):
                self.pos_err[i].append(pos_err_3[i])
                self.ori_err[i].append(ori_err_3[i])
            self.jerk.append(jerk_norm)

    def _loop(self):
        fig, (ax_p, ax_o, ax_j) = plt.subplots(3, 1, figsize=(11, 10))
        fig.suptitle("UR5e Acados NMPC — Real-time")

        ax_p.set_title("EE Position Error (m)")
        ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("Error (m)"); ax_p.grid(True)
        ax_o.set_title("EE Orientation Error (rad, axis-angle)")
        ax_o.set_xlabel("Time (s)"); ax_o.set_ylabel("Error (rad)"); ax_o.grid(True)
        ax_j.set_title("EE Jerk Magnitude (m/s³)")
        ax_j.set_xlabel("Time (s)"); ax_j.set_ylabel("|jerk| (m/s³)"); ax_j.grid(True)

        colors_xyz = ["r", "g", "b"]
        lines_pos = [ax_p.plot([], [], c, label=l, lw=1.5)[0]
                     for c, l in zip(colors_xyz, ["X", "Y", "Z"])]
        lines_ori = [ax_o.plot([], [], c, label=l, lw=1.5)[0]
                     for c, l in zip(colors_xyz, ["rx", "ry", "rz"])]
        line_jerk, = ax_j.plot([], [], "m", lw=1.5, label="|jerk|")
        ax_p.legend(ncol=3, fontsize=8)
        ax_o.legend(ncol=3, fontsize=8)
        ax_j.legend(fontsize=8)

        all_lines = lines_pos + lines_ori + [line_jerk]

        def _set_ylim(ax, datasets, margin=0.1, min_span=1e-6):
            """根据当前可见窗口内的数据自动设置纵轴范围。"""
            x0, x1 = ax.get_xlim()
            vals = []
            for xs, ys in datasets:
                for xi, yi in zip(xs, ys):
                    if x0 <= xi <= x1:
                        vals.append(yi)
            if not vals:
                return
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, min_span)
            ax.set_ylim(lo - margin * span, hi + margin * span)

        def animate(_):
            if not self.running:
                return all_lines
            with self.lock:
                if not self.t_data:
                    return all_lines
                t   = list(self.t_data)
                pe  = [list(d) for d in self.pos_err]
                oe  = [list(d) for d in self.ori_err]
                jk  = list(self.jerk)
            cur = t[-1]
            x0  = max(0, cur - window_width)
            for ax in (ax_p, ax_o, ax_j):
                ax.set_xlim(x0, cur + 0.5)
            for i, ln in enumerate(lines_pos):
                ln.set_data(t, pe[i])
            for i, ln in enumerate(lines_ori):
                ln.set_data(t, oe[i])
            line_jerk.set_data(t, jk)
            _set_ylim(ax_p, [(t, pe[i]) for i in range(3)])
            _set_ylim(ax_o, [(t, oe[i]) for i in range(3)])
            _set_ylim(ax_j, [(t, jk)])
            return all_lines

        anim = FuncAnimation(fig, animate, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory visualisation helper
# ═══════════════════════════════════════════════════════════════════════════════
def draw_trajectory(viewer, positions, color=(0, 1, 0, 1), width=0.002, clear=False):
    if clear:
        viewer.user_scn.ngeom = 0
    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width,
                             np.array(positions[i][:3]),
                             np.array(positions[i+1][:3]))
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

    joint_names   = ["shoulder_pan","shoulder_lift","elbow",
                     "wrist_1","wrist_2","wrist_3"]
    actuator_ids  = np.array([m.actuator(n).id for n in joint_names])
    
    # 夹爪致动器 ID
    try:
        gripper_actuator_id = m.actuator("robotiq_85_left_knuckle_joint").id
    except KeyError:
        gripper_actuator_id = None
        
    site_id       = m.site("attachment_site").id
    mocap_id      = m.body("target").mocapid[0]
    nq            = len(joint_names)

    # 初始关节角从 keyframe 读取，无需手工硬编码
    try:
        key_name = "csv" if USE_CSV_TRAJ else "home"
        key = m.key(key_name)
    except KeyError:
        key = m.key("home")
    
    key_id   = key.id
    q0 = key.qpos[:nq].copy()
    d.qpos[:] = key.qpos.copy()
    mujoco.mj_forward(m, d)

    # ── Build acados solver ───────────────────────────────────────────────────
    print("Building acados solver …")
    ctrl = UR5eNMPC(N=N, Tf=Tf, rebuild=REBUILD_SOLVER)
    print("Solver ready.\n")

    # ── TrajLoader & 轨迹源选择 ───────────────────────────────────────────────
    loader = TrajLoader(_CSV, pre_hold=4.0, post_hold=4.0)

    if USE_CSV_TRAJ:
        _get_pos = loader.get_pos
        _get_ori = loader.get_ori
    else:
        _get_pos = get_circle_target
        _get_ori = get_target_orientation

    # ── Reference queue (pre-filled at t=0) ──────────────────────────────────
    ref_queue = ReferenceQueue(N, dt, _get_pos, _get_ori)
    p_init = _get_pos(0.0)
    R_init = _get_ori(0.0)
    ref_queue.init(p_init, R_init)

    # ── 暖启动 solver：以 q0 填满整个预测域，避免初始化 NaN ─────────────────
    x0_warm = np.concatenate([q0, np.zeros(nq)])
    for k in range(N + 1):
        ctrl.solver.set(k, "x", x0_warm)
    for k in range(N):
        ctrl.solver.set(k, "u", np.zeros(nq))

    # ── Plotter ───────────────────────────────────────────────────────────────
    plotter = None
    if ENABLE_PLOT:
        plotter = RealtimePlotter()
        plotter.start()

    # ── State ─────────────────────────────────────────────────────────────────
    paused     = {"value": False}
    traj_start = time.time()
    traj_pts   = []   # actual EE  (MuJoCo frame)
    ref_pts    = []   # reference  (MuJoCo frame)
    max_pts    = 60

    # Jerk 估算：保存最近三帧的末端笛卡尔速度和对应时间戳
    # jerk ≈ (v[k] - 2*v[k-1] + v[k-2]) / dt²
    _ee_vel_buf = deque(maxlen=3)   # 每帧存 (t, ee_vel (3,))
    data_log    = []                # [(t, pos_norm, ori_norm, jerk_norm), ...]

    # NMPC 求解计时
    _solve_times = deque(maxlen=200)   # 最近200次求解耗时 (s)

    # ── Simulation loop ───────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model=m, data=d,
        show_left_ui=False, show_right_ui=False,
        key_callback=make_pause_toggle(paused),
    ) as viewer:
        # ── 重置到所选 keyframe（消除 IK 多解问题）
        mujoco.mj_resetDataKeyframe(m, d, key_id)
        mujoco.mj_forward(m, d)

        print("=== UR5e Acados NMPC ===")
        print("SPACE to pause/resume\n")
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            t_wall = time.time()

            if not paused["value"]:
                t_traj = t_wall - traj_start

                # Current state
                q_cur = d.qpos[:nq].copy()
                v_cur = d.qvel[:nq].copy()
                x0    = np.concatenate([q_cur, v_cur])

                # Rolling reference: push 1 new point, discard oldest
                ref_queue.step(t_traj)
                ref_pos, ref_rot = ref_queue.get()

                # Solve NMPC
                u_opt = ctrl.solve(x0, ref_pos, ref_rot)
                _solve_times.append(ctrl.time_tot)
                u_opt = np.clip(u_opt, -np.array([150,150,150,28,28,28]),
                                        np.array([150,150,150,28,28,28]))
                d.ctrl[actuator_ids] = u_opt

                # 夹爪随轨迹开闭 (与 vis_ur5e_gripper 保持一致)
                if gripper_actuator_id is not None:
                    w = 0.8 * np.pi
                    pos_min = 0.096
                    pos_max = 0.64
                    ratio = 0.5 * np.sin(5 * w * t_traj) + 0.5
                    gripper_ctrl = pos_min + (pos_max - pos_min) * ratio
                    d.ctrl[gripper_actuator_id] = gripper_ctrl

                # Visualise target pose via mocap body
                t0_pos = _get_pos(t_traj)
                t0_rot = _get_ori(t_traj)

                d.mocap_pos[mocap_id] = t0_pos
                q_mj = np.zeros(4)
                mujoco.mju_mat2Quat(q_mj, t0_rot.flatten())
                d.mocap_quat[mocap_id] = q_mj

                # Position & orientation error for plotter  (Pinocchio frame)
                ee_pin  = ctrl.f_fk_pos(q_cur).full().flatten()
                pos_err = ee_pin - ref_pos[:, 0]

                ee_rot_pin = ctrl.f_fk_rot(q_cur).full()          # (3,3)
                ref_rot_mat = ref_rot[:, 0].reshape(3, 3, order='F')  # col-major → 3×3
                R_err = ref_rot_mat.T @ ee_rot_pin
                tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1 + 1e-6, 1 - 1e-6)
                theta = np.arccos(tr)
                coeff = theta / (2.0 * np.sin(theta) + 1e-9)
                ori_err = coeff * np.array([R_err[2,1]-R_err[1,2],
                                            R_err[0,2]-R_err[2,0],
                                            R_err[1,0]-R_err[0,1]])

                if plotter:
                    # 末端笛卡尔线速度：J_transl(q) · qvel
                    jacp = np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, jacp, None, site_id)
                    ee_vel = jacp[:, :nq] @ v_cur   # (3,)
                    _ee_vel_buf.append((t_traj, ee_vel))

                    # 需要至少3帧才能算 jerk
                    if len(_ee_vel_buf) == 3:
                        t0, v0 = _ee_vel_buf[0]
                        t1, v1 = _ee_vel_buf[1]
                        t2, v2 = _ee_vel_buf[2]
                        dt0 = max(t1 - t0, 1e-6)
                        dt1 = max(t2 - t1, 1e-6)
                        acc0 = (v1 - v0) / dt0          # 加速度 (t0→t1)
                        acc1 = (v2 - v1) / dt1          # 加速度 (t1→t2)
                        jerk_vec  = (acc1 - acc0) / max(0.5*(dt0+dt1), 1e-6)
                        jerk_norm = float(np.linalg.norm(jerk_vec))
                    else:
                        jerk_norm = 0.0

                    plotter.update(t_traj, pos_err, ori_err, jerk_norm)
                    # 始终记录（不受绘图频率限制）
                    _pos_norm = float(np.linalg.norm(pos_err))
                    _ori_norm = float(np.linalg.norm(ori_err))
                    data_log.append((t_traj, _pos_norm, _ori_norm, jerk_norm))

                else:
                    # plotter 关闭时也计算并记录
                    jacp = np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, jacp, None, site_id)
                    ee_vel = jacp[:, :nq] @ v_cur
                    _ee_vel_buf.append((t_traj, ee_vel))
                    if len(_ee_vel_buf) == 3:
                        t0, v0 = _ee_vel_buf[0]
                        t1, v1 = _ee_vel_buf[1]
                        t2, v2 = _ee_vel_buf[2]
                        dt0 = max(t1 - t0, 1e-6)
                        dt1 = max(t2 - t1, 1e-6)
                        acc0 = (v1 - v0) / dt0
                        acc1 = (v2 - v1) / dt1
                        jerk_vec  = (acc1 - acc0) / max(0.5*(dt0+dt1), 1e-6)
                        jerk_norm = float(np.linalg.norm(jerk_vec))
                    else:
                        jerk_norm = 0.0
                    _pos_norm = float(np.linalg.norm(pos_err))
                    _ori_norm = float(np.linalg.norm(ori_err))
                    data_log.append((t_traj, _pos_norm, _ori_norm, jerk_norm))

                # Record trajectory points
                now_ee = d.site(site_id).xpos.copy()
                if not traj_pts or t_traj - traj_pts[-1][3] > 0.05:
                    traj_pts.append(np.append(now_ee, t_traj))
                    if len(traj_pts) > max_pts:
                        traj_pts.pop(0)
                if not ref_pts or t_traj - ref_pts[-1][3] > 0.05:
                    ref_pts.append(np.append(t0_pos, t_traj))
                    if len(ref_pts) > max_pts:
                        ref_pts.pop(0)

                # Print status
                if t_traj % 0.5 < dt:
                    err_norm = np.linalg.norm(pos_err)
                    jk_print = jerk_norm if len(_ee_vel_buf) == 3 else 0.0
                    if _solve_times:
                        _t_avg = sum(_solve_times) / len(_solve_times)
                        _t_max = max(_solve_times)
                        _freq  = 1.0 / _t_avg if _t_avg > 0 else float('inf')
                        solve_str = (f"  solve={_t_avg*1e3:.2f}ms"
                                     f"(max {_t_max*1e3:.2f}ms)"
                                     f"  {_freq:.1f}Hz"
                                     f"  qp={ctrl.time_qp*1e3:.2f}ms"
                                     f"  iter={ctrl.sqp_iter}")
                    else:
                        solve_str = ""
                    print(f"t={t_traj:6.2f}s  |pos_err|={err_norm*100:.1f} cm"
                          f"  |jerk|={jk_print:.2f} m/s³"
                          f"  τ_max={np.abs(u_opt).max():.1f} Nm"
                          + solve_str)

                mujoco.mj_step(m, d)

            # Draw trajectories
            draw_trajectory(viewer, ref_pts,  color=(0,1,0,1), clear=True)
            draw_trajectory(viewer, traj_pts, color=(1,0,0,1))

            viewer.sync()
            t_sleep = max(0, dt - (time.time() - t_wall))
            time.sleep(t_sleep)

    if plotter:
        plotter.stop()

    # ── 保存数据到 CSV ────────────────────────────────────────────────────────
    if data_log and SAVE_TRAJ:
        raw_stem = Path(FILE_NAME).stem
        if len(raw_stem) >= 8 and raw_stem[:8].isdigit():
            raw_stem = raw_stem[8:]
            if raw_stem.startswith("_"):
                raw_stem = raw_stem[1:]
        clean_stem = raw_stem if raw_stem else "traj"

        date_prefix = datetime.now().strftime("%Y%m%d")
        out_name = f"{date_prefix}_{clean_stem}_NMPC.csv"

        data_dir = _SCRIPT_DIR / "Data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / out_name

        arr = np.array(data_log)   # (M, 4)
        np.savetxt(out_path, arr, delimiter=",",
                   header="Time_s,Pos_err_norm_m,Ori_err_norm_rad,Jerk_norm_m_s3",
                   comments="")
        print(f"[data] 已保存 {len(data_log)} 条记录 → {out_path}")


if __name__ == "__main__":
    main()
