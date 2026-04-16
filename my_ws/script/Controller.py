import numpy as np
import mujoco
# from nmpc_controller_ur5e import mj2pin_pos, mj2pin_rot
from acados_template import AcadosOcpSolver
import yaml

from export_model_ur5e import export_ur5e_model
import casadi as ca
from acados_template import AcadosOcp
import os
from pathlib import Path

# ── Coordinate-transform constant ─────────────────────────────────────────────
# Measured: R_pin = R_mj @ Rz90
_Rz90 = np.array([[0., -1., 0.],
                   [1.,  0., 0.],
                   [0.,  0., 1.]])

# MuJoCo position → Pinocchio position: [x,y,z]_mj → [y, -x, z]_pin
def mj2pin_pos(p_mj: np.ndarray) -> np.ndarray:
    return np.array([p_mj[1], -p_mj[0], p_mj[2]])

def mj2pin_rot(R_mj: np.ndarray) -> np.ndarray:
    return R_mj @ _Rz90

class BaseController:
    """控制器基类，提供统一的任务空间控制接口"""
    def __init__(self, model):
        self.model = model
        self.jac = np.zeros((6, model.nv))
        self.F_task = np.zeros(6)

    def compute_errors(self, target_pos, target_quat, current_pos, current_mat):
        """
        计算任务空间的位置误差和旋转误差
        target_pos: [3]
        target_quat: [4] (w, x, y, z)
        current_pos: [3]
        current_mat: [3x3] 旋转矩阵
        """
        # 1. 位置误差
        pos_err = target_pos - current_pos

        # 2. 旋转误差 (使用四元数计算轴角, 与目标四元数同半球)
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, current_mat.flatten())
        if np.dot(target_quat, curr_quat) < 0:
            curr_quat = -curr_quat

        quat_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_inv, curr_quat)

        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, target_quat, quat_inv)

        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
        
        return pos_err, rot_err

    def compute_torque(self, data, site_id, dx, drot, v_ee, v_des):
        """子类需实现此方法"""
        raise NotImplementedError("Subclasses must implement compute_torque")

class PDController(BaseController):
    """
    参考 decoupledPD.py 的任务空间解耦 PD 控制器。

    控制律:
        tau = J^T * (Kp_task * e6 - Kd_task * (J * dq))

    并按位置/姿态分别投影到关节空间后再相加。
    """
    def __init__(
        self,
        model,
        task_kp,
        task_kd,
        torque_limits=None,
    ):
        super().__init__(model)
        self.task_kp = np.asarray(task_kp, dtype=float)
        self.task_kd = np.asarray(task_kd, dtype=float)
        self.torque_limits = None if torque_limits is None else np.asarray(torque_limits, dtype=float)

    def compute_torque(self, data, site_id, target_pos, target_quat, v_ee, v_des):
        current_pos = data.site_xpos[site_id]
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        pos_err, rot_err = self.compute_errors(target_pos, target_quat, current_pos, current_mat)

        mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], site_id)
        J6 = self.jac[:, :6]

        e6 = np.concatenate([pos_err, rot_err])
        v6 = J6 @ np.asarray(data.qvel[:6], dtype=float)

        w_task = self.task_kp * e6 - self.task_kd * v6
        u_pos = J6.T @ np.concatenate([w_task[:3], np.zeros(3)])
        u_ori = J6.T @ np.concatenate([np.zeros(3), w_task[3:]])
        tau = u_pos + u_ori
        self.F_task = w_task

        if self.torque_limits is not None:
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)

        return np.asarray(tau).flatten()

class ImpedanceController(BaseController):
    """
    任务空间阻抗控制器 (考虑惯性匹配)
    公式: tau = J^T * [ Lambda * (a_ref) + mu + eta ]
    其中 Lambda 是任务空间质量阵: Lambda = (J * M^-1 * J^T)^-1
    a_ref = M_m^-1 * (K * err + D * v_err)
    """
    def __init__(self, model, stiffness, damping, inertia_m=None, vel_limits=None):
        super().__init__(model)
        self.stiffness = stiffness
        self.damping = damping
        self.vel_limits = vel_limits
        
        # 如果是 1D 数组，将其转为对角阵
        if inertia_m is not None:
            if inertia_m.ndim == 1:
                self.M_m = np.diag(inertia_m)
            else:
                self.M_m = inertia_m
        else:
            self.M_m = np.eye(6)
            
        self.M_m_inv = np.linalg.inv(self.M_m)
        
        # 内部缓存
        self.M = np.zeros((model.nv, model.nv))

    def compute_torque(self, data, site_id, target_pos, target_quat, v_ee, v_des):
        # 1. 运动学与误差
        current_pos = data.site_xpos[site_id]
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        pos_err, rot_err = self.compute_errors(target_pos, target_quat, current_pos, current_mat)
        err = np.concatenate([pos_err, rot_err])
        v_err = v_des - v_ee

        # 2. 获取雅可比 J 和 关节空间惯性阵 M
        mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], site_id)
        J = self.jac[:, :6]
        mujoco.mj_fullM(self.model, self.M, data.qM)
        M_joint = self.M[:6, :6]
        M_inv = np.linalg.inv(M_joint)

        # 3. 计算任务空间等效质量阵 Lambda = (J * M^-1 * J^T)^-1
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + np.eye(6) * 1e-6) # 正则化

        # print("[DEBUG] Task-Space Inertia Matrix (Lambda):")
        # with np.printoptions(precision=5, suppress=True, formatter={'float': '{: 8.5f}'.format}):
        #     print(Lambda)


        # 4. 阻抗行为 a_ref
        # 标准阻抗方程: M_m * dd_x + D * d_x + K * x = F_ext
        # 这里解算期望加速度 a_ref:
        a_ref = self.M_m_inv @ (self.stiffness @ err + self.damping @ v_err)

        # 5. 计算任务空间力 (这里由于 UR 自带重力补偿和摩擦补偿，我们主要关注惯性解耦)
        self.F_task = Lambda @ a_ref

        # 7. 映射回关节力矩
        tau = J.T @ self.F_task
        return np.asarray(tau).flatten()


class ImpedanceControllerLegacy(ImpedanceController):
    """兼容旧的阻抗扭矩实现，保留原有接口。"""


ImpedanceController = ImpedanceControllerLegacy


class PDJointController(BaseController):
    """关节空间 PD 控制器: tau = Kp*(q_des-q) + Kd*(dq_des-dq)."""

    def __init__(self, model, kp, kd, torque_limits=None):
        super().__init__(model)
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.torque_limits = None if torque_limits is None else np.array(torque_limits, dtype=float)

    def compute_torque(self, q_des, q, dq_des, dq):
        q_des = np.asarray(q_des, dtype=float)
        q = np.asarray(q, dtype=float)
        dq_des = np.asarray(dq_des, dtype=float)
        dq = np.asarray(dq, dtype=float)

        tau = self.kp * (q_des - q) + self.kd * (dq_des - dq)
        return np.asarray(tau).flatten()


class NMPCController(BaseController):
    """
    任务空间 NMPC 控制器，直接在类中初始化 Acados 求解器，参数从 yaml 读取。
    """
    def __init__(self, model, config_path, rebuild=False):
        super().__init__(model)
        
        # 1. 加载配置
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        nmpc_cfg = cfg.get("nmpc_controller")
        if nmpc_cfg is None:
            raise ValueError(f"Config file {config_path} missing 'nmpc_controller' section")
            
        self.N = nmpc_cfg["horizon_steps"]
        self.Tf = nmpc_cfg["horizon_time"]
        self.dt = self.Tf / self.N
        self.torque_limits = np.array(nmpc_cfg.get("torque_limits", [150, 150, 150, 28, 28, 28]))

        # 2. 导出模型并获取运动学函数
        acados_model, f_fk_pos, f_fk_rot, nq, nv = export_ur5e_model()
        self.nx = nq + nv
        self.nu = nq

        # 获取权重并计算对角矩阵
        w_q = nmpc_cfg["weight_q"]
        w_vel = nmpc_cfg["weight_vel"]
        w_tau = nmpc_cfg["weight_tau"]
        
        W_diag = np.array(list(w_q) + list(w_vel) + list(w_tau))
        W_diag = np.maximum(W_diag, 1e-6)
        terminal_q_scale = float(nmpc_cfg.get("terminal_q_scale", 10.0))
        W_e_diag = np.array(list(np.array(w_q, dtype=float) * terminal_q_scale))
        W_e_diag = np.maximum(W_e_diag, 1e-6)
        
        W_mat = np.diag(W_diag)
        W_e_mat = np.diag(W_e_diag)
        
        json_file = nmpc_cfg.get("json_file", "acados_ocp_ur5e.json")

        if rebuild:
            # 3. 构建 OCP 描述 (编译模式)
            x_sym = acados_model.x
            u_sym = acados_model.u
            q_sym = x_sym[:nq]
            v_sym = x_sym[nq:]

            # 参数 p: [ref_q(6), ref_dq(6)]
            p_sym = ca.SX.sym("p", 12)
            ref_q = p_sym[:6]
            ref_dq = p_sym[6:12]
            acados_model.p = p_sym

            # 直接计算关节空间的误差残差
            res_q = q_sym - ref_q
            res_v = v_sym - ref_dq
            
            res_stage = ca.vertcat(res_q, res_v, u_sym) # 6+6+6=18维
            res_terminal = ca.vertcat(res_q)           # 6维

            ocp = AcadosOcp()
            ocp.model = acados_model
            ocp.dims.N = self.N
            ocp.dims.np = 12

            ocp.cost.cost_type = "NONLINEAR_LS"
            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr = res_stage
            ocp.model.cost_y_expr_e = res_terminal

            ocp.cost.W = W_mat
            ocp.cost.W_e = W_e_mat
            ocp.cost.yref = np.zeros(18)
            ocp.cost.yref_e = np.zeros(6)
            ocp.parameter_values = np.zeros(12)

            # 约束和求解器选项
            ocp.constraints.lbu = -self.torque_limits
            ocp.constraints.ubu =  self.torque_limits
            ocp.constraints.idxbu = np.arange(self.nu)
            ocp.constraints.x0 = np.zeros(self.nx)

            s_cfg = nmpc_cfg.get("solver", {})
            ocp.solver_options.qp_solver = s_cfg.get("qp_solver", "PARTIAL_CONDENSING_HPIPM")
            ocp.solver_options.hessian_approx = s_cfg.get("hessian_approx", "GAUSS_NEWTON")
            ocp.solver_options.integrator_type = s_cfg.get("integrator_type", "ERK")
            ocp.solver_options.nlp_solver_type = s_cfg.get("nlp_solver_type", "SQP_RTI")
            ocp.solver_options.tf = self.Tf
            ocp.solver_options.qp_solver_iter_max = s_cfg.get("qp_solver_iter_max", 50)
            ocp.solver_options.levenberg_marquardt = s_cfg.get("levenberg_marquardt", 1e-2)
            
            if os.path.exists(json_file):
                os.remove(json_file)
            
            print("[INFO] Rebuilding Acados OCP solver...")
            self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        else:
            print("[INFO] Loading Acados OCP solver from existing build...")
            self.solver = AcadosOcpSolver(None, json_file=json_file, generate=False, build=False)

        # [关键] 无论是否 rebuild，都在线更新可变配置，从而保证修改 yaml 文件后无需重新编译也能生效！
        # 注意: 如果修改了步长 N, 求解时间 Tf, 或者底层求解器类型(qp_solver等)，由于模型维度/架构改变，必须设置 rebuild: true ！
        
        # 1. 权重参数在线更新
        for k in range(self.N):
            self.solver.cost_set(k, "W", W_mat)
        self.solver.cost_set(self.N, "W", W_e_mat)

        # 2. 控制器扭矩软/硬限幅约束 (lbu, ubu) 在线更新
        for k in range(self.N):
            self.solver.constraints_set(k, "lbu", -self.torque_limits)
            self.solver.constraints_set(k, "ubu", self.torque_limits)

        # 3. 求解器选项在线更新 (仅限 acados 允许在线设置的部分参数，如 levenberg_marquardt)
        s_cfg = nmpc_cfg.get("solver", {})
        if "levenberg_marquardt" in s_cfg:
            try:
                self.solver.options_set("levenberg_marquardt", float(s_cfg["levenberg_marquardt"]))
            except Exception:
                pass


    def compute_torque(self, data, q, dq, ref_q_batch, ref_dq_batch):
        """
        计算 NMPC 力矩。
        ref_q_batch: [6, N+1] 参考关节位置序列
        ref_dq_batch: [6, N+1] 参考关节速度序列
        """
        # 1. 状态
        x0 = np.concatenate([q, dq])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # 2. 设置参数 p 为关节空间参考序列
        for k in range(self.N + 1):
            p_k = np.concatenate([ref_q_batch[:, k], ref_dq_batch[:, k]])
            self.solver.set(k, "p", p_k)
            
            # 由于使用的是非线性 LS 且残差在 cost_y_expr 中减去了参考，这里 yref 设为 0
            ny = 18 if k < self.N else 6
            self.solver.set(k, "yref", np.zeros(ny))

        status = self.solver.solve()
        
        # 1. 提取第 0 步的最优前馈力矩
        u_nmpc = self.solver.get(0, "u")
        
        # 2. 提取第 1 步的预测期望状态
        x_next = self.solver.get(1, "x") 
        q_nmpc_des = x_next[:6]
        dq_nmpc_des = x_next[6:]
        
        # 3. 力矩限幅与重力扣除
        u_nmpc = np.clip(u_nmpc, -self.torque_limits, self.torque_limits)
        tau_nmpc_ff = u_nmpc - data.qfrc_bias[:6]
        
        # 返回前馈力矩、期望位置、期望速度
        return np.asarray(tau_nmpc_ff).flatten(), q_nmpc_des, dq_nmpc_des


class FrictionCompensator:
    """
    关节摩擦力补偿器，使用 Stribeck 模型。
    """
    def __init__(self, param_dir, enabled=True, comp_factor=0.25, vel_threshold=0.01):
        self.enabled = enabled
        self.comp_factor = comp_factor
        self.vel_threshold = vel_threshold
        self.param_dir = Path(param_dir)
        self.models = {}
        
        if self.enabled:
            for i in range(6):
                p_path = self.param_dir / f"joint_{i}_param.yaml"
                if p_path.exists():
                    with open(p_path, 'r') as f:
                        params = yaml.safe_load(f)
                    S = params.get("Stribeck", {})
                    self.models[i] = {
                        "Fc": S.get("Fc", 0.0),
                        "Fs": S.get("Fs", 0.0),
                        "vs": S.get("vs", 0.01),
                        "B": S.get("B", 0.0),
                        "bias": params.get("bias", 0.0)
                    }
                else:
                    print(f"[WARN] Friction parameters for joint {i} not found at {p_path}")

    def compute_torque(self, dq):
        """
        根据当前关节速度计算补偿力矩。
        dq: [6] 关节速度
        """
        if not self.enabled or not self.models:
            return np.zeros(6)
        
        tau_comp = np.zeros(6)
        dq = np.asarray(dq)
        for i, m in self.models.items():
            v = dq[i]
            # 平滑符号函数，避免零速附近的抖动
            if abs(v) > self.vel_threshold:
                f_sign = np.sign(v)
            else:
                f_sign = v / self.vel_threshold
            
            # Stribeck 摩擦力模型
            stribeck = m['Fc'] + (m['Fs'] - m['Fc']) * np.exp(-(v/m['vs'])**2)
            tau_fric = stribeck * f_sign + m['B'] * v
            
            # 综合补偿 (摩擦力 + 偏置) * 安全系数
            tau_comp[i] = self.comp_factor * (tau_fric + m['bias'])
            
        return tau_comp
