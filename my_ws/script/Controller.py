import numpy as np
import mujoco
from nmpc_controller_ur5e import mj2pin_pos, mj2pin_rot
from acados_template import AcadosOcpSolver
import yaml

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

        # 2. 旋转误差 (使用四元数计算轴角)
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, current_mat.flatten())
        
        quat_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_inv, curr_quat)
        
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, target_quat, quat_inv)
        
        # 3. 确保四元数在同一半球 (shortest path)
        if quat_err[0] < 0:
            quat_err = -quat_err

        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
        
        return pos_err, rot_err

    def compute_torque(self, data, site_id, dx, drot, v_ee, v_des):
        """子类需实现此方法"""
        raise NotImplementedError("Subclasses must implement compute_torque")

class PDController(BaseController):
    """
    任务空间 PD 控制器
    公式: F_task = Kp * err + Kd * v_err
    """
    def __init__(self, model, stiffness, damping, vel_limits=None):
        super().__init__(model)
        self.stiffness = stiffness
        self.damping = damping
        self.vel_limits = vel_limits

    def compute_torque(self, data, site_id, target_pos, target_quat, v_ee, v_des):
        """
        计算 PD 控制力矩
        """
        # 1. 获取当前状态并计算误差
        current_pos = data.site_xpos[site_id]
        current_mat = data.site_xmat[site_id].reshape(3, 3)
        pos_err, rot_err = self.compute_errors(target_pos, target_quat, current_pos, current_mat)

        # 2. 计算雅可比矩阵
        mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], site_id)
        J = self.jac[:, :6] 

        # 3. 计算基本控制力
        err = np.concatenate([pos_err, rot_err])
        v_err = v_des - v_ee
        self.F_task = self.stiffness @ err + self.damping @ v_err

        # 4. 速度限制逻辑
        if self.vel_limits is not None:
            v_trans = v_ee[:3]
            v_max_trans = np.min(self.vel_limits[:3])
            speed_trans = np.linalg.norm(v_trans)
            if speed_trans > v_max_trans:
                penalty_k = 200.0
                self.F_task[:3] -= penalty_k * (speed_trans - v_max_trans) * (v_trans / speed_trans)

            v_rot = v_ee[3:]
            v_max_rot = np.min(self.vel_limits[3:])
            speed_rot = np.linalg.norm(v_rot)
            if speed_rot > v_max_rot:
                penalty_k_rot = 50.0
                self.F_task[3:] -= penalty_k_rot * (speed_rot - v_max_rot) * (v_rot / speed_rot)

        # 5. 投影到关节空间
        tau = J.T @ self.F_task
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
        # 注意: 实际计算中常用 Lambda * J * M^-1 来实现，或者直接求逆
        Lambda_inv = J @ M_inv @ J.T
        Lambda = np.linalg.inv(Lambda_inv + np.eye(6) * 1e-6) # 正则化

        # 4. 阻抗行为 a_ref
        # 标准阻抗方程: M_m * dd_x + D * d_x + K * x = F_ext
        # 这里解算期望加速度 a_ref:
        a_ref = self.M_m_inv @ (self.stiffness @ err + self.damping @ v_err)

        # 5. 计算任务空间力 (这里由于 UR 自带重力补偿和摩擦补偿，我们主要关注惯性解耦)
        self.F_task = Lambda @ a_ref

        # # 6. 速度限制惩罚 (同 PD 控制器)
        # if self.vel_limits is not None:
        #     v_trans = v_ee[:3]
        #     v_max_trans = np.min(self.vel_limits[:3])
        #     speed_trans = np.linalg.norm(v_trans)
        #     if speed_trans > v_max_trans:
        #         print(f"[WARN] Translational velocity limit exceeded: {speed_trans:.2f} > {v_max_trans:.2f}")
        #         self.F_task[:3] -= 200.0 * (speed_trans - v_max_trans) * (v_trans / speed_trans)
            
        #     v_rot = v_ee[3:]
        #     v_max_rot = np.min(self.vel_limits[3:])
        #     speed_rot = np.linalg.norm(v_rot)
        #     if speed_rot > v_max_rot:
        #         print(f"[WARN] Rotational velocity limit exceeded: {speed_rot:.2f} > {v_max_rot:.2f}")
        #         self.F_task[3:] -= 50.0 * (speed_rot - v_max_rot) * (v_rot / speed_rot)

        # 7. 映射回关节力矩
        tau = J.T @ self.F_task
        return np.asarray(tau).flatten()


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
        from export_model_ur5e import export_ur5e_model
        import casadi as ca
        from acados_template import AcadosOcp
        import os
        
        acados_model, f_fk_pos, f_fk_rot, nq, nv = export_ur5e_model()
        self.nx = nq + nv
        self.nu = nq

        # 获取权重并计算对角矩阵
        w_pos = nmpc_cfg["weight_pos"]
        w_ori = nmpc_cfg["weight_ori"]
        w_vel = nmpc_cfg["weight_vel"]
        w_tau = nmpc_cfg["weight_tau"]
        
        W_diag = np.array(list(w_pos) + list(w_ori) + [w_vel] * 6 + list(w_tau))
        W_diag = np.maximum(W_diag, 1e-6)
        terminal_pos_scale = float(nmpc_cfg.get("terminal_pos_scale", 10.0))
        terminal_ori_scale = float(nmpc_cfg.get("terminal_ori_scale", 10.0))
        W_e_diag = np.array(list(np.array(w_pos, dtype=float) * terminal_pos_scale) +
                    list(np.array(w_ori, dtype=float) * terminal_ori_scale))
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

            # 参数 p: [ref_pos(3), ref_rot(9)]
            p_sym = ca.SX.sym("p", 12)
            ref_pos = p_sym[:3]
            ref_rot = ca.reshape(p_sym[3:12], 3, 3)
            acados_model.p = p_sym

            # 计算残差 (NONLINEAR_LS)
            ee_pos = f_fk_pos(q_sym)
            res_pos = ee_pos - ref_pos

            ee_rot = f_fk_rot(q_sym)
            R_err = ca.mtimes(ref_rot.T, ee_rot)
            tr = R_err[0,0] + R_err[1,1] + R_err[2,2]
            cos_th = ca.fmin(ca.fmax((tr - 1.0) / 2.0, -1.0 + 1e-6), 1.0 - 1e-6)
            theta = ca.acos(cos_th)
            coeff = theta / (2.0 * ca.sin(theta) + 1e-9)
            res_ori = coeff * ca.vertcat(R_err[2,1] - R_err[1,2],
                                         R_err[0,2] - R_err[2,0],
                                         R_err[1,0] - R_err[0,1])
            
            res_stage = ca.vertcat(res_pos, res_ori, v_sym, u_sym) # 3+3+6+6=18
            res_terminal = ca.vertcat(res_pos, res_ori)           # 6

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


    def compute_torque(self, data, q, dq, ref_pos_batch, ref_rot_batch):
        """
        计算 NMPC 力矩。
        ref_pos_batch: [3, N+1] 参考位置序列 (MuJoCo 坐标系)
        ref_rot_batch: [9, N+1] 参考旋转矩阵序列 (MuJoCo 坐标系)
        """
        # 1. 状态
        x0 = np.concatenate([q, dq])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # 2. 设置参数 p 而非 yref (同步 nmpc_controller_ur5e.py 的逻辑)
        for k in range(self.N + 1):
            p_pin = mj2pin_pos(ref_pos_batch[:, k])
            R_mj = ref_rot_batch[:, k].reshape(3, 3, order='F')
            R_pin = mj2pin_rot(R_mj)
            
            p_k = np.concatenate([p_pin, R_pin.flatten(order='F')])
            self.solver.set(k, "p", p_k)
            
            # 由于使用的是非线性 LS 且残差在 cost_y_expr 中减去了参考，这里 yref 设为 0
            ny = 18 if k < self.N else 6
            self.solver.set(k, "yref", np.zeros(ny))

        # 3. 求解
        status = self.solver.solve()
        u_nmpc = self.solver.get(0, "u")
        
        # 4. 限幅与重力补偿
        u_nmpc = np.clip(u_nmpc, -self.torque_limits, self.torque_limits)
        tau_nmpc = u_nmpc - data.qfrc_bias[:6]
        
        return np.asarray(tau_nmpc).flatten()
