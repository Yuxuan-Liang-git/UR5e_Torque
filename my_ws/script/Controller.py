import numpy as np
import mujoco

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
        if self.torque_limits is not None:
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return np.asarray(tau).flatten()
