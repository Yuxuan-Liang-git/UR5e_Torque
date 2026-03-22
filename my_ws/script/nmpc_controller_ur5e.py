"""
UR5e Acados NMPC Controller

Cost structure  (LINEAR_LS, per stage):
    y_k = [ee_pos(q_k) — ref_pos_k,   # 3   position error in Pinocchio frame
           rot_err_k,                  # 3   SO3 log-map error  (axis-angle vec)
           v_k,                        # 6   joint velocity
           τ_k]                        # 6   joint torque
    ny = 3 + 3 + 6 + 6 = 18

    y_e = [ee_pos(q_N) — ref_pos_N,   # 3
           rot_err_N]                  # 3
    ny_e = 6

Because the position / orientation residuals are nonlinear in q, we use
cost type = NONLINEAR_LS and provide the residual expression via
  ocp.cost.cost_expr_ext_cost  →  0.5 * r^T W r
"""

import sys
from pathlib import Path
import numpy as np
import casadi as ca
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
from export_model_ur5e import export_ur5e_model

_HERE = Path(__file__).parent

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


class UR5eNMPC:
    """
    Acados NMPC for UR5e task-space tracking.

    Parameters
    ----------
    N        : prediction horizon (steps)
    Tf       : prediction horizon (seconds)  →  dt = Tf / N
    json_file: where acados writes its generated C code manifest
    """

    def __init__(self, N: int = 30, Tf: float = 0.3,
                 json_file: str = "acados_ocp_ur5e.json",
                 rebuild: bool = True):

        self.N  = N
        self.Tf = Tf
        self.dt = Tf / N
        self.ny   = 18
        self.ny_e = 6
        json_path = str(_HERE / json_file)

        # ── 始终需要：加载 Pinocchio FK 函数供 solve() 使用 ──────────────────
        _, f_fk_pos, f_fk_rot, nq, nv = export_ur5e_model()
        self.nq       = nq
        self.nv       = nv
        self.nx       = nq + nv   # 12
        self.nu       = nq        # 6
        self.f_fk_pos = f_fk_pos
        self.f_fk_rot = f_fk_rot

        if rebuild:
            # ── 重新构建 OCP 并生成/编译 C 代码 ──────────────────────────────
            import shutil, os
            acados_model, _, _, _, _ = export_ur5e_model()

            x_sym = acados_model.x
            u_sym = acados_model.u
            q_sym = x_sym[:nq]
            v_sym = x_sym[nq:]

            p_sym   = ca.SX.sym("p", 12)
            ref_pos = p_sym[:3]
            ref_rot = ca.reshape(p_sym[3:12], 3, 3)
            acados_model.p = p_sym

            ee_pos  = f_fk_pos(q_sym)
            res_pos = ee_pos - ref_pos

            ee_rot  = f_fk_rot(q_sym)
            R_err   = ca.mtimes(ref_rot.T, ee_rot)
            tr      = R_err[0,0] + R_err[1,1] + R_err[2,2]
            cos_th  = ca.fmin(ca.fmax((tr - 1.0) / 2.0, -1.0 + 1e-6), 1.0 - 1e-6)
            theta   = ca.acos(cos_th)
            coeff   = theta / (2.0 * ca.sin(theta) + 1e-9)
            res_ori = coeff * ca.vertcat(R_err[2,1] - R_err[1,2],
                                         R_err[0,2] - R_err[2,0],
                                         R_err[1,0] - R_err[0,1])
            res_vel = v_sym
            res_tau = u_sym

            res_stage    = ca.vertcat(res_pos, res_ori, res_vel, res_tau)
            res_terminal = ca.vertcat(res_pos, res_ori)

            ocp = AcadosOcp()
            ocp.model = acados_model
            ocp.dims.N  = N
            ocp.dims.np = 12

            ocp.cost.cost_type   = "NONLINEAR_LS"
            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr   = res_stage
            ocp.model.cost_y_expr_e = res_terminal

            # w_pos = 500.0; w_ori = 100.0; w_vel = 0.1
            # w_tau_large = 0.001; w_tau_small = 0.01
            w_pos = 2000.0; w_ori = 500.0; w_vel = 0.1
            w_tau_large = 0.001; w_tau_small = 0.01

            W_diag   = np.array([w_pos]*3 + [w_ori]*3 + [w_vel]*6 +
                                 [w_tau_large]*3 + [w_tau_small]*3)
            W_e_diag = np.array([w_pos*20]*3 + [w_ori*20]*3)
            ocp.cost.W   = np.diag(W_diag)
            ocp.cost.W_e = np.diag(W_e_diag)
            ocp.cost.yref   = np.zeros(self.ny)
            ocp.cost.yref_e = np.zeros(self.ny_e)
            ocp.parameter_values = np.zeros(12)

            tau_max = np.array([150., 150., 150., 28., 28., 28.])
            ocp.constraints.lbu   = -tau_max
            ocp.constraints.ubu   =  tau_max
            ocp.constraints.idxbu = np.arange(self.nu)
            ocp.constraints.x0    = np.zeros(self.nx)

            ocp.solver_options.qp_solver               = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx          = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type         = "ERK"
            ocp.solver_options.nlp_solver_type         = "SQP_RTI"
            ocp.solver_options.print_level             = 0
            ocp.solver_options.tf                      = Tf
            ocp.solver_options.sim_method_num_stages   = 4
            ocp.solver_options.sim_method_num_steps    = 1
            ocp.solver_options.qp_solver_iter_max      = 50
            ocp.solver_options.regularize_method       = "MIRROR"

            # 删除旧产物，确保从零生成
            if os.path.exists(json_path):
                os.remove(json_path)
            gen_dir = _HERE / "c_generated_code"
            if gen_dir.exists():
                shutil.rmtree(gen_dir)

            print("[UR5eNMPC] regenerating C code and compiling …")
            self.solver = AcadosOcpSolver(ocp, json_file=json_path)
            print(f"[UR5eNMPC] solver built (C code regenerated)  N={N}  Tf={Tf}s  dt={self.dt*1000:.1f}ms")

        else:
            # ── 直接加载已编译的共享库，完全跳过 OCP / CasADi 构建 ────────────
            # ocp=None + generate=False + build=False → 只 dlopen 已有 .so
            self.solver = AcadosOcpSolver(None, json_file=json_path,
                                          generate=False, build=False, verbose=False)
            print(f"[UR5eNMPC] solver loaded from existing build  N={N}  Tf={Tf}s  dt={self.dt*1000:.1f}ms")

    # ──────────────────────────────────────────────────────────────────────────
    def solve(self,
              x0:       np.ndarray,
              ref_pos:  np.ndarray,   # [3, N+1]  Pinocchio frame
              ref_rot:  np.ndarray,   # [3,3, N+1] or [9, N+1] col-major
             ) -> np.ndarray:
        """
        Run one SQP-RTI step.

        Parameters
        ----------
        x0      : current state [q(6), v(6)]
        ref_pos : reference positions,   shape (3, N+1),  Pinocchio frame
        ref_rot : reference rotations,   shape (9, N+1),  column-major flattened

        Returns
        -------
        u0 : optimal torques for the current step  (6,)
        """
        N = self.N

        # Set initial state constraint
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set rolling-horizon reference and parameters stage by stage
        for k in range(N):
            # Parameter: [ref_pos(3), ref_rot_colmaj(9)]
            p_k = np.concatenate([ref_pos[:, k], ref_rot[:, k]])
            self.solver.set(k, "p", p_k)

            # yref = zeros: residuals are built into the nonlinear expression
            # (residual = 0 means "match the reference encoded in p")
            self.solver.set(k, "yref", np.zeros(self.ny))

        # Terminal stage
        p_N = np.concatenate([ref_pos[:, N], ref_rot[:, N]])
        self.solver.set(N, "p", p_N)
        self.solver.set(N, "yref", np.zeros(self.ny_e))

        # Solve
        status = self.solver.solve()
        if status not in (0, 2):   # 0=success, 2=max-iter (acceptable)
            print(f"[UR5eNMPC] solver status {status}")

        # ── acados 内置统计（可直接读取）───────────────────────────────
        # time_tot : 整个求解周期总耗时 (s)
        # time_qp  : QP 子问题耗时 (s)
        # sqp_iter : 本次 SQP 迭代次数
        self.time_tot  = float(self.solver.get_stats("time_tot"))
        self.time_qp   = float(self.solver.get_stats("time_qp"))
        self.sqp_iter  = int(self.solver.get_stats("sqp_iter"))

        return self.solver.get(0, "u")


if __name__ == "__main__":
    import numpy as np
    ctrl = UR5eNMPC(N=30, Tf=0.3, rebuild=False)
    q0   = np.array([0., -1.5708, 1.5708, -1.5708, -1.5708, 0.])
    x0   = np.concatenate([q0, np.zeros(6)])
    ref_p = np.tile(np.array([0.1, 0.4, 0.49]), (31, 1)).T   # (3, N+1)
    R_id  = np.eye(3).flatten(order='F')
    ref_r = np.tile(R_id, (31, 1)).T                          # (9, N+1)
    u = ctrl.solve(x0, ref_p, ref_r)
    print("u0 =", u)
