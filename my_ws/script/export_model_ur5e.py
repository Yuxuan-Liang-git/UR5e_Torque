"""
UR5e Acados Model Export
Defines the continuous-time ODE  ẋ = f(x, u)  using Pinocchio + CasADi.

State  x  = [q (6), v (6)]  — joint positions & velocities   (12-dim)
Input  u  = τ  (6)          — joint torques                   (6-dim)

The dynamics are computed via the Articulated-Body Algorithm (ABA):
    v̇ = ABA(q, v, τ)   →   ẋ = [v, ABA(q,v,τ)]
"""

import sys
from pathlib import Path
from acados_template import AcadosModel
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

_HERE = Path(__file__).parent
_MJCF = _HERE / "universal_robots_ur5e" / "ur5e.xml"


def export_ur5e_model():
    # ── Pinocchio model ────────────────────────────────────────────────────────
    pin_model = pin.buildModelFromMJCF(str(_MJCF))
    cmodel    = cpin.Model(pin_model)
    cdata     = cmodel.createData()

    nq = pin_model.nq   # 6
    nv = pin_model.nv   # 6
    nx = nq + nv        # 12
    nu = nq             # 6

    # ── CasADi symbolic state & input ─────────────────────────────────────────
    q   = ca.SX.sym("q",   nq)
    v   = ca.SX.sym("v",   nv)
    tau = ca.SX.sym("tau", nu)

    # State and state-derivative vectors
    x    = ca.vertcat(q, v)
    xdot = ca.SX.sym("xdot", nx)

    # Articulated-Body Algorithm: v̇ = ABA(q, v, τ)
    vdot = cpin.aba(cmodel, cdata, q, v, tau)
    f_expl = ca.vertcat(v, vdot)       # explicit ODE  ẋ = f(x,u)
    f_impl = xdot - f_expl             # implicit form (required by acados)

    # ── Acados model ──────────────────────────────────────────────────────────
    model = AcadosModel()
    model.name   = "ur5e_aba"
    model.x      = x
    model.xdot   = xdot
    model.u      = tau
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    # ── Forward kinematics functions (used externally for cost / plotting) ────
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    ee_id = pin_model.getFrameId("attachment_site")
    ee_pos = cdata.oMf[ee_id].translation          # 3-vec  (Pinocchio frame)
    ee_rot = cdata.oMf[ee_id].rotation             # 3×3    (Pinocchio frame)

    f_fk_pos = ca.Function("f_fk_pos", [q], [ee_pos])
    f_fk_rot = ca.Function("f_fk_rot", [q], [ee_rot])

    return model, f_fk_pos, f_fk_rot, nq, nv


if __name__ == "__main__":
    model, f_pos, f_rot, nq, nv = export_ur5e_model()
    print(f"Model name : {model.name}")
    print(f"State dim  : {model.x.size()[0]}  (nq={nq}, nv={nv})")
    print(f"Input dim  : {model.u.size()[0]}")
    print("export_model_ur5e OK")
