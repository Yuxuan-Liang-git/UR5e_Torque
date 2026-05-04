"""
UR5e Acados Model Export
Defines the continuous-time ODE  ẋ = f(x, u)  using Pinocchio + CasADi.

State  x  = [q (6), v (6)]  — joint positions & velocities   (12-dim)
Input  u  = τ  (6)          — joint torques                   (6-dim)

The dynamics are computed via the Articulated-Body Algorithm (ABA):
    v̇ = ABA(q, v, τ)   →   ẋ = [v, ABA(q,v,τ)]
"""

import sys
sys.path = [p for p in sys.path if "/opt/ros" not in p and ".local" not in p]

import argparse
from pathlib import Path
from acados_template import AcadosModel
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import yaml

_HERE = Path(__file__).parent.parent


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _HERE / path


def export_ur5e_model(mjcf_path: str | Path, frame_name: str = "attachment_site"):
    # ── Pinocchio model ────────────────────────────────────────────────────────
    mjcf_path = resolve_repo_path(mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(f"NMPC MJCF file not found: {mjcf_path}")

    pin_model = pin.buildModelFromMJCF(str(mjcf_path))
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
    ee_id = pin_model.getFrameId(frame_name)
    if ee_id >= pin_model.nframes:
        raise ValueError(f"Frame '{frame_name}' not found in NMPC MJCF: {mjcf_path}")
    ee_pos = cdata.oMf[ee_id].translation          # 3-vec  (Pinocchio frame)
    ee_rot = cdata.oMf[ee_id].rotation             # 3×3    (Pinocchio frame)

    f_fk_pos = ca.Function("f_fk_pos", [q], [ee_pos])
    f_fk_rot = ca.Function("f_fk_rot", [q], [ee_rot])

    return model, f_fk_pos, f_fk_rot, nq, nv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UR5e Acados model export")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    nmpc_cfg = cfg.get("nmpc_controller", {})

    model, f_pos, f_rot, nq, nv = export_ur5e_model(
        nmpc_cfg["mjcf_path"],
        nmpc_cfg.get("frame_name", "attachment_site"),
    )
    print(f"Model name : {model.name}")
    print(f"State dim  : {model.x.size()[0]}  (nq={nq}, nv={nv})")
    print(f"Input dim  : {model.u.size()[0]}")
    print("export_model_ur5e OK")
