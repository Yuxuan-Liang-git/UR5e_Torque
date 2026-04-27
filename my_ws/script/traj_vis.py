#!/usr/bin/env python3
"""Visualize a recorded joint trajectory CSV in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_XML = Path("script/ur5e_gripper/scene.xml")
FALLBACK_XML = Path("script/universal_robots_ur5e/scene.xml")
DEFAULT_TRAJ = Path("config/trajectory/final_exec_160704.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CSV joint trajectory in MuJoCo")
    parser.add_argument("--csv", type=Path, default=DEFAULT_TRAJ, help="CSV trajectory file")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MuJoCo scene XML path")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop playback after the trajectory ends")
    parser.add_argument("--trail-step", type=int, default=8, help="Downsample step for precomputed EE trail")
    return parser.parse_args()


def load_joint_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV trajectory file not found: {csv_path}")

    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if table.size == 0:
        raise ValueError(f"CSV trajectory file is empty: {csv_path}")
    table = np.atleast_1d(table)

    names = table.dtype.names or ()
    required = ["timestamp", "real_q1", "real_q2", "real_q3", "real_q4", "real_q5", "real_q6"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    raw_time = np.asarray(table["timestamp"], dtype=float)
    raw_q = np.column_stack([np.asarray(table[f"real_q{i}"], dtype=float) for i in range(1, 7)])

    finite_mask = np.isfinite(raw_time) & np.all(np.isfinite(raw_q), axis=1)
    raw_time = raw_time[finite_mask]
    raw_q = raw_q[finite_mask]
    if raw_time.size < 2:
        raise ValueError(f"Need at least two valid samples in {csv_path}")

    order = np.argsort(raw_time)
    raw_time = raw_time[order]
    raw_q = raw_q[order]

    keep = np.concatenate(([True], np.diff(raw_time) > 0.0))
    raw_time = raw_time[keep]
    raw_q = raw_q[keep]
    if raw_time.size < 2:
        raise ValueError(f"Need at least two unique timestamps in {csv_path}")

    return raw_time - raw_time[0], raw_q


def sample_q(times: np.ndarray, q_data: np.ndarray, t: float) -> np.ndarray:
    tk = float(np.clip(t, 0.0, times[-1]))
    return np.array([np.interp(tk, times, q_data[:, j]) for j in range(6)], dtype=float)


def find_ee_site_id(model: mujoco.MjModel) -> int:
    for name in ("attachment_site", "eef_site", "tool0", "ee_site"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id != -1:
            return site_id
    raise ValueError("No end-effector site found. Tried: attachment_site, eef_site, tool0, ee_site")


def compute_ee_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    q_data: np.ndarray,
    step: int,
) -> np.ndarray:
    positions = []
    step = max(1, int(step))
    indices = list(range(0, q_data.shape[0], step))
    if indices[-1] != q_data.shape[0] - 1:
        indices.append(q_data.shape[0] - 1)

    for idx in indices:
        data.qpos[:6] = q_data[idx]
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        positions.append(data.site_xpos[site_id].copy())

    return np.asarray(positions, dtype=float)


def draw_trajectory(viewer, positions, color=(0.0, 1.0, 0.0, 1.0), width=0.003, clear=False):
    if clear:
        viewer.user_scn.ngeom = 0
    if len(positions) < 2:
        return

    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        from_pos = np.asarray(positions[i][:3], dtype=np.float64)
        to_pos = np.asarray(positions[i + 1][:3], dtype=np.float64)
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            from_pos,
            to_pos,
        )
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = color
        viewer.user_scn.ngeom += 1


def main() -> int:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be > 0")

    xml_path = args.xml
    if not xml_path.exists() and args.xml == DEFAULT_XML:
        xml_path = FALLBACK_XML
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    times, q_data = load_joint_csv(args.csv)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    path_data = mujoco.MjData(model)
    site_id = find_ee_site_id(model)
    full_ee_path = compute_ee_path(model, path_data, site_id, q_data, args.trail_step)

    data.qpos[:6] = q_data[0]
    mujoco.mj_forward(model, data)

    print(f"[INFO] Loaded model: {xml_path}")
    print(f"[INFO] Loaded CSV: {args.csv}")
    print(f"[INFO] Samples: {q_data.shape[0]}, duration: {times[-1]:.3f}s, speed: {args.speed:.3f}x")
    print("[INFO] Green line: full end-effector trajectory; red line: played trajectory.")

    played_points = []
    start_wall = time.perf_counter()
    last_sample_time = -1.0

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.opt.sitegroup[:] = True

        while viewer.is_running():
            elapsed = (time.perf_counter() - start_wall) * args.speed
            if args.loop and times[-1] > 0.0:
                t_traj = elapsed % times[-1]
                if t_traj < last_sample_time:
                    played_points.clear()
            else:
                t_traj = min(elapsed, times[-1])

            q = sample_q(times, q_data, t_traj)
            data.qpos[:6] = q
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)

            if t_traj == 0.0 or t_traj - last_sample_time >= 0.03 or not played_points:
                played_points.append(data.site_xpos[site_id].copy())
                last_sample_time = t_traj

            draw_trajectory(viewer, full_ee_path, color=(0.0, 0.8, 0.2, 0.55), width=0.002, clear=True)
            draw_trajectory(viewer, played_points, color=(1.0, 0.1, 0.05, 1.0), width=0.005)
            viewer.sync()

            if not args.loop and elapsed >= times[-1]:
                time.sleep(0.02)
            else:
                time.sleep(0.002)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
