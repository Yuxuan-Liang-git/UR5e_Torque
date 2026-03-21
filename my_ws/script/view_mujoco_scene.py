#!/usr/bin/env python3
"""Load and visualize a MuJoCo XML scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

DEFAULT_XML = Path("/home/amdt/ur_force_ws/my_ws/script/ur5e_gripper/scene.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo XML scene")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to MuJoCo XML scene")
    parser.add_argument("--no-viewer", action="store_true", help="Only test parsing without opening viewer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()

    if not xml_path.exists():
        print(f"[ERROR] XML not found: {xml_path}")
        return 2

    try:
        model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        data = mujoco.MjData(model)
    except Exception as exc:
        print(f"[ERROR] Failed to load XML: {exc}")
        return 3

    print(f"[OK] Loaded: {xml_path}")
    print(f"[OK] njnt={model.njnt}, nbody={model.nbody}, nq={model.nq}, nv={model.nv}")

    if args.no_viewer:
        return 0

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        print("[INFO] Viewer started. Close the window or Ctrl+C to exit.")
        while viewer.is_running():
            step_t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_t0)
            if dt > 0:
                time.sleep(dt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
