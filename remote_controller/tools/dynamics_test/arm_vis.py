#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mockway arm visualization in MuJoCo.

Loads the scene defined in:
  mockway_description/urdf/scene.xml

Controls:
  - Space: pause/resume animation
  - R: reset to home pose
"""

from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer


HERE = Path(__file__).resolve().parent
SCENE_XML = (HERE / "../../mockway_description/urdf/scene.xml").resolve()


def make_key_callback(state, model, data, home_qpos):
    """Keyboard callback for pause/reset."""

    def key_callback(keycode):
        # Space
        if keycode == 32:
            state["paused"] = not state["paused"]
            print(f"\n>>> {'暂停' if state['paused'] else '继续'} <<<")
        # R/r
        elif keycode in (82, 114):
            data.qpos[:] = home_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print("\n>>> 已重置到 home 位姿 <<<")

    return key_callback


def draw_trajectory(viewer, points, color, width=0.003, clear=False):
    """Draw line trajectory in viewer user scene."""
    if clear:
        viewer.user_scn.ngeom = 0

    for i in range(len(points) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        a = np.asarray(points[i], dtype=np.float64)
        b = np.asarray(points[i + 1], dtype=np.float64)
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            a,
            b,
        )
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        geom.rgba[0] = color[0]
        geom.rgba[1] = color[1]
        geom.rgba[2] = color[2]
        geom.rgba[3] = color[3]
        viewer.user_scn.ngeom += 1


def find_joint_qpos_indices(model, max_count=6):
    """Collect qpos indices for hinge/slide joints (first max_count)."""
    indices = []
    for j in range(model.njnt):
        jnt_type = model.jnt_type[j]
        if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            indices.append(model.jnt_qposadr[j])
        if len(indices) >= max_count:
            break
    return np.array(indices, dtype=int)


def main():
    if not SCENE_XML.exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_XML}")

    model = mujoco.MjModel.from_xml_path(SCENE_XML.as_posix())
    data = mujoco.MjData(model)

    # Initialize to keyframe "home" if present.
    try:
        key_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except KeyError:
        mujoco.mj_resetData(model, data)

    qpos_indices = find_joint_qpos_indices(model, max_count=6)
    home_qpos = data.qpos.copy()

    # End effector tracking target (prefer site, fallback to body link6).
    ee_site_id = -1
    ee_body_id = -1
    try:
        ee_site_id = model.site("attachment_site").id
    except KeyError:
        try:
            ee_body_id = model.body("link6").id
        except KeyError:
            ee_body_id = model.nbody - 1

    # Target mocap id (optional).
    mocap_id = -1
    try:
        mocap_id = model.body("target").mocapid[0]
    except KeyError:
        pass

    # Animation params
    dt = 0.01
    model.opt.timestep = dt
    amplitude = np.array([0.4, 0.5, 0.4, 0.6, 0.6, 0.8])
    frequency = np.array([0.15, 0.18, 0.21, 0.25, 0.28, 0.33])  # Hz
    max_traj_points = 200
    ee_traj = []
    target_traj = []

    state = {"paused": False}
    key_callback = make_key_callback(state, model, data, home_qpos)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 110
        viewer.cam.elevation = -25
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            if not state["paused"]:
                t = data.time

                # Move target mocap in a smooth 3D path if target exists.
                if mocap_id >= 0:
                    center = np.array([0.35, 0.0, 0.35])
                    target = center.copy()
                    target[0] += 0.12 * np.sin(0.8 * np.pi * t)
                    target[1] += 0.08 * np.sin(1.6 * np.pi * t)
                    target[2] += 0.08 * np.cos(0.8 * np.pi * t)
                    data.mocap_pos[mocap_id] = target

                # Joint-space animation around home pose.
                n = min(len(qpos_indices), len(amplitude))
                q = home_qpos[qpos_indices[:n]] + amplitude[:n] * np.sin(
                    2.0 * np.pi * frequency[:n] * t
                )
                data.qpos[qpos_indices[:n]] = q
                data.qvel[:] = 0.0

                # Update kinematics without dynamics integration.
                mujoco.mj_forward(model, data)
                data.time += dt

                # Record trajectories
                if ee_site_id >= 0:
                    ee_pos = data.site(ee_site_id).xpos.copy()
                else:
                    ee_pos = data.xpos[ee_body_id].copy()
                ee_traj.append(ee_pos)
                if len(ee_traj) > max_traj_points:
                    ee_traj.pop(0)

                if mocap_id >= 0:
                    target_traj.append(data.mocap_pos[mocap_id].copy())
                    if len(target_traj) > max_traj_points:
                        target_traj.pop(0)

            # Draw target and ee trajectories.
            draw_trajectory(viewer, target_traj, color=[0.0, 1.0, 0.0, 1.0], clear=True)
            draw_trajectory(viewer, ee_traj, color=[1.0, 0.0, 0.0, 1.0], clear=False)

            mujoco.mj_camlight(model, data)
            viewer.sync()

            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

