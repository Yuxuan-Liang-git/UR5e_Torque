#!/usr/bin/env python3
"""Task space impedance control for UR5e using ur-rtde and MuJoCo for kinematics.

Formula: tau = J^T * (K_p * (x_des - x) + K_d * (v_des - v)) + gravity_comp
Note: UR direct torque mode expects torques AFTER internal gravity compensation.
So we only send J^T * F_task if we want the robot to handle gravity.
"""

import argparse
import time
import numpy as np
import yaml
from pathlib import Path
import mujoco
import mujoco.viewer
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def draw_trajectory(viewer, positions, color=[0, 1, 0, 1], width=0.002, clear=False):
    """Render trajectory segments in MuJoCo viewer."""
    if clear:
        viewer.user_scn.ngeom = 0

    for i in range(len(positions) - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        from_pos = np.array(positions[i][:3], dtype=np.float64)
        to_pos = np.array(positions[i + 1][:3], dtype=np.float64)
        
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            from_pos,
            to_pos
        )
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[0:4] = color
        viewer.user_scn.ngeom += 1

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Task Space Impedance Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--config", default="config/ctrl_config.yaml", help="Control config file")
    parser.add_argument("--sys-config", default="config/sys_config.yaml", help="System config file")
    return parser.parse_args()

def rotation_error(R_d, R):
    """Compute the orientation error between two rotation matrices using quaternions."""
    # Convert rotation matrices to quaternions
    q_d = np.zeros(4)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q_d, R_d.flatten())
    mujoco.mju_mat2Quat(q, R.flatten())
    
    # Compute conjugate of current quaternion
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q)
    
    # Compute error quaternion: q_err = q_d * q_inv
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_d, q_inv)
    
    # Convert error quaternion to 3D velocity/error vector
    res = np.zeros(3)
    mujoco.mju_quat2Vel(res, q_err, 1.0)
    return res

def main():
    args = parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        ctrl_cfg = yaml.safe_load(f)
    with open(args.sys_config, 'r') as f:
        sys_cfg = yaml.safe_load(f)

    # Impedance parameters
    stiffness = np.array(ctrl_cfg['impedance_controller']['stiffness'])
    damping_ratio = np.array(ctrl_cfg['impedance_controller']['damping_ratio'])
    damping = damping_ratio * 2 * np.sqrt(stiffness) 

    # Trajectory parameters
    circle_radius = ctrl_cfg['trajectory']['circle_radius']
    circle_omega = ctrl_cfg['trajectory']['circle_omega']
    dt = ctrl_cfg['trajectory']['control_dt']

    torque_limits = np.array(ctrl_cfg['safety']['torque_limits'])

    # Load MuJoCo model for kinematics/Jacobian
    xml_path = Path("script/universal_robots_ur5e/scene_torque.xml")
    if not xml_path.exists():
        # Fallback to scene.xml if unique torque one doesn't exist
        xml_path = Path("script/universal_robots_ur5e/scene.xml")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        # try another common name
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    # Connect to robot
    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # Initial state
    q = rtde_r.getActualQ()
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    
    # Set target as current pose
    x_des_pos_init = data.site_xpos[ee_site_id].copy()
    x_des_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    # Trajectory visualization containers
    trajectory_points = []  # Actual (red)
    target_trajectory = []  # Target (green)
    max_points = 200
    start_time = None

    print(f"[INFO] Initial EE pos: {x_des_pos_init}")
    print("[INFO] Starting impedance control with circular trajectory. Press Ctrl+C to stop.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25

            if start_time is None:
                start_time = time.time()

            while viewer.is_running():
                t_now = time.time() - start_time
                t_start = rtde_c.initPeriod()
                
                # 1. Generate circular trajectory in XY plane
                # x = x0 + R * cos(wt), y = y0 + R * sin(wt)
                x_des_pos = x_des_pos_init.copy()
                x_des_pos[0] += circle_radius * (np.cos(circle_omega * t_now) - 1.0)
                x_des_pos[1] += circle_radius * np.sin(circle_omega * t_now)
                
                v_des_pos = np.array([
                    -circle_radius * circle_omega * np.sin(circle_omega * t_now),
                     circle_radius * circle_omega * np.cos(circle_omega * t_now),
                     0.0
                ])

                # 2. Get current state
                q = rtde_r.getActualQ()
                dq = rtde_r.getActualQd()
                
                # 3. Update MuJoCo model
                data.qpos[:6] = q
                data.qvel[:6] = dq
                mujoco.mj_forward(model, data)
                
                # 4. Compute Jacobian
                # jacp: 3xNV, jacr: 3xNV
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
                J = np.vstack([jacp[:, :6], jacr[:, :6]]) # 6x6 for UR5e
                
                # 5. Compute EE velocity
                v_ee = J @ dq
                
                # 6. Compute pose error
                x_curr_pos = data.site_xpos[ee_site_id]
                x_curr_rot = data.site_xmat[ee_site_id].reshape(3, 3)
                
                pos_err = x_des_pos - x_curr_pos
                rot_err = rotation_error(x_des_rot, x_curr_rot)
                err = np.concatenate([pos_err, rot_err])
                
                # Extended velocity error (assuming no orientation velocity for now)
                v_err = np.concatenate([v_des_pos, np.zeros(3)]) - v_ee

                # 7. Task space force
                # F = K * err + D * v_err
                F_task = stiffness * err + damping * v_err
                
                # 8. Map to joint torques
                tau = J.T @ F_task
                
                # 9. Limit torques for safety
                tau = np.clip(tau, -torque_limits, torque_limits)
                
                # 10. Send to robot (Robot handles gravity compensation)
                ok = rtde_c.directTorque(tau.tolist(), True)
                if not ok:
                    print("[ERROR] directTorque failed")
                    break
                    
                # 11. Update Trajectory Visualization
                if not trajectory_points or (t_now % 0.05 < 0.002):
                    trajectory_points.append(x_curr_pos.copy())
                    target_trajectory.append(x_des_pos.copy())
                    if len(trajectory_points) > max_points:
                        trajectory_points.pop(0)
                        target_trajectory.pop(0)

                draw_trajectory(viewer, target_trajectory, color=[0, 1, 0, 1], clear=True, width=0.003)
                draw_trajectory(viewer, trajectory_points, color=[1, 0, 0, 1], width=0.003)

                # 12. Update viewer
                viewer.sync()

                rtde_c.waitPeriod(t_start)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        rtde_c.stopScript()
        print("[INFO] Script stopped")

if __name__ == "__main__":
    main()
