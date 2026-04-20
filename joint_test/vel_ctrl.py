#!/usr/bin/env python3
"""Velocity control for UR5e.

Initializes the robot to a target pose, then runs velocity control.
Joint 6 tracking a sine wave, while other 5 joints hold reference speed 0.
"""

import argparse
import time
import signal
import threading
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from visualization import VisualizationWorker, UDPLogger

try:
    from live_plotter import live_plot_worker
except ImportError:
    live_plot_worker = None

# --- Joints test parameters ---
VIS = True        # Enable online matplotlib plotting of tracking error

# --- Global stop event ---
stop_event = threading.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    stop_event.set()

CONTROL_DT = 0.002 
VIS_FREQ = 50.0

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Joint-Space Velocity Control")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--no-vis", action="store_true", help="Disable MuJoCo visualization thread")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="UDP destination IP for PlotJuggler")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP destination port for PlotJuggler")
    parser.add_argument("--udp-div", type=int, default=2, help="Send one packet every N control loops")
    parser.add_argument("--traj-file", type=str, default="Data/opt_sltn_N5T20_full_trajectory.csv", help="CSV file containing joint trajectories")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Load MuJoCo environment ---
    xml_path = Path("ur5e_gripper/scene.xml").absolute()
    if not xml_path.exists():
        xml_path = Path("universal_robots_ur5e/scene.xml").absolute()
    if not xml_path.exists():
        xml_path = Path("scene.xml").absolute()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site_id == -1:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)
    rtde_r = RTDEReceiveInterface(args.robot_ip)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    visualizer = None
    if not args.no_vis:
        visualizer = VisualizationWorker(xml_path, render_hz=VIS_FREQ)

    logger = UDPLogger(args.udp_ip, args.udp_port, send_every_n=args.udp_div)

    # --- Load CSV Trajectory ---
    traj_path = Path(args.traj_file).absolute()
    if not traj_path.exists():
        print(f"[ERROR] Trajectory file not found: {traj_path}")
        return
        
    print(f"[INFO] Loading trajectory from {traj_path}")
    traj_df = pd.read_csv(traj_path)
    t_traj_data = traj_df['time'].values

    q1_data = traj_df['q1'].values
    q2_data = traj_df['q2'].values
    q3_data = traj_df['q3'].values
    q4_data = traj_df['q4'].values
    q5_data = traj_df['q5'].values
    q6_data = traj_df['q6'].values
    qd1_data = traj_df['qd1'].values
    qd2_data = traj_df['qd2'].values
    qd3_data = traj_df['qd3'].values
    qd4_data = traj_df['qd4'].values
    qd5_data = traj_df['qd5'].values
    qd6_data = traj_df['qd6'].values

    # --- Setup Live Plotting ---
    plot_queue = None
    plot_process = None
    if VIS and live_plot_worker is not None:
        print("[INFO] Starting live plotter process...")
        plot_queue = mp.Queue()
        plot_process = mp.Process(target=live_plot_worker, args=(plot_queue,))
        plot_process.start()

    # q_target = np.array([1.5326, -1.30735, -2.06397, -1.36756, 1.59805, -1.55037], dtype=float)

    q_target = np.array([0,	-1.5707963267949,	0,	-1.5707963267949,	0,	0], dtype=float)

    data.qpos[:6] = q_target
    mujoco.mj_forward(model, data)

    print("[INFO] Moving to initial pose...")
    rtde_c.moveJ(q_target.tolist(), 1.05, 1.4)

    trajectory_points = []
    target_trajectory = []
    max_points = 200
    traj_sample_period = 0.05
    next_traj_sample = 0.0

    start_time = time.perf_counter()
    next_tick = time.perf_counter()
    total_loop_count = 0
    freq_start_time = time.perf_counter()
    freq_loop_count = 0

    accel = 2.0  # joint acceleration for speedJ command
    
    recorded_data = []
    prev_dq = np.zeros(6)

    try:
        while not stop_event.is_set():
            if visualizer is not None and not visualizer.is_running():
                print("[INFO] Visualization closed, stopping controller loop.")
                break

            t_now = time.perf_counter() - start_time

            q = np.array(rtde_r.getActualQ(), dtype=float)
            dq = np.array(rtde_r.getActualQd(), dtype=float)

            # [Fx, Fy, Fz, Tx, Ty, Tz]
            tcp_force = np.array(rtde_r.getActualTCPForce(), dtype=float)

            data.qpos[:6] = q
            data.qvel[:6] = dq
            mujoco.mj_forward(model, data)
            x_curr_pos = data.site_xpos[ee_site_id].copy()
            
            dq_des = np.zeros(6)

            t_traj = t_now
            
            if t_traj > t_traj_data[-1]:
                print("[INFO] Trajectory finished.")
                break
            
            # Since we just moved to initial position, our absolute target position 
            # for non-tracking joints remains q_target, and they are holding steady
            q_des = q_target.copy()
            
            # Joints track the trajectory loaded from CSV
            # q_des[0] = np.interp(t_traj, t_traj_data, q1_data)
            q_des[1] = np.interp(t_traj, t_traj_data, q2_data)
            q_des[2] = np.interp(t_traj, t_traj_data, q3_data)
            q_des[3] = np.interp(t_traj, t_traj_data, q4_data)
            q_des[4] = np.interp(t_traj, t_traj_data, q5_data)
            q_des[5] = np.interp(t_traj, t_traj_data, q6_data)
            
            # dq_des[0] = np.interp(t_traj, t_traj_data, qd1_data)
            dq_des[1] = np.interp(t_traj, t_traj_data, qd2_data)
            dq_des[2] = np.interp(t_traj, t_traj_data, qd3_data)
            dq_des[3] = np.interp(t_traj, t_traj_data, qd4_data)
            dq_des[4] = np.interp(t_traj, t_traj_data, qd5_data)
            dq_des[5] = np.interp(t_traj, t_traj_data, qd6_data)
            
            # Get actual torques from the real robot (TargetMoment is just target payload/gravity torque)
            # To get better approximation of actual motor effort, we can use getActualCurrent or getJointTorques from control interface.
            # Using actual current (since tau = Kt * I, this reflects actual friction). Or rtde_c.getJointTorques()
            tau_actual = np.array(rtde_r.getActualCurrent(), dtype=float)
            
            if total_loop_count > 0:
                qdd = (dq - prev_dq) / CONTROL_DT
            else:
                qdd = np.zeros(6)
            prev_dq = dq.copy()
            
            recorded_data.append([t_traj] + q.tolist() + dq.tolist() + qdd.tolist() + tau_actual.tolist())
            
            # Send velocity command
            ok = rtde_c.speedJ(dq_des.tolist(), accel, CONTROL_DT)
            if not ok:
                print("[ERROR] speedJ failed")
                break

            total_loop_count += 1
            
            extra_data = {
                "dq": dq,
                "dq_des": dq_des,
                "tcp_force": tcp_force,
            }
            # use np.zeros(6) for torque as it's purely speed control
            logger.update(total_loop_count, q_target, q, np.zeros(3), x_curr_pos, np.zeros(6), extra=extra_data)

            if (not trajectory_points) or (t_now >= next_traj_sample):
                trajectory_points.append(x_curr_pos.copy())
                target_trajectory.append(np.zeros(3))
                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)
                next_traj_sample = t_now + traj_sample_period

            if visualizer is not None:
                visualizer.update(q, trajectory_points, target_trajectory, np.zeros(3), np.eye(3))

            # Print freq
            freq_loop_count += 1
            if freq_loop_count >= 500:
                now = time.perf_counter()
                elapsed = now - freq_start_time
                print(f"[INFO] Control Frequency: {freq_loop_count / elapsed:.2f} Hz")
                freq_loop_count = 0
                freq_start_time = now

            if plot_queue is not None:
                # Downsample for live plotting
                if total_loop_count % 10 == 0:
                    try:
                        plot_queue.put_nowait((t_now, q_des, q, dq_des, dq, tau_actual))
                    except mp.queues.Full:
                        pass

            next_tick += CONTROL_DT
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    except (KeyboardInterrupt, SystemExit):
        print("\n[INFO] Stopped by user")
    finally:
        print("[INFO] Cleaning up...")
        if plot_queue is not None:
            plot_queue.put(None)
            if plot_process is not None:
                plot_process.join(timeout=1.0)
        
        if visualizer is not None:
            visualizer.stop()
        logger.stop()
        try:
            rtde_c.speedStop(10.0)
            rtde_c.stopScript()
            rtde_c.disconnect()
        except:
            pass
        print("[INFO] Script stopped")
        
        if recorded_data:
            print("[INFO] Saving recorded data to CSV...")
            cols = ['time'] + [f'q{i}' for i in range(1, 7)] + \
                   [f'dq{i}' for i in range(1, 7)] + [f'qdd{i}' for i in range(1, 7)] + \
                   [f'I{i}' for i in range(1, 7)]
            df_out = pd.DataFrame(recorded_data, columns=cols)
            ref_name = Path(args.traj_file).name
            save_path = Path("Data") / f"data_{ref_name}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(save_path, index=False)
            print(f"[INFO] Successfully saved to {save_path}")

if __name__ == "__main__":
    main()
