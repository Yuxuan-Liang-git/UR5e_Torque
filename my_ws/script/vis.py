#!/usr/bin/env python3
"""Visualize real UR5e joint angles in MuJoCo.

This script reads actual joint positions from a real robot through RTDE and maps
those joint angles to the UR5e model in MuJoCo for live visualization.
"""

from __future__ import annotations

import argparse
import sys
import time
import socket
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    from rtde_receive import RTDEReceiveInterface
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'ur-rtde'. Install it in your mujoco_env: pip install ur-rtde"
    ) from exc


HERE = Path(__file__).resolve().parent
XML_PATH = HERE / "universal_robots_ur5e" / "scene.xml"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map real UR5e joint angles to MuJoCo")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP address")
    parser.add_argument("--rtde-freq", type=float, default=125.0, help="RTDE receive frequency (Hz)")
    parser.add_argument("--xml", type=Path, default=XML_PATH, help="MuJoCo scene XML path")
    parser.add_argument("--udp-port", type=int, default=9870, help="UDP port to listen for target position")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.exists():
        print(f"[ERROR] XML not found: {xml_path}")
        return 1

    print(f"[INFO] Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)

    dof_ids = np.array([model.joint(name).id for name in JOINT_NAMES], dtype=int)

    print(f"[INFO] Connecting RTDE receive: ip={args.robot_ip}, freq={args.rtde_freq}Hz")
    rtde = RTDEReceiveInterface(args.robot_ip, args.rtde_freq)

    # UDP 接收初始化（端口被占用时不终止程序，降级为仅显示实际轨迹）
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_enabled = True
    try:
        sock.bind(("0.0.0.0", args.udp_port))
        sock.setblocking(False)
        print(f"[INFO] Listening for UDP target position on port {args.udp_port}")
    except OSError as exc:
        udp_enabled = False
        sock.close()
        sock = None
        print(f"[WARN] UDP port {args.udp_port} is busy: {exc}")
        print("[WARN] Running without target trajectory input. Only actual trajectory will be shown.")
        print("[WARN] Use --udp-port <new_port> or stop the process occupying the port.")

    # 轨迹可视化参数
    trajectory_points = []  # 实际末端轨迹 (红色)
    target_trajectory = []  # 期望目标轨迹 (绿色)
    max_points = 150
    site_id = model.site("attachment_site").id
    target_pos = np.zeros(3) # 默认目标点

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mj_resetData(model, data)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        print("[INFO] Running live visualization. Press Ctrl+C to stop.")
        while viewer.is_running():
            try:
                q_actual = np.asarray(rtde.getActualQ(), dtype=float)
                if q_actual.shape[0] != 6:
                    raise RuntimeError(f"Unexpected joint vector size: {q_actual.shape}")
            except Exception as exc:
                print(f"[WARN] RTDE read failed: {exc}")
                time.sleep(0.02)
                continue

            data.qpos[dof_ids] = q_actual
            data.qvel[dof_ids] = 0.0
            mujoco.mj_forward(model, data)

            # --- UDP 接收目标点数据 ---
            if udp_enabled and sock is not None:
                try:
                    # 尝试读取最新的 UDP 包 (非阻塞)
                    while True:
                        data_udp, _ = sock.recvfrom(4096)
                        # 假定发送的是 JSON 格式: {"pos_des":{"x":0,"y":0,"z":0}}
                        msg = json.loads(data_udp.decode("utf-8"))
                        if "pos_des" in msg:
                            target_pos[1] = msg["pos_des"]["x"]
                            target_pos[0] = -msg["pos_des"]["y"]
                            target_pos[2] = msg["pos_des"]["z"]
                except (BlockingIOError, json.JSONDecodeError, KeyError):
                    pass

            # --- 更新轨迹点 ---
            if not trajectory_points or time.time() - trajectory_points[-1][3] > 0.05:
                # 实际末端位置
                curr_pos = data.site(site_id).xpos.copy()
                trajectory_points.append(np.append(curr_pos, time.time()))
                
                # 期望目标位置 (来自上面的 UDP 接收)
                target_trajectory.append(np.append(target_pos.copy(), time.time()))

                if len(trajectory_points) > max_points:
                    trajectory_points.pop(0)
                    target_trajectory.pop(0)

            # 绘制轨迹
            draw_trajectory(viewer, target_trajectory, color=[0, 1, 0, 1], clear=True, width=0.003)  # 绿色期望
            draw_trajectory(viewer, trajectory_points, color=[1, 0, 0, 1], width=0.003)             # 红色实际

            viewer.sync()
            time.sleep(0.002)

    if sock is not None:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
