#!/usr/bin/env python3
"""Pure gravity-compensation test for UR5e using ur-rtde.

UR direct torque mode expects torques after internal gravity compensation.
Sending a zero vector keeps the robot near current pose (payload must be set correctly).
"""

from __future__ import annotations

import argparse
import time
from rtde_control import RTDEControlInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR5e pure gravity-compensation hold test")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="UR robot IP")
    parser.add_argument("--freq", type=float, default=500.0, help="Control frequency in Hz")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds; <=0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--disable-friction-comp",
        action="store_true",
        help="Disable robot internal friction compensation (enabled by default)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print status every N control cycles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    friction_comp = not args.disable_friction_comp

    print(f"[INFO] Connecting to robot {args.robot_ip}")
    rtde_c = RTDEControlInterface(args.robot_ip)

    print("[INFO] Starting pure gravity-compensation hold")
    duration_text = "infinite (Ctrl+C to stop)" if args.duration <= 0 else f"{args.duration:.2f}s"
    print(f"[INFO] freq={args.freq:.1f}Hz, duration={duration_text}")
    print(f"[INFO] friction_comp={friction_comp}")
    print("[INFO] command torque = [0, 0, 0, 0, 0, 0]")

    t0 = time.time()
    step = 0
    zero_torque = [0.0] * 6

    try:
        while True:
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break

            t_start = rtde_c.initPeriod()

            # ok = rtde_c.directTorque(zero_torque, friction_comp)
            ok = rtde_c.directTorque(zero_torque, True)

            if not ok:
                print("[ERROR] directTorque returned False; stopping test")
                break

            if step % args.print_every == 0:
                print(f"[STEP {step:6d}] sending zero torque for gravity hold")

            rtde_c.waitPeriod(t_start)
            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Send zero torque for a few cycles to leave controller in a safe state.
    for _ in range(30):
        t_start = rtde_c.initPeriod()
        rtde_c.directTorque(zero_torque, friction_comp)
        rtde_c.waitPeriod(t_start)

    rtde_c.stopScript()
    print("[INFO] Gravity-compensation hold test finished, script stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
