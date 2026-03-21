#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="/home/amdt/ur_force_ws"

# ROS setup scripts may reference unset vars; avoid nounset errors while sourcing.
set +u
source /opt/ros/humble/setup.bash
source "$WS_ROOT/install/setup.bash"
set -u

ros2 launch ur5_description rviz_with_gripper.launch.py "$@"
