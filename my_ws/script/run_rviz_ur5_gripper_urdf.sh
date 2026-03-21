#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="/home/amdt/ur_force_ws"
URDF_FILE="${1:-$WS_ROOT/my_ws/urdf/ur5_gripper.urdf}"
RVIZ_CONFIG="${2:-$WS_ROOT/my_ws/urdf/ur5_description/rviz/ur5.rviz}"

if [[ ! -f "$URDF_FILE" ]]; then
  echo "[ERROR] URDF file not found: $URDF_FILE"
  echo "Usage: $0 [URDF_FILE] [RVIZ_CONFIG]"
  exit 1
fi

if [[ ! -f "$RVIZ_CONFIG" ]]; then
  echo "[ERROR] RViz config not found: $RVIZ_CONFIG"
  echo "Usage: $0 [URDF_FILE] [RVIZ_CONFIG]"
  exit 1
fi

# ROS setup scripts may reference unset vars; avoid nounset errors while sourcing.
set +u
source /opt/ros/humble/setup.bash
if [[ -f "$WS_ROOT/install/setup.bash" ]]; then
  source "$WS_ROOT/install/setup.bash"
fi
set -u

TMP_PARAMS="$(mktemp /tmp/ur5_gripper_rsp_XXXXXX.yaml)"

cleanup() {
  local code=$?
  if [[ -n "${RSP_PID:-}" ]]; then kill "$RSP_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${JSP_PID:-}" ]]; then kill "$JSP_PID" >/dev/null 2>&1 || true; fi
  rm -f "$TMP_PARAMS"
  exit $code
}
trap cleanup EXIT INT TERM

{
  echo "robot_state_publisher:"
  echo "  ros__parameters:"
  echo "    use_sim_time: false"
  echo "    robot_description: |"
  sed 's/^/      /' "$URDF_FILE"
} > "$TMP_PARAMS"

echo "[INFO] Using URDF: $URDF_FILE"
echo "[INFO] Using RViz config: $RVIZ_CONFIG"
echo "[INFO] Params file: $TMP_PARAMS"

ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$TMP_PARAMS" &
RSP_PID=$!

ros2 run joint_state_publisher_gui joint_state_publisher_gui &
JSP_PID=$!

rviz2 -d "$RVIZ_CONFIG"
