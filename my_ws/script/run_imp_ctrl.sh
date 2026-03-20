#!/bin/bash
# Run the newly created task space impedance control script.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/home/amdt/app_ws/my_mjctrl/mujoco_env"
# 自动定位到 ur_client_library 的编译输出目录
LIB_PATH="/home/amdt/ur_force_ws/ur_client_library/build"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Python environment not found at $VENV_PATH"
    exit 1
fi

echo "--- Setting LD_LIBRARY_PATH to $LIB_PATH ---"
export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"

echo "--- Activating environment from $VENV_PATH ---"
. "$VENV_PATH/bin/activate"

echo "--- Starting Task Space Impedance Control ---"
cd "$SCRIPT_DIR"/..
python3 script/torque_ctrl.py "$@"
