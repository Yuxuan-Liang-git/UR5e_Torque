#!/bin/bash
# Run task space impedance control script with a deterministic Python interpreter.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# RTDE python bindings on this machine are installed for system Python 3.10.
PYTHON_BIN="/usr/bin/python3"
# 自动定位到 ur_client_library 的编译输出目录
LIB_PATH="/home/amdt/ur_force_ws/ur_client_library/build"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

echo "--- Setting LD_LIBRARY_PATH to $LIB_PATH ---"
export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"

echo "--- Using Python: $PYTHON_BIN ---"
"$PYTHON_BIN" -c 'import sys; print(sys.version)'

echo "--- Starting Task Space Impedance Control ---"
cd "$SCRIPT_DIR"/..
"$PYTHON_BIN" script/torque_ctrl.py "$@"
