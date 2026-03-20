#!/bin/bash
# 自动激活指定环境并运行可视化脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="/home/amdt/app_ws/my_mjctrl/mujoco_env"

# 检查环境是否存在
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Python environment not found at $VENV_PATH"
    exit 1
fi

# 激活环境
echo "--- Activating environment from $VENV_PATH ---"
. "$VENV_PATH/bin/activate"

# 运行 Python 脚本
echo "--- Starting vis.py ---"
cd "$SCRIPT_DIR"
# 可以通过 $@ 传递参数，如 ./run_vis.sh --robot-ip 192.168.56.101
python3 vis.py "$@"
