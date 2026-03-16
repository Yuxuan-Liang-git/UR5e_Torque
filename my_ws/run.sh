#!/bin/bash

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$DIR/build"
TARGET="$BUILD_DIR/torque_ctrl"

# 设置 UR 库路径 (相对于脚本所在目录)
UR_LIB_DIR="$DIR/../ur_client_library"
UR_BUILD_DIR="$UR_LIB_DIR/build"

# 设置 LD_LIBRARY_PATH 优先加载本地编译的库（解决 undefined symbol 问题）
export LD_LIBRARY_PATH="${UR_BUILD_DIR}:${LD_LIBRARY_PATH}"

# 自动编译
echo "--- Checking and Compiling torque_ctrl ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" && cmake .. && make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi
echo "--- Compilation successful ---"

# 运行可执行文件
if [ -f "$TARGET" ]; then
    echo "Running torque_ctrl..."
    # 默认使用 IP 地址，或者从命令行传递
    ROBOT_IP=${1:-"192.168.56.101"}
    # 进入项目根目录运行，以确保相对路径 ./ur_client_library/... 生效
    cd "$DIR/.."
    exec "$TARGET" "$ROBOT_IP"
else
    echo "Error: Executable not found at $TARGET"
    exit 1
fi
