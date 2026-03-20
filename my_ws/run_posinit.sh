#!/bin/bash
# 运行 init_pos 初始化文件

# 获取脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 UR 库路径 (相对于脚本所在目录)
UR_LIB_DIR="$DIR/../ur_client_library"
UR_BUILD_DIR="$UR_LIB_DIR/build"

# 设置 LD_LIBRARY_PATH 优先加载本地编译的库
export LD_LIBRARY_PATH="${UR_BUILD_DIR}:${LD_LIBRARY_PATH}"

# 运行 init_pos 可执行文件
# 注意：init_pos 位于 ur_client_library 的 build/examples 目录下
INIT_POS_EXE="$UR_BUILD_DIR/examples/init_pos"

if [ ! -f "$INIT_POS_EXE" ]; then
    echo "Error: $INIT_POS_EXE not found. Please build ur_client_library first."
    exit 1
fi

echo "--- Starting init_pos ---"
# 进入 ur_client_library 目录执行，以确保其内部使用的相对路径（如 recipes）能被正确找到
cd "$UR_LIB_DIR"
exec "$INIT_POS_EXE" "$@"
