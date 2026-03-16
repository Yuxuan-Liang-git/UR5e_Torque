#!/bin/bash
# 运行 torquetest 并正确加载本地编译的库
# 用法：./run_torquetest.sh

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置 LD_LIBRARY_PATH 优先加载本地编译的库
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build:${LD_LIBRARY_PATH}"

# # 自动编译 init_pos 和 torquetest
# echo "--- Compiling init_pos and torquetest ---"
# (cd "${SCRIPT_DIR}/build" && make -j$(nproc) torquetest)
# if [ $? -ne 0 ]; then
#     echo "Compilation failed! Exiting."
#     exit 1
# fi
# echo "--- Compilation successful ---"

# 运行 torquetest 可执行文件
# exec "${SCRIPT_DIR}/build/examples/torquetest" "$@"
cd "${SCRIPT_DIR}"
exec "./build/examples/init_pos" "$@"
