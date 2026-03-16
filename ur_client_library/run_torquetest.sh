#!/bin/bash
# 运行 torquetest 并正确加载本地编译的库
# 用法：./run_torquetest.sh

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置 LD_LIBRARY_PATH 优先加载本地编译的库
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build:${LD_LIBRARY_PATH}"

# 运行 torquetest 可执行文件
exec "${SCRIPT_DIR}/build/examples/torquetest" "$@"
