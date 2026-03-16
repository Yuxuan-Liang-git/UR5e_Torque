#!/bin/bash
# 运行 UR Client Library 示例时，确保使用本地编译的库
# 用法: ./run_example.sh <可执行文件名> [参数]

# 设置 LD_LIBRARY_PATH 以优先使用本地编译的库
export LD_LIBRARY_PATH="/home/amdt/ur_force_ws/ur_client_library/build:$LD_LIBRARY_PATH"

# 运行指定的可执行文件
exec "$@"
