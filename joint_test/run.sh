#!/bin/bash

# 默认机器人IP (和代码里的默认IP保持一致)
ROBOT_IP=${1:-"192.168.56.101"}

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}[INFO] 准备运行 torque_smc 控制器... 目标IP: ${ROBOT_IP}${NC}"

if [ ! -f "build/torque_smc" ]; then
    echo -e "\033[0;31m[ERROR] 找不到可执行文件，请先运行 ./build.sh 进行编译${NC}"
    exit 1
fi

cd build/
# 执行控制程序并传入机器人的IP参数
./torque_smc ${ROBOT_IP}
