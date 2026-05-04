#!/bin/bash

# 打印颜色和前缀设置
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO] 开始配置并编译 torque_smc 控制器...${NC}"

# 创建 build 文件夹并进入
mkdir -p build
cd build

# 如果 ur_client_library 之前并未全局安装而是单独编译的，请在此处的cmake命令上附带其路径
# 例如：cmake -Dur_client_library_DIR=/home/amdt/ur_force_ws/ur_client_library/build ..
cmake ..

# 开始并行编译
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] 编译成功！可执行文件已生成在 build/torque_smc${NC}"
else
    echo -e "\033[0;31m[ERROR] 编译失败，请检查 CMake 和源代码报错信息。\033[0m"
fi
