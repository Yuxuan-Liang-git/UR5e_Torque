cmake_minimum_required(VERSION 3.10)
project(torque_ctrl_project)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找依赖项
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Threads REQUIRED)

# 设置 ur_client_library 路径 (使用相对 CMakeLists.txt 的路径)
set(URCL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../ur_client_library)
set(URCL_INCLUDE_DIR ${URCL_ROOT}/include)
set(URCL_3RDPARTY_DIR ${URCL_ROOT}/3rdparty)
# 假设 ur_client_library 已经在其 build 文件夹下编译完成
set(URCL_LIB_DIR ${URCL_ROOT}/build)

# 包含路径
include_directories(
    ${URCL_INCLUDE_DIR}
    ${URCL_3RDPARTY_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# 添加可执行文件
add_executable(torque_ctrl src/torque_ctrl.cpp)

# 链接库
# 注意：ur_client_library 的库文件名可能是 liburcl.so
target_link_libraries(torque_ctrl
    ${URCL_LIB_DIR}/liburcl.so
    pinocchio::pinocchio
    Threads::Threads
)

# 设置运行时库搜索路径，这样运行可执行文件时能找到 libur_client_library.so
set_target_properties(torque_ctrl PROPERTIES INSTALL_RPATH "${URCL_LIB_DIR}")
set_target_properties(torque_ctrl PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
