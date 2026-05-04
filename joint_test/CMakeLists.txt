cmake_minimum_required(VERSION 3.10)
project(torque_smc_project)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 Eigen3 库
find_package(Eigen3 REQUIRED)

# 寻找 ur_client_library
# 如果提示找不到包，请取消下面这行的注释并修改为您的 ur_client_library/build 路径
# set(ur_client_library_DIR "/home/amdt/ur_force_ws/ur_client_library/build")
find_package(ur_client_library REQUIRED)

# 如果后续需要引入 mujoco，可以取消下面的注释
# find_package(mujoco REQUIRED)

# 添加可执行文件
add_executable(torque_smc src/torque_smc.cpp)

# 链接库文件
target_link_libraries(torque_smc 
    PRIVATE
    Eigen3::Eigen
    ur_client_library::urcl
    # mujoco::mujoco
)

# 添加包含路径（如果系统环境变量未配置Eigen3的话可能需要）
target_include_directories(torque_smc PRIVATE 
    ${EIGEN3_INCLUDE_DIR}
)
