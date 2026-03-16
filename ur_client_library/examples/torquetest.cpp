#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// UR 客户端库
#include <ur_client_library/ur/dashboard_client.h>
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/types.h>
#include <ur_client_library/example_robot_wrapper.h>

// Pinocchio 相关
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"

// --- UDP 日志类 ---
class PJRealtimeLog {
private:
    int sock_;
    struct sockaddr_in servaddr_;
public:
    PJRealtimeLog(const std::string& ip, int port) {
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_port = htons(port);
        servaddr_.sin_addr.s_addr = inet_addr(ip.c_str());
    }
    void send(const std::string& json_str) {
        sendto(sock_, json_str.c_str(), json_str.length(), 0, 
               (const struct sockaddr *)&servaddr_, sizeof(servaddr_));
    }
    ~PJRealtimeLog() { ::close(sock_); }
};

// --- 常量设置 ---
const std::string ur5e_model_path = "/home/amdt/ur_force_ws/ur_client_library/urdf/ur5e.urdf";
const std::string DEFAULT_ROBOT_IP = "192.168.56.101";
const std::string OUTPUT_RECIPE = "examples/resources/rtde_output_recipe.txt";
const std::string INPUT_RECIPE = "examples/resources/rtde_input_recipe.txt";

// 安全限制：调试阶段建议设小
const urcl::vector6d_t SAFE_TORQUE_LIMITS = { 10.0, 10.0, 10.0, 5.0, 5.0, 5.0 };

// --- 全局变量 ---
std::unique_ptr<urcl::ExampleRobotWrapper> g_my_robot;
Eigen::VectorXd q(6), v(6), F_ext(6);
urcl::vector6d_t target_torques;

// 控制矩阵
Eigen::MatrixXd Kd = Eigen::MatrixXd::Zero(6, 6);
Eigen::MatrixXd Dd = Eigen::MatrixXd::Zero(6, 6);
Eigen::MatrixXd J(6, 6);

// --- 辅助函数 ---
void configureImpedance() {
    // 调低刚度以确保安全测试
    const double pos_k = 400.0; 
    const double ori_k = 50.0;
    const double pos_d = 2 * std::sqrt(pos_k * 5.0) * 0.7; // 阻尼比 0.7
    const double ori_d = 2 * std::sqrt(ori_k * 0.5) * 0.7;

    Kd.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * pos_k;
    Kd.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * ori_k;
    Dd.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * pos_d;
    Dd.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * ori_d;
}

int main(int argc, char* argv[]) {
    urcl::setLogLevel(urcl::LogLevel::INFO);
    std::string robot_ip = (argc > 1) ? argv[1] : DEFAULT_ROBOT_IP;

    // 1. 初始化机械臂
    g_my_robot = std::make_unique<urcl::ExampleRobotWrapper>(robot_ip, OUTPUT_RECIPE, INPUT_RECIPE, true, "external_control.urp");
    if (!g_my_robot->isHealthy()) return 1;

    // 2. 初始化 Pinocchio
    pinocchio::Model model;
    pinocchio::urdf::buildModel(ur5e_model_path, model);
    pinocchio::Data data(model);
    int ee_frame_id = model.getFrameId("tool0");

    // 3. 配置控制参数
    configureImpedance();
    PJRealtimeLog logger("127.0.0.1", 9870);

    // 4. 状态变量
    bool initial_captured = false;
    Eigen::Vector3d x_start, x_des;
    Eigen::Quaterniond quat_des;
    double t_traj = 0.0;
    double dt = 0.002; // 500Hz
    int log_cnt = 0;

    g_my_robot->getUrDriver()->startRTDECommunication();
    URCL_LOG_INFO("Impedance Control Loop Started.");

    while (true) {
        // 读取 RTDE 数据
        auto data_pkg = g_my_robot->getUrDriver()->getDataPackage();
        if (!data_pkg) continue;

        urcl::vector6d_t q_raw, qd_raw;
        data_pkg->getData("actual_q", q_raw);
        data_pkg->getData("actual_qd", qd_raw);
        for(int i=0; i<6; i++) { q[i] = q_raw[i]; v[i] = qd_raw[i]; }

        // 运动学更新
        pinocchio::forwardKinematics(model, data, q, v);
        pinocchio::updateFramePlacements(model, data);
        pinocchio::computeJointJacobians(model, data, q);
        pinocchio::getFrameJacobian(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);

        pinocchio::SE3 curr_pose = data.oMf[ee_frame_id];
        Eigen::Vector3d x_curr = curr_pose.translation();
        Eigen::Matrix3d R_curr = curr_pose.rotation();

        // 【关键】捕捉初始位姿
        if (!initial_captured) {
            x_start = x_curr;
            quat_des = Eigen::Quaterniond(R_curr);
            initial_captured = true;
            URCL_LOG_INFO("Home Position Captured.");
        }

        // 生成圆形轨迹 (相对于初始点在 YZ 平面移动)
        double r = 0.1; // 10cm 半径
        double w = 1.0; // 1 rad/s
        x_des = x_start;
        x_des[1] = x_start[1] + r * std::sin(w * t_traj);
        x_des[2] = x_start[2] + r * (1.0 - std::cos(w * t_traj)); 
        t_traj += dt;

        // 计算误差
        Eigen::Vector3d pos_err = x_curr - x_des;
        Eigen::Quaterniond q_curr(R_curr);
        if (q_curr.dot(quat_des) < 0.0) q_curr.coeffs() *= -1.0;
        Eigen::Quaterniond q_err = q_curr * quat_des.inverse();
        Eigen::Vector3d ori_err = 2.0 * q_err.vec();

        Eigen::VectorXd err_6d(6), vel_6d(6);
        err_6d.head(3) = pos_err; err_6d.tail(3) = ori_err;
        vel_6d = J * v; // 假设期望速度为 0

        // 阻抗力计算
        Eigen::VectorXd F_task = -(Kd * err_6d + Dd * vel_6d);

        // 动力学补偿：重力 + 科氏力
        Eigen::VectorXd tau_dyn = pinocchio::computeCoriolisMatrix(model, data, q, v) * v + 
                                  pinocchio::computeGeneralizedGravity(model, data, q);

        // 最终力矩映射
        Eigen::VectorXd tau_cmd = tau_dyn + J.transpose() * F_task;

        // 安全限幅
        for (int i = 0; i < 6; i++) {
            tau_cmd[i] = std::max(-SAFE_TORQUE_LIMITS[i], std::min(SAFE_TORQUE_LIMITS[i], tau_cmd[i]));
            target_torques[i] = tau_cmd[i];
        }

        // 发送指令
        if (!g_my_robot->getUrDriver()->writeJointCommand(target_torques, urcl::comm::ControlMode::MODE_TORQUE)) {
            URCL_LOG_ERROR("Robot Disconnected.");
            break;
        }

        // 日志 (100Hz)
        if (++log_cnt % 5 == 0) {
            std::stringstream ss;
            ss << "{\"ex\":" << pos_err[0] << ",\"ey\":" << pos_err[1] << ",\"ez\":" << pos_err[2] << "}";
            logger.send(ss.str());
        }
    }
    return 0;
}