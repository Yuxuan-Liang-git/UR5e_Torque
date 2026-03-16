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
#include <fstream>

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

// --- UDP 日志类 (适配 PlotJuggler JSON) ---
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

// --- 常量与路径设置 ---
const std::string ur5e_model_path = "/home/amdt/ur_force_ws/ur_client_library/urdf/ur5e.urdf";
const std::string INIT_POS_FILE = "/home/amdt/ur_force_ws/ur_client_library/init_pos.txt";
const std::string DEFAULT_ROBOT_IP = "192.168.56.101";
const std::string OUTPUT_RECIPE = "examples/resources/rtde_output_recipe.txt";
const std::string INPUT_RECIPE = "examples/resources/rtde_input_recipe.txt";

// 安全限制：力矩限幅 (Nm)
const urcl::vector6d_t SAFE_TORQUE_LIMITS = { 20.0, 20.0, 20.0, 10.0, 10.0, 10.0 };

// --- 全局变量 ---
std::unique_ptr<urcl::ExampleRobotWrapper> g_my_robot;
Eigen::VectorXd q(6), v(6);
urcl::vector6d_t target_torques;

// 控制参数矩阵
Eigen::MatrixXd Kd = Eigen::MatrixXd::Zero(6, 6);
Eigen::MatrixXd Dd = Eigen::MatrixXd::Zero(6, 6);

// --- 阻抗参数配置 ---
void configureImpedance() {
    // 刚度系数 (平移 400 N/m, 旋转 50 Nm/rad)
    const double pos_k = 3000.0; 
    const double ori_k = 50.0;
    // 阻尼系数 (临界阻尼计算)
    const double pos_d = 2 * std::sqrt(pos_k * 5.0) * 0.7; 
    const double ori_d = 2 * std::sqrt(ori_k * 0.5) * 0.7;

    Kd.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * pos_k;
    Kd.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * ori_k;
    Dd.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * pos_d;
    Dd.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * ori_d;
}

int main(int argc, char* argv[]) {
    urcl::setLogLevel(urcl::LogLevel::INFO);
    std::string robot_ip = (argc > 1) ? argv[1] : DEFAULT_ROBOT_IP;

    // 1. 初始化机械臂驱动
    g_my_robot = std::make_unique<urcl::ExampleRobotWrapper>(robot_ip, OUTPUT_RECIPE, INPUT_RECIPE, true, "external_control.urp");
    if (!g_my_robot->isHealthy()) {
        URCL_LOG_ERROR("Failed to connect to robot.");
        return 1;
    }

    // 2. 初始化 Pinocchio 动力学模型
    pinocchio::Model model;
    pinocchio::urdf::buildModel(ur5e_model_path, model);
    pinocchio::Data data(model);
    int ee_frame_id = model.getFrameId("tool0");

    // 3. 配置控制与频率监控
    configureImpedance();
    PJRealtimeLog logger("127.0.0.1", 9870);
    
    // 频率监控变量
    auto loop_start_time = std::chrono::steady_clock::now();
    int loop_count = 0;
    int log_cnt = 0;

    // 4. 状态与轨迹变量
    bool initial_captured = false;
    Eigen::Vector3d x_start, x_des;
    Eigen::Quaterniond quat_des;
    double t_traj = 0.0;
    const double dt = 0.002; // 500Hz
    
    g_my_robot->getUrDriver()->startRTDECommunication();
    URCL_LOG_INFO("Impedance Control Loop Started. Sending data to PlotJuggler on Port 9870.");

    while (true) {
        // A. 读取 RTDE 传感器数据
        auto data_pkg = g_my_robot->getUrDriver()->getDataPackage();
        if (!data_pkg) continue;

        urcl::vector6d_t q_raw, qd_raw;
        data_pkg->getData("actual_q", q_raw);
        data_pkg->getData("actual_qd", qd_raw);
        for(int i=0; i<6; i++) { q[i] = q_raw[i]; v[i] = qd_raw[i]; }

        // B. 运动学更新
        pinocchio::forwardKinematics(model, data, q, v);
        pinocchio::updateFramePlacements(model, data);
        
        // 获取雅可比矩阵 (世界坐标系对齐)
        Eigen::MatrixXd J(6, 6);
        pinocchio::computeJointJacobians(model, data, q);
        pinocchio::getFrameJacobian(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);

        // 获取当前末端位姿
        pinocchio::SE3 curr_pose = data.oMf[ee_frame_id];
        Eigen::Vector3d x_curr = curr_pose.translation();
        Eigen::Matrix3d R_curr = curr_pose.rotation();

        // C. 捕捉初始位置作为参考点
        if (!initial_captured) {
            std::ifstream infile(INIT_POS_FILE);
            if (infile.is_open()) {
                urcl::vector6d_t q_init;
                for (int i = 0; i < 6; ++i) {
                    if (!(infile >> q_init[i])) {
                        URCL_LOG_ERROR("Failed to read joint %d from %s. Using current pose instead.", i, INIT_POS_FILE.c_str());
                        x_start = x_curr;
                        quat_des = Eigen::Quaterniond(R_curr);
                        initial_captured = true;
                        break;
                    }
                }
                if (!initial_captured) {
                    Eigen::VectorXd q_vec(6);
                    for (int i = 0; i < 6; ++i) q_vec[i] = q_init[i];
                    
                    // 使用读取到的关节角计算参考位姿
                    pinocchio::Data data_init(model);
                    pinocchio::forwardKinematics(model, data_init, q_vec);
                    pinocchio::updateFramePlacements(model, data_init);
                    pinocchio::SE3 init_pose = data_init.oMf[ee_frame_id];
                    
                    x_start = init_pose.translation();
                    quat_des = Eigen::Quaterniond(init_pose.rotation());
                    initial_captured = true;
                    URCL_LOG_INFO("Reference Position loaded from %s.", INIT_POS_FILE.c_str());
                    URCL_LOG_INFO("Loaded Joints: %.4f %.4f %.4f %.4f %.4f %.4f", q_init[0], q_init[1], q_init[2], q_init[3], q_init[4], q_init[5]);
                }
            } else {
                URCL_LOG_WARN("Could not open %s. Using current pose as reference.", INIT_POS_FILE.c_str());
                x_start = x_curr;
                quat_des = Eigen::Quaterniond(R_curr);
                initial_captured = true;
                URCL_LOG_INFO("Reference Position Captured (Current).");
            }
        }

        // D. 生成圆形轨迹 (YZ平面)
        double r = 0.08; // 8cm 半径
        double w = 1.2;  // 旋转速度
        x_des = x_start;
        x_des[1] = x_start[1] + r * std::sin(w * t_traj);
        x_des[2] = x_start[2] + r * (1.0 - std::cos(w * t_traj)); 
        t_traj += dt;

        // E. 计算偏差 (平移 + 姿态)
        Eigen::Vector3d pos_err = x_curr - x_des;
        
        Eigen::Quaterniond q_curr(R_curr);
        // 确保四元数最短路径
        if (q_curr.dot(quat_des) < 0.0) q_curr.coeffs() *= -1.0;
        Eigen::Quaterniond q_err = q_curr * quat_des.inverse();
        Eigen::Vector3d ori_err = 2.0 * q_err.vec(); // 轴角误差近似

        Eigen::VectorXd err_6d(6), vel_6d(6);
        err_6d.head(3) = pos_err; 
        err_6d.tail(3) = ori_err;
        vel_6d = J * v; // 末端当前速度

        // F. 阻抗力计算: F = -(Kp*e + Kd*v)
        Eigen::VectorXd F_task = -(Kd * err_6d + Dd * vel_6d);

        // G. 动力学补偿 (重力 + 科氏力)
        // 使用 RNEA 算法计算 tau = M(q)ddq + C(q,v)v + g(q)，此处令 ddq=0
        Eigen::VectorXd tau_dyn = pinocchio::nonLinearEffects(model, data, q, v);

        // H. 最终力矩映射: tau = J^T * F + tau_dyn
        Eigen::VectorXd tau_cmd = tau_dyn + J.transpose() * F_task;

        // I. 安全限幅与发送指令
        for (int i = 0; i < 6; i++) {
            tau_cmd[i] = std::max(-SAFE_TORQUE_LIMITS[i], std::min(SAFE_TORQUE_LIMITS[i], tau_cmd[i]));
            target_torques[i] = tau_cmd[i];
        }

        if (!g_my_robot->getUrDriver()->writeJointCommand(target_torques, urcl::comm::ControlMode::MODE_TORQUE)) {
            URCL_LOG_ERROR("Robot Disconnected or Safety Stop.");
            break;
        }

        // J. PlotJuggler UDP 日志发送 (100Hz)
        if (++log_cnt % 5 == 0) {
            std::stringstream ss;
            ss << "{"
               << "\"pos_curr\":{\"x\":" << x_curr[0] << ",\"y\":" << x_curr[1] << ",\"z\":" << x_curr[2] << "},"
               << "\"pos_des\":{\"x\":" << x_des[0] << ",\"y\":" << x_des[1] << ",\"z\":" << x_des[2] << "},"
               << "\"pos_err\":{\"x\":" << pos_err[0] << ",\"y\":" << pos_err[1] << ",\"z\":" << pos_err[2] << "},"
               << "\"tau\":{"
               << "\"j0\":" << target_torques[0] << ",\"j1\":" << target_torques[1] << ",\"j2\":" << target_torques[2] << ","
               << "\"j3\":" << target_torques[3] << ",\"j4\":" << target_torques[4] << ",\"j5\":" << target_torques[5]
               << "},"
               << "\"q_actual\":{"
               << "\"j0\":" << q[0] << ",\"j1\":" << q[1] << ",\"j2\":" << q[2] << ","
               << "\"j3\":" << q[3] << ",\"j4\":" << q[4] << ",\"j5\":" << q[5]
               << "}"
               << "}";
            logger.send(ss.str());
        }

        // 频率监控输出 (每 500 次循环计算一次平均频率，约 1 秒)
        loop_count++;
        if (loop_count >= 500) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - loop_start_time;
            double frequency = loop_count / elapsed.count();
            URCL_LOG_INFO("Control Frequency: %.2f Hz", frequency);
            
            // 重置计数
            loop_count = 0;
            loop_start_time = now;
        }
    }
    return 0;
}