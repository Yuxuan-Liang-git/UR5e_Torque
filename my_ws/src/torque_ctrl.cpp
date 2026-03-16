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

// UR Client Library
#include <ur_client_library/ur/dashboard_client.h>
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/types.h>
#include <ur_client_library/example_robot_wrapper.h>

// Pinocchio
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"

// --- UDP Logging for PlotJuggler ---
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

// --- Config ---
const std::string ur5e_model_path = "./ur_client_library/urdf/ur5e.urdf";
const std::string INIT_POS_FILE = "./ur_client_library/init_pos.txt";
const std::string DEFAULT_ROBOT_IP = "192.168.56.101";
const std::string OUTPUT_RECIPE = "./ur_client_library/examples/resources/rtde_output_recipe.txt";
const std::string INPUT_RECIPE = "./ur_client_library/examples/resources/rtde_input_recipe.txt";
const std::string SCRIPT_FILE = "./ur_client_library/resources/external_control.urscript";

const urcl::vector6d_t SAFE_TORQUE_LIMITS = { 20.0, 20.0, 20.0, 10.0, 10.0, 10.0 };

std::unique_ptr<urcl::ExampleRobotWrapper> g_my_robot;
Eigen::VectorXd q(6), v(6);
urcl::vector6d_t target_torques;

Eigen::MatrixXd Kd = Eigen::MatrixXd::Zero(6, 6);
Eigen::MatrixXd Dd = Eigen::MatrixXd::Zero(6, 6);

void configureImpedance() {
    const double pos_k = 4000.0; 
    const double ori_k = 50.0;
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

    g_my_robot = std::make_unique<urcl::ExampleRobotWrapper>(robot_ip, OUTPUT_RECIPE, INPUT_RECIPE, true, "external_control.urp", SCRIPT_FILE);
    if (!g_my_robot->isHealthy()) {
        URCL_LOG_ERROR("Failed to connect to robot.");
        return 1;
    }

    pinocchio::Model model;
    pinocchio::urdf::buildModel(ur5e_model_path, model);
    pinocchio::Data data(model);
    int ee_frame_id = model.getFrameId("tool0");

    configureImpedance();
    PJRealtimeLog logger("127.0.0.1", 9870);
    
    auto loop_start_time = std::chrono::steady_clock::now();
    int loop_count = 0;
    int log_cnt = 0;

    bool initial_captured = false;
    Eigen::Vector3d x_start, x_des;
    Eigen::Quaterniond quat_des;
    double t_traj = 0.0;
    const double dt = 0.002; 
    
    g_my_robot->getUrDriver()->startRTDECommunication();
    URCL_LOG_INFO("Impedance Control Loop Started.");

    while (true) {
        auto data_pkg = g_my_robot->getUrDriver()->getDataPackage();
        if (!data_pkg) continue;

        urcl::vector6d_t q_raw, qd_raw;
        data_pkg->getData("actual_q", q_raw);
        data_pkg->getData("actual_qd", qd_raw);
        for(int i=0; i<6; i++) { q[i] = q_raw[i]; v[i] = qd_raw[i]; }

        pinocchio::forwardKinematics(model, data, q, v);
        pinocchio::updateFramePlacements(model, data);
        
        Eigen::MatrixXd J(6, 6);
        pinocchio::computeJointJacobians(model, data, q);
        pinocchio::getFrameJacobian(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);

        pinocchio::SE3 curr_pose = data.oMf[ee_frame_id];
        Eigen::Vector3d x_curr = curr_pose.translation();
        Eigen::Matrix3d R_curr = curr_pose.rotation();

        if (!initial_captured) {
            std::ifstream infile(INIT_POS_FILE);
            if (infile.is_open()) {
                urcl::vector6d_t q_init;
                for (int i = 0; i < 6; ++i) {
                    if (!(infile >> q_init[i])) {
                        x_start = x_curr;
                        quat_des = Eigen::Quaterniond(R_curr);
                        initial_captured = true;
                        break;
                    }
                }
                if (!initial_captured) {
                    Eigen::VectorXd q_vec(6);
                    for (int i = 0; i < 6; ++i) q_vec[i] = q_init[i];
                    pinocchio::Data data_init(model);
                    pinocchio::forwardKinematics(model, data_init, q_vec);
                    pinocchio::updateFramePlacements(model, data_init);
                    x_start = data_init.oMf[ee_frame_id].translation();
                    quat_des = Eigen::Quaterniond(data_init.oMf[ee_frame_id].rotation());
                    initial_captured = true;
                }
            } else {
                x_start = x_curr;
                quat_des = Eigen::Quaterniond(R_curr);
                initial_captured = true;
            }
        }

        double r = 0.08; 
        double w = 1.2;  
        x_des = x_start;
        x_des[1] = x_start[1] + r * std::sin(w * t_traj);
        x_des[2] = x_start[2] + r * (1.0 - std::cos(w * t_traj)); 
        t_traj += dt;

        Eigen::Vector3d pos_err = x_curr - x_des;
        Eigen::Quaterniond q_curr(R_curr);
        if (q_curr.dot(quat_des) < 0.0) q_curr.coeffs() *= -1.0;
        Eigen::Quaterniond q_err = q_curr * quat_des.inverse();
        Eigen::Vector3d ori_err = 2.0 * q_err.vec(); 

        Eigen::VectorXd err_6d(6), vel_6d(6);
        err_6d.head(3) = pos_err; 
        err_6d.tail(3) = ori_err;
        vel_6d = J * v; 

        Eigen::VectorXd F_task = -(Kd * err_6d + Dd * vel_6d);
        Eigen::VectorXd tau_dyn = pinocchio::nonLinearEffects(model, data, q, v);
        Eigen::VectorXd tau_cmd = tau_dyn + J.transpose() * F_task;

        for (int i = 0; i < 6; i++) {
            tau_cmd[i] = std::max(-SAFE_TORQUE_LIMITS[i], std::min(SAFE_TORQUE_LIMITS[i], tau_cmd[i]));
            target_torques[i] = tau_cmd[i];
        }

        if (!g_my_robot->getUrDriver()->writeJointCommand(target_torques, urcl::comm::ControlMode::MODE_TORQUE)) {
            break;
        }

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

        loop_count++;
        if (loop_count >= 500) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - loop_start_time;
            double frequency = loop_count / elapsed.count();
            URCL_LOG_INFO("Control Frequency: %.2f Hz", frequency);
            loop_count = 0;
            loop_start_time = now;
        }
    }
    return 0;
}
