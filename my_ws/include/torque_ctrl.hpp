#ifndef TORQUE_CTRL_HPP
#define TORQUE_CTRL_HPP

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
#include <yaml-cpp/yaml.h>

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

struct Config {
    std::string robot_ip;
    std::string ur5e_model_path;
    std::string init_pos_file;
    std::string rtde_output_recipe;
    std::string rtde_input_recipe;
    std::string script_file;

    Eigen::Matrix<double, 6, 1> stiffness_diag;
    Eigen::Matrix<double, 6, 1> damping_ratio;
    urcl::vector6d_t torque_limits;

    double circle_radius;
    double circle_omega;
    double control_dt;

    void loadSysConfig(const std::string& sys_cfg) {
        YAML::Node path_node = YAML::LoadFile(sys_cfg);
        ur5e_model_path = path_node["paths"]["ur5e_model"].as<std::string>();
        init_pos_file = path_node["paths"]["init_pos_file"].as<std::string>();
        rtde_output_recipe = path_node["paths"]["rtde_output_recipe"].as<std::string>();
        rtde_input_recipe = path_node["paths"]["rtde_input_recipe"].as<std::string>();
        script_file = path_node["paths"]["external_script"].as<std::string>();
        robot_ip = path_node["robot"]["ip"].as<std::string>();
    }

    void loadCtrlConfig(const std::string& ctrl_cfg) {
        YAML::Node ctrl_node = YAML::LoadFile(ctrl_cfg);
        
        // 阻抗控制器配置
        YAML::Node imp_node = ctrl_node["impedance_controller"];
        std::vector<double> stiff = imp_node["stiffness"].as<std::vector<double>>();
        std::vector<double> damp = imp_node["damping_ratio"].as<std::vector<double>>();
        
        // 安全配置
        std::vector<double> limits = ctrl_node["safety"]["torque_limits"].as<std::vector<double>>();

        for(int i=0; i<6; ++i) {
            stiffness_diag[i] = stiff[i];
            damping_ratio[i] = damp[i];
            torque_limits[i] = limits[i];
        }

        // 轨迹与周期
        YAML::Node traj_node = ctrl_node["trajectory"];
        circle_radius = traj_node["circle_radius"].as<double>();
        circle_omega = traj_node["circle_omega"].as<double>();
        control_dt = traj_node["control_dt"].as<double>();
    }
};

#endif
