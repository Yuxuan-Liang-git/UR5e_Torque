/**
 * @file torque_smc.cpp
 * @brief C++ implementation of Joint-space PD + SMC torque control using ur_client_library
 *
 * This file replaces the Python RTDE interface with the official C++ ur_client_library (URCL),
 * directly exposing and leveraging the `actual_current_as_torque` RTDE output variable.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <csignal>
#include <memory>
#include <Eigen/Dense>

// UR Client Library includes
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/comm/control_mode.h>

// Note: To compile this code, you will need to modify your CMakeLists.txt to link against 
// ur_client_library, mujoco, and Eigen3.

bool g_running = true;

void signalHandler(int signum) {
    std::cout << "\n[INFO] Interrupt signal received. Stopping..." << std::endl;
    g_running = false;
}

// Controller Parameters
const Eigen::VectorXd KP = (Eigen::VectorXd(6) << 1500.0, 1500.0, 1500.0, 150.0, 150.0, 10.0).finished();
const Eigen::VectorXd KD = (Eigen::VectorXd(6) << 40.0, 40.0, 40.0, 5.0, 5.0, 1.0).finished();
const Eigen::VectorXd TORQUE_LIMITS = (Eigen::VectorXd(6) << 20.0, 20.0, 20.0, 10.0, 10.0, 10.0).finished();

// SMC Parameters
const double LAMBDA_SMC = 10.0;
const Eigen::VectorXd K_SMC = (Eigen::VectorXd(6) << 50.0, 50.0, 50.0, 10.0, 10.0, 10.0).finished();
const Eigen::VectorXd U_SMC = (Eigen::VectorXd(6) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished(); // Set to 0 in Python script originally
const double TANH_BETA = 40.0;

int main(int argc, char* argv[]) {
    // 1. Setup Signal Handler
    std::signal(SIGINT, signalHandler);

    std::string robot_ip = "192.168.56.101";
    if (argc > 1) {
        robot_ip = argv[1];
    }
    
    std::cout << "[INFO] Connecting to UR Robot via ur_client_library... IP: " << robot_ip << std::endl;

    // 2. Define exactly which variables we want from the RTDE data stream
    std::vector<std::string> rtde_recipe = {
        "actual_q", 
        "actual_qd", 
        "actual_current_as_torque", // <-- This is the authentic Torque variable we couldn't get in Python!
        "target_moment"
    };

    // 3. Initialize the UR Driver using UrDriverConfiguration
    urcl::UrDriverConfiguration config;
    config.robot_ip = robot_ip;
    config.script_file = "/opt/ros/humble/share/ur_client_library/resources/external_control.urscript";
    config.output_recipe = rtde_recipe;
    
    std::unique_ptr<urcl::UrDriver> driver;
    try {
        driver = std::make_unique<urcl::UrDriver>(config);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize UR Driver: " << e.what() << std::endl;
        return 1;
    }

    driver->startRTDECommunication();

    std::cout << "[INFO] Control loop started. Running at ~500Hz." << std::endl;

    urcl::vector6d_t actual_q_vec, actual_qd_vec, tau_actual_vec;
    urcl::vector6d_t torque_command = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Initial Targets (Based on your Python script logic)
    Eigen::VectorXd q_target = (Eigen::VectorXd(6) << 1.5326, -1.30735, -2.06397, -1.36756, 1.59805, -1.55037).finished();
    Eigen::VectorXd dq_target = Eigen::VectorXd::Zero(6);

    auto start_time = std::chrono::steady_clock::now();
    double stabilize_time = 1.0;
    
    // 4. Real-time control loop
    while (g_running) {
        // Read data package (blocks until a new 500Hz packet arrives from the robot)
        auto data_pkg = driver->getDataPackage();
        if (data_pkg) {
            // Extract the RTDE data directly
            data_pkg->getData("actual_q", actual_q_vec);
            data_pkg->getData("actual_qd", actual_qd_vec);
            data_pkg->getData("actual_current_as_torque", tau_actual_vec); // The Authentic Torque!
            
            // Convert to Eigen formats
            Eigen::Map<Eigen::VectorXd> q(actual_q_vec.data(), 6);
            Eigen::Map<Eigen::VectorXd> dq(actual_qd_vec.data(), 6);
            Eigen::Map<Eigen::VectorXd> tau_actual(tau_actual_vec.data(), 6);

            // Time calculation
            auto current_time = std::chrono::steady_clock::now();
            double t_now = std::chrono::duration<double>(current_time - start_time).count();

            Eigen::VectorXd tau = Eigen::VectorXd::Zero(6);

            // Compute SMC Control Law (Skeleton mapping)
            if (t_now > stabilize_time) {
                // Tracking errors
                Eigen::VectorXd e_joint = q - q_target;    // In practice, q_target comes from MuJoCo path generator
                Eigen::VectorXd de_joint = dq - dq_target; // dq_target comes from MuJoCo Jacobian logic
                
                // Sliding surface: sigma = de + lambda * e
                Eigen::VectorXd sigma = de_joint + LAMBDA_SMC * e_joint;

                // SMC Output: u_nom + u_smc
                Eigen::VectorXd u_nom = -K_SMC.cwiseProduct(sigma);
                
                // tanh(sigma * TANH_BETA)
                Eigen::VectorXd tanh_term = (sigma * TANH_BETA).array().tanh().matrix();
                Eigen::VectorXd u_smc = -U_SMC.cwiseProduct(tanh_term);
                
                tau = u_nom + u_smc;
            } else {
                // Simple PD Initialization mode to hold pose
                tau = KP.cwiseProduct(q_target - q) + KD.cwiseProduct(dq_target - dq);
            }

            // Clip torque output for safety limits
            for (int i = 0; i < 6; ++i) {
                tau(i) = std::clamp(tau(i), -TORQUE_LIMITS(i), TORQUE_LIMITS(i));
                torque_command[i] = tau(i);
            }

            // --- Send the direct Torque Command using ControlMode::MODE_TORQUE ---
            // Inside URCL, this instructs the robot script to read these targets and apply direct torque.
            driver->writeJointCommand(torque_command, urcl::comm::ControlMode::MODE_TORQUE);

            /* Note: 
            For visualization and UDP logging, you would traditionally publish the 'tau', 'tau_actual', 
            'sigma', 'actual_q' variables down a UDP socket here like your Python UDPLogger does. 
            */
        }
    }

    std::cout << "[INFO] Shutting down RTDE communication..." << std::endl;
    // Stop robot movement safely
    driver->stopControl();

    return 0;
}