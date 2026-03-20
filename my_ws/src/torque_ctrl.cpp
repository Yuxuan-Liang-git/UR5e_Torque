#include "torque_ctrl.hpp"

// --- 全局变量 ---
std::unique_ptr<urcl::ExampleRobotWrapper> g_my_robot;
Eigen::VectorXd q(6), v(6);
urcl::vector6d_t target_torques;
Config cfg;

Eigen::MatrixXd Kd = Eigen::MatrixXd::Zero(6, 6);
Eigen::MatrixXd Dd = Eigen::MatrixXd::Zero(6, 6);

// 从配置文件中读取设置的控制器参数
void configureImpedance() {
    for (int i = 0; i < 6; ++i) {
        Kd(i, i) = cfg.stiffness_diag[i];
        double m_approx = (i < 3) ? 5.0 : 0.5; 
        Dd(i, i) = 2.0 * cfg.damping_ratio[i] * std::sqrt(Kd(i, i) * m_approx);
    }
}

int main(int argc, char* argv[]) {
    urcl::setLogLevel(urcl::LogLevel::INFO);
    
    try {
        cfg.loadSysConfig("./my_ws/config/sys_config.yaml");
        cfg.loadCtrlConfig("./my_ws/config/ctrl_config.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return 1;
    }

    std::string robot_ip = (argc > 1) ? argv[1] : cfg.robot_ip;

    g_my_robot = std::make_unique<urcl::ExampleRobotWrapper>(
        robot_ip, cfg.rtde_output_recipe, cfg.rtde_input_recipe, true, "external_control.urp", cfg.script_file
    );
    if (!g_my_robot->isHealthy()) {
        URCL_LOG_ERROR("Failed to connect to robot.");
        return 1;
    }

    pinocchio::Model model;
    pinocchio::urdf::buildModel(cfg.ur5e_model_path, model);
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
    const double dt = cfg.control_dt; 
    
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
            std::ifstream infile(cfg.init_pos_file);
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
                    // 使用Pinocchio计算初始末端位置和姿态
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

        x_des = x_start;
        x_des[0] = x_start[0] + cfg.circle_radius * std::sin(cfg.circle_omega * t_traj);
        x_des[1] = x_start[1] + cfg.circle_radius * (1.0 - std::cos(cfg.circle_omega * t_traj));
        t_traj += dt;

        Eigen::Vector3d pos_err = x_curr - x_des;
        // 把当前旋转矩阵转换为四元数，并确保与目标四元数在同一半空间（q与-q表示相同旋转）
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
            tau_cmd[i] = std::max(-cfg.torque_limits[i], std::min(cfg.torque_limits[i], tau_cmd[i]));
            target_torques[i] = tau_cmd[i];
        }

        if (!g_my_robot->getUrDriver()->writeJointCommand(target_torques, urcl::comm::ControlMode::MODE_TORQUE)) {
            break;
        }

        if (++log_cnt % 5 == 0) {
            std::stringstream ss;
            ss << "{"
               << "\"pos_curr\":{\"x\":" << x_curr[0] << ",\"y\":" << x_curr[1] << ",\"z\":" << x_curr[2] << "},"
               << "\"pos_des\":{\"x\":" << x_des[0] << ",\"y\":" << x_des[1] << ",\"z\":" << x_des[2] << "}"
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
