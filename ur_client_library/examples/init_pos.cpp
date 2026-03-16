#include <ur_client_library/ur/dashboard_client.h>
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/types.h>
#include <ur_client_library/example_robot_wrapper.h>
#include <ur_client_library/rtde/data_package.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <fstream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

using namespace urcl;

const std::string DEFAULT_ROBOT_IP = "192.168.56.101";
const std::string OUTPUT_RECIPE = "examples/resources/rtde_output_recipe.txt";
const std::string INPUT_RECIPE = "examples/resources/rtde_input_recipe.txt";
const std::string SAVE_FILE = "init_pos.txt";

std::unique_ptr<ExampleRobotWrapper> g_my_robot;

// 非阻塞读取按键
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

void sendFreedriveMessageOrDie(const control::FreedriveControlMessage freedrive_action) {
    bool ret = g_my_robot->getUrDriver()->writeFreedriveControlMessage(freedrive_action);
    if (!ret) {
        URCL_LOG_ERROR("Could not send freedrive command. Is there an external_control program running on the robot?");
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    urcl::setLogLevel(urcl::LogLevel::INFO);
    std::string robot_ip = (argc > 1) ? argv[1] : DEFAULT_ROBOT_IP;

    // 1. 初始化机械臂驱动
    g_my_robot = std::make_unique<ExampleRobotWrapper>(robot_ip, OUTPUT_RECIPE, INPUT_RECIPE, true, "external_control.urp");

    if (!g_my_robot->isHealthy()) {
        URCL_LOG_ERROR("Failed to connect to robot.");
        return 1;
    }

    // 2. 启动 RTDE 通讯以获取数据
    g_my_robot->getUrDriver()->startRTDECommunication();

    URCL_LOG_INFO("Starting freedrive mode. Press 's' to save current pose, 'q' to quit.");
    sendFreedriveMessageOrDie(control::FreedriveControlMessage::FREEDRIVE_START);

    bool running = true;
    while (running) {
        sendFreedriveMessageOrDie(control::FreedriveControlMessage::FREEDRIVE_NOOP);

        if (kbhit()) {
            char c = getchar();
            if (c == 's') {
                // 从 RTDE 数据包读取当前关节角度
                auto data_pkg = g_my_robot->getUrDriver()->getDataPackage();
                if (data_pkg) {
                    vector6d_t q;
                    if (data_pkg->getData("actual_q", q)) {
                        std::ofstream outfile(SAVE_FILE);
                        if (outfile.is_open()) {
                            for (size_t i = 0; i < 6; ++i) {
                                outfile << q[i] << (i == 5 ? "" : " ");
                            }
                            outfile << std::endl;
                            outfile.close();
                            URCL_LOG_INFO("Successfully saved current pose to %s", SAVE_FILE.c_str());
                            URCL_LOG_INFO("Pose: %.4f %.4f %.4f %.4f %.4f %.4f", q[0], q[1], q[2], q[3], q[4], q[5]);
                        } else {
                            URCL_LOG_ERROR("Failed to open file %s for writing", SAVE_FILE.c_str());
                        }
                    } else {
                        URCL_LOG_ERROR("Failed to get 'actual_q' from data package. Check if it's in the recipe.");
                    }
                } else {
                    URCL_LOG_ERROR("Failed to get data package from robot.");
                }
            } else if (c == 'q') {
                running = false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    URCL_LOG_INFO("Stopping freedrive mode");
    sendFreedriveMessageOrDie(control::FreedriveControlMessage::FREEDRIVE_STOP);

    return 0;
}
