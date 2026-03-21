from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model = LaunchConfiguration("model")
    model_dir = LaunchConfiguration("model_dir")
    rvizconfig = LaunchConfiguration("rvizconfig")
    gui = LaunchConfiguration("gui")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Treat robot_description as a raw string to avoid YAML parsing issues.
    robot_description = ParameterValue(Command(["xacro ", model]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        cwd=model_dir,
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        cwd=model_dir,
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(gui),
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        cwd=model_dir,
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": use_sim_time},
        ],
        condition=UnlessCondition(gui),
        output="screen",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        cwd=model_dir,
        arguments=["-d", rvizconfig],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="/home/amdt/ur_force_ws/my_ws/urdf/ur5_gripper.mujoco.urdf",
            ),
            DeclareLaunchArgument(
                "model_dir",
                default_value="/home/amdt/ur_force_ws/my_ws/urdf",
            ),
            DeclareLaunchArgument(
                "rvizconfig",
                default_value="/home/amdt/ur_force_ws/my_ws/urdf/robotiq_description/rviz/robotiq_85_ros2.rviz",
            ),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            robot_state_publisher,
            joint_state_publisher_gui,
            joint_state_publisher,
            rviz2,
        ]
    )
