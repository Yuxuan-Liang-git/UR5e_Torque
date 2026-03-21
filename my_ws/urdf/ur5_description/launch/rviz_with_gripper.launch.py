from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")
    rvizconfig = LaunchConfiguration("rvizconfig")

    urdf_xacro = PathJoinSubstitution(
        [FindPackageShare("ur5_description"), "urdf", "ur5_robotiq85_gripper.urdf.xacro"]
    )

    robot_description = ParameterValue(Command(["xacro ", urdf_xacro]), value_type=str)

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description},
        ],
        condition=IfCondition(gui),
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description},
        ],
        condition=UnlessCondition(gui),
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_description": robot_description},
        ],
        output="screen",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rvizconfig],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "rvizconfig",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ur5_description"), "rviz", "ur5.rviz"]
                ),
            ),
            joint_state_publisher_gui,
            joint_state_publisher,
            robot_state_publisher,
            rviz2,
        ]
    )
