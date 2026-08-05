"""
MoveIt2 launch for RB5-850E with Isaac Sim as hardware backend.

Architecture:
  Isaac Sim (02_isaac_rb5_scene.py)
    ↑ publishes /joint_states, /tf
    ↓ subscribes /isaac_joint_commands
  trajectory_bridge (03_trajectory_bridge.py)
    ↑ FollowJointTrajectory action server ← MoveIt2 move_group
    ↓ forwards commands to /isaac_joint_commands
  MoveIt2 (move_group + RViz2)
  
    reads /joint_states from Isaac Sim

Usage:
  # Terminal 1 — Isaac Sim
  ~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_isaac/scripts/02_isaac_rb5_scene.py

  # Terminal 2 — MoveIt2 stack
  ros2 launch rb5_isaac moveit_isaac.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    model_id_arg = DeclareLaunchArgument(
        "model_id",
        default_value="rb5_850e",
        description="RB model ID",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="moveit.rviz",
        description="RViz config file",
    )

    model_id = LaunchConfiguration("model_id")

    # MoveIt2 config — use fake hardware only for robot_description/SRDF,
    # actual execution is handled by trajectory_bridge → Isaac Sim
    moveit_config = (
        MoveItConfigsBuilder("rbpodo")
        .robot_description(
            file_path="config/rbpodo.urdf.xacro",
            mappings={
                "model_id": model_id,
                "use_fake_hardware": "true",
                "fake_sensor_commands": "false",
                "cb_simulation": "Simulation",
                "robot_ip": "127.0.0.1",
            },
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    # move_group node
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            # Isaac Sim의 joint_states를 시작 상태로 사용 — 검증 허용 오차 완화
            {"trajectory_execution.allowed_start_tolerance": 0.0},
            {"trajectory_execution.allowed_execution_duration_scaling": 2.0},
            {"publish_planning_scene": True},
            {"publish_geometry_updates": True},
            {"publish_state_updates": True},
            {"publish_transforms_updates": True},
        ],
    )

    # Static TF: world → link0
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "link0"],
    )

    # robot_state_publisher — reads /joint_states (from Isaac Sim)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # RViz2
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("rbpodo_moveit_config"), "config",
         LaunchConfiguration("rviz_config")]
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Trajectory bridge — FollowJointTrajectory ↔ Isaac Sim
    trajectory_bridge = Node(
        package="rb5_isaac",
        executable="trajectory_bridge",
        name="trajectory_bridge",
        output="screen",
    )

    # Camera image viewer (color + depth)
    rqt_color = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_color_view",
        arguments=["/camera/color/image_raw"],
        output="log",
    )

    rqt_depth = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_depth_view",
        arguments=["/camera/depth/image_rect_raw"],
        output="log",
    )

    return LaunchDescription([
        model_id_arg,
        rviz_config_arg,
        static_tf,
        robot_state_publisher,
        move_group,
        rviz,
        trajectory_bridge,
        rqt_color,
        rqt_depth,
    ])
