"""
빈피킹 MoveIt2 launch — rb5_binpicking 패키지.

Isaac Sim 씬은 별도 터미널에서 먼저 실행:
  ~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_binpicking/scripts/binpicking_scene.py

이후 이 launch 파일 실행:
  ros2 launch rb5_binpicking binpicking.launch.py
"""

import os
import xml.etree.ElementTree as ET
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from moveit_configs_utils import MoveItConfigsBuilder


def _warn_if_pcd_offset_nonzero(context, *args, **kwargs):
    """pcd_offset_x/y/z are an empirical patch on top of
    camera_depth_optical_frame, not camera calibration (README2.md §10 —
    the root TF/frame mismatch that made this "necessary" is still
    unresolved). Defaults are 0.0; warn loudly whenever a caller overrides
    them so the patch doesn't go unnoticed again.
    """
    values = {
        name: float(LaunchConfiguration(name).perform(context))
        for name in ("pcd_offset_x", "pcd_offset_y", "pcd_offset_z")
    }
    if any(abs(v) > 1e-9 for v in values.values()):
        return [
            LogInfo(
                msg=(
                    f"[binpicking.launch.py] WARNING: non-zero pcd_offset override {values} "
                    "applied to camera_depth_optical_frame -> camera_depth_points_frame. "
                    "This is an empirical patch, NOT camera calibration — see README2.md §10."
                )
            )
        ]
    return []


GRIPPER_JOINTS_FIXED_FOR_MOVEIT = {
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
}


def _fix_gripper_joints_for_moveit(robot_description: str) -> str:
    """Keep tool collision geometry but remove gripper active joints for MoveIt."""
    root = ET.fromstring(robot_description)
    for joint in root.findall("joint"):
        if joint.attrib.get("name") not in GRIPPER_JOINTS_FIXED_FOR_MOVEIT:
            continue
        joint.set("type", "fixed")
        for tag in (
            "axis",
            "limit",
            "mimic",
            "dynamics",
            "safety_controller",
            "calibration",
        ):
            for element in list(joint.findall(tag)):
                joint.remove(element)
    return ET.tostring(root, encoding="unicode")


def _package_file(package_name: str, *relative_path_parts: str) -> str:
    try:
        path = os.path.join(
            get_package_share_directory(package_name),
            *relative_path_parts,
        )
    except PackageNotFoundError:
        path = ""
    return path


def generate_launch_description():
    model_id_arg = DeclareLaunchArgument("model_id", default_value="rb5_850e")
    rviz_config_arg = DeclareLaunchArgument("rviz_config", default_value="moveit.rviz")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")
    # Defaults are 0.0 — this is a diagnostic knob, not calibration. See
    # _warn_if_pcd_offset_nonzero() above and README2.md §10.
    pcd_offset_x_arg = DeclareLaunchArgument("pcd_offset_x", default_value="0.0")
    pcd_offset_y_arg = DeclareLaunchArgument("pcd_offset_y", default_value="0.0")
    pcd_offset_z_arg = DeclareLaunchArgument("pcd_offset_z", default_value="0.0")
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    moveit_config = (
        MoveItConfigsBuilder("rbpodo")
        .robot_description(
            file_path="config/rbpodo.urdf.xacro",
            mappings={
                "model_id": LaunchConfiguration("model_id"),
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
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    # rb5_with_tools.urdf contains the actual mounted Robotiq gripper and camera.
    _urdf_path = _package_file("rb5_isaac", "urdf", "rb5_with_tools.urdf")
    if not os.path.isfile(_urdf_path):
        # fallback: source-tree path for an uninstalled development checkout
        _urdf_path = os.path.expanduser(
            "~/asl_ws/Manipulator/rb5_isaac/urdf/rb5_with_tools.urdf"
        )
    with open(_urdf_path, "r") as _f:
        _full_urdf = _f.read()

    full_robot_description = {"robot_description": _full_urdf}
    moveit_robot_description = {
        "robot_description": _fix_gripper_joints_for_moveit(_full_urdf)
    }

    _srdf_path = _package_file(
        "rb5_binpicking",
        "config",
        "rb5_binpicking_with_tools.srdf",
    )
    if not os.path.isfile(_srdf_path):
        _srdf_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                "rb5_binpicking_with_tools.srdf",
            )
        )
    with open(_srdf_path, "r") as _f:
        _tool_srdf = _f.read()
    moveit_robot_description_semantic = {
        "robot_description_semantic": _tool_srdf
    }

    sensors_3d_path = _package_file("rb5_binpicking", "config", "sensors_3d.yaml")
    if not os.path.isfile(sensors_3d_path):
        sensors_3d_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                "sensors_3d.yaml",
            )
        )
    with open(sensors_3d_path, "r") as _f:
        sensors_3d_config = yaml.safe_load(_f)

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_robot_description,
            moveit_robot_description_semantic,
            sensors_3d_config,
            use_sim_time,
            {"trajectory_execution.allowed_start_tolerance": 0.0},
            {"trajectory_execution.allowed_execution_duration_scaling": 2.0},
            {"publish_planning_scene": True},
            {"publish_geometry_updates": True},
            {"publish_state_updates": True},
            {"publish_transforms_updates": True},
        ],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "link0"],
    )

    pcd_correction_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            LaunchConfiguration("pcd_offset_x"),
            LaunchConfiguration("pcd_offset_y"),
            LaunchConfiguration("pcd_offset_z"),
            "0",
            "0",
            "0",
            "camera_depth_optical_frame",
            "camera_depth_points_frame",
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[full_robot_description, use_sim_time],
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare("rbpodo_moveit_config"),
        "config",
        LaunchConfiguration("rviz_config"),
    ])
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_robot_description,
            moveit_robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            use_sim_time,
        ],
    )

    # source/dest bin collision geometry → planning scene, so RViz shows the
    # boxes as soon as the stack comes up (independent of the pick-place expert).
    scene_setup = Node(
        package="rb5_binpicking",
        executable="scene_setup.py",
        name="scene_setup",
        output="screen",
        parameters=[use_sim_time],
    )

    # rb5_isaac 의 trajectory_bridge 재사용
    trajectory_bridge = Node(
        package="rb5_isaac",
        executable="trajectory_bridge",
        name="trajectory_bridge",
        output="screen",
        parameters=[use_sim_time],
    )

    depth_to_pointcloud = Node(
        package="rb5_binpicking",
        executable="depth_to_pointcloud.py",
        name="depth_to_pointcloud",
        output="screen",
        parameters=[
            use_sim_time,
            {
                "depth_topic": "/camera/depth/image_rect_raw",
                "camera_info_topic": "/camera/depth/camera_info",
                "points_topic": "/camera/depth/points",
                "pointcloud_frame": "camera_depth_points_frame",
                "stride": 4,
                "max_range": 1.2,
            },
        ],
    )

    rqt_color = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_color",
        arguments=["/camera/color/image_raw"],
        parameters=[use_sim_time],
    )

    rqt_depth = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_depth",
        arguments=["/camera/depth/image_rect_raw"],
        parameters=[use_sim_time],
    )

    return LaunchDescription([
        model_id_arg,
        rviz_config_arg,
        use_sim_time_arg,
        pcd_offset_x_arg,
        pcd_offset_y_arg,
        pcd_offset_z_arg,
        OpaqueFunction(function=_warn_if_pcd_offset_nonzero),
        static_tf,
        pcd_correction_tf,
        robot_state_publisher,
        move_group,
        rviz,
        scene_setup,
        trajectory_bridge,
        depth_to_pointcloud,
        rqt_color,
        rqt_depth,
    ])
