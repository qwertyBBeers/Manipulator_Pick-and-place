"""MoveIt2 launch for the two-robot relay demo -- new file, does not touch
rb5_binpicking/launch/binpicking.launch.py.

Runs two complete, independent copies of the existing single-robot MoveIt
stack (robot_state_publisher + move_group + trajectory_bridge + scene_setup),
each wrapped in its own ROS2 namespace ("robot_a" / "robot_b"). Most of those
nodes only ever use *relative* topic names (isaac_joint_commands,
joint_states, etc.), so namespacing alone is enough for them -- no source
changes needed. trajectory_bridge.py is the one exception: its two action
names (arm FollowJointTrajectory, gripper GripperCommand) are hardcoded as
ROS *global* names, and empirically, rclpy's ActionServer does not apply
launch-time remapping to an already-global name (confirmed with an isolated
test node) -- so two copies would collide on the same global action. Rather
than edit that file, this launch runs a fork
(namespaced_trajectory_bridge.py, this directory) whose only diff is those
two constants changed from global to relative, so each copy correctly lands
under its own robot_a/robot_b namespace.

Skipped vs. binpicking.launch.py: rviz2, depth_to_pointcloud, rqt_image_view,
pcd_correction_tf -- this scene has no cameras (see dual_binpicking_scene.py).

Isaac Sim scene must already be running:
  ~/isaacsim/python.sh ~/asl_ws/Manipulator/lerobot_ws/dual_robot/dual_binpicking_scene.py

Usage:
  ros2 launch <this file>
"""

import os
import xml.etree.ElementTree as ET
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from moveit_configs_utils import MoveItConfigsBuilder


GRIPPER_JOINTS_FIXED_FOR_MOVEIT = {
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
}


def _fix_gripper_joints_for_moveit(robot_description: str) -> str:
    root = ET.fromstring(robot_description)
    for joint in root.findall("joint"):
        if joint.attrib.get("name") not in GRIPPER_JOINTS_FIXED_FOR_MOVEIT:
            continue
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics", "safety_controller", "calibration"):
            for element in list(joint.findall(tag)):
                joint.remove(element)
    return ET.tostring(root, encoding="unicode")


def _package_file(package_name: str, *relative_path_parts: str) -> str:
    try:
        return os.path.join(get_package_share_directory(package_name), *relative_path_parts)
    except PackageNotFoundError:
        return ""


def _robot_group(namespace: str) -> GroupAction:
    use_sim_time = {"use_sim_time": True}

    moveit_config = (
        MoveItConfigsBuilder("rbpodo")
        .robot_description(
            file_path="config/rbpodo.urdf.xacro",
            mappings={
                "model_id": "rb5_850e",
                "use_fake_hardware": "true",
                "fake_sensor_commands": "false",
                "cb_simulation": "Simulation",
                "robot_ip": "127.0.0.1",
            },
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .to_moveit_configs()
    )

    urdf_path = _package_file("rb5_isaac", "urdf", "rb5_with_tools.urdf") or os.path.expanduser(
        "~/asl_ws/Manipulator/rb5_isaac/urdf/rb5_with_tools.urdf"
    )
    with open(urdf_path) as f:
        full_urdf = f.read()
    full_robot_description = {"robot_description": full_urdf}
    moveit_robot_description = {"robot_description": _fix_gripper_joints_for_moveit(full_urdf)}

    srdf_path = _package_file("rb5_binpicking", "config", "rb5_binpicking_with_tools.srdf") or os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "rb5_binpicking", "config", "rb5_binpicking_with_tools.srdf"
        )
    )
    with open(srdf_path) as f:
        moveit_robot_description_semantic = {"robot_description_semantic": f.read()}

    sensors_3d_path = _package_file("rb5_binpicking", "config", "sensors_3d.yaml")
    sensors_3d_params = []
    if sensors_3d_path and os.path.isfile(sensors_3d_path):
        import yaml
        with open(sensors_3d_path) as f:
            sensors_3d_params = [yaml.safe_load(f)]

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_robot_description,
            moveit_robot_description_semantic,
            *sensors_3d_params,
            use_sim_time,
            # rbpodo_moveit_config/config/kinematics.yaml ships the MoveIt Setup
            # Assistant default of 0.005s -- five milliseconds for KDL's
            # numerical IK, per call. Measured effect: compute_cartesian_path
            # returned fractions of 0.00, 0.50, 0.25 and 0.88 for the *same* 4cm
            # straight-up move, and disabling collision checking sometimes made
            # the fraction worse -- the signature of an IK solver randomly
            # running out of time, not of anything geometric. It is also the
            # likely source of the constant -31 NO_IK_SOLUTION from Pilz LIN.
            # Overridden here rather than in rbpodo_ros2/, which this workspace
            # does not modify; these keys come after to_dict() so they win.
            {"robot_description_kinematics.mainpulation.kinematics_solver_timeout": 0.05},
            {"robot_description_kinematics.mainpulation.kinematics_solver_attempts": 3},
            # KDL -> LMA. KDL's Newton-Raphson restarts from a random seed when
            # it stalls, so the same grasp pose comes back as a different arm
            # configuration each call -- which is exactly the observed problem:
            # tool error jumped between 1.6mm and 12.0mm for identical commands,
            # and grasps landing within 3.4mm held the block while ones at 5.4mm
            # or worse dropped it. Levenberg-Marquardt damps its steps instead of
            # restarting, so it converges to the same branch far more
            # consistently. moveit_lma_kinematics_plugin ships with Humble, so
            # this needs nothing installed.
            #
            # Better still would be pick_ik (`sudo apt install ros-humble-pick-ik`,
            # then set this to "pick_ik/PickIkPlugin") -- MoveIt 2's current
            # recommendation and gradient-based rather than Jacobian-iterative.
            # Not done here because installing it needs a sudo password.
            {"robot_description_kinematics.mainpulation.kinematics_solver":
                "lma_kinematics_plugin/LMAKinematicsPlugin"},
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

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[full_robot_description, use_sim_time],
    )

    # dual_scene_setup.py (this dir), not rb5_binpicking's scene_setup.py: that
    # one publishes to the absolute /collision_object (move_group listens on the
    # namespaced relative one, so it never arrived) and describes the
    # single-robot bin layout, which has no handoff tray in it. Run as a script
    # for the same reason as the bridge below -- it isn't an installed
    # package executable.
    scene_setup = ExecuteProcess(
        cmd=[
            "python3",
            os.path.join(os.path.dirname(__file__), "dual_scene_setup.py"),
            "--robot", namespace,
            "--ros-args",
            "-r", f"__ns:=/{namespace}",
            "-r", "__node:=dual_scene_setup",
            "-p", "use_sim_time:=true",
        ],
        output="screen",
    )

    # See module docstring: namespaced_trajectory_bridge.py (this dir) is a
    # fork of rb5_isaac's trajectory_bridge with its two action names made
    # relative, run directly as a script rather than a Node(package=...)
    # since it isn't installed as a colcon package executable. use_sim_time
    # isn't passed via ros-args here (the original Node form used a
    # parameters=[] list) since the script doesn't declare that parameter
    # explicitly -- ros2 accepts unknown --params-file entries, but simplest
    # to just rely on /clock being published and default node behavior.
    trajectory_bridge = ExecuteProcess(
        cmd=[
            "python3",
            os.path.join(os.path.dirname(__file__), "namespaced_trajectory_bridge.py"),
            "--ros-args",
            "-r", f"__ns:=/{namespace}",
            "-r", "__node:=trajectory_bridge",
            "-p", "use_sim_time:=true",
            # The arm lags the commanded trajectory most while accelerating out
            # of rest, and the default 0.2s grace expires inside that window.
            # (The lag itself is kept small by the low velocity scaling in
            # relay_pick_place.py -- this only covers the startup transient.)
            "-p", "path_tolerance_grace_period:=1.0",
            # Raised from the upstream 0.08. This check exists to catch a
            # trajectory that has gone wild (the spline-extrapolation bug that
            # once produced sudden fast rotation), which shows up as a huge
            # error -- but at 0.08/0.15 it was instead tripping on benign
            # lag during slow descents, and each abort left the arm
            # half-executed where it clipped the block and ruined the grasp.
            # Final positioning accuracy is enforced separately and precisely
            # by goal_tolerance below, which is at its tight upstream value.
            # Raised again, 0.6 -> 1.2. What trips it is always the same thing:
            # the base joint is the one that has to swing the whole arm, and it
            # simply cannot track the commanded profile even at low velocity
            # scaling (measured mid-descent: base cur=-0.7931 cmd=-0.1700, every
            # other joint inside 0.08rad). Aborting there is the worst possible
            # response -- it leaves the arm half-way through a descent, right
            # where it can clip the block. The base does catch up by the end, so
            # let the trajectory finish; goal_tolerance below (0.02rad, tight)
            # and relay_pick_place.py's FK check are what actually decide
            # whether the arm ended up where it was asked to.
            "-p", "path_tolerance:=1.2",
            # goal_tolerance stays at the upstream 0.02rad. It was briefly
            # raised to 0.05 while the arm could not settle accurately, but
            # that turned out to be an under-converged PhysX solver (fixed in
            # dual_binpicking_scene.py) -- and 0.05rad per joint is several cm
            # of TCP error at this reach, enough for the gripper to close
            # beside the 42mm block instead of on it. Left explicit as a
            # reminder not to loosen it again to paper over a settling bug.
            "-p", "goal_tolerance:=0.02",
            "-p", "stopped_velocity_tolerance:=0.02",
            # 3.0 -> 12.0. Same root cause as path_tolerance above: the base
            # joint lags the commanded profile by up to ~0.6rad and needs
            # several seconds past the nominal trajectory end to close that gap.
            # At 3s the long source->handoff transit aborted with CONTROL_FAILED
            # on all three planner fallbacks, and each partially-executed abort
            # shook the block out of the gripper. This only grants more *time*;
            # goal_tolerance still decides whether the pose was actually reached.
            "-p", "goal_time_tolerance:=12.0",
            # Gripper ramp/settle times. Measured on this scene: from fully
            # closed, the knuckle drive needs ~1.8s to actually reach 0 rad,
            # but the upstream defaults read the angle back after only
            # 0.5+0.3s -- so every open reported reached_goal=False while the
            # fingers were still moving and doing fine. That false negative is
            # what made "release failed" look like a real failure earlier.
            # Closing measured ~2.4s, so its 2.0s ramp was left alone at first.
            #
            # These were briefly inflated (2.0/3.5/1.2) to compensate for the
            # bridge timing its gripper ramp on the wall clock while the ~1.8s
            # drive response is in simulated seconds -- the simulator runs at
            # 0.77 real-time with one camera and 0.58 with three, so the ramp
            # kept shrinking underneath the numbers and picks started missing
            # with "reached_goal=True stalled=False" and nothing in the hand.
            # That is fixed at the source now (the ramp runs on the ROS clock,
            # like the trajectory path already did), so these are simulated
            # seconds and can go back to being sized against the measured drive
            # response instead of against the frame rate.
            "-p", "gripper_open_duration:=1.5",
            "-p", "gripper_close_duration:=2.5",
            "-p", "gripper_settle_duration:=1.0",
        ],
        output="screen",
    )

    return GroupAction([
        PushRosNamespace(namespace),
        static_tf,
        robot_state_publisher,
        move_group,
        scene_setup,
        trajectory_bridge,
    ])


def generate_launch_description():
    return LaunchDescription([
        _robot_group("robot_a"),
        _robot_group("robot_b"),
    ])
