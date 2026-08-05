from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

# Must match rb5_isaac/rb5_isaac/trajectory_bridge.py's JOINT_NAMES order.
DEFAULT_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


@RobotConfig.register_subclass("rb5_850e")
@dataclass
class RB5850EConfig(RobotConfig):
    """RB5-850E (Rainbow Robotics), reached through a ZMQ bridge to a ROS2 node
    (see Manipulator/lerobot_ws/ros_bridge/rb5_lerobot_bridge.py) that talks to
    either Isaac Sim or the real arm via the same /joint_states, /isaac_joint_commands,
    /gripper_joint_commands topics used by rb5_isaac/trajectory_bridge.py.
    """

    # ZMQ REQ/REP endpoint of the ROS2-side bridge process.
    bridge_host: str = "localhost"
    bridge_port: int = 5555
    # Timeout for a single request/response round trip.
    timeout_s: float = 2.0

    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_NAMES))

    # Camera stream keys the bridge is expected to serve (relayed from ROS image
    # topics, e.g. /camera/color/image_raw -> "color"). Populated per observation,
    # not through lerobot's local Camera/CameraConfig classes since the frames
    # originate on the ROS2 side, not on this machine's local device bus.
    camera_keys: list[str] = field(default_factory=lambda: ["color"])
    camera_height: int = 480
    camera_width: int = 640

    # Safety cap, in radians, on how far a single send_action may move a joint
    # from its last known position (mirrors ensure_safe_goal_position elsewhere
    # in lerobot's robot implementations).
    max_relative_target: float | None = None
