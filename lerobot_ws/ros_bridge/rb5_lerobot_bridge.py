#!/usr/bin/env python3
"""ROS2 <-> LeRobot bridge for the RB5-850E.

Runs standalone (no colcon package, no launch file) in a ROS2 Humble sourced
shell under a Python-3.10 interpreter (e.g. the `physical` or `isaaclab` conda
env from the main Manipulator workspace) -- NOT inside the `lerobot` conda env,
which is Python 3.12 and can't import rclpy against Humble's ABI.

Talks over ZMQ REQ/REP (JSON) to lerobot_robot_rb5.RB5850E, which runs inside
the `lerobot` conda env. This lets a LeRobot `Robot` (Python 3.12, no ROS2 on
its PYTHONPATH) drive the arm without either side needing the other's stack.

Publishes /isaac_joint_commands + /gripper_joint_commands directly (same
topics rb5_isaac/trajectory_bridge.py publishes to for Isaac Sim) -- this
bridge takes over that role for teleoperation/rollout, bypassing MoveIt/the
scripted pick-and-place state machine. Don't run both against the same sim/
robot at once; they'll fight over the same command topics.

Usage:
    source /opt/ros/humble/setup.bash
    conda activate physical   # or isaaclab, whichever has this checked out
    pip install pyzmq         # if not already present in that env
    python3 rb5_lerobot_bridge.py --port 5555

Status: scaffold, not yet run against a live robot/sim. The wire protocol
(see lerobot_robot_rb5/robot_rb5.py) and the gripper mimic-joint constants
below (copied from trajectory_bridge.py -- keep them in sync manually if
that file changes, same as the other duplicated grasp-pose constants in
this repo) are the parts most likely to need adjustment once tested live.
"""

from __future__ import annotations

import argparse
import base64
import threading

import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

# Copied from rb5_isaac/rb5_isaac/trajectory_bridge.py -- must stay in sync.
GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GRIPPER_MIMIC = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
KNUCKLE_MAX = 0.8  # rad, fully closed

CAMERA_TOPICS = {"color": "/camera/color/image_raw"}


class RB5Bridge(Node):
    def __init__(self):
        super().__init__("rb5_lerobot_bridge")

        self._joint_pos: dict[str, float] = dict.fromkeys(JOINT_NAMES, 0.0)
        self._gripper_pos = 1.0  # 0=closed, 1=open
        self._images: dict[str, Image] = {}
        self._lock = threading.Lock()

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        for key, topic in CAMERA_TOPICS.items():
            self.create_subscription(Image, topic, lambda msg, key=key: self._on_image(key, msg), 1)

        self._cmd_pub = self.create_publisher(JointState, "isaac_joint_commands", 10)
        self._gripper_pub = self.create_publisher(JointState, "gripper_joint_commands", 10)

    def _on_joint_states(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                if name in self._joint_pos:
                    self._joint_pos[name] = pos
                elif name == GRIPPER_JOINTS[0]:
                    self._gripper_pos = 1.0 - pos / KNUCKLE_MAX

    def _on_image(self, key: str, msg: Image) -> None:
        with self._lock:
            self._images[key] = msg

    def get_observation(self, camera_keys: list[str]) -> dict:
        with self._lock:
            joint_pos = dict(self._joint_pos)
            gripper_pos = self._gripper_pos
            images = {}
            for key in camera_keys:
                msg = self._images.get(key)
                if msg is None:
                    continue
                channels = len(msg.data) // (msg.height * msg.width)
                images[key] = {
                    "shape": [msg.height, msg.width, channels],
                    "data_b64": base64.b64encode(bytes(msg.data)).decode("ascii"),
                }
        return {"joint_pos": joint_pos, "gripper_pos": gripper_pos, "images": images}

    def send_action(self, joint_pos: dict[str, float], gripper_pos: float) -> None:
        arm_msg = JointState()
        arm_msg.name = JOINT_NAMES
        arm_msg.position = [joint_pos[name] for name in JOINT_NAMES]
        self._cmd_pub.publish(arm_msg)

        knuckle = KNUCKLE_MAX * (1.0 - max(0.0, min(1.0, gripper_pos)))
        gripper_msg = JointState()
        gripper_msg.name = GRIPPER_JOINTS
        gripper_msg.position = [knuckle * mimic for mimic in GRIPPER_MIMIC]
        self._gripper_pub.publish(gripper_msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    rclpy.init()
    node = RB5Bridge()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    node.get_logger().info(f"rb5_lerobot_bridge listening on tcp://*:{args.port}")

    try:
        while rclpy.ok():
            req = sock.recv_json()
            cmd = req.get("cmd")
            if cmd == "ping":
                sock.send_json({"ok": True})
            elif cmd == "get_observation":
                obs = node.get_observation(req.get("camera_keys", []))
                sock.send_json(obs)
            elif cmd == "send_action":
                node.send_action(req["joint_pos"], req["gripper_pos"])
                sock.send_json({"ok": True})
            else:
                sock.send_json({"ok": False, "error": f"unknown cmd '{cmd}'"})
    finally:
        sock.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
