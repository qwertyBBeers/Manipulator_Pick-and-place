"""RB5-850E LeRobot Robot backend.

Talks to `Manipulator/lerobot_ws/ros_bridge/rb5_lerobot_bridge.py` over a ZMQ
REQ/REP socket instead of importing rclpy directly: rclpy under ROS2 Humble is
built against Python 3.10, while this package runs under the `lerobot` conda
env's Python 3.12, so the two live in separate processes/interpreters and the
bridge script is the only thing that needs ROS2 sourced.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np
import zmq

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_rb5 import RB5850EConfig


class RB5850E(Robot):
    """Rainbow Robotics RB5-850E, driven via rb5_lerobot_bridge.py (Isaac Sim or real hardware)."""

    config_class = RB5850EConfig
    name = "rb5_850e"

    def __init__(self, config: RB5850EConfig):
        super().__init__(config)
        self.config = config
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._last_pos: dict[str, float] = {}

    @property
    def _joint_pos_keys(self) -> list[str]:
        return [f"{name}.pos" for name in self.config.joint_names]

    @property
    def observation_features(self) -> dict[str, Any]:
        motors = dict.fromkeys([*self._joint_pos_keys, "gripper.pos"], float)
        cameras = {
            key: (self.config.camera_height, self.config.camera_width, 3) for key in self.config.camera_keys
        }
        return {**motors, **cameras}

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys([*self._joint_pos_keys, "gripper.pos"], float)

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    @property
    def is_calibrated(self) -> bool:
        # Calibration (joint offsets, gripper stroke) lives on the bridge/ROS2 side,
        # already baked into rb5_isaac's URDF + trajectory_bridge conversions.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(self.config.timeout_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.config.timeout_s * 1000))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.config.bridge_host}:{self.config.bridge_port}")
        self._sock = sock

        reply = self._request({"cmd": "ping"})
        if not reply.get("ok"):
            self.disconnect()
            raise ConnectionError(f"rb5_lerobot_bridge did not respond to ping: {reply}")

    def disconnect(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _request(self, payload: dict) -> dict:
        if self._sock is None:
            raise ConnectionError("RB5850E is not connected")
        self._sock.send_json(payload)
        return self._sock.recv_json()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError("RB5850E is not connected")

        before_read_t = time.perf_counter()
        reply = self._request({"cmd": "get_observation", "camera_keys": self.config.camera_keys})
        read_dt_s = time.perf_counter() - before_read_t

        obs: RobotObservation = {}
        joint_pos = reply["joint_pos"]
        for name in self.config.joint_names:
            obs[f"{name}.pos"] = float(joint_pos[name])
        obs["gripper.pos"] = float(reply["gripper_pos"])
        self._last_pos = dict(joint_pos)

        for key, img in reply.get("images", {}).items():
            raw = base64.b64decode(img["data_b64"])
            h, w, c = img["shape"]
            obs[key] = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c)

        obs["read_dt_s"] = read_dt_s
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError("RB5850E is not connected")

        joint_pos = {name: float(action[f"{name}.pos"]) for name in self.config.joint_names}

        if self.config.max_relative_target is not None and self._last_pos:
            goal_present = {
                name: (joint_pos[name], self._last_pos[name])
                for name in self.config.joint_names
                if name in self._last_pos
            }
            safe = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            joint_pos.update(safe)

        gripper_pos = float(action["gripper.pos"])

        reply = self._request(
            {"cmd": "send_action", "joint_pos": joint_pos, "gripper_pos": gripper_pos}
        )
        if not reply.get("ok"):
            raise RuntimeError(f"rb5_lerobot_bridge rejected action: {reply}")

        self._last_pos = joint_pos
        return {**{f"{k}.pos": v for k, v in joint_pos.items()}, "gripper.pos": gripper_pos}
