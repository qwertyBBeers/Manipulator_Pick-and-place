#!/usr/bin/env python3
"""Passively records RB5-850E episodes for later LeRobotDataset conversion.

Purely additive: subscribes to topics the *existing*, untouched pick-and-place
pipeline already publishes (binpicking_scene.py + trajectory_bridge.py) --
doesn't modify or interfere with it. Run this alongside a normal
`ros2 launch rb5_binpicking binpicking.launch.py` + `moveit_pick_place.py`
session to harvest its scripted runs as demonstrations for LeRobot.

Same interpreter/env requirement as rb5_lerobot_bridge.py: ROS2 Humble
sourced, Python 3.10 (e.g. `physical` or `isaaclab` conda env), NOT the
`lerobot` conda env.

Recorded topics:
    /joint_states             -> observation.state (6 arm joints + gripper)
    /isaac_joint_commands     -> action (6 arm joints), as commanded by
    /gripper_joint_commands   -> action (gripper), whatever is driving the
                                  robot at the time (currently: trajectory_bridge.py,
                                  driven by moveit_pick_place.py's state machine)
    /camera/color/image_raw   -> observation.images.color

Usage:
    source /opt/ros/humble/setup.bash
    conda activate physical   # or isaaclab
    python3 episode_logger.py --out-dir ./episodes --task "pick up the cube and place it in the bin"

Episode boundaries are marked interactively by default: press Enter to start
recording, press Enter again to stop and save. Repeat for each episode;
Ctrl+C to exit. Pass --duration/--count for non-interactive fixed-length
back-to-back episodes instead (e.g. for scripting alongside moveit_pick_place.py's
autonomous state machine, which has no external start/stop hook to trigger on).

Output layout is consumed by ../tools/convert_isaac_episodes_to_lerobot.py's
`iter_episodes`.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
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
KNUCKLE_MAX = 0.8  # rad, fully closed


def _knuckle_to_gripper_pos(knuckle_rad: float) -> float:
    return 1.0 - knuckle_rad / KNUCKLE_MAX  # 0=closed, 1=open


class EpisodeLogger(Node):
    def __init__(self):
        super().__init__("rb5_episode_logger")
        self._lock = threading.Lock()
        self._recording = False
        self._frames: list[dict] = []

        # Latest-known values, updated continuously regardless of recording state.
        self._joint_pos = dict.fromkeys(JOINT_NAMES, 0.0)
        self._gripper_obs_pos = 1.0
        self._action_joint_pos = dict.fromkeys(JOINT_NAMES, 0.0)
        self._action_gripper_pos = 1.0
        self._latest_image: Image | None = None

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(JointState, "/isaac_joint_commands", self._on_arm_command, 10)
        self.create_subscription(JointState, "/gripper_joint_commands", self._on_gripper_command, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self._on_image, 1)

    def _on_joint_states(self, msg: JointState) -> None:
        # Frames are paced by _on_image (camera fps), not this callback -- /joint_states
        # arrives much faster (sim physics rate) and would otherwise flood the episode
        # buffer with near-duplicate frames sharing one stale image.
        with self._lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                if name in self._joint_pos:
                    self._joint_pos[name] = pos
                elif name == GRIPPER_JOINTS[0]:
                    self._gripper_obs_pos = _knuckle_to_gripper_pos(pos)

    def _on_arm_command(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                if name in self._action_joint_pos:
                    self._action_joint_pos[name] = pos

    def _on_gripper_command(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position, strict=False):
                if name == GRIPPER_JOINTS[0]:
                    self._action_gripper_pos = _knuckle_to_gripper_pos(pos)

    def _on_image(self, msg: Image) -> None:
        with self._lock:
            self._latest_image = msg
            self._maybe_log_frame()

    def _maybe_log_frame(self) -> None:
        """Called with self._lock held, on every /joint_states or /camera tick."""
        if not self._recording or self._latest_image is None:
            return
        channels = len(self._latest_image.data) // (self._latest_image.height * self._latest_image.width)
        image = np.frombuffer(self._latest_image.data, dtype=np.uint8).reshape(
            self._latest_image.height, self._latest_image.width, channels
        )
        self._frames.append(
            {
                "state": np.array(
                    [self._joint_pos[n] for n in JOINT_NAMES] + [self._gripper_obs_pos], dtype=np.float32
                ),
                "action": np.array(
                    [self._action_joint_pos[n] for n in JOINT_NAMES] + [self._action_gripper_pos],
                    dtype=np.float32,
                ),
                "image_color": image.copy(),
            }
        )

    def start_episode(self) -> None:
        with self._lock:
            self._frames = []
            self._recording = True

    def stop_episode(self) -> list[dict]:
        with self._lock:
            self._recording = False
            frames = self._frames
            self._frames = []
        return frames


def save_episode(frames: list[dict], task: str, out_dir: Path, index: int) -> None:
    episode_dir = out_dir / f"episode_{index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        episode_dir / "state.npz",
        state=np.stack([f["state"] for f in frames]),
        action=np.stack([f["action"] for f in frames]),
    )
    np.save(episode_dir / "images_color.npy", np.stack([f["image_color"] for f in frames]))
    (episode_dir / "task.txt").write_text(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Language instruction stored with every episode")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="If set, run non-interactively: record fixed-length episodes of this "
        "many seconds back-to-back instead of waiting on Enter keypresses.",
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of episodes to record in --duration mode."
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = EpisodeLogger()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    index = len(list(args.out_dir.glob("episode_*")))
    try:
        if args.duration is not None:
            for _ in range(args.count):
                print(f"[episode {index}] recording for {args.duration}s...")
                node.start_episode()
                time.sleep(args.duration)
                frames = node.stop_episode()
                if not frames:
                    print(f"[episode {index}] no frames captured, discarding")
                    continue
                save_episode(frames, args.task, args.out_dir, index)
                print(f"[episode {index}] saved {len(frames)} frames to {args.out_dir}/episode_{index:04d}")
                index += 1
        else:
            while True:
                input(f"[episode {index}] press Enter to start recording...")
                node.start_episode()
                input(f"[episode {index}] recording -- press Enter to stop and save...")
                frames = node.stop_episode()
                if not frames:
                    print(f"[episode {index}] no frames captured, discarding")
                    continue
                save_episode(frames, args.task, args.out_dir, index)
                print(f"[episode {index}] saved {len(frames)} frames to {args.out_dir}/episode_{index:04d}")
                index += 1
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
