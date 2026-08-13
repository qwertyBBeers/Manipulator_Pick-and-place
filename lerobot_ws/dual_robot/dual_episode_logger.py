#!/usr/bin/env python3
"""Records two-robot relay episodes, with images, for LeRobot/VLA fine-tuning.

A separate file from ros_bridge/episode_logger.py rather than an extension of
it: that one records a single un-namespaced robot with one camera and marks
episode boundaries by wall-clock duration or a keypress. Neither assumption
survives here -- there are two namespaced arms, three cameras, and the boundary
that matters is "one pick-and-place attempt", which only the controller knows.

Episode boundaries come from relay_pick_place.py on /relay/episode (JSON in a
std_msgs/String): {"event": "start"|"end", "index": N, "task": "...",
"arm": "robot_a", "attempt": 1, "success": true}. The `arm` field is what makes
the recording single-robot at any instant -- the relay only ever moves one arm
at a time, and a VLA is trained to control one. State/action come from that
arm's namespace; the other arm is scenery.

Frames are paced by the wrist camera, the slowest-changing required input: the
joint states arrive at physics rate and pairing them all with one stale image
would just inflate the dataset with near-duplicates.

Recorded per frame:
    observation.state           <arm>/joint_states       6 arm joints + gripper
    action                      <arm>/isaac_joint_commands + gripper_joint_commands
    observation.images.scene    /scene_camera/rgb        front overview
    observation.images.wrist    <arm>/wrist_camera/rgb   that arm's wrist

Images are written as per-frame JPEGs under images_scene/ and images_wrist/;
state/action/timestamps go in state.npz.

Usage (alongside dual_binpicking_scene.py + dual_binpicking.launch.py):
    source /opt/ros/humble/setup.bash
    python3 dual_episode_logger.py --out-dir ~/asl_ws/Manipulator/lerobot_ws/datasets/relay
    # then, in another shell:
    python3 relay_pick_place.py --cycles 20 --randomize

Output layout is one directory per episode. Converted to a LeRobotDataset by
../tools/convert_relay_episodes_to_lerobot.py -- NOT by
convert_isaac_episodes_to_lerobot.py, which reads the single-robot,
single-camera layout and cannot read this one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from PIL import Image as PILImage

ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
# Must stay in sync with namespaced_trajectory_bridge.py.
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
KNUCKLE_MAX = 0.8  # rad, fully closed

EPISODE_TOPIC = "/relay/episode"
JPEG_QUALITY = 92
SCENE_CAMERA_TOPIC = "/scene_camera/rgb"
NAMESPACES = ("robot_a", "robot_b")


def _knuckle_to_gripper_pos(knuckle_rad: float) -> float:
    """0 = closed, 1 = open -- the LeRobot convention, not the knuckle angle."""
    return 1.0 - knuckle_rad / KNUCKLE_MAX


def _image_to_array(msg: Image) -> np.ndarray:
    channels = len(msg.data) // (msg.height * msg.width)
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)[:, :, :3]


class RelayEpisodeLogger(Node):
    def __init__(self, out_dir: Path):
        super().__init__("relay_episode_logger")
        self._lock = threading.Lock()
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._recording_arm: str | None = None
        self._episode: dict | None = None
        self._frames: list[dict] = []
        self._saved = 0
        self._dropped = 0
        # Episode numbering belongs to the logger, not to the controller.
        # relay_pick_place.py counts from 1 in each process, and unattended
        # collection runs it in repeated short batches -- so its indices repeat
        # every batch. Writing into an existing episode_0001/ only overwrote the
        # frames the new episode happened to reach: an episode of 263 frames
        # landed in a directory that still held 766 JPEGs from a longer earlier
        # one, and meta.json disagreed with the directory. Silent, and it makes
        # the whole dataset untrustworthy. Continue from whatever is already
        # there so restarts append instead of colliding.
        self._next_index = self._highest_existing_index() + 1

        # Latest values per namespace, kept whether or not we are recording, so
        # a frame is never missing state just because a topic went quiet during
        # the moment an episode opened.
        self._state = {ns: dict.fromkeys(ARM_JOINTS, 0.0) for ns in NAMESPACES}
        self._gripper_state = {ns: 1.0 for ns in NAMESPACES}
        self._action = {ns: dict.fromkeys(ARM_JOINTS, 0.0) for ns in NAMESPACES}
        self._gripper_action = {ns: 1.0 for ns in NAMESPACES}
        self._wrist_image: dict[str, Image | None] = {ns: None for ns in NAMESPACES}
        self._scene_image: Image | None = None

        for ns in NAMESPACES:
            self.create_subscription(JointState, f"/{ns}/joint_states",
                                     lambda m, n=ns: self._on_joint_states(n, m), 10)
            self.create_subscription(JointState, f"/{ns}/isaac_joint_commands",
                                     lambda m, n=ns: self._on_arm_command(n, m), 10)
            self.create_subscription(JointState, f"/{ns}/gripper_joint_commands",
                                     lambda m, n=ns: self._on_gripper_command(n, m), 10)
            # Frame pacing comes from here, so queue depth 1: an old wrist frame
            # is worse than a dropped one.
            self.create_subscription(Image, f"/{ns}/wrist_camera/rgb",
                                     lambda m, n=ns: self._on_wrist_image(n, m), 1)
        self.create_subscription(Image, SCENE_CAMERA_TOPIC, self._on_scene_image, 1)
        self.create_subscription(String, EPISODE_TOPIC, self._on_episode_marker, 10)

        self.get_logger().info(
            f"recording to {self.out_dir}; waiting for episode markers on {EPISODE_TOPIC}"
        )

    def _highest_existing_index(self) -> int:
        highest = 0
        for tree in ("success", "failure"):
            for path in (self.out_dir / tree).glob("episode_*"):
                try:
                    highest = max(highest, int(path.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return highest

    # ── inputs ──────────────────────────────────────────────────────────
    def _on_joint_states(self, ns: str, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._state[ns]:
                    self._state[ns][name] = pos
                elif name == GRIPPER_JOINT:
                    self._gripper_state[ns] = _knuckle_to_gripper_pos(pos)

    def _on_arm_command(self, ns: str, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name in self._action[ns]:
                    self._action[ns][name] = pos

    def _on_gripper_command(self, ns: str, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name == GRIPPER_JOINT:
                    self._gripper_action[ns] = _knuckle_to_gripper_pos(pos)

    def _on_scene_image(self, msg: Image):
        with self._lock:
            self._scene_image = msg

    def _on_wrist_image(self, ns: str, msg: Image):
        with self._lock:
            self._wrist_image[ns] = msg
            if self._recording_arm == ns:
                self._append_frame(ns)

    # ── episode control ─────────────────────────────────────────────────
    def _on_episode_marker(self, msg: String):
        try:
            marker = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"unparseable episode marker: {msg.data!r}")
            return
        event = marker.get("event")
        if event == "start":
            self._start(marker)
        elif event == "end":
            self._end(marker)

    def _start(self, marker: dict):
        arm = marker.get("arm")
        if arm not in NAMESPACES:
            self.get_logger().warn(f"episode marker for unknown arm {arm!r}; ignoring")
            return
        with self._lock:
            self._episode = marker
            self._frames = []
            self._recording_arm = arm
        self.get_logger().info(
            f"episode {marker.get('index')} start: {arm} attempt {marker.get('attempt')} "
            f"-- {marker.get('task')!r}"
        )

    def _end(self, marker: dict):
        with self._lock:
            episode, frames = self._episode, self._frames
            self._recording_arm, self._episode, self._frames = None, None, []
        if episode is None:
            return
        if not frames:
            self._dropped += 1
            self.get_logger().warn(
                f"episode {marker.get('index')} produced no frames -- nothing saved "
                f"(are the cameras publishing?)"
            )
            return
        self._save(episode, marker, frames)

    # ── recording ───────────────────────────────────────────────────────
    def _append_frame(self, ns: str):
        """Called with the lock held, from the wrist-camera callback."""
        wrist, scene = self._wrist_image[ns], self._scene_image
        if wrist is None or scene is None:
            return
        self._frames.append({
            "state": np.array([self._state[ns][j] for j in ARM_JOINTS] + [self._gripper_state[ns]],
                              dtype=np.float32),
            "action": np.array([self._action[ns][j] for j in ARM_JOINTS] + [self._gripper_action[ns]],
                               dtype=np.float32),
            "image_scene": _image_to_array(scene).copy(),
            "image_wrist": _image_to_array(wrist).copy(),
            "t": time.time(),
        })

    def _save(self, episode: dict, end_marker: dict, frames: list[dict]):
        index = self._next_index
        self._next_index += 1
        success = bool(end_marker.get("success", False))
        # Successes and failures go in separate trees. Mixing them in one
        # directory makes it far too easy to train on both by accident, and a
        # failed attempt is a demonstration of dropping the block.
        root = self.out_dir / ("success" if success else "failure")
        episode_dir = root / f"episode_{index:04d}"
        # Should not exist -- the index is fresh -- but a leftover from an
        # earlier layout would silently blend into this episode, so start clean.
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)
        np.savez_compressed(
            episode_dir / "state.npz",
            state=np.stack([f["state"] for f in frames]),
            action=np.stack([f["action"] for f in frames]),
            timestamps=np.array([f["t"] for f in frames], dtype=np.float64),
        )
        # JPEG frames, not one stacked .npy per camera. Raw uint8 at these
        # resolutions is ~1.15MB per frame across the two cameras, which at
        # ~12Hz over a ~45s transfer is ~600MB for a single episode -- a
        # thousand-episode dataset would not fit anywhere. JPEG at q=92 is
        # visually lossless for this content and ~40x smaller, and per-frame
        # files are what the LeRobot converters expect to walk anyway.
        for name, key in (("images_scene", "image_scene"), ("images_wrist", "image_wrist")):
            frame_dir = episode_dir / name
            frame_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                PILImage.fromarray(frame[key]).save(frame_dir / f"{i:06d}.jpg", quality=JPEG_QUALITY)
        task = episode.get("task", "")
        (episode_dir / "task.txt").write_text(task)
        (episode_dir / "meta.json").write_text(json.dumps({
            "index": index,
            "controller_index": episode.get("index"),
            "arm": episode.get("arm"),
            "attempt": episode.get("attempt"),
            "task": task,
            "success": success,
            "frames": len(frames),
            "duration_sec": frames[-1]["t"] - frames[0]["t"],
        }, indent=2))
        self._saved += 1
        self.get_logger().info(
            f"episode {index} {'OK ' if success else 'FAIL'} saved: {len(frames)} frames, "
            f"{frames[-1]['t'] - frames[0]['t']:.1f}s -> {episode_dir}"
        )

    def summary(self) -> str:
        return f"{self._saved} episode(s) saved, {self._dropped} dropped (no frames)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Where episode directories go.")
    args = parser.parse_args()

    rclpy.init()
    node = RelayEpisodeLogger(args.out_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(node.summary())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
