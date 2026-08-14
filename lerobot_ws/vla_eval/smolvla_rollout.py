#!/usr/bin/env python3
"""Closed-loop pick-and-place in Isaac Sim, driven by the fine-tuned SmolVLA.

The robot half of the pair; the policy half is smolvla_policy_server.py, which
runs in the `lerobot` conda env because LeRobot needs Python 3.12 and rclpy
does not exist there. This file runs under ROS 2 Humble's Python 3.10.

MoveIt is NOT in the loop. The relay recorded `action` as the joint command the
trajectory bridge was streaming to Isaac -- absolute joint positions plus a
gripper opening -- so the policy's output goes onto exactly the same two topics
the bridge used, and nothing plans or collision-checks. That is the point of the
experiment: whether the policy alone reproduces the behaviour the planner had to
be engineered into.

Pacing follows the wrist camera, not a wall-clock timer. The logger appended one
training frame per wrist image, so the demonstrated spacing between consecutive
actions IS one wrist frame, whatever the simulator's real-time factor happened
to be. A fixed-rate loop would replay them at the wrong speed as soon as the RTF
moved.

Usage (scene must already be running; MoveIt need not be):
    conda activate lerobot   # in another shell
    python smolvla_policy_server.py --checkpoint .../020000/pretrained_model

    source /opt/ros/humble/setup.bash
    /usr/bin/python3 smolvla_rollout.py --arm robot_a --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "dual_robot"))
from ipc import recv_msg, send_msg  # noqa: E402
from layout import DEST_D, DEST_W, DEST_WORLD, HANDOFF_WORLD, SOURCE_WORLD  # noqa: E402

# Must match namespaced_trajectory_bridge.py and dual_episode_logger.py.
ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GRIPPER_MIMIC = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
KNUCKLE_MAX = 0.8

RANDOMIZE_TOPIC = "/binpicking/randomize_object"
OBJECT_POSE_TOPIC = "/binpicking/object_pose"
SCENE_CAMERA_TOPIC = "/scene_camera/rgb"

# The pose every demonstration starts from, taken as the median first-frame
# state over the 504 recorded episodes rather than re-derived from the watch
# pose by IK: it is the distribution the policy was actually trained to open
# from, and it is the same for both arms (WATCH_B == WATCH_A, and each arm
# plans in its own link0).
START_JOINTS = [0.1482, -0.2195, 1.0679, 0.5910, 1.5212, -0.2256]
START_GRIPPER = 1.0  # 1 = open, the LeRobot convention used by the logger
HOME_RAMP_SEC = 4.0
SETTLE_AFTER_RANDOMIZE_SEC = 2.0

TASKS = {
    "a_to_handoff": ("robot_a", "pick up the block and place it on the blue tray in the middle",
                     HANDOFF_WORLD),
    "b_to_dest": ("robot_b", "pick up the block from the blue tray and place it in the far bin",
                  DEST_WORLD),
    "a_to_source": ("robot_a", "pick up the block from the blue tray and place it in the near bin",
                    SOURCE_WORLD),
}

# Landing inside the target tray, using the same settled-and-near test the
# expert used to confirm a handoff (relay_pick_place.wait_until_settled_near),
# plus a height ceiling that test did not need.
#
# The ceiling matters here in a way it never did for the expert: a policy that
# carries the block over the tray and simply holds it there is stationary and
# on target, so near-and-settled alone scores it a success. Measured, a block
# resting in a tray sits at z=0.028 (0.007 tray floor + half of the 42mm cube)
# and one still in the gripper sits at z=0.223.
SUCCESS_RADIUS_M = min(DEST_W, DEST_D) / 2.0
SUCCESS_MAX_Z_M = 0.08
SETTLED_MOVE_M = 0.005
SETTLED_FOR_SEC = 1.0

# A cap on how far one control step may move a joint. Actions are absolute
# positions on a stiff drive, so an out-of-distribution observation returning a
# wild target is not a wobble, it is a slam. Demonstrations move well under this
# per frame, so it only ever clips something that was already wrong.
MAX_STEP_RAD = 0.30


def _image_to_array(msg: Image) -> np.ndarray:
    channels = len(msg.data) // (msg.height * msg.width)
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)[:, :, :3]


def _knuckle_to_gripper_pos(knuckle_rad: float) -> float:
    return 1.0 - knuckle_rad / KNUCKLE_MAX


class PolicyClient:
    def __init__(self, port: int, timeout_sec: float = 30.0):
        self._socket = socket.create_connection(("127.0.0.1", port), timeout=timeout_sec)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _call(self, request: dict) -> dict:
        send_msg(self._socket, request)
        reply = recv_msg(self._socket)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "policy server error"))
        return reply

    def ping(self) -> dict:
        return self._call({"cmd": "ping"})

    def reset(self) -> None:
        self._call({"cmd": "reset"})

    def act(self, state, scene, wrist, task) -> tuple[np.ndarray, float]:
        reply = self._call({"cmd": "act", "state": state, "scene": scene,
                            "wrist": wrist, "task": task})
        return reply["action"], reply["infer_ms"]


class VlaRollout(Node):
    def __init__(self, arm: str, port: int):
        super().__init__("smolvla_rollout")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.arm = arm

        self._joints = dict.fromkeys(ARM_JOINTS, 0.0)
        self._gripper = 1.0
        self._scene_image = None
        self._wrist_image = None
        self._wrist_seq = 0
        self._object_pose = None

        self.create_subscription(JointState, f"/{arm}/joint_states", self._on_joint_states, 10)
        self.create_subscription(Image, SCENE_CAMERA_TOPIC, self._on_scene_image, 1)
        self.create_subscription(Image, f"/{arm}/wrist_camera/rgb", self._on_wrist_image, 1)
        self.create_subscription(PoseStamped, OBJECT_POSE_TOPIC, self._on_object_pose, 10)

        self._arm_pub = self.create_publisher(JointState, f"/{arm}/isaac_joint_commands", 10)
        self._grip_pub = self.create_publisher(JointState, f"/{arm}/gripper_joint_commands", 10)
        self._randomize_pub = self.create_publisher(Empty, RANDOMIZE_TOPIC, 10)

        self.policy = PolicyClient(port)
        self.action_low = None
        self.action_high = None

    # ── inputs ──────────────────────────────────────────────────────────
    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joints:
                self._joints[name] = pos
            elif name == GRIPPER_JOINTS[0]:
                self._gripper = _knuckle_to_gripper_pos(pos)

    def _on_scene_image(self, msg: Image):
        self._scene_image = msg

    def _on_wrist_image(self, msg: Image):
        self._wrist_image = msg
        self._wrist_seq += 1

    def _on_object_pose(self, msg: PoseStamped):
        self._object_pose = msg

    # ── outputs ─────────────────────────────────────────────────────────
    def _publish_arm(self, positions):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = ARM_JOINTS
        cmd.position = [float(p) for p in positions]
        self._arm_pub.publish(cmd)

    def _publish_gripper(self, gripper_pos: float):
        knuckle = (1.0 - min(1.0, max(0.0, gripper_pos))) * KNUCKLE_MAX
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = GRIPPER_JOINTS
        cmd.position = [knuckle * m for m in GRIPPER_MIMIC]
        self._grip_pub.publish(cmd)

    # ── helpers ─────────────────────────────────────────────────────────
    def state_vector(self) -> np.ndarray:
        return np.array([self._joints[j] for j in ARM_JOINTS] + [self._gripper], dtype=np.float32)

    def spin(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_for_inputs(self, timeout_sec=60.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._scene_image is not None and self._wrist_image is not None \
                    and self._object_pose is not None:
                return True
        missing = [n for n, v in (("scene camera", self._scene_image),
                                  ("wrist camera", self._wrist_image),
                                  ("object pose", self._object_pose)) if v is None]
        self.get_logger().error(f"timed out waiting for: {', '.join(missing)}")
        return False

    def go_home(self):
        """Ramp to the pose every demonstration opens from.

        Ramped rather than commanded in one step because these go straight to
        Isaac's position drives with no trajectory behind them.
        """
        start = [self._joints[j] for j in ARM_JOINTS]
        t0 = self.get_clock().now()
        while True:
            rclpy.spin_once(self, timeout_sec=0.01)
            elapsed = (self.get_clock().now() - t0).nanoseconds * 1e-9
            fraction = min(1.0, elapsed / HOME_RAMP_SEC)
            self._publish_arm([s + (h - s) * fraction for s, h in zip(start, START_JOINTS)])
            self._publish_gripper(START_GRIPPER)
            if fraction >= 1.0:
                break
        self.spin(1.0)

    def randomize_block(self):
        self._randomize_pub.publish(Empty())
        self.spin(SETTLE_AFTER_RANDOMIZE_SEC)

    def object_xyz(self):
        if self._object_pose is None:
            return None
        p = self._object_pose.pose.position
        return (p.x, p.y, p.z)

    def clamp_action(self, action: np.ndarray, previous: np.ndarray):
        """Apply the demonstrated envelope and the per-step cap. Returns
        (clamped, reasons) so a clipped rollout is never silently reported as a
        clean one."""
        reasons, kinds = [], set()
        clamped = action.copy()
        if self.action_low is not None:
            outside = (clamped < self.action_low) | (clamped > self.action_high)
            if outside.any():
                # Separated from the step cap because they mean different
                # things: a gripper command a hair past fully-open is noise,
                # a joint target the arm never visited is the policy leaving
                # the distribution.
                kinds.add("gripper_range" if set(np.where(outside)[0]) == {6} else "joint_range")
                reasons.append(f"outside demonstrated range at {np.where(outside)[0].tolist()}")
                clamped = np.clip(clamped, self.action_low, self.action_high)
        delta = clamped[:6] - previous[:6]
        big = np.abs(delta) > MAX_STEP_RAD
        if big.any():
            kinds.add("step_cap")
            reasons.append(f"step capped at {np.where(big)[0].tolist()} "
                           f"(max |d|={np.abs(delta).max():.3f}rad)")
            clamped[:6] = previous[:6] + np.clip(delta, -MAX_STEP_RAD, MAX_STEP_RAD)
        return clamped, reasons, kinds


def run_episode(node: VlaRollout, task: str, target_xy, max_steps: int):
    """One rollout. Returns a result dict."""
    node.policy.reset()
    node.go_home()
    start_xyz = node.object_xyz()

    states, actions = [], []
    previous = node.state_vector()
    last_seq = node._wrist_seq
    clips = {"gripper_range": 0, "joint_range": 0, "step_cap": 0}
    stable_since = None
    last_xy = None
    success = False
    t_start = time.monotonic()

    for step in range(max_steps):
        # One action per wrist frame -- the cadence the demonstrations were
        # recorded at.
        while node._wrist_seq == last_seq:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t_start > max_steps * 2.0 + 60.0:
                return {"success": False, "reason": "wrist camera stopped publishing",
                        "steps": step, "states": states, "actions": actions}
        last_seq = node._wrist_seq

        state = node.state_vector()
        action, _ = node.policy.act(state,
                                    _image_to_array(node._scene_image),
                                    _image_to_array(node._wrist_image),
                                    task)
        action, reasons, kinds = node.clamp_action(action, previous)
        for kind in kinds:
            clips[kind] += 1
        if "joint_range" in kinds or "step_cap" in kinds:
            node.get_logger().warn(f"step {step}: {'; '.join(reasons)}")
        node._publish_arm(action[:6])
        node._publish_gripper(float(action[6]))
        previous = action
        states.append(state)
        actions.append(action)

        xyz = node.object_xyz()
        if xyz is not None:
            near = (math.hypot(xyz[0] - target_xy[0], xyz[1] - target_xy[1]) < SUCCESS_RADIUS_M
                    and xyz[2] < SUCCESS_MAX_Z_M)
            moved = math.hypot(xyz[0] - last_xy[0], xyz[1] - last_xy[1]) if last_xy else 1.0
            last_xy = (xyz[0], xyz[1])
            if near and moved < SETTLED_MOVE_M:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= SETTLED_FOR_SEC:
                    success = True
                    break
            else:
                stable_since = None

    end_xyz = node.object_xyz()
    return {
        "success": success,
        "reason": "settled in target tray" if success else "ran out of steps",
        "steps": len(actions),
        "duration_sec": time.monotonic() - t_start,
        "block_start": start_xyz,
        "block_end": end_xyz,
        "block_moved_m": (math.dist(start_xyz, end_xyz) if start_xyz and end_xyz else None),
        "clips": clips,
        "states": states,
        "actions": actions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="a_to_handoff")
    parser.add_argument("--arm", default=None, help="Override the arm the task implies")
    parser.add_argument("--episodes", type=int, default=5)
    # Demonstrations of these tasks run 530-761 frames (median 662), and the
    # rollout is paced one step per wrist frame just as they were, so anything
    # under ~800 cuts the policy off mid-task rather than measuring it.
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/mnt/hdd/relay_datasets/vla_rollouts"))
    args = parser.parse_args()

    arm, task, target = TASKS[args.task]
    arm = args.arm or arm
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = VlaRollout(arm, args.port)
    try:
        info = node.policy.ping()
        if "action_min" in info:
            node.action_low, node.action_high = info["action_min"], info["action_max"]
            node.get_logger().info("policy server up; clamping to the demonstrated action range")
        else:
            node.get_logger().warn("policy server reported no action range; only the step cap applies")

        node.get_logger().info(f"waiting for {arm} cameras, joint states and object pose...")
        if not node.wait_for_inputs():
            return
        node.get_logger().info(f"task: {task!r}  target xy=({target[0]:.3f}, {target[1]:.3f}) "
                               f"radius={SUCCESS_RADIUS_M:.3f}m")

        results = []
        for episode in range(1, args.episodes + 1):
            if not args.no_randomize:
                node.randomize_block()
            node.get_logger().info(f"--- episode {episode}/{args.episodes} ---")
            result = run_episode(node, task, target, args.max_steps)
            states = np.stack(result.pop("states")) if result["steps"] else np.zeros((0, 7))
            actions = np.stack(result.pop("actions")) if result["steps"] else np.zeros((0, 7))
            stamp = time.strftime("%Y%m%d_%H%M%S")
            np.savez_compressed(args.out_dir / f"rollout_{stamp}_{episode:03d}.npz",
                                state=states, action=actions)
            results.append(result)
            node.get_logger().info(
                f"episode {episode}: {'SUCCESS' if result['success'] else 'FAIL'} "
                f"({result['reason']}) steps={result['steps']} "
                f"block moved {(result['block_moved_m'] or 0) * 100:.1f}cm "
                f"end z={(result['block_end'] or (0, 0, 0))[2]:.3f} clips={result['clips']}"
            )

        successes = sum(r["success"] for r in results)
        node.get_logger().info(f"=== {successes}/{len(results)} succeeded ===")
        (args.out_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps({"task": task, "arm": arm, "results": results}, indent=2, default=str))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
