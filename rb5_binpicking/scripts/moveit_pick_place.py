#!/usr/bin/env python3
"""MoveIt2 pick-and-place expert for RB5 bin picking (Phase 1 baseline).

Only the object's XY(Z) position is perceived dynamically today
(/binpicking/object_pose is Isaac Sim ground truth). Grasp orientation, the
overall approach strategy, and several geometric assumptions (object size)
are still Phase-1 simplifications — each is called out in a comment at the
point it's used rather than hidden.

Execution is NOT considered successful just because a MoveIt goal reported
SUCCESS. This expert explicitly verifies: grasp (gripper stall signal +
short test lift with observed object-pose rise) and placement (object
settled inside the destination bin). See README2.md for the full writeup of
what "verification" here does and does not cover.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Iterable

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped support on Buffer.transform)

from rb5_binpicking.bin_geometry import (
    BinSpec,
    load_bin_geometry,
    make_bin_collision_objects,
    make_box_collision_object,
)


GROUP_NAME = "mainpulation"  # SRDF typo, preserved intentionally — see README2.md §26
PIPELINE_OMPL = "ompl"
PIPELINE_PILZ = "pilz_industrial_motion_planner"
PLANNER_ID_LIN = "LIN"
BASE_LINK = "link0"
PLANNING_FRAME = BASE_LINK
EEF_LINK = "tcp"
ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
OBJECT_POSE_TOPIC = "/binpicking/object_pose"
OBJECT_POSE_TIMEOUT_SEC = 3.0

# Robotiq 2F-85 links allowed to touch the grasped object once attached in the
# planning scene (from rb5_isaac/urdf/rb5_with_tools.urdf). Must be the WHOLE
# gripper assembly, not just the fingertips: the attached object is a coarse
# cube approximation centered near the grasp point, close enough to
# robotiq_85_base_link/knuckles that omitting them makes MoveIt see the
# attach itself as a self-collision — which poisons the start state for
# every subsequent plan (including the very motion that just attached the
# object) until the object is detached. Confirmed via move_group's log:
# "Found a contact between 'robotiq_85_base_link' ... 'target_object'" /
# "Start state appears to be in collision" immediately after attach.
GRIPPER_TOUCH_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_right_finger_tip_link",
]
TARGET_OBJECT_ID = "target_object"

DEFAULT_DYNAMIC_PICK_PARAMS = {
    # Object pose is the target object's center. These offsets produce TCP goals.
    "grasp_offset_x": 0.0,
    "grasp_offset_y": 0.0,
    # Was 0.16, then 0.108 (README2.md §7.20) -- that second value was wrong
    # in a new way: it was derived from robotiq_85_left_finger_tip_link's
    # JOINT ORIGIN only, via URDF forward kinematics cross-checked against a
    # live TF reading of that same origin. What that missed: the fingertip's
    # collision MESH extends up to 51mm *beyond* its own origin (measured
    # directly from left_finger_tip.stl's vertex bounds), so the pad's real
    # lowest reach is far past what origin-only FK suggested. Confirmed live
    # via direct /compute_ik + /check_state_validity calls against the
    # running move_group: at 0.108 the fingertip penetrated source_bin_floor
    # by 9.4cm; a binary search over TCP z (same calls) found the actual
    # floor-just-touching height empirically, independent of any hand FK/mesh
    # math (README2.md §7.25): grasp_offset_z=0.130 is the exact knife-edge
    # (zero penetration). 0.135 (only 5mm of margin) still occasionally
    # scraped the floor in practice (README2.md §7.26) -- normal system noise
    # (re-observed object pose jitter, execution tolerance, TF timing) easily
    # eats 5mm. Widened to 15mm margin. The fingertip pad is ~5.7cm long
    # (see §7.25's mesh measurement) so this still lands well within the
    # object's side face, not near an edge. If placement is still off, retune
    # live with `-p grasp_offset_z:=<value>` and re-run the same
    # /check_state_validity sweep before hand-editing this default again --
    # hand-derived offsets have now been wrong twice.
    "grasp_offset_z": 0.145,
    "pre_grasp_lift": 0.18,
    # Was 0.22 -- with the source bin's 0.22m wall height, that left only a
    # few cm of clearance once the object's own resting height was
    # subtracted, and the arm clipped the bin wall while swinging toward the
    # destination bin. See lift_clearance_margin below for the geometry-based
    # floor that's enforced on top of this.
    "post_grasp_lift": 0.30,
    "lift_clearance_margin": 0.05,       # m, object must clear source_bin height by at least this much before transfer
    "position_tolerance": 0.02,
    "orientation_tolerance": 0.20,
    # Was False -- with this off, OMPL was free to spin the wrist mid-path
    # even though start/end orientation matched, which is consistent with
    # objects dropping "while turning" during the lift->pre_place swing.
    # Trade-off: OMPL has to work harder to satisfy the path constraint, so
    # planning failures during transfer may become more common; that's a
    # legible failure (retried/reported) rather than a silent drop.
    "use_path_orientation_constraint": True,
    # Re-observation (see README2.md §11)
    "max_object_pose_age": 1.0,          # s, reject a stored pose older than this
    "refine_pose_timeout": 2.0,          # s, how long to wait for a fresh pose at pre-grasp
    "max_object_pose_jump": 0.08,        # m, initial vs refined position sanity bound
    "max_object_tilt": 0.436,            # rad (~25deg), reject top-down grasp beyond this tilt (README2.md §7.13)
    # Grasp/lift/place verification (README2.md §16, §20)
    "test_lift_height": 0.04,            # m, small lift used to confirm the grasp
    "test_lift_z_threshold": 0.015,      # m, min observed object-z rise to count as success
    "place_settle_wait": 0.4,            # s, wait before re-sampling object pose after release
    "place_settle_move_threshold": 0.01,  # m, max object movement between samples to call it "settled"
    # Retry bounds
    "grasp_max_attempts": 4,
    "place_max_attempts": 2,
    "lift_max_attempts": 3,
    "lift_step_count": 3,
    # Trial loop (README2.md §7.6). 0 = unlimited trials.
    "max_trials": 10,
    "max_consecutive_failures": 3,
    # Place-pose generation (README2.md §19) — Phase 1 assumes a single known
    # cube size because /binpicking/object_pose carries no object dimensions.
    "object_half_height": 0.021,
    "destination_wall_clearance": 0.03,
    "place_approach_lift": 0.15,
    # Diagnostics
    "pcd_offset_x": 0.0,
    "pcd_offset_y": 0.0,
    "pcd_offset_z": 0.0,
}

SAVED_POSES = {
    "watch": {
        "position": [0.331, -0.069, 0.766],
        "orientation": [0.475, 0.449, 0.507, 0.562],
    },
    "gripping": {
        "position": [0.301, -0.055, 0.194],
        "orientation": [0.532, 0.528, 0.458, 0.478],
    },
    "second_box": {
        "position": [0.304, 0.400, 0.561],
        "orientation": [0.531, 0.529, 0.457, 0.479],
    },
    "end": {
        "position": [0.261, 0.414, 0.235],
        "orientation": [-0.001, 0.708, 0.701, -0.089],
    },
}

GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GRIPPER_MIMIC = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
GRIPPER_MAX_OPEN = 0.085
KNUCKLE_MAX = 0.8
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"


class StageResult(Enum):
    SUCCESS = auto()
    PLANNING_FAILED = auto()
    EXECUTION_FAILED = auto()
    POSE_TIMEOUT = auto()
    TF_FAILED = auto()
    GRASP_FAILED = auto()
    LIFT_FAILED = auto()
    PLACE_FAILED = auto()
    COLLISION_DETECTED = auto()
    CANCELED = auto()


class MotionClass(Enum):
    """Which planner policy a given EEF motion should use (README2.md §13)."""

    FREE_SPACE = auto()          # OMPL fallback allowed if LIN fails
    CONTACT_SENSITIVE = auto()   # LIN only — a failed LIN plan rejects the
                                  # candidate rather than letting OMPL choose
                                  # an unpredictable approach direction near
                                  # the object/bin.


@dataclass
class GraspCandidate:
    """One candidate grasp target.

    UPDATED (README2.md §7.13): /binpicking/object_pose DOES publish a real
    orientation (an earlier version of this docstring incorrectly assumed it
    was position-only ground truth). generate_grasp_candidates() now yaw-
    aligns the fixed top-down "gripping" approach to the object's actual
    heading, and rejects objects tilted too far off a resting face (no
    object type/dimensions are still published, so full 6-DOF shape-aware
    grasping remains out of scope — see §14). What varies between candidates
    is grasp depth and a small XY offset on top of the yaw-aligned
    orientation, so a rejected candidate has something meaningfully
    different to retry with.
    """

    grasp_xyz: list
    orientation: Quaternion
    pre_grasp_lift: float
    post_grasp_lift: float
    label: str


@dataclass
class TrialRecord:
    trial_id: str
    timestamp: str
    object_initial_pose: dict | None
    stages: dict = field(default_factory=dict)
    grasp_attempts: int = 0
    lift_attempts: int = 0
    place_attempts: int = 0
    full_task_success: bool | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    total_cycle_time_sec: float | None = None


class TrialLogger:
    """Structured per-trial JSONL logging (README2.md §21).

    Deliberately scoped to what this node can actually measure without
    additional plumbing: stage outcomes, retry counts, wall-clock timing,
    and the pass/fail signals from grasp/lift/place verification. Finer
    metrics (joint-tracking RMS error, perception latency breakdown) would
    need trajectory_bridge to publish them separately — not done here, see
    README2.md "remaining issues".
    """

    def __init__(self, node: Node, path: str):
        self._node = node
        self._path = os.path.expanduser(path)
        d = os.path.dirname(self._path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._record: TrialRecord | None = None
        self._t0 = 0.0

    def start_trial(self, object_pose: PoseStamped | None):
        self._record = TrialRecord(
            trial_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            object_initial_pose=_pose_to_dict(object_pose) if object_pose else None,
        )
        self._t0 = time.monotonic()

    def stage(self, name: str, result: StageResult, **extra):
        if self._record is None:
            return
        self._record.stages[name] = {"result": result.name, **extra}

    def note(self, **fields):
        """Set arbitrary top-level TrialRecord fields (e.g. grasp_attempts)."""
        if self._record is None:
            return
        for k, v in fields.items():
            setattr(self._record, k, v)

    def finish(self, success: bool, failure_stage: str | None, failure_reason: str | None):
        if self._record is None:
            return
        self._record.full_task_success = success
        self._record.failure_stage = failure_stage
        self._record.failure_reason = failure_reason
        self._record.total_cycle_time_sec = time.monotonic() - self._t0
        line = json.dumps(self._record.__dict__)
        try:
            with open(self._path, "a") as f:
                f.write(line + "\n")
        except OSError as exc:
            self._node.get_logger().error(f"Failed to write trial log to {self._path}: {exc}")
        self._record = None


def _tilt_from_vertical(q: Quaternion) -> float:
    """Angle (radians) between the object's local +Z axis (rotated into the
    world frame by q) and world +Z. 0 = resting flat on a face, pi/2 = on its
    side. Standard quaternion-to-rotation-matrix third-column formula
    (q assumed unit-norm -- see README2.md §7.13 for why that mattered)."""
    zz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    zz = max(-1.0, min(1.0, zz))
    return math.acos(zz)


def _yaw_from_quaternion(q: Quaternion) -> float:
    """Rotation about world Z (radians), standard ZYX-Euler yaw extraction."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _yaw_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """q1 ⊗ q2 (apply q2 first, then q1 -- i.e. q1 is the extrinsic/world-frame
    follow-up rotation)."""
    result = Quaternion()
    result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    return result


def _pose_to_dict(pose: PoseStamped) -> dict:
    return {
        "frame_id": pose.header.frame_id,
        "position": [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
        "orientation": [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ],
    }


class RB5MoveItPickPlace(Node):
    """Stage-based scripted expert for RB5 bin picking (Phase 1 baseline)."""

    def __init__(self):
        super().__init__("rb5_moveit_pick_place")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        for name, value in DEFAULT_DYNAMIC_PICK_PARAMS.items():
            self.declare_parameter(name, value)
        self.declare_parameter("trial_log_path", "~/.ros/rb5_binpicking_trials.jsonl")

        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.gripper_client = ActionClient(self, GripperCommand, GRIPPER_ACTION)
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self.attached_pub = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )

        self.object_pose: PoseStamped | None = None
        self.create_subscription(PoseStamped, OBJECT_POSE_TOPIC, self._object_pose_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Single source of truth for bin geometry — see config/bin_geometry.yaml.
        # Raises RuntimeError (crashing this node with a clear message) if the
        # file is missing/malformed, by design: no silent unrelated fallback.
        self.source_bin: BinSpec
        self.dest_bin: BinSpec
        self.source_bin, self.dest_bin = load_bin_geometry()

        for name in ("pcd_offset_x", "pcd_offset_y", "pcd_offset_z"):
            v = self.get_parameter(name).value
            if abs(v) > 1e-9:
                self.get_logger().warn(
                    f"{name}={v:.4f} is a non-zero empirical camera correction, applied once "
                    "in binpicking.launch.py's pcd_correction_tf (camera_depth_optical_frame -> "
                    "camera_depth_points_frame). It is NOT camera calibration and should be zero "
                    "once the underlying TF/frame mismatch is fixed (README2.md §10)."
                )

        self.trial_logger = TrialLogger(self, self.get_parameter("trial_log_path").value)

        self.get_logger().info(
            "MoveIt expert config: "
            f"group={GROUP_NAME}, base={BASE_LINK}, eef={EEF_LINK}, joints={ARM_JOINTS}, "
            f"source_bin_center={self.source_bin.center}, dest_bin_center={self.dest_bin.center}"
        )

    # ── top-level run: bounded multi-trial loop ─────────────────────────────

    # Stages reached only while the gripper is still physically closed on the
    # object (failure paths before release/detach — see _transfer_and_place).
    # After one of these, physical grasp state is unknown; the loop stops
    # rather than attempting another pick blind.
    HOLDING_OBJECT_FAILURE_STAGES = {"LIFT", "PRE_PLACE", "PLACE"}

    def run(self):
        self.get_logger().info("Waiting for /move_action...")
        if not self.move_group.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup action server /move_action is not available")
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Gripper action server {GRIPPER_ACTION} is not available")

        max_trials = int(self.get_parameter("max_trials").value)
        max_consecutive_failures = int(self.get_parameter("max_consecutive_failures").value)

        # IMPORTANT CAVEAT (README2.md §7.6): /binpicking/object_pose always
        # tracks one fixed prim (/World/Objects/Cube0 in binpicking_scene.py),
        # not "whichever object is next in the bin". Looping does NOT clear
        # multiple distinct objects from the bin — it repeats pick-and-place
        # on whatever pose that one topic currently reports (which may now be
        # inside the destination bin after a successful trial). This is a
        # perception limitation, not something this loop can work around.
        trial_num = 0
        consecutive_failures = 0
        while rclpy.ok():
            trial_num += 1
            if max_trials > 0 and trial_num > max_trials:
                self.get_logger().info(f"Reached max_trials={max_trials}, stopping.")
                break

            self.get_logger().info(f"=== Trial {trial_num}{f'/{max_trials}' if max_trials > 0 else ''} ===")
            success, stage, reason = self._run_one_trial()

            if success:
                consecutive_failures = 0
                self.get_logger().info(f"Trial {trial_num}: SUCCESS.")
                continue

            self.get_logger().error(f"Trial {trial_num}: FAILED at stage={stage}: {reason}")

            if stage in self.HOLDING_OBJECT_FAILURE_STAGES:
                self.get_logger().error(
                    "Failure happened while still holding the object — physical grasp state "
                    "is unknown. Stopping the trial loop rather than attempting another pick "
                    "blind. Check the physical/sim state before running again."
                )
                break

            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                self.get_logger().error(
                    f"{consecutive_failures} consecutive failures (not holding an object) — "
                    "stopping rather than retrying indefinitely. This usually means something "
                    "structural is wrong (TF, perception, planning scene), not bad luck."
                )
                break

        self.get_logger().info(f"Trial loop finished after {trial_num} trial(s).")

    def _run_one_trial(self) -> tuple[bool, str, str]:
        """Run exactly one pick-and-place attempt: reset planning-scene state,
        (re-)publish bin geometry, wait for an object pose, dispatch to the
        dynamic or saved-pose sequence, and log the trial."""
        # move_group's planning scene outlives this process (it's owned by the
        # long-running move_group node from binpicking.launch.py). If a
        # previous trial (or a previous killed process) ended while still
        # holding an attached object, that attachment survives and poisons
        # every plan afterward (see GRIPPER_TOUCH_LINKS comment). REMOVE is a
        # no-op if nothing is attached, so this is always safe to call.
        # Repeated like publish_bin_collision_objects(): /attached_collision_object
        # is not latched, and the publisher may not have a matched subscriber
        # yet this early in startup.
        for _ in range(5):
            self._detach_object()
            rclpy.spin_once(self, timeout_sec=0.1)
        self.publish_bin_collision_objects()

        initial_msg = self.wait_for_object_pose()
        self.trial_logger.start_trial(initial_msg)

        if initial_msg is None:
            self.get_logger().warn(
                f"No {OBJECT_POSE_TOPIC} received within {OBJECT_POSE_TIMEOUT_SEC:.1f}s. "
                "Using saved-pose fallback sequence."
            )
            success, stage, reason = self.run_saved_pose_sequence()
        else:
            success, stage, reason = self.run_dynamic_object_sequence(initial_msg)

        self.trial_logger.finish(success, failure_stage=None if success else stage, failure_reason=None if success else reason)
        if success:
            self.get_logger().info("Pick-and-place sequence complete.")
        else:
            self.get_logger().error(f"Pick-and-place sequence FAILED at stage={stage}: {reason}")
        return success, stage, reason

    # ── fallback sequence (no perception) ───────────────────────────────────

    def run_saved_pose_sequence(self) -> tuple[bool, str, str]:
        """No object pose available at all — there is nothing to derive a
        grasp target from, so grasp position/orientation stay hardcoded.
        The place side still uses bin-geometry-derived poses for consistency.
        """
        self.command_gripper(GRIPPER_MAX_OPEN)
        if not self.move_to_saved_pose("watch", MotionClass.FREE_SPACE):
            return False, "WATCH", "planning/execution failed"
        if not self.move_to_saved_pose("gripping", MotionClass.CONTACT_SENSITIVE):
            return False, "GRASP", "planning/execution failed (no OMPL fallback for contact motion)"
        gripper_result = self.command_gripper(0.0)
        if not self._grasp_verified(gripper_result):
            self.command_gripper(GRIPPER_MAX_OPEN)
            self.move_to_saved_pose("watch", MotionClass.FREE_SPACE)
            return False, "VERIFY_GRASP", "gripper closed without a stall — likely missed object"

        place_xyz, pre_place_xyz, retreat_xyz, orientation = self._place_targets()
        ok = (
            self.move_to_xyz("pre_place", pre_place_xyz, orientation, MotionClass.FREE_SPACE)
            and self.move_to_xyz("place", place_xyz, orientation, MotionClass.CONTACT_SENSITIVE)
        )
        if not ok:
            return False, "PLACE", "planning/execution failed while carrying object"
        self.command_gripper(GRIPPER_MAX_OPEN)
        if not self._verify_placement():
            self.get_logger().warn("Placement verification failed (object not settled in destination bin)")
        self.move_to_xyz("retreat", retreat_xyz, orientation, MotionClass.CONTACT_SENSITIVE)
        self.move_to_saved_pose("watch", MotionClass.FREE_SPACE)
        return True, "", ""

    # ── dynamic (perception-driven) sequence ────────────────────────────────

    def run_dynamic_object_sequence(self, initial_msg: PoseStamped) -> tuple[bool, str, str]:
        initial_pose = self._transform_object_pose(initial_msg)
        if initial_pose is None:
            return False, "TF_FAILED", "could not transform initial object pose into planning frame"
        obj = initial_pose.pose.position
        self.get_logger().info(
            f"Initial object pose in {PLANNING_FRAME}: ({obj.x:.3f}, {obj.y:.3f}, {obj.z:.3f})"
        )

        candidates = self.generate_grasp_candidates([obj.x, obj.y, obj.z], initial_pose.pose.orientation)
        if not candidates:
            return False, "GRASP", "object orientation too tilted for a reliable top-down grasp"
        max_attempts = min(int(self.get_parameter("grasp_max_attempts").value), len(candidates))

        self.command_gripper(GRIPPER_MAX_OPEN)
        if not self.move_to_saved_pose("watch", MotionClass.FREE_SPACE):
            return False, "WATCH", "planning/execution failed"

        grasp_result = None
        actual_grasp_xyz = None
        winning_candidate = None
        attempts = 0
        for candidate in candidates[:max_attempts]:
            attempts += 1
            grasp_result, actual_grasp_xyz = self._attempt_grasp(candidate, attempt_idx=attempts)
            self.trial_logger.stage(f"grasp_attempt_{attempts}_{candidate.label}", grasp_result)
            if grasp_result is StageResult.SUCCESS:
                winning_candidate = candidate
                break
        self.trial_logger.note(grasp_attempts=attempts)

        if grasp_result is not StageResult.SUCCESS or actual_grasp_xyz is None:
            self.move_to_saved_pose("watch", MotionClass.FREE_SPACE)
            return False, "GRASP", f"exhausted {attempts} grasp candidate(s), last result={grasp_result}"

        # We are now holding the object (best-effort verified). Transfer +
        # place from where the object was ACTUALLY grasped (the re-observed,
        # refined position) — not the stale initial estimate the candidate
        # was generated from.
        result, stage, reason = self._transfer_and_place(winning_candidate, actual_grasp_xyz)
        return (result is StageResult.SUCCESS), stage, reason

    # ── grasp attempt (one candidate) ───────────────────────────────────────

    def _attempt_grasp(self, candidate: GraspCandidate, attempt_idx: int) -> tuple[StageResult, list | None]:
        pre_grasp_xyz = [
            candidate.grasp_xyz[0],
            candidate.grasp_xyz[1],
            candidate.grasp_xyz[2] + candidate.pre_grasp_lift,
        ]
        self.get_logger().info(f"Grasp attempt {attempt_idx} ('{candidate.label}'): pre_grasp={pre_grasp_xyz}")

        if not self.move_to_xyz("dynamic_pre_grasp", pre_grasp_xyz, candidate.orientation, MotionClass.FREE_SPACE):
            return StageResult.PLANNING_FAILED, None

        # Re-observe the object right before final approach (README2.md §11):
        # a stale/ground-truth-only pose from several seconds ago is not
        # trustworthy enough to commit to a contact-sensitive motion.
        refined_xyz = self._refine_grasp_target(candidate.grasp_xyz)
        if refined_xyz is None:
            return StageResult.POSE_TIMEOUT, None

        if not self.move_to_xyz("dynamic_grasp", refined_xyz, candidate.orientation, MotionClass.CONTACT_SENSITIVE):
            # Contact-sensitive motion failed to plan/execute — do not linger
            # at pre_grasp indefinitely, but do not attempt anything fancy
            # either. Caller will move on to the next candidate from watch.
            return StageResult.PLANNING_FAILED, None

        gripper_result = self.command_gripper(0.0)
        if not self._grasp_verified(gripper_result):
            self.command_gripper(GRIPPER_MAX_OPEN)
            self.move_to_xyz("dynamic_pre_grasp_retreat", pre_grasp_xyz, candidate.orientation, MotionClass.CONTACT_SENSITIVE)
            return StageResult.GRASP_FAILED, None

        if not self._test_lift_verify(refined_xyz, candidate):
            self.command_gripper(GRIPPER_MAX_OPEN)
            self.move_to_xyz("dynamic_pre_grasp_retreat", pre_grasp_xyz, candidate.orientation, MotionClass.CONTACT_SENSITIVE)
            return StageResult.LIFT_FAILED, None

        self._attach_object(refined_xyz)
        return StageResult.SUCCESS, refined_xyz

    def _refine_grasp_target(self, initial_grasp_xyz: list) -> list | None:
        timeout = self.get_parameter("refine_pose_timeout").value
        max_age = self.get_parameter("max_object_pose_age").value
        max_jump = self.get_parameter("max_object_pose_jump").value

        refined = self._await_fresh_object_pose(after_ns=self.get_clock().now().nanoseconds, timeout_sec=timeout)
        if refined is None:
            self.get_logger().error("Re-observation failed: no fresh object pose before final approach")
            return None

        age = self._pose_age_sec(refined)
        if age > max_age:
            self.get_logger().error(f"Re-observed pose is stale ({age:.2f}s > {max_age:.2f}s)")
            return None

        p = refined.pose.position
        offset_x = self.get_parameter("grasp_offset_x").value
        offset_y = self.get_parameter("grasp_offset_y").value
        offset_z = self.get_parameter("grasp_offset_z").value
        refined_xyz = [p.x + offset_x, p.y + offset_y, p.z + offset_z]

        jump = math.dist(refined_xyz, initial_grasp_xyz)
        if jump > max_jump:
            self.get_logger().error(
                f"Re-observed grasp target jumped {jump:.3f}m from the initial estimate "
                f"(> max_object_pose_jump={max_jump:.3f}m) — rejecting as unreliable"
            )
            return None

        if not self._within_source_bin(p.x, p.y):
            self.get_logger().error(f"Re-observed object ({p.x:.3f},{p.y:.3f}) is outside the source bin footprint")
            return None

        return refined_xyz

    # ── verification ─────────────────────────────────────────────────────────

    def _grasp_verified(self, gripper_result: GripperCommand.Result | None) -> bool:
        """See trajectory_bridge.py's _execute_gripper docstring for exactly
        what `stalled` does and does not mean (heuristic, not a contact
        sensor)."""
        if gripper_result is None:
            self.get_logger().error("Grasp verification failed: no gripper result (action call failed)")
            return False
        if gripper_result.reached_goal and not gripper_result.stalled:
            self.get_logger().warn("Grasp verification failed: gripper closed fully with no stall (likely missed object)")
            return False
        return bool(gripper_result.stalled)

    def _test_lift_verify(self, grasp_xyz: list, candidate: GraspCandidate) -> bool:
        lift_h = self.get_parameter("test_lift_height").value
        threshold = self.get_parameter("test_lift_z_threshold").value

        pre_lift_pose = self._await_fresh_object_pose(
            after_ns=self.get_clock().now().nanoseconds - int(0.5e9), timeout_sec=1.0
        )
        if pre_lift_pose is None:
            self.get_logger().warn("Test-lift verification skipped: no object-pose feedback available before lift")
            return False
        pre_z = pre_lift_pose.pose.position.z

        test_xyz = [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + lift_h]
        if not self.move_to_xyz(
            "test_lift", test_xyz, candidate.orientation, MotionClass.CONTACT_SENSITIVE, linear_only=True
        ):
            self.get_logger().error("Test-lift motion failed to execute")
            return False

        post_lift_pose = self._await_fresh_object_pose(after_ns=self.get_clock().now().nanoseconds, timeout_sec=1.0)
        if post_lift_pose is None:
            self.get_logger().warn("Test-lift verification skipped: no object-pose feedback available after lift")
            return False

        rise = post_lift_pose.pose.position.z - pre_z
        self.get_logger().info(f"Test-lift: object z rose {rise*1000:.1f}mm (threshold {threshold*1000:.1f}mm)")
        return rise >= threshold

    def _verify_placement(self) -> bool:
        settle_wait = self.get_parameter("place_settle_wait").value
        move_threshold = self.get_parameter("place_settle_move_threshold").value

        p1 = self._await_fresh_object_pose(after_ns=self.get_clock().now().nanoseconds, timeout_sec=1.0)
        if p1 is None:
            self.get_logger().warn("Placement verification skipped: no object-pose feedback")
            return False
        p2 = self._await_fresh_object_pose(
            after_ns=p1.header.stamp.sec * 1_000_000_000 + p1.header.stamp.nanosec + int(settle_wait * 1e9),
            timeout_sec=settle_wait + 1.0,
        )
        if p2 is None:
            self.get_logger().warn("Placement verification skipped: no settle-check sample")
            return False

        moved = math.dist(
            [p1.pose.position.x, p1.pose.position.y, p1.pose.position.z],
            [p2.pose.position.x, p2.pose.position.y, p2.pose.position.z],
        )
        settled = moved < move_threshold

        cx, cy, floor_z = self.dest_bin.center
        w, d, _ = self.dest_bin.inner_size
        wall = self.dest_bin.wall_thickness
        inside_xy = (
            abs(p2.pose.position.x - cx) <= (w / 2.0 - wall)
            and abs(p2.pose.position.y - cy) <= (d / 2.0 - wall)
        )
        # Was bounded by dest_bin's inner_size height, which worked back when
        # that height (0.17m) was far taller than any object -- but that
        # height is now a near-zero "open table" lip (README2.md §7.24), so
        # it's no longer a sane proxy for "how high can a resting object's
        # center plausibly be". Use object size instead: generous 3x half-height
        # covers a normal resting object plus landing on top of another one.
        object_half_height = self.get_parameter("object_half_height").value
        plausible_z = (floor_z - 0.02) <= p2.pose.position.z <= (floor_z + 3.0 * object_half_height)

        ok = settled and inside_xy and plausible_z
        self.get_logger().info(
            f"Placement verification: settled={settled} (moved {moved*1000:.1f}mm) "
            f"inside_xy={inside_xy} plausible_z={plausible_z} -> {'OK' if ok else 'FAILED'}"
        )
        return ok

    def _within_source_bin(self, x: float, y: float) -> bool:
        cx, cy, _ = self.source_bin.center
        w, d, _ = self.source_bin.inner_size
        margin = 0.02
        return abs(x - cx) <= w / 2.0 + margin and abs(y - cy) <= d / 2.0 + margin

    # ── transfer + place ─────────────────────────────────────────────────────

    def _attempt_lift(self, candidate: GraspCandidate, actual_grasp_xyz: list) -> tuple[int, float]:
        """Lift in `lift_step_count` staged increments rather than one big
        straight-line jump, with up to `lift_max_attempts` bounded retries of
        the whole staged sequence.

        We are already at grasp_z + test_lift_height (the test-lift that
        verified the grasp), so this only needs to cover the remaining
        distance to post_grasp_lift. A single long CONTACT_SENSITIVE LIN
        segment right next to the source bin walls is exactly the kind of
        motion that has been observed to fail in practice (see README2.md
        §7) — several short segments have more IK/collision-free options
        than one long one. Returns (1-based attempt number, actual lift
        height used) on success, or (0, 0.0) if all attempts were exhausted.

        IMPORTANT: a *failed* attempt still physically moves the arm partway
        through its steps before the failure is detected, and the abrupt
        stop can shake a marginal grasp loose without triggering any of our
        checks (grasp verification and test-lift both already happened
        earlier, before this method runs). The caller MUST re-confirm the
        object is still actually held after this returns success — see
        _confirm_still_holding() — rather than assuming a successful lift
        means the object is still in the gripper.
        """
        test_lift_h = self.get_parameter("test_lift_height").value
        max_attempts = int(self.get_parameter("lift_max_attempts").value)
        step_count = max(1, int(self.get_parameter("lift_step_count").value))

        # Enforce a geometry-derived floor on top of the configured
        # post_grasp_lift: the OBJECT (not just the TCP) must clear the
        # source bin's wall height by lift_clearance_margin before any
        # horizontal transfer motion, regardless of where in the bin/pile it
        # was picked from. A fixed post_grasp_lift alone left only a few cm
        # of margin for objects near the bin floor and the arm clipped the
        # wall while swinging toward the destination bin.
        grasp_offset_z = self.get_parameter("grasp_offset_z").value
        clearance_margin = self.get_parameter("lift_clearance_margin").value
        object_z_at_grasp = actual_grasp_xyz[2] - grasp_offset_z
        _, _, bin_floor_z = self.source_bin.center
        _, _, bin_height = self.source_bin.inner_size
        min_lift_for_clearance = (bin_floor_z + bin_height + clearance_margin) - object_z_at_grasp
        target_h = max(candidate.post_grasp_lift, min_lift_for_clearance)
        if target_h > candidate.post_grasp_lift + 1e-6:
            self.get_logger().info(
                f"Raising lift target {candidate.post_grasp_lift:.3f}m -> {target_h:.3f}m to clear "
                f"source bin wall (height {bin_height:.3f}m) with {clearance_margin:.3f}m margin"
            )

        for attempt in range(1, max_attempts + 1):
            ok = True
            for step in range(1, step_count + 1):
                frac = step / step_count
                h = test_lift_h + frac * (target_h - test_lift_h)
                waypoint = [actual_grasp_xyz[0], actual_grasp_xyz[1], actual_grasp_xyz[2] + h]
                if not self.move_to_xyz(
                    f"dynamic_lift_a{attempt}_s{step}", waypoint, candidate.orientation,
                    MotionClass.CONTACT_SENSITIVE, linear_only=True,
                ):
                    ok = False
                    self.get_logger().warn(
                        f"Lift attempt {attempt}/{max_attempts} failed at step {step}/{step_count} "
                        f"(height {h:.3f}m)"
                    )
                    break
            if ok:
                return attempt, target_h
        return 0, 0.0

    def _confirm_still_holding(self, actual_grasp_xyz: list, achieved_lift_h: float) -> bool:
        """Re-observe the object after a (possibly retried) lift and check it
        is actually still up where the gripper is holding it, rather than
        having been shaken loose and left back on the bin floor by a failed-
        then-retried lift attempt (see _attempt_lift docstring)."""
        grasp_offset_z = self.get_parameter("grasp_offset_z").value
        expected_object_z = (actual_grasp_xyz[2] - grasp_offset_z) + achieved_lift_h
        tolerance = 0.05  # m -- generous enough for settling, tight enough to catch "back on the floor"

        pose = self._await_fresh_object_pose(
            after_ns=self.get_clock().now().nanoseconds - int(0.5e9), timeout_sec=1.0
        )
        if pose is None:
            self.get_logger().warn("Could not re-confirm grasp after lift: no object-pose feedback")
            return False
        actual_z = pose.pose.position.z
        ok = abs(actual_z - expected_object_z) <= tolerance
        if not ok:
            self.get_logger().error(
                f"Object is not where a held object should be after lift: expected z~"
                f"{expected_object_z:.3f}m, observed z={actual_z:.3f}m (diff="
                f"{abs(actual_z - expected_object_z):.3f}m > {tolerance:.3f}m) — likely shaken loose "
                f"during a failed-then-retried lift attempt. Not proceeding to place an empty gripper."
            )
        return ok

    def _transfer_and_place(self, candidate: GraspCandidate, actual_grasp_xyz: list) -> tuple[StageResult, str, str]:
        lift_attempts, achieved_lift_h = self._attempt_lift(candidate, actual_grasp_xyz)
        self.trial_logger.note(lift_attempts=lift_attempts)
        if lift_attempts == 0:
            # Still (physically) holding the object — do NOT open the gripper
            # blind over an unknown location; that stays a manual-recovery
            # situation. But we MUST detach it from the MoveIt *planning
            # scene* regardless, or every future plan in this move_group
            # session inherits a permanently-in-collision start state (see
            # GRIPPER_TOUCH_LINKS comment / README2.md — this bit us for
            # real: a failed LIFT here poisoned the next run's "watch" move
            # too, because the attach was never undone).
            self.get_logger().error(
                "Lift-off failed while holding the object (all staged-lift retries exhausted). "
                "Not releasing the gripper blind, but detaching from the MoveIt planning scene "
                "so future plans aren't blocked. Manual recovery of the physical grasp likely required."
            )
            self._detach_object()
            return StageResult.LIFT_FAILED, "LIFT", "planning/execution failed while holding object"

        if not self._confirm_still_holding(actual_grasp_xyz, achieved_lift_h):
            # The lift "succeeded" (reached the target height) but the object
            # itself isn't there anymore -- almost certainly shaken loose by
            # an earlier failed-then-retried lift attempt. Don't carry an
            # empty gripper to the destination bin and release nothing.
            self._detach_object()
            return StageResult.LIFT_FAILED, "LIFT", "lift succeeded but object is no longer in the gripper"

        place_xyz, pre_place_xyz, retreat_xyz, orientation = self._place_targets()
        max_attempts = int(self.get_parameter("place_max_attempts").value)
        place_jitters = [(0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02)][:max_attempts]

        pre_place_ok = False
        for i in range(max_attempts):
            if self.move_to_xyz("dynamic_pre_place", pre_place_xyz, orientation, MotionClass.FREE_SPACE):
                pre_place_ok = True
                break
            self.get_logger().warn(f"pre_place attempt {i + 1}/{max_attempts} failed, retrying")
        if not pre_place_ok:
            self._detach_object()
            return StageResult.PLANNING_FAILED, "PRE_PLACE", "planning/execution failed"

        placed = False
        place_attempts = 0
        for i, (dx, dy) in enumerate(place_jitters):
            place_attempts += 1
            jittered = [place_xyz[0] + dx, place_xyz[1] + dy, place_xyz[2]]
            if self.move_to_xyz(
                f"dynamic_place_{i}", jittered, orientation, MotionClass.CONTACT_SENSITIVE, linear_only=True
            ):
                placed = True
                break
            self.get_logger().warn(f"Place attempt {i + 1} failed (dx={dx}, dy={dy}), trying next offset")
        self.trial_logger.note(place_attempts=place_attempts)

        if not placed:
            self.get_logger().error(
                "All place attempts failed while holding the object. Not releasing the gripper "
                "blind, but detaching from the MoveIt planning scene so future plans aren't blocked. "
                "Manual recovery of the physical grasp likely required."
            )
            self._detach_object()
            return StageResult.PLACE_FAILED, "PLACE", "planning/execution failed while holding object"

        self.command_gripper(GRIPPER_MAX_OPEN)
        self._detach_object()
        placement_ok = self._verify_placement()

        self.move_to_xyz(
            "dynamic_retreat", retreat_xyz, orientation, MotionClass.CONTACT_SENSITIVE, linear_only=True
        )
        self.move_to_saved_pose("watch", MotionClass.FREE_SPACE)

        if not placement_ok:
            return StageResult.PLACE_FAILED, "VERIFY_PLACE", "object not settled inside destination bin after release"
        return StageResult.SUCCESS, "", ""

    def _place_targets(self) -> tuple[list, list, list, Quaternion]:
        """Place/pre-place/retreat poses derived from destination-bin geometry
        (README2.md §19), not from a saved fixed pose. Accounts for the TCP
        being `grasp_offset_z` above the object center while carrying it —
        without that correction the object would be dropped `grasp_offset_z`
        too high.
        """
        cx, cy, floor_z = self.dest_bin.center
        _, _, height = self.dest_bin.inner_size
        object_half_height = self.get_parameter("object_half_height").value
        clearance = self.get_parameter("destination_wall_clearance").value
        grasp_offset_z = self.get_parameter("grasp_offset_z").value
        approach_lift = self.get_parameter("place_approach_lift").value

        object_place_z = floor_z + object_half_height + clearance
        tcp_place_z = object_place_z + grasp_offset_z
        place_xyz = [cx, cy, tcp_place_z]
        transit_z = floor_z + height + approach_lift
        pre_place_xyz = [cx, cy, transit_z]
        retreat_xyz = list(pre_place_xyz)
        orientation = self.saved_orientation("gripping")
        return place_xyz, pre_place_xyz, retreat_xyz, orientation

    # ── grasp candidate generation ──────────────────────────────────────────

    def generate_grasp_candidates(self, object_xyz: list, object_orientation: Quaternion) -> list[GraspCandidate]:
        """Build grasp candidates, now yaw-aligned to the object's actual
        orientation (README2.md §7.13) instead of always using a fixed
        compass heading regardless of how the object is rotated.

        Still top-down only: if the object is tilted too far from resting
        flat (on an edge/corner rather than a face), no fixed-tilt top-down
        approach can reliably capture it, so we reject rather than attempt a
        grasp that's likely to fail and knock the object into an even worse
        pose (observed in practice -- see README2.md §7.13).
        """
        tilt = _tilt_from_vertical(object_orientation)
        max_tilt = self.get_parameter("max_object_tilt").value
        if tilt > max_tilt:
            self.get_logger().error(
                f"Object tilt {math.degrees(tilt):.1f}deg exceeds max_object_tilt="
                f"{math.degrees(max_tilt):.1f}deg -- it's resting on an edge/corner, "
                "not a face. A fixed-tilt top-down grasp can't reliably capture this; "
                "rejecting rather than attempting a doomed grasp."
            )
            return []

        object_yaw = _yaw_from_quaternion(object_orientation)
        base_orientation = _quaternion_multiply(_yaw_quaternion(object_yaw), self.saved_orientation("gripping"))

        offset_x = self.get_parameter("grasp_offset_x").value
        offset_y = self.get_parameter("grasp_offset_y").value
        offset_z = self.get_parameter("grasp_offset_z").value
        pre_grasp_lift = self.get_parameter("pre_grasp_lift").value
        post_grasp_lift = self.get_parameter("post_grasp_lift").value

        # (dx, dy, dz, label) — orientation is now yaw-aligned per candidate
        # generation above; what still varies between candidates is grasp
        # depth and a small XY jitter (see GraspCandidate docstring).
        jitters = [
            (0.0, 0.0, 0.0, "nominal"),
            (0.0, 0.0, 0.02, "deeper"),
            (0.01, 0.0, 0.0, "shift+x"),
            (-0.01, 0.0, 0.0, "shift-x"),
        ]
        candidates = []
        for dx, dy, dz, label in jitters:
            grasp_xyz = [
                object_xyz[0] + offset_x + dx,
                object_xyz[1] + offset_y + dy,
                object_xyz[2] + offset_z + dz,
            ]
            candidates.append(
                GraspCandidate(grasp_xyz, base_orientation, pre_grasp_lift, post_grasp_lift, label)
            )
        return candidates

    # ── planning scene: bins + attach/detach ────────────────────────────────

    def publish_bin_collision_objects(self):
        objects = make_bin_collision_objects("source_bin", self.source_bin)
        objects += make_bin_collision_objects("dest_bin", self.dest_bin)

        for _ in range(5):
            for obj in objects:
                obj.header.stamp = self.get_clock().now().to_msg()
                self.collision_pub.publish(obj)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Published {len(objects)} bin collision objects to /collision_object")

    def _attach_object(self, object_xyz: list):
        """Add the grasped object to the planning scene and attach it to the
        EEF link so subsequent transfer motions plan with its geometry.
        This is planning-scene bookkeeping only — it does not physically
        attach anything in Isaac Sim (README2.md §18); physical holding is
        whatever the gripper/friction is actually doing.
        """
        object_half_height = self.get_parameter("object_half_height").value
        size = object_half_height * 2.0
        world_obj = make_box_collision_object(TARGET_OBJECT_ID, object_xyz, [size, size, size], frame_id=PLANNING_FRAME)
        self.collision_pub.publish(world_obj)

        attached = AttachedCollisionObject()
        attached.link_name = EEF_LINK
        attached.object = world_obj
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = GRIPPER_TOUCH_LINKS
        self.attached_pub.publish(attached)

    def _detach_object(self):
        """Detach AND fully remove target_object from the planning scene.

        MoveIt's AttachedCollisionObject REMOVE only detaches the object from
        the link -- it does NOT delete it. The object reappears as a static
        WORLD collision object at wherever it was when detached, i.e. right
        next to the gripper (since that's where release happens). Confirmed
        via move_group's log immediately after a real run: "Found a contact
        between 'target_object' (type 'Object') and 'robotiq_85_base_link'"
        followed by "Start state appears to be in collision" -- this silently
        blocked every subsequent plan (the next retreat/watch move in the
        same trial, and every move in the next trial) until the process was
        restarted. Explicitly removing the world copy too avoids leaving a
        phantom obstacle behind. We don't track the object's true final
        resting pose precisely enough to re-publish it correctly anyway, so
        just dropping it from the scene is the safer choice here.
        """
        detach = AttachedCollisionObject()
        detach.link_name = EEF_LINK
        detach.object.id = TARGET_OBJECT_ID
        detach.object.operation = CollisionObject.REMOVE
        self.attached_pub.publish(detach)

        world_remove = CollisionObject()
        world_remove.id = TARGET_OBJECT_ID
        world_remove.header.frame_id = PLANNING_FRAME
        world_remove.operation = CollisionObject.REMOVE
        self.collision_pub.publish(world_remove)

    # ── object pose: subscription, TF transform, re-observation ────────────

    def _object_pose_cb(self, msg: PoseStamped):
        self.object_pose = msg

    def wait_for_object_pose(self) -> PoseStamped | None:
        deadline = time.monotonic() + OBJECT_POSE_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            if self.object_pose is not None:
                return self.object_pose
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def _pose_age_sec(self, pose: PoseStamped) -> float:
        stamp_ns = pose.header.stamp.sec * 1_000_000_000 + pose.header.stamp.nanosec
        return (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9

    def _await_fresh_object_pose(self, after_ns: int, timeout_sec: float) -> PoseStamped | None:
        """Block (with manual spinning) until a NEW /binpicking/object_pose
        message (stamped after `after_ns`) arrives, then return it transformed
        into PLANNING_FRAME. Never returns a stale or untransformed pose."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            msg = self.object_pose
            if msg is not None:
                stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                if stamp_ns > after_ns:
                    return self._transform_object_pose(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(f"No fresh {OBJECT_POSE_TOPIC} within {timeout_sec:.2f}s")
        return None

    def _transform_object_pose(self, msg: PoseStamped, timeout_sec: float = 1.0) -> PoseStamped | None:
        """Transform a stamped object pose into PLANNING_FRAME via TF2 — never
        interpret raw XYZ as already being in the planning frame unless it is
        (README2.md §10). Uses manual spin_once polling because this node
        does not run a background executor thread, so Buffer.transform's
        internal wait would otherwise block without ever processing /tf.
        """
        if msg.header.frame_id == PLANNING_FRAME:
            return msg
        target_time = Time.from_msg(msg.header.stamp)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.tf_buffer.can_transform(
                PLANNING_FRAME, msg.header.frame_id, target_time, timeout=Duration(seconds=0.0)
            ):
                try:
                    return self.tf_buffer.transform(msg, PLANNING_FRAME, timeout=Duration(seconds=0.0))
                except TransformException as exc:
                    self.get_logger().error(
                        f"TF transform {msg.header.frame_id}->{PLANNING_FRAME} failed: {exc}"
                    )
                    return None
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            f"TF {msg.header.frame_id}->{PLANNING_FRAME} not available within {timeout_sec:.1f}s "
            f"at stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        )
        return None

    # ── EEF orientation / TF lookups (unchanged from prior version) ────────

    def lookup_eef_orientation(self, timeout_sec: float = 5.0) -> Quaternion | None:
        """Return the current EEF orientation when TF is available.

        The running stack may publish two TF trees:
        - robot_state_publisher frames: link0, link1..link6, tcp
        - Isaac Sim frames: world, base_link, link_1..link_6

        If the URDF dynamic chain is not connected, link0->tcp is unavailable.
        In that case, use Isaac's world->link_6 orientation as a practical
        fallback because tcp has the same fixed orientation as link6 in the URDF.
        """
        pose = self.lookup_pose_transform(PLANNING_FRAME, EEF_LINK, timeout_sec=timeout_sec, log_frames=False)
        if pose is not None:
            return pose.pose.orientation

        for parent, child in [("world", "link_6"), ("world", "tcp")]:
            pose = self.lookup_pose_transform(parent, child, timeout_sec=1.0, log_frames=False)
            if pose is not None:
                self.get_logger().warn(
                    f"Using fallback orientation from TF {parent}->{child}; "
                    f"{PLANNING_FRAME}->{EEF_LINK} is not connected."
                )
                return pose.pose.orientation

        frames = self.tf_buffer.all_frames_as_string()
        if frames:
            self.get_logger().error(f"Known TF frames:\n{frames}")
        self.get_logger().warn("No usable EEF orientation TF found. Continuing with position-only goals.")
        return None

    def lookup_pose_transform(
        self, parent_frame: str, child_frame: str, timeout_sec: float = 5.0, log_frames: bool = True
    ) -> PoseStamped | None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(parent_frame, child_frame, Time(), timeout=Duration(seconds=0.2))
                pose = PoseStamped()
                pose.header = tf.header
                pose.pose.position.x = tf.transform.translation.x
                pose.pose.position.y = tf.transform.translation.y
                pose.pose.position.z = tf.transform.translation.z
                pose.pose.orientation = tf.transform.rotation
                return pose
            except TransformException as exc:
                self.get_logger().debug(f"Waiting for TF {parent_frame}->{child_frame}: {exc}")
                rclpy.spin_once(self, timeout_sec=0.1)
        if log_frames:
            frames = self.tf_buffer.all_frames_as_string()
            if frames:
                self.get_logger().error(f"Known TF frames:\n{frames}")
        return None

    # ── motion helpers ───────────────────────────────────────────────────────

    def move_to_xyz(
        self,
        name: str,
        xyz: Iterable[float],
        orientation: Quaternion | None,
        motion_class: MotionClass,
        linear_only: bool = False,
    ) -> bool:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
        if orientation is not None:
            pose.orientation = orientation
        return self.move_to_pose(
            name, pose, use_orientation=orientation is not None, motion_class=motion_class, linear_only=linear_only
        )

    def move_to_saved_pose(self, name: str, motion_class: MotionClass) -> bool:
        saved = SAVED_POSES[name]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) for v in saved["position"]]
        qx, qy, qz, qw = [float(v) for v in saved["orientation"]]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        return self.move_to_pose(name, pose, use_orientation=True, motion_class=motion_class)

    def saved_orientation(self, name: str) -> Quaternion:
        saved = SAVED_POSES[name]
        qx, qy, qz, qw = [float(v) for v in saved["orientation"]]
        q = Quaternion()
        q.x, q.y, q.z, q.w = qx, qy, qz, qw
        return q

    def move_to_pose(
        self,
        name: str,
        pose: Pose,
        use_orientation: bool,
        motion_class: MotionClass,
        linear_only: bool = False,
    ) -> bool:
        """Plan+execute one EEF pose goal.

        Planner policy (README2.md §13, revised §7.10/§7.12):
          FREE_SPACE:         Pilz LIN first (if orientation given), OMPL fallback.
          CONTACT_SENSITIVE:  Pilz LIN first, OMPL fallback -- but the OMPL
                               fallback is ALWAYS run with a path orientation
                               constraint (independent of the
                               use_path_orientation_constraint param), so the
                               wrist still cannot spin freely mid-path even
                               when OMPL is used here.
          linear_only=True:   No OMPL fallback at all, regardless of
                               motion_class. Only Pilz LIN can satisfy an
                               exact straight-line Cartesian path (OMPL only
                               constrains orientation along the path, never
                               position) -- use this when the motion must be
                               a true straight line in the world frame, e.g.
                               lifting an object straight up in +Z after a
                               successful grasp (README2.md §7.12). Fails
                               rather than wandering off-line.

        Originally CONTACT_SENSITIVE was LIN-only with no OMPL fallback at
        all, to avoid OMPL picking an unpredictable approach direction near
        the object/bin. In practice this was too strict: Pilz LIN returning
        NO_IK_SOLUTION can mean a grasp target is permanently unreachable via
        a straight line even though a (safely orientation-constrained) bent
        path exists, and the arm would bob at pre-grasp forever across every
        candidate, never actually descending. The path constraint on the
        fallback keeps the original safety property (no free wrist spin)
        while no longer refusing outright. linear_only reintroduces the
        strict LIN-only behavior for the specific motions where a bent path
        would defeat the purpose of the motion itself (a "lift" that
        wanders in X/Y isn't a lift).
        """
        attempts = []
        if use_orientation:
            attempts.append((PIPELINE_PILZ, PLANNER_ID_LIN, False))

        if linear_only:
            pass  # LIN (if appended above) or nothing -- no OMPL fallback.
        elif motion_class is MotionClass.FREE_SPACE:
            requested_path_constraint = (
                use_orientation and bool(self.get_parameter("use_path_orientation_constraint").value)
            )
            attempts.append((PIPELINE_OMPL, "", requested_path_constraint))
            if requested_path_constraint:
                attempts.append((PIPELINE_OMPL, "", False))
        else:
            # CONTACT_SENSITIVE: OMPL fallback forced to respect orientation
            # along the whole path whenever we have an orientation to
            # constrain to (see docstring).
            attempts.append((PIPELINE_OMPL, "", use_orientation))

        last_error = None
        for pipeline_id, planner_id, use_path_constraint in attempts:
            error_code = self._move_to_pose_once(
                name, pose, use_orientation=use_orientation, use_path_constraint=use_path_constraint,
                pipeline_id=pipeline_id, planner_id=planner_id,
            )
            if error_code == MoveItErrorCodes.SUCCESS:
                return True
            last_error = error_code
            self.get_logger().warn(
                f"MoveGroup failed at '{name}' with pipeline={pipeline_id} "
                f"planner={planner_id or 'default'} (error {error_code}); "
            )

        self.get_logger().error(f"MoveGroup exhausted all planners at '{name}' (last error {last_error})")
        return False

    def _move_to_pose_once(
        self, name: str, pose: Pose, use_orientation: bool, use_path_constraint: bool,
        pipeline_id: str = PIPELINE_OMPL, planner_id: str = "",
    ) -> int:
        self.get_logger().info(
            f"Moving to {name}: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}), "
            f"pipeline={pipeline_id}, planner={planner_id or 'default'}, path_orientation_constraint={use_path_constraint}"
        )
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.2
        goal.request.max_acceleration_scaling_factor = 0.2
        goal.request.pipeline_id = pipeline_id
        if planner_id:
            goal.request.planner_id = planner_id
        goal.request.start_state.is_diff = True
        goal.request.workspace_parameters.header.frame_id = PLANNING_FRAME
        goal.request.workspace_parameters.min_corner.x = 0.0
        goal.request.workspace_parameters.min_corner.y = -1.0
        goal.request.workspace_parameters.min_corner.z = 0.0
        goal.request.workspace_parameters.max_corner.x = 1.0
        goal.request.workspace_parameters.max_corner.y = 1.0
        goal.request.workspace_parameters.max_corner.z = 1.2
        goal.request.goal_constraints = [self.make_pose_constraints(pose, use_orientation=use_orientation)]
        if use_orientation and use_path_constraint:
            goal.request.path_constraints.orientation_constraints = [
                self.make_orientation_constraint(pose.orientation)
            ]

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"MoveGroup rejected goal '{name}'")
            return MoveItErrorCodes.FAILURE

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return result.error_code.val
        self.get_logger().info(f"Reached {name}")
        return MoveItErrorCodes.SUCCESS

    def make_pose_constraints(self, pose: Pose, use_orientation: bool) -> Constraints:
        constraints = Constraints()
        constraints.name = "eef_pose_goal"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(self.get_parameter("position_tolerance").value)]

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = PLANNING_FRAME
        pos_constraint.link_name = EEF_LINK
        pos_constraint.constraint_region.primitives = [sphere]
        pos_constraint.constraint_region.primitive_poses = [pose]
        pos_constraint.weight = 1.0

        constraints.position_constraints = [pos_constraint]
        if use_orientation:
            constraints.orientation_constraints = [self.make_orientation_constraint(pose.orientation)]
        return constraints

    def make_orientation_constraint(self, orientation: Quaternion) -> OrientationConstraint:
        tolerance = float(self.get_parameter("orientation_tolerance").value)
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = PLANNING_FRAME
        orient_constraint.link_name = EEF_LINK
        orient_constraint.orientation = orientation
        orient_constraint.absolute_x_axis_tolerance = tolerance
        orient_constraint.absolute_y_axis_tolerance = tolerance
        orient_constraint.absolute_z_axis_tolerance = tolerance
        orient_constraint.weight = 1.0
        return orient_constraint

    # ── gripper (via GripperCommand action, so trajectory_bridge's stall
    #    signal is actually usable — see README2.md root-cause list) ────────

    def command_gripper(self, opening_m: float) -> GripperCommand.Result | None:
        opening_m = max(0.0, min(GRIPPER_MAX_OPEN, float(opening_m)))
        label = "open" if math.isclose(opening_m, GRIPPER_MAX_OPEN, abs_tol=1e-5) else "close"
        self.get_logger().info(f"Gripper {label}: opening={opening_m:.3f} m")

        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Gripper action server {GRIPPER_ACTION} unavailable")
            return None

        goal = GripperCommand.Goal()
        goal.command.position = opening_m
        goal.command.max_effort = 0.0

        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(
            f"Gripper result: position={result.position:.4f} reached_goal={result.reached_goal} "
            f"stalled={result.stalled}"
        )
        return result


def main(args=None):
    rclpy.init(args=args)
    node = RB5MoveItPickPlace()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
