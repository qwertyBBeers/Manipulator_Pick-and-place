#!/usr/bin/env python3
"""
Fork of rb5_isaac/rb5_isaac/trajectory_bridge.py for the two-robot relay demo.

The only change from the original is ACTION_NAME/GRIPPER_ACTION below: the
original hardcodes them as ROS *global* names (leading "/"), which rclpy's
ActionServer does not apply remapping/namespacing to (verified empirically --
a `-r /name:=relative_name` launch remap is silently ignored for actions with
an absolute name, at least on this ROS2 Humble build). Two robots running the
original file would both register on the exact same global action name and
collide. Forked (not edited in place) because that file is heavily tuned
production code for the single-robot demo and this avoids touching it.

Trajectory bridge between MoveIt2 and Isaac Sim.

Follows the same pattern as ros2_controllers/joint_trajectory_controller:
  - Cubic Hermite spline when waypoints include velocities
  - Linear fallback when velocities are absent
  - Runs a control loop at COMMAND_HZ (default 100 Hz)
  - Validates trajectories before executing them
  - Reports SUCCESS only after actual /joint_states feedback confirms the
    goal was reached and has settled (not merely "nominal time elapsed")

Bridges:
  MoveIt2  →  FollowJointTrajectory action  →  /isaac_joint_commands  →  Isaac Sim
  Isaac Sim →  /joint_states  →  feedback to MoveIt2

Known limitation: this node does not parse the URDF, so it has no independent
knowledge of joint position/velocity limits. The `max_command_jump` guard is a
cheap regression check against wild single-cycle discontinuities (the class of
bug that caused the old "sudden fast rotation" issue), not a substitute for
real joint-limit enforcement, which MoveIt/OMPL/Pilz already do at planning
time.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time as RclpyTime

from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES    = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
ACTION_NAME    = "joint_trajectory_controller/follow_joint_trajectory"
COMMAND_HZ     = 100.0   # control loop rate sent to Isaac Sim

# FollowJointTrajectory.Result error codes (control_msgs). We only have these
# five buckets to work with, so some are deliberately repurposed with a
# documented meaning specific to this bridge:
#   INVALID_GOAL            malformed trajectory (bad lengths / non-finite /
#                            non-monotonic timestamps / duplicate timestamps)
#   INVALID_JOINTS           trajectory names a joint we don't know about, or
#                            is missing a required arm joint
#   OLD_HEADER_TIMESTAMP     /joint_states feedback is stale — we cannot trust
#                            the current state enough to start or continue
#   PATH_TOLERANCE_VIOLATED  start-state mismatch, mid-execution tracking
#                            error, or an unsafe single-cycle command jump
#   GOAL_TOLERANCE_VIOLATED  did not settle within goal_time_tolerance
SUCCESSFUL              = FollowJointTrajectory.Result.SUCCESSFUL
INVALID_GOAL             = FollowJointTrajectory.Result.INVALID_GOAL
INVALID_JOINTS           = FollowJointTrajectory.Result.INVALID_JOINTS
OLD_HEADER_TIMESTAMP     = FollowJointTrajectory.Result.OLD_HEADER_TIMESTAMP
PATH_TOLERANCE_VIOLATED  = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
GOAL_TOLERANCE_VIOLATED  = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED

# Robotiq 2F-85 gripper
# left_knuckle is primary; right/inner/tip are mimics driven simultaneously.
# Mimic ratios (multiplier relative to left_knuckle):
#   right_knuckle      : -1   (mirror)
#   left_inner_knuckle : -1
#   right_inner_knuckle: -1
#   left_finger_tip    : +1
#   right_finger_tip   : +1
GRIPPER_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GRIPPER_MIMIC  = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0]  # multipliers vs left_knuckle (from URDF)
# Slowest simulation the wall-clock backstops assume. Only used to size those
# backstops, never the motion itself: at RTF 0.1 a 2s simulated ramp is 20s of
# wall time, and waiting that long beats aborting a ramp that is progressing.
MIN_EXPECTED_RTF = 0.1
GRIPPER_ACTION = "gripper_controller/gripper_cmd"
GRIPPER_MAX_OPEN   = 0.085   # 85 mm opening → knuckle = 0 rad
GRIPPER_MAX_CLOSED = 0.0     # 0 mm  opening → knuckle = 0.8 rad
KNUCKLE_MAX        = 0.8     # rad, from Robotiq URDF limit


# ── Cubic Hermite spline ─────────────────────────────────────────────────────
# Same math as ros2_controllers: uses (t, pos, vel) at each waypoint end.
# Gives C1-continuous (smooth velocity) motion across segment boundaries.
#
#   h00(s) =  2s³ - 3s²+ 1   (position blend from p0)
#   h10(s) =   s³ - 2s²+ s   (velocity blend from v0, scaled by Δt)
#   h01(s) = -2s³ + 3s²      (position blend from p1)
#   h11(s) =   s³ -  s²      (velocity blend from v1, scaled by Δt)
#   s = (t - t0) / Δt

def _cubic_hermite(t0: float, p0: float, v0: float,
                   t1: float, p1: float, v1: float,
                   t:  float):
    """Return (position, velocity, clamped) at time t using cubic Hermite spline."""
    dt = t1 - t0
    if dt < 1e-9:
        return p1, v1, False
    s  = (t - t0) / dt
    clamped = s < -0.05 or s > 1.05
    s = min(1.0, max(0.0, s))
    s2 = s * s
    s3 = s2 * s

    h00 = 2*s3 - 3*s2 + 1
    h10 = dt * (s3 - 2*s2 + s)
    h01 = -2*s3 + 3*s2
    h11 = dt * (s3 - s2)
    pos = h00*p0 + h10*v0 + h01*p1 + h11*v1

    # d/dt of the spline (velocity)
    dh00 = (6*s2 - 6*s) / dt
    dh10 = 3*s2 - 4*s + 1
    dh01 = (-6*s2 + 6*s) / dt
    dh11 = 3*s2 - 2*s
    vel  = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1

    return pos, vel, clamped


def _interp_segment(t0, p0, v0, t1, p1, v1, t, has_vel):
    """Cubic if velocities available, linear otherwise."""
    if has_vel:
        return _cubic_hermite(t0, p0, v0, t1, p1, v1, t)
    # linear position, constant velocity
    s = (t - t0) / max(t1 - t0, 1e-9)
    clamped = s < -0.05 or s > 1.05
    s = min(1.0, max(0.0, s))
    pos = p0 + s * (p1 - p0)
    vel = (p1 - p0) / max(t1 - t0, 1e-9)
    return pos, vel, clamped


class TrajectoryValidationError(Exception):
    """Raised when an incoming trajectory fails validation. Carries the
    FollowJointTrajectory.Result error code that should be returned."""

    def __init__(self, error_code: int, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass
class PreparedTrajectory:
    joint_names: list
    times: list          # length N+1, times[0] == 0.0 (current state)
    pos_arr: np.ndarray  # (N+1, n_joints)
    vel_arr: np.ndarray  # (N+1, n_joints)
    has_vel: bool


# ── Node ─────────────────────────────────────────────────────────────────────

class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__("trajectory_bridge")
        self._cb_group = ReentrantCallbackGroup()

        # ── Tunable execution-verification parameters (see README2.md §5) ──
        self.declare_parameter("path_tolerance", 0.08)              # rad, mid-execution tracking error bound
        self.declare_parameter("goal_tolerance", 0.02)               # rad, final position tolerance
        self.declare_parameter("stopped_velocity_tolerance", 0.02)   # rad/s, "settled" velocity bound
        self.declare_parameter("goal_time_tolerance", 1.0)           # s, max extra time to wait for settling
        self.declare_parameter("joint_state_timeout", 0.5)           # s, max age of /joint_states we'll trust
        self.declare_parameter("settling_time", 0.2)                 # s, goal+vel tolerance must hold this long
        self.declare_parameter("allowed_start_tolerance", 0.05)      # rad, max |current - first_waypoint| at t=0
        self.declare_parameter("max_command_jump", 0.3)              # rad, max |cmd[k] - cmd[k-1]| per control cycle
        self.declare_parameter("path_tolerance_grace_period", 0.2)   # s, no path-tolerance check during startup transient
        # Was a single fixed 0.5s ramp for both directions. A fast close on a
        # small/light object can knock it away before the fingers actually
        # capture it -- lower momentum at first contact by closing slower
        # than opening (README2.md §7.14). Opening doesn't need to be gentle.
        # Was 1.2 -- still fast enough that the passive-linkage joints
        # (independent PD drives, not a true mechanical constraint -- see
        # README2.md §7.18/§7.19/§7.27) could overshoot/tilt under contact
        # momentum, letting an edge of the fingertip pad mesh poke inward and
        # push the object away before the pad closed flat against it.
        # Slower closing gives those joints more time to settle into a
        # roughly-parallel shape at each step instead of arriving with speed.
        # All three gripper durations are SIMULATED seconds (ROS clock), not wall
        # seconds -- see _execute_gripper.
        self.declare_parameter("gripper_close_duration", 2.0)        # s, ramp time when closing (opening_m decreases)
        self.declare_parameter("gripper_open_duration", 0.5)         # s, ramp time when opening
        self.declare_parameter("gripper_settle_duration", 0.3)       # s, wait after ramp before reading back actual position

        self._path_tolerance = self.get_parameter("path_tolerance").value
        self._goal_tolerance = self.get_parameter("goal_tolerance").value
        self._stopped_velocity_tolerance = self.get_parameter("stopped_velocity_tolerance").value
        self._goal_time_tolerance = self.get_parameter("goal_time_tolerance").value
        self._joint_state_timeout = self.get_parameter("joint_state_timeout").value
        self._settling_time = self.get_parameter("settling_time").value
        self._allowed_start_tolerance = self.get_parameter("allowed_start_tolerance").value
        self._max_command_jump = self.get_parameter("max_command_jump").value
        self._path_tolerance_grace_period = self.get_parameter("path_tolerance_grace_period").value

        # Arm command publisher → Isaac Sim
        self._cmd_pub = self.create_publisher(
            JointState, "isaac_joint_commands", 10
        )
        # Gripper command publisher → Isaac Sim
        self._gripper_pub = self.create_publisher(
            JointState, "gripper_joint_commands", 10
        )

        # Current joint states (arm + gripper knuckle angle, open=0 rad)
        self._known_joints = set(JOINT_NAMES) | set(GRIPPER_JOINTS)
        self._current_pos = {j: 0.0 for j in self._known_joints}
        self._current_vel = {j: 0.0 for j in self._known_joints}
        self._prev_sample_pos = dict(self._current_pos)
        self._prev_sample_time = None  # rclpy Time, for finite-difference velocity fallback
        self._last_joint_state_stamp = None  # rclpy Time of last /joint_states message
        self._gripper_knuckle = 0.0  # left_knuckle angle (0=open, 0.8=closed)
        self.create_subscription(
            JointState, "joint_states", self._joint_states_cb, 10,
            callback_group=self._cb_group,
        )

        # Arm trajectory action server ← MoveIt2
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
            execute_callback=self._execute_trajectory,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        # Gripper action server ← GripperCommand (open/close)
        # goal.command.position: 0.0 = closed, FINGER_MAX_OPEN = fully open
        self._gripper_server = ActionServer(
            self,
            GripperCommand,
            GRIPPER_ACTION,
            execute_callback=self._execute_gripper,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"TrajectoryBridge ready.\n"
            f"  Arm action:     {ACTION_NAME} (cubic Hermite @ {COMMAND_HZ:.0f} Hz)\n"
            f"  Gripper action: {GRIPPER_ACTION}\n"
            f"  Publishing:     /isaac_joint_commands  /gripper_joint_commands\n"
            f"  Subscribing:    /joint_states\n"
            f"  Verification:   goal_tol={self._goal_tolerance:.3f}rad "
            f"path_tol={self._path_tolerance:.3f}rad "
            f"settle={self._settling_time:.2f}s "
            f"goal_time_tol={self._goal_time_tolerance:.2f}s"
        )

    # ── callbacks ────────────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState):
        now = self.get_clock().now()
        stamp = RclpyTime.from_msg(msg.header.stamp) if msg.header.stamp.sec or msg.header.stamp.nanosec else now
        has_msg_vel = len(msg.velocity) == len(msg.name)

        for i, name in enumerate(msg.name):
            if name not in self._current_pos:
                continue
            self._current_pos[name] = msg.position[i]
            if has_msg_vel:
                self._current_vel[name] = msg.velocity[i]

        if not has_msg_vel and self._prev_sample_time is not None:
            dt = (stamp - self._prev_sample_time).nanoseconds * 1e-9
            if dt > 1e-4:
                for name in msg.name:
                    if name not in self._current_pos:
                        continue
                    prev = self._prev_sample_pos.get(name, self._current_pos[name])
                    self._current_vel[name] = (self._current_pos[name] - prev) / dt

        if not has_msg_vel:
            self._prev_sample_pos.update(self._current_pos)
            self._prev_sample_time = stamp

        self._last_joint_state_stamp = now

    def _joint_state_age_sec(self) -> float:
        if self._last_joint_state_stamp is None:
            return float("inf")
        return (self.get_clock().now() - self._last_joint_state_stamp).nanoseconds * 1e-9

    # ── trajectory validation ────────────────────────────────────────────────

    def _prepare_trajectory(self, traj) -> PreparedTrajectory:
        """Validate an incoming trajectory and build interpolation arrays.

        Raises TrajectoryValidationError on any malformed input. Never
        silently coerces bad data into "safe-looking" defaults — an unknown
        joint name or a non-finite value must abort, not fall back to 0.0.
        """
        joint_names = list(traj.joint_names)
        raw_points  = traj.points

        if not joint_names:
            raise TrajectoryValidationError(INVALID_GOAL, "trajectory has no joint_names")
        if len(set(joint_names)) != len(joint_names):
            raise TrajectoryValidationError(INVALID_GOAL, f"duplicate joint names: {joint_names}")
        unknown = [j for j in joint_names if j not in self._known_joints]
        if unknown:
            raise TrajectoryValidationError(
                INVALID_JOINTS, f"unknown joint(s) in trajectory: {unknown}"
            )
        missing_arm = [j for j in JOINT_NAMES if j not in joint_names]
        if missing_arm and any(j in joint_names for j in JOINT_NAMES):
            # Partial arm trajectories are ambiguous: we can't tell whether the
            # missing joints should hold position or are simply omitted.
            raise TrajectoryValidationError(
                INVALID_JOINTS,
                f"trajectory covers some but not all arm joints; missing {missing_arm}",
            )
        if not raw_points:
            raise TrajectoryValidationError(INVALID_GOAL, "trajectory has no points")

        n_joints = len(joint_names)
        has_vel = bool(raw_points[0].velocities)

        prev_t = -1.0
        for idx, p in enumerate(raw_points):
            if len(p.positions) != n_joints:
                raise TrajectoryValidationError(
                    INVALID_GOAL,
                    f"point {idx}: {len(p.positions)} positions, expected {n_joints}",
                )
            point_has_vel = bool(p.velocities)
            if point_has_vel != has_vel:
                raise TrajectoryValidationError(
                    INVALID_GOAL,
                    f"point {idx}: inconsistent velocity availability "
                    f"(mixing waypoints with/without velocities is not supported)",
                )
            if point_has_vel and len(p.velocities) != n_joints:
                raise TrajectoryValidationError(
                    INVALID_GOAL,
                    f"point {idx}: {len(p.velocities)} velocities, expected {n_joints}",
                )
            if not all(np.isfinite(p.positions)):
                raise TrajectoryValidationError(INVALID_GOAL, f"point {idx}: non-finite position")
            if point_has_vel and not all(np.isfinite(p.velocities)):
                raise TrajectoryValidationError(INVALID_GOAL, f"point {idx}: non-finite velocity")

            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            if not np.isfinite(t) or t < 0.0:
                raise TrajectoryValidationError(INVALID_GOAL, f"point {idx}: invalid time_from_start={t}")
            if t <= prev_t:
                raise TrajectoryValidationError(
                    INVALID_GOAL,
                    f"point {idx}: time_from_start={t:.4f} not strictly increasing "
                    f"(previous={prev_t:.4f}); duplicate/out-of-order timestamps rejected",
                )
            prev_t = t

        if self._joint_state_age_sec() > self._joint_state_timeout:
            raise TrajectoryValidationError(
                OLD_HEADER_TIMESTAMP,
                f"/joint_states is stale ({self._joint_state_age_sec():.2f}s > "
                f"{self._joint_state_timeout:.2f}s) — refusing to trust current state",
            )

        first = raw_points[0]
        start_err = max(
            abs(self._current_pos[j] - first.positions[i]) for i, j in enumerate(joint_names)
        )
        if start_err > self._allowed_start_tolerance:
            raise TrajectoryValidationError(
                PATH_TOLERANCE_VIOLATED,
                f"start-state mismatch {start_err:.4f}rad exceeds allowed_start_tolerance="
                f"{self._allowed_start_tolerance:.4f}rad — replan required, not silently jumping",
            )

        times = [0.0] + [
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in raw_points
        ]
        pos_arr = np.array(
            [[self._current_pos[j] for j in joint_names]] +
            [list(p.positions) for p in raw_points]
        )
        if has_vel:
            vel_arr = np.array(
                [[0.0] * n_joints] + [list(p.velocities) for p in raw_points]
            )
        else:
            vel_arr = np.zeros_like(pos_arr)

        return PreparedTrajectory(joint_names, times, pos_arr, vel_arr, has_vel)

    def _execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory

        try:
            prepared = self._prepare_trajectory(traj)
        except TrajectoryValidationError as exc:
            self.get_logger().error(f"Trajectory rejected: {exc.message}")
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = exc.error_code
            result.error_string = exc.message
            return result

        joint_names = prepared.joint_names
        times = prepared.times
        pos_arr = prepared.pos_arr
        vel_arr = prepared.vel_arr
        has_vel = prepared.has_vel
        n_joints = len(joint_names)

        self.get_logger().info(
            f"Executing: {len(times) - 1} waypoints, joints={joint_names}, "
            f"has_vel={has_vel}, duration={times[-1]:.2f}s"
        )

        total_duration = times[-1]
        dt = 1.0 / COMMAND_HZ
        start_ros = self.get_clock().now()

        prev_cmd_pos = pos_arr[0].copy()
        clamp_warn_count = 0
        settled_since: RclpyTime | None = None
        goal_pos = pos_arr[-1]

        # ── Phase 1: follow the interpolated trajectory ─────────────────────
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Trajectory cancelled.")
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "canceled"
                return result

            if self._joint_state_age_sec() > self._joint_state_timeout:
                self.get_logger().error(
                    f"Aborting: /joint_states went stale mid-execution "
                    f"({self._joint_state_age_sec():.2f}s)"
                )
                goal_handle.abort()
                return self._make_result(OLD_HEADER_TIMESTAMP, "joint_states stale mid-execution")

            elapsed = (self.get_clock().now() - start_ros).nanoseconds * 1e-9
            t = min(elapsed, total_duration)

            seg = max(0, np.searchsorted(times, t, side='right') - 1)
            seg = min(seg, len(times) - 2)
            t0, t1 = times[seg], times[seg + 1]

            interp_pos = np.empty(n_joints)
            interp_vel = np.empty(n_joints)
            seg_clamped = False
            for j in range(n_joints):
                p, v, clamped = _interp_segment(
                    t0, pos_arr[seg, j], vel_arr[seg, j],
                    t1, pos_arr[seg + 1, j], vel_arr[seg + 1, j],
                    t, has_vel,
                )
                interp_pos[j] = p
                interp_vel[j] = v
                seg_clamped = seg_clamped or clamped

            if seg_clamped:
                clamp_warn_count += 1
                if clamp_warn_count <= 3 or clamp_warn_count % 100 == 0:
                    self.get_logger().warn(
                        f"Hermite interpolation clamped s outside [0,1] at t={t:.3f}s "
                        f"(segment {seg}); this should be rare — investigate if frequent."
                    )

            if not np.all(np.isfinite(interp_pos)) or not np.all(np.isfinite(interp_vel)):
                self.get_logger().error("Aborting: interpolated command contains NaN/Inf")
                goal_handle.abort()
                return self._make_result(INVALID_GOAL, "interpolated command was non-finite")

            jump = np.max(np.abs(interp_pos - prev_cmd_pos))
            if jump > self._max_command_jump:
                self.get_logger().error(
                    f"Aborting: command jump {jump:.4f}rad exceeds max_command_jump="
                    f"{self._max_command_jump:.4f}rad at t={t:.3f}s (segment {seg}) — "
                    f"refusing to publish a potentially dangerous discontinuous command"
                )
                goal_handle.abort()
                return self._make_result(PATH_TOLERANCE_VIOLATED, "unsafe single-cycle command jump")

            # Mid-execution tracking-error check, after a short startup grace
            # period (the real robot/sim needs a moment to start accelerating).
            if elapsed > self._path_tolerance_grace_period:
                track_err = max(
                    abs(self._current_pos[j] - interp_pos[i]) for i, j in enumerate(joint_names)
                )
                if track_err > self._path_tolerance:
                    # Per-joint breakdown (added in this fork), same reasoning
                    # as the settling abort below: one joint far off means it
                    # is jammed against something, all joints slightly off
                    # means the arm is merely lagging a too-fast command.
                    self.get_logger().error(
                        "  per-joint lag: "
                        + ", ".join(
                            f"{j}: cur={self._current_pos[j]:+.4f} cmd={interp_pos[i]:+.4f} "
                            f"err={self._current_pos[j] - interp_pos[i]:+.4f}"
                            for i, j in enumerate(joint_names)
                        )
                    )
                    self.get_logger().error(
                        f"Aborting: tracking error {track_err:.4f}rad exceeds path_tolerance="
                        f"{self._path_tolerance:.4f}rad at t={t:.3f}s"
                    )
                    goal_handle.abort()
                    return self._make_result(PATH_TOLERANCE_VIOLATED, "mid-execution tracking error")

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = joint_names
            cmd.position = interp_pos.tolist()
            cmd.velocity = interp_vel.tolist()
            self._cmd_pub.publish(cmd)
            prev_cmd_pos = interp_pos

            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = joint_names
            desired = JointTrajectoryPoint()
            desired.positions = interp_pos.tolist()
            desired.velocities = interp_vel.tolist()
            feedback.desired = desired
            actual = JointTrajectoryPoint()
            actual.positions = [self._current_pos[j] for j in joint_names]
            actual.velocities = [self._current_vel[j] for j in joint_names]
            feedback.actual = actual
            goal_handle.publish_feedback(feedback)

            if elapsed >= total_duration:
                break
            time.sleep(dt)

        # ── Phase 2: hold final waypoint, wait for actual settling ──────────
        # Nominal trajectory time elapsing is NOT success — we only report
        # success once /joint_states confirms the goal was actually reached
        # and the arm has stopped moving, sustained for settling_time.
        settle_start = self.get_clock().now()
        hold_cmd = JointState()
        hold_cmd.name = joint_names
        hold_cmd.position = goal_pos.tolist()
        hold_cmd.velocity = [0.0] * n_joints

        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Trajectory cancelled while settling.")
                return self._make_result(SUCCESSFUL, "canceled while settling")

            if self._joint_state_age_sec() > self._joint_state_timeout:
                self.get_logger().error("Aborting during settling: /joint_states went stale")
                goal_handle.abort()
                return self._make_result(OLD_HEADER_TIMESTAMP, "joint_states stale during settling")

            hold_cmd.header.stamp = self.get_clock().now().to_msg()
            self._cmd_pub.publish(hold_cmd)

            pos_err = max(abs(self._current_pos[j] - goal_pos[i]) for i, j in enumerate(joint_names))
            vel_mag = max(abs(self._current_vel[j]) for j in joint_names)
            now = self.get_clock().now()

            if pos_err <= self._goal_tolerance and vel_mag <= self._stopped_velocity_tolerance:
                if settled_since is None:
                    settled_since = now
                elif (now - settled_since).nanoseconds * 1e-9 >= self._settling_time:
                    self.get_logger().info(
                        f"Trajectory execution complete: pos_err={pos_err:.4f}rad "
                        f"vel={vel_mag:.4f}rad/s, settled {self._settling_time:.2f}s"
                    )
                    goal_handle.succeed()
                    return self._make_result(SUCCESSFUL, "")
            else:
                settled_since = None

            waited = (now - settle_start).nanoseconds * 1e-9
            if waited > self._goal_time_tolerance:
                # Per-joint breakdown (added in this fork): a single max is
                # not enough to tell "one joint is stuck against something"
                # apart from "every joint droops a bit under a weak drive",
                # and those have completely different fixes.
                per_joint = ", ".join(
                    f"{j}: cur={self._current_pos[j]:+.4f} goal={goal_pos[i]:+.4f} "
                    f"err={self._current_pos[j] - goal_pos[i]:+.4f}"
                    for i, j in enumerate(joint_names)
                )
                self.get_logger().error(
                    f"Aborting: goal not settled within goal_time_tolerance="
                    f"{self._goal_time_tolerance:.2f}s (pos_err={pos_err:.4f}rad, "
                    f"vel={vel_mag:.4f}rad/s, goal_tolerance={self._goal_tolerance:.4f}rad, "
                    f"stopped_velocity_tolerance={self._stopped_velocity_tolerance:.4f}rad/s)\n"
                    f"  per-joint: {per_joint}"
                )
                goal_handle.abort()
                return self._make_result(GOAL_TOLERANCE_VIOLATED, "did not settle in time")

            time.sleep(dt)

    @staticmethod
    def _make_result(error_code: int, message: str) -> FollowJointTrajectory.Result:
        result = FollowJointTrajectory.Result()
        result.error_code = error_code
        result.error_string = message
        return result

    # ── Gripper action callback ──────────────────────────────────────────────
    def _execute_gripper(self, goal_handle):
        """
        Execute a GripperCommand goal for Robotiq 2F-85.
        goal.command.position: opening in meters
          0.0   = fully closed (knuckle = 0.8 rad)
          0.085 = fully open   (knuckle = 0.0 rad)
        All 6 Robotiq joints are published with mimic ratios applied.

        This bridge does not have access to Isaac Sim contact sensors, so it
        cannot directly detect "gripper is holding an object". As a coarse,
        documented fallback signal (see README2.md §16), it compares the
        *commanded* knuckle target against the *actual* knuckle angle read
        back from /joint_states after the move: if we commanded a close and
        the fingers stopped meaningfully short of the commanded target, that
        is consistent with something being caught between them, and we report
        `stalled=True, reached_goal=False`. This is NOT a substitute for a
        real contact/force signal — it cannot distinguish "gripping an
        object" from "mechanically jammed", and a soft/small object may not
        produce a large enough gap to be detected this way.
        """
        target_m = float(goal_handle.request.command.position)
        target_m = max(GRIPPER_MAX_CLOSED, min(GRIPPER_MAX_OPEN, target_m))

        target_knuckle = KNUCKLE_MAX * (1.0 - target_m / GRIPPER_MAX_OPEN)
        start_knuckle  = self._gripper_knuckle
        is_closing = target_knuckle > start_knuckle

        self.get_logger().info(
            f"Gripper: {target_m*1000:.1f} mm → knuckle {target_knuckle:.3f} rad"
        )

        MOVE_DURATION = (
            self.get_parameter("gripper_close_duration").value
            if is_closing
            else self.get_parameter("gripper_open_duration").value
        )
        SETTLE_DURATION = self.get_parameter("gripper_settle_duration").value
        dt    = 1.0 / 50.0
        knuckle_vel = (target_knuckle - start_knuckle) / MOVE_DURATION

        # Progress is measured on the ROS clock, which is Isaac's simulated time
        # (use_sim_time), not on the wall clock. These durations are sized
        # against a measured *simulated* drive response (~1.8s for the knuckle
        # to reach a commanded angle), and the simulator does not run at real
        # time: adding the scene camera took the real-time factor to 0.77 and
        # the two wrist cameras took it to 0.58, so a wall-clock ramp silently
        # became 58% as long in the only units that matter. The symptom is a
        # close that returns reached_goal=True stalled=False holding nothing,
        # and it moves every time the scene's rendering cost changes. The
        # trajectory path in this same file already times itself this way.
        # `time.sleep(dt)` stays as the *pacing* mechanism -- it only decides
        # how often to publish, not how far along the ramp is.
        start_ros = self.get_clock().now()
        # Wall-clock backstop: if the sim clock stops advancing (paused sim,
        # dead publisher) the loop above would otherwise never terminate.
        wall_deadline = time.monotonic() + MOVE_DURATION / MIN_EXPECTED_RTF + 5.0
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = GripperCommand.Result()
                result.position     = self._gripper_knuckle / KNUCKLE_MAX * GRIPPER_MAX_OPEN
                result.reached_goal = False
                result.stalled      = False
                return result

            elapsed = (self.get_clock().now() - start_ros).nanoseconds * 1e-9
            s       = min(1.0, elapsed / MOVE_DURATION) if MOVE_DURATION > 0 else 1.0
            knuckle = start_knuckle + s * (target_knuckle - start_knuckle)

            cmd              = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = GRIPPER_JOINTS
            cmd.position     = [knuckle * m for m in GRIPPER_MIMIC]
            cmd.velocity     = [knuckle_vel * m for m in GRIPPER_MIMIC]
            self._gripper_pub.publish(cmd)

            self._gripper_knuckle = knuckle
            if s >= 1.0:
                break
            if time.monotonic() > wall_deadline:
                self.get_logger().warn(
                    f"gripper ramp hit its wall-clock backstop at {elapsed:.2f}/{MOVE_DURATION:.2f} "
                    f"simulated seconds -- is the simulation clock advancing?"
                )
                break
            time.sleep(dt)

        # Let Isaac Sim's physics settle, then read back the actual left-knuckle
        # angle from /joint_states to compare against what we commanded. Also on
        # the sim clock, for the same reason as the ramp.
        settle_start = self.get_clock().now()
        settle_wall_deadline = time.monotonic() + SETTLE_DURATION / MIN_EXPECTED_RTF + 5.0
        while (self.get_clock().now() - settle_start).nanoseconds * 1e-9 < SETTLE_DURATION:
            if time.monotonic() > settle_wall_deadline:
                break
            time.sleep(dt)

        actual_knuckle = self._current_pos.get(GRIPPER_JOINTS[0], target_knuckle)
        knuckle_gap = target_knuckle - actual_knuckle  # positive: closed less than commanded

        STALL_GAP_THRESHOLD = 0.05  # rad (~6mm at the finger), see class docstring
        stalled = is_closing and knuckle_gap > STALL_GAP_THRESHOLD
        reached_goal = abs(knuckle_gap) <= STALL_GAP_THRESHOLD

        if stalled:
            self.get_logger().info(
                f"Gripper stalled while closing: commanded {target_knuckle:.3f}rad, "
                f"actual {actual_knuckle:.3f}rad (gap {knuckle_gap:.3f}rad) — "
                f"possible object between fingers (heuristic, not a contact sensor)"
            )

        goal_handle.succeed()
        result = GripperCommand.Result()
        # Report the *actual* opening, not blindly the commanded one.
        result.position     = max(0.0, min(GRIPPER_MAX_OPEN, actual_knuckle / KNUCKLE_MAX * GRIPPER_MAX_OPEN))
        result.reached_goal = reached_goal
        result.stalled      = stalled
        self.get_logger().info(
            f"Gripper command complete: reached_goal={reached_goal}, stalled={stalled}"
        )
        return result


def main():
    rclpy.init()
    node     = TrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
