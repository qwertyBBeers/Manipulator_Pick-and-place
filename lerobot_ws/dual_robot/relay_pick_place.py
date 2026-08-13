#!/usr/bin/env python3
"""Two-robot relay pick-and-place: robot A moves the block from the source
bin to a handoff tray; robot B picks it up from there and moves it to its
own destination bin. One block. With --cycles N the block is then carried
back the same way (B -> handoff -> A -> source) and the relay repeats.

New file -- does not modify rb5_binpicking/scripts/moveit_pick_place.py.
Reuses that file's proven core techniques (pose-constraint construction,
Pilz-LIN-then-OMPL planner fallback, GripperCommand-based gripper control,
the grasp_offset_z/pre_grasp_lift/post_grasp_lift constants and SAVED_POSES
"gripping" down-orientation) but drops its retry loops, tilt rejection, pose
re-observation, and trial logging -- this is a single scripted pass for a
single block, not a hardened many-trial expert.

Talks to both robots' namespaced move_action/gripper_cmd action servers
brought up by dual_binpicking.launch.py. Robot B's target poses are computed
in *its own* local link0 frame by subtracting B_OFFSET from the shared
world-frame /binpicking/object_pose reading: each robot's MoveIt plans purely
in its own local frame, and B_OFFSET is only a physical placement in Isaac
Sim, invisible to MoveIt. All positional layout is imported from layout.py,
shared with dual_binpicking_scene.py so the controller and the scene cannot
drift apart.

Usage (after dual_binpicking_scene.py and dual_binpicking.launch.py are up):
  source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
  python3 relay_pick_place.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory, GripperCommand
from std_msgs.msg import Empty, String
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints, JointConstraint, MoveItErrorCodes, OrientationConstraint, PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (  # noqa: E402
    SRC_X, SRC_Y, DEST_X, DEST_Y, WALL_T,
    B_OFFSET, SOURCE_WORLD, DEST_WORLD,
    HANDOFF_WORLD, HANDOFF_LOCAL_A, HANDOFF_LOCAL_B,
)

GROUP_NAME = "mainpulation"  # SRDF typo, preserved -- see rb5_binpicking/scripts/moveit_pick_place.py
PLANNING_FRAME = "link0"
EEF_LINK = "tcp"
# The only joints move_group knows about. dual_binpicking.launch.py rewrites the
# six Robotiq knuckle joints to `fixed` before handing the URDF to MoveIt, so
# they are not variables of the planning model -- and naming one in a service
# request throws moveit::Exception *uncaught*, killing the move_group process
# outright (seen once: "Variable 'robotiq_85_left_inner_knuckle_joint' is not
# known to model 'rb5_850e'" followed by terminate). /joint_states carries all
# 12, so anything sent to move_group has to be filtered down to these.
ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
# Every arm joint is revolute +/-3.14rad in rb5_with_tools.urdf.
JOINT_LIMIT_RAD = 3.14
PIPELINE_PILZ = "pilz_industrial_motion_planner"
PIPELINE_OMPL = "ompl"

GRASP_OFFSET_Z = 0.145      # from moveit_pick_place.py DEFAULT_DYNAMIC_PICK_PARAMS -- empirically tuned, same gripper/object
# Extra clearance above that, only for the grasp. The tool lands consistently
# 1-2mm below what it was asked for, and the fingertips at 0.145 sit within a
# few mm of the tray floor -- close enough that a failed grasp pressed the block
# 7mm INTO the floor, which PhysX then resolved by flinging it 0.6m out of the
# workspace. Kept small: at 5mm the pads gripped high enough on the 42mm cube
# that it pivoted out of them during the carry, which is a worse failure than
# the one this prevents.
# 0.002 -> 0.006. At 2mm the fingertips ended a couple of mm off the tray floor,
# and since the tool lands 1-2mm low the descent pressed them into it: the
# position drives (stiffness 10000) keep pushing toward a z the floor will not
# allow, which is the arm visibly shaking at the bottom of every pick. The
# earlier note that 5mm let the cube pivot out of the pads mid-carry was measured
# at a pad friction of 1.2; that is 3.0 now.
GRASP_FLOOR_MARGIN = 0.004
PRE_GRASP_LIFT = 0.18
# Lowered 0.30 -> 0.18. The block was slipping out of a confirmed grasp during
# the long source->handoff transit, on trajectories the controller executed
# perfectly (pos_err ~0.003rad, no aborts) -- so it is inertia during the swing,
# not a tracking failure and not the grip decaying (the knuckle angle was
# measured holding 0.5854rad dead steady for 65s). Carrying lower shortens the
# swing; the trays are 1cm lips, so 18cm is still ample clearance.
POST_GRASP_LIFT = 0.18
OBJECT_HALF_HEIGHT = 0.021
# Every tray stands on the ground, so the top of its floor slab -- what the
# block actually rests on -- is one wall thickness up, not z=0. Placing at
# half-height above z=0 instead buried the block 7mm in that slab while it was
# still gripped, and PhysX resolved the overlap by launching it: one release
# put the block 5m away, out of both robots' reach. The extra clearance means
# the block is let go just above its resting height and drops the last few mm.
# 0.005 -> 0.020. Same problem at the other end: the block was set down until it
# touched and the arm kept driving down against the tray floor through the block.
# Releasing 2cm up lets the cube drop the last stretch instead, which its
# zero-restitution material absorbs without bouncing.
PLACE_CLEARANCE = 0.020
PLACE_REST_Z = WALL_T + OBJECT_HALF_HEIGHT + PLACE_CLEARANCE
# After a carry, the block must still be where the tool is. A move that
# "succeeded" for move_group can still have thrown the payload -- one Pilz LIN
# place plan was reported executed while the controller aborted 2.6rad from the
# goal -- and without this the arm cheerfully opens an empty gripper over the
# tray and the relay continues with the block somewhere off the table.
CARRY_CHECK_MIN_Z = 0.10
PLACED_CHECK_RADIUS = 0.06
# Grasp verification, same technique as moveit_pick_place.py: lift a little
# and check the block's ground-truth z rose with the gripper.
TEST_LIFT_HEIGHT = 0.04
TEST_LIFT_MIN_RISE = 0.015
# The gripper closes in two stages, because the two failure modes pull in
# opposite directions and neither single target avoids both:
#
#   too hard (32mm, and far worse at a full close) -- the pads reach the 42mm
#     cube at a knuckle angle of ~0.42rad while the drive keeps driving toward
#     0.50. An isolated 4-run probe had the cube squirt sideways out from
#     between the pads half the time (both pads confirmed in contact on every
#     run, so this is the grip failing, not the approach missing it), and a full
#     close crushed it hard enough that the later open command did not move the
#     fingers at all.
#   too soft (36mm) -- holds through the 4cm test lift every time, then drops
#     the block partway through the 0.5m carry to the handoff tray.
#
# Making contact softly and only then squeezing gets both: the first stage
# captures the cube between the pads without the impulse that ejects it, and
# once it is captured the second stage cannot squirt it anywhere.
GRASP_CONTACT_TARGET_M = 0.036
# 0.024 -> 0.030. 24mm on a 42mm cube is a large overshoot that only stops
# because the cube stalls the fingers, and the leftover drive is what ejects it:
# a failed close stalled at 6.7mm, i.e. the pads met each other, with the block
# found 10cm away. Successful closes all stall at 28-35mm, so 30mm asks for
# about what actually happens. Clamping force stopped being the scarce resource
# when the pad friction went 1.2 -> 3.0 -- holding a 0.1kg cube needs ~0.33N of
# normal force at mu=1.5 and half that now.
# 0.024 -> 0.030 -> 0.024. The ejection that motivated 30mm was measured with
# the bridge's 2.0s close ramp; that ramp is 3.5s now (see the launch file), so
# the same target is approached far more slowly and the impulse that squirted
# the cube out is gone. 30mm meanwhile grips too lightly to be worth it: with
# randomized block poses it missed half the picks outright ("block rose 0.0mm
# on test lift"), mostly on the handoff tray, and a missed close knocks the
# block off the tray onto the floor.
GRASP_CLOSE_TARGET_M = 0.030
# The gripper is commanded through a fixed-duration ramp; if the fingers do not
# reach the commanded angle (reached_goal False) it is worth simply asking
# again before treating it as a failure.
GRIPPER_RETRIES = 3
# RB5-850E reach is 850mm; stay inside it with margin for the wrist/gripper.
MAX_REACH_M = 0.80
POSITION_TOLERANCE = 0.02
ORIENTATION_TOLERANCE = 0.20
# Tighter goal region for the two moves where the tool actually has to be where
# it was asked to be. At the loose 20mm/0.20rad values a satisfied goal could
# still put the fingertips ~2cm low and ~11deg tilted, which is enough to drive
# them into the tray floor: observed as the fingers jamming a quarter of the
# way closed while the block was pushed 10mm *down* and 65mm sideways. Not
# tighter than this, though: at 5mm the planners started failing outright, and
# the goal region is only a hint -- accuracy is enforced by the FK check below,
# so it is better to leave the planner room and reject what it comes back with.
GRASP_POSITION_TOLERANCE = 0.010
GRASP_ORIENTATION_TOLERANCE = 0.05
# The actual acceptance test: how far the tool may sit from the commanded
# grasp/place pose, measured by FK on the real joint angles rather than trusted
# from the planner. move_to() tries the next planner when this is exceeded.
# 5mm, from the outcomes: every logged grasp landing within 3.4mm held the block
# (1.3/1.6/1.7/1.7/2.3/2.4/2.6/3.4mm), and almost every one at 5.4mm or worse
# missed it (5.4/6.5/7.9/8.6/9.0/9.5mm). Rejecting a sloppy solution and letting
# the next planner try costs seconds; attempting the grasp anyway knocks the
# block somewhere new and costs a whole retry.
TCP_VERIFY_TOLERANCE = 0.005
# Tool tilt limit. The fingertips hang ~150mm below the tool frame, so 1deg of
# tilt swings them ~2.6mm sideways -- comparable to the whole position budget
# above. Checked separately because a pose can satisfy the position goal
# perfectly and still present the pads to the block at an angle.
# 3deg = ~7.9mm of fingertip offset, which still leaves most of the 21.5mm of
# per-side clearance around the 42mm cube. Not tighter: at 1.5deg this rejected
# a 2.34deg grasp from robot A, the arm that had been picking reliably all
# along, and the rejection was far more expensive than the tilt.
TCP_VERIFY_ANGLE = math.radians(3.0)
# A knocked block is recoverable: re-observe where it ended up and pick again.
PICK_ATTEMPTS = 4
# Straight-line (Cartesian) descent onto the block, and straight-line lift off it.
# This was tried once with the KDL solver and removed, because every "line" came
# back as fraction=1.00 over 800+ waypoints totalling 37-50 RADIANS of joint
# travel -- KDL walking the arm around singular IK branches. With LMA that cause
# is gone, and a straight descent is what the grasp actually needs: with IK now
# deterministic (the same grasp reached an identical 3.2mm tool pose four times
# running) the only thing still varying between a hit and a miss was OMPL's
# free-space approach sweeping in from a different direction and clipping the
# block. CARTESIAN_MAX_JOINT_TRAVEL stays as insurance -- fraction==1.0 is not
# proof the path is sane, and a bad one must never be executed again.
CARTESIAN_MAX_STEP = 0.005        # m between interpolated waypoints
CARTESIAN_MIN_FRACTION = 0.99     # anything less is not the line we asked for
CARTESIAN_REVOLUTE_JUMP = 0.1     # rad, max per-step motion of any revolute joint
CARTESIAN_MAX_JOINT_TRAVEL = 1.5  # rad, summed along the path, worst joint
CARTESIAN_JOINT_SPEED = 0.35      # rad/s for the busiest joint
CARTESIAN_MIN_DURATION = 1.0      # s
# Planner-move speed, as a fraction of the URDF's limits. Was 0.03 -- 3% of
# rated speed, which is why the arm crawled. That was set when the block was
# sliding out of the gripper mid-carry and slowing the swing was the only lever
# that helped; the pad friction coefficient has since gone 1.2 -> 3.0, so the
# grip has margin the crawl was substituting for. Raise these together with
# CARTESIAN_JOINT_SPEED and re-measure the carry -- a dropped block during a
# transfer is the symptom that says they went too far.
VELOCITY_SCALING = 0.15
ACCELERATION_SCALING = 0.10
# ...but only while the hand is empty. Everything above was raised after the pad
# friction went 1.2 -> 3.0, and the one failure that survived is the block
# leaving a *confirmed* grasp during the 18cm lift right after it ("aborted at
# 'carrying?'", with the block having risen 36.9mm on the 4cm test lift moments
# earlier). Friction is not the scarce resource -- holding 0.1kg at mu=3.0 needs
# ~0.16N of pad force -- so what breaks the grip is the acceleration of the
# swing, and the fix is to spend the speed where nothing is being carried and
# give it back where something is. Approach moves stay quick; carrying moves go
# back to roughly the old crawl.
CARRY_VELOCITY_SCALING = 0.04
CARRY_ACCELERATION_SCALING = 0.03
CARRY_CARTESIAN_JOINT_SPEED = 0.12
GRIPPER_MAX_OPEN = 0.085

# SAVED_POSES["gripping"] orientation from moveit_pick_place.py -- top-down grasp orientation at yaw=0.
DOWN_ORIENTATION = (0.532, 0.528, 0.458, 0.478)  # x,y,z,w
WATCH_A = {"position": (0.331, -0.069, 0.766), "orientation": (0.475, 0.449, 0.507, 0.562)}
# Robot B's watch pose is the same joint-space-equivalent Cartesian pose as
# robot A's (identical robot, identical local-frame problem -- see module
# docstring), reused verbatim.
WATCH_B = WATCH_A
# Joint-space recovery pose: the arm straight up, which is where both robots
# start and is provably a configuration the planners can reach `watch` from.
# A failed execution leaves the arm wherever it stopped, and that can be a
# configuration OMPL cannot plan out of while also honouring the watch pose's
# orientation path constraint ("Path constraints not satisfied for start state",
# then every sampled state invalid). A joint-space goal has no such constraint
# and no IK to solve, so it works from anywhere the arm is not actually stuck.
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# How far straight up to extract the tool when it is stuck among the trays.
RECOVERY_LIFT = 0.20
# Seed for every IK call -- see Arm.solve_ik(). Its only job is to pick which
# of the arm's several solutions for a pose gets used, so it has to be one
# configuration used everywhere, not a per-pose guess. Value measured with
# ik_seed_probe.py: the descent from a pre-grasp solved with this seed costs
# ~0.4rad of joint travel for both arms at all three trays, vs 3.5-6.8rad from
# the branches OMPL's pose goals were picking.
IK_SEED = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Blind retract to HOME when move_group will not plan at all -- see
# Arm.escape_to_home(). Step size stays under the bridge's max_command_jump
# (0.3rad) with margin; the speed matches the Cartesian moves.
# Episode boundaries for dataset recording. The controller is the only thing
# that knows where one demonstration ends and the next begins -- a passive
# logger watching /joint_states cannot tell a successful transfer from a failed
# attempt followed by a retry, and a dataset that mixes the two teaches the
# policy to drop things. One episode = one pick-and-place attempt, labelled with
# its own task string and whether it worked.
# What each leg of the relay is, in the plain English a VLA is conditioned on.
# Named by what the arm can see rather than by internal ids ("handoff tray" is
# the blue one in the middle), because that is what grounds the instruction in
# the image.
TASK_A_TO_HANDOFF = "pick up the block and place it on the blue tray in the middle"
TASK_B_TO_DEST = "pick up the block from the blue tray and place it in the far bin"
TASK_B_TO_HANDOFF = "pick up the block and place it on the blue tray in the middle"
TASK_A_TO_SOURCE = "pick up the block from the blue tray and place it in the near bin"

EPISODE_TOPIC = "/relay/episode"                     # std_msgs/String, JSON
RANDOMIZE_TOPIC = "/binpicking/randomize_object"     # std_msgs/Empty
# Time for the scene to teleport the block and for the settled pose to reach us.
RANDOMIZE_SETTLE_SEC = 2.0

ESCAPE_MAX_STEP_RAD = 0.05
ESCAPE_JOINT_SPEED = 0.15


def _yaw_from_quaternion(x, y, z, w) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tilt_from_quaternion(x, y, z, w) -> float:
    """Angle between the object's own +Z and world +Z, in radians.

    A cube resting flat reads ~0. Anything else means it is up on an edge or a
    corner, which matters because the grasp only corrects for yaw: the pads come
    straight down expecting two parallel vertical faces, and a tilted cube
    presents them an edge instead.
    """
    # Third column of the rotation matrix = the body's +Z expressed in world.
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, abs(zz) / math.sqrt(zx * zx + zy * zy + zz * zz))))


def _wrap_to_cube_symmetry(yaw: float) -> float:
    """Fold a yaw into [-45deg, +45deg).

    The object is a cube, so yaw, yaw+90, yaw+180 and yaw+270 are the same grasp
    as far as the fingers are concerned -- but they are NOT the same for the arm,
    and that difference broke the second half of the relay. The block starts
    axis-aligned, so robot A grasps it at a well-conditioned wrist angle; A then
    places it rotated to the gripper's own ~89deg yaw, and robot B read that back
    and added another 89deg on top. B's wrist ended up in a completely different
    IK branch, and it closed on empty air three attempts running (the gripper
    reported reaching its commanded closure with no stall -- nothing between the
    pads) while A, picking the same block from the same kind of tray, succeeded.
    Folding into the symmetric range keeps every grasp near the configuration
    that works, no matter how many times the block has been handed over.
    """
    quarter = math.pi / 2.0
    return (yaw + math.pi / 4.0) % quarter - math.pi / 4.0


def _yaw_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quaternion_angle(q1, q2) -> float:
    """Smallest rotation angle (rad) between two orientations."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _fmt(xyz) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in xyz) + ")"


def _make_pose(xyz, quat_xyzw) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat_xyzw
    return p


class RobotArm:
    """One robot's move_action + gripper_cmd clients, namespaced."""

    def __init__(self, node: Node, ns: str):
        self.node = node
        self.ns = ns
        self.move_client = ActionClient(node, MoveGroup, f"/{ns}/move_action")
        self.gripper_client = ActionClient(node, GripperCommand, f"/{ns}/gripper_controller/gripper_cmd")
        # Where the tool REALLY is, from the measured joint angles. The global
        # /tf tree is unusable here: both robots' robot_state_publishers and
        # both Isaac PublishTransformTree nodes push identically-named frames
        # (link0..tcp) onto the one global /tf, so a tf lookup returns whichever
        # robot published last. Planning is unaffected (both move_groups plan in
        # their own link0, which needs no lookup), but any TF-based measurement
        # is meaningless -- so ask this robot's own move_group for FK instead.
        self.fk_client = node.create_client(GetPositionFK, f"/{ns}/compute_fk")
        self.ik_client = node.create_client(GetPositionIK, f"/{ns}/compute_ik")
        self.cartesian_client = node.create_client(GetCartesianPath, f"/{ns}/compute_cartesian_path")
        self.execute_client = ActionClient(node, ExecuteTrajectory, f"/{ns}/execute_trajectory")
        # The controller itself, for the one case move_group cannot serve:
        # escaping a start state it considers in collision.
        self.escape_client = ActionClient(
            node, FollowJointTrajectory, f"/{ns}/joint_trajectory_controller/follow_joint_trajectory"
        )
        # Set while the gripper is holding the block: every motion planned in
        # this state is slowed down. See CARRY_VELOCITY_SCALING.
        self.carrying = False
        self._joint_state: dict | None = None
        node.create_subscription(JointState, f"/{ns}/joint_states", self._joint_state_cb, 10)

    @property
    def velocity_scaling(self) -> float:
        return CARRY_VELOCITY_SCALING if self.carrying else VELOCITY_SCALING

    @property
    def acceleration_scaling(self) -> float:
        return CARRY_ACCELERATION_SCALING if self.carrying else ACCELERATION_SCALING

    @property
    def cartesian_joint_speed(self) -> float:
        return CARRY_CARTESIAN_JOINT_SPEED if self.carrying else CARTESIAN_JOINT_SPEED

    def _joint_state_cb(self, msg: JointState):
        self._joint_state = dict(zip(msg.name, msg.position))

    def joints_out_of_bounds(self):
        """[(joint, value)] for any arm joint outside the URDF's +/-3.14rad."""
        if self._joint_state is None:
            return []
        return [(j, self._joint_state[j]) for j in ARM_JOINTS
                if j in self._joint_state and abs(self._joint_state[j]) > JOINT_LIMIT_RAD]

    def wait_for_servers(self, timeout_sec=15.0) -> bool:
        return self.move_client.wait_for_server(timeout_sec=timeout_sec) and self.gripper_client.wait_for_server(
            timeout_sec=timeout_sec
        )

    def tcp_pose(self, timeout_sec=5.0):
        """Actual ((x, y, z), (qx, qy, qz, qw)) of `tcp` in this robot's link0."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._joint_state is not None and all(j in self._joint_state for j in ARM_JOINTS):
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
        else:
            return None
        if not self.fk_client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().warn(f"[{self.ns}] compute_fk service unavailable")
            return None
        request = GetPositionFK.Request()
        request.header.frame_id = PLANNING_FRAME
        request.fk_link_names = [EEF_LINK]
        # ARM_JOINTS only -- see the constant's comment; sending the gripper
        # joints here kills move_group.
        request.robot_state.joint_state.name = ARM_JOINTS
        request.robot_state.joint_state.position = [self._joint_state[j] for j in ARM_JOINTS]
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        response = future.result()
        if response is None or not response.pose_stamped:
            return None
        pose = response.pose_stamped[0].pose
        p, q = pose.position, pose.orientation
        return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)

    def _make_pose_constraints(self, pose: Pose, use_orientation: bool, pos_tol: float, ori_tol: float) -> Constraints:
        constraints = Constraints()
        constraints.name = "eef_pose_goal"
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = PLANNING_FRAME
        pos_constraint.link_name = EEF_LINK
        pos_constraint.constraint_region.primitives = [sphere]
        pos_constraint.constraint_region.primitive_poses = [pose]
        pos_constraint.weight = 1.0
        constraints.position_constraints = [pos_constraint]
        if use_orientation:
            constraints.orientation_constraints = [self._make_orientation_constraint(pose.orientation, ori_tol)]
        return constraints

    def _make_orientation_constraint(self, orientation: Quaternion, ori_tol: float) -> OrientationConstraint:
        oc = OrientationConstraint()
        oc.header.frame_id = PLANNING_FRAME
        oc.link_name = EEF_LINK
        oc.orientation = orientation
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol
        oc.weight = 1.0
        return oc

    def _move_once(self, name, pose, use_orientation, use_path_constraint, pipeline_id, planner_id="",
                   pos_tol=POSITION_TOLERANCE, ori_tol=ORIENTATION_TOLERANCE):
        self.node.get_logger().info(
            f"[{self.ns}] -> {name}: ({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f}) "
            f"pipeline={pipeline_id} planner={planner_id or 'default'}"
        )
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        # 0.2 (the single-robot demo's value) commanded the arm faster than
        # this two-robot sim actually tracks: the trajectory bridge measured
        # 0.25-0.39rad of lag the moment its startup grace period expired and
        # aborted mid-descent, and the half-executed motion knocked the block
        # out from under the gripper. Final positioning is accurate (settles
        # to ~0 error), so the problem is purely commanded speed.
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
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
        goal.request.goal_constraints = [self._make_pose_constraints(pose, use_orientation, pos_tol, ori_tol)]
        if use_orientation and use_path_constraint:
            goal.request.path_constraints.orientation_constraints = [
                self._make_orientation_constraint(pose.orientation, ori_tol)
            ]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"[{self.ns}] MoveGroup rejected goal '{name}'")
            return MoveItErrorCodes.FAILURE
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().warn(f"[{self.ns}] '{name}' failed (pipeline={pipeline_id}): code={result.error_code.val}")
        else:
            self.node.get_logger().info(f"[{self.ns}] reached {name}")
        return result.error_code.val

    def _tool_reached(self, name: str, xyz, quat_xyzw) -> bool:
        """Did the tool actually end up at this pose, position AND orientation?

        Orientation is checked because it dominates where the fingertips land:
        they sit ~150mm below the tool frame, so the 0.05rad (2.9deg) goal
        tolerance alone lets them swing ~7.6mm sideways -- more than twice the
        position error the arm actually achieves. Robot A picks straight ahead
        at 0.51m and is nearly untilted; robot B picks at 0.67m off to one side,
        where the arm is far more extended, and it was missing the block over and
        over at a *deterministic* 3.2mm position error, which position alone
        could not explain.
        """
        measured = self.tcp_pose()
        if measured is None:
            self.node.get_logger().warn(f"[{self.ns}] '{name}': FK unavailable, cannot verify tool pose")
            return True
        actual_xyz, actual_quat = measured
        position_error = math.dist(actual_xyz, xyz)
        angle_error = _quaternion_angle(actual_quat, quat_xyzw)
        self.node.get_logger().info(
            f"[{self.ns}] '{name}' tool at {_fmt(actual_xyz)} -- {position_error * 1000:.1f}mm, "
            f"{math.degrees(angle_error):.2f}deg off"
        )
        if position_error <= TCP_VERIFY_TOLERANCE and angle_error <= TCP_VERIFY_ANGLE:
            return True
        self.node.get_logger().warn(
            f"[{self.ns}] '{name}' is off by {position_error * 1000:.1f}mm / "
            f"{math.degrees(angle_error):.2f}deg (limits {TCP_VERIFY_TOLERANCE * 1000:.0f}mm / "
            f"{math.degrees(TCP_VERIFY_ANGLE):.1f}deg)"
        )
        return False

    @staticmethod
    def _joint_travel(trajectory) -> float:
        """Total joint-space distance along the path, for the busiest joint."""
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return 0.0
        travel = [0.0] * len(points[0].positions)
        for a, b in zip(points, points[1:]):
            for i, (pa, pb) in enumerate(zip(a.positions, b.positions)):
                travel[i] += abs(pb - pa)
        return max(travel)

    def _retime(self, trajectory):
        """Retime a Cartesian solution to a joint speed the arm actually tracks.

        Humble's GetCartesianPath has no velocity_scaling field, so move_group
        times the path at full speed. Rescaling a fixed path in time is exact
        (t*k, v/k, a/k^2); the factor comes from the path's own joint-space
        length so the busiest joint stays near CARTESIAN_JOINT_SPEED. A fixed
        factor is wrong here -- one once turned an 18cm descent into a
        1039-second trajectory.
        """
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return trajectory
        original = points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9
        if original <= 0.0:
            return trajectory
        target = max(CARTESIAN_MIN_DURATION, self._joint_travel(trajectory) / self.cartesian_joint_speed)
        k = target / original
        for point in points:
            t = (point.time_from_start.sec + point.time_from_start.nanosec * 1e-9) * k
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            point.velocities = [v / k for v in point.velocities]
            point.accelerations = [a / (k * k) for a in point.accelerations]
        return trajectory

    def _execute(self, name: str, trajectory) -> bool:
        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error(f"[{self.ns}] execute_trajectory server missing")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"[{self.ns}] execute_trajectory rejected '{name}'")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.node.get_logger().warn(f"[{self.ns}] linear '{name}' execution failed: code={result.error_code.val}")
            return False
        return True

    def move_linear(self, name: str, xyz, quat_xyzw=DOWN_ORIENTATION, verify_tcp=True,
                    fallback=True) -> bool:
        """Move the tool in a straight Cartesian line, falling back to a planner.

        Only worth using for the short vertical approach/retreat moves -- see the
        CARTESIAN_* constants for why this exists and why it is guarded.

        `fallback=False` means "this move is a straight line or it is nothing".
        Use it for the descent onto the block: a free-space planner asked to get
        from above the tray down into it produces a wandering path, and when the
        controller aborts partway through one the arm is left wherever that path
        had reached. Observed exactly once and it cost the run -- robot B ended
        up pressed against robot A's base, which is inside the `robot_a_base`
        collision object, so from there move_group refused every subsequent
        request (start state in collision) including the joint-space escape to
        home, and the arm sat grinding into the floor until the sim was
        restarted. Returning False instead sends the caller back through the
        outer pick retry, which retreats and re-approaches cleanly.
        """
        if self.cartesian_client.wait_for_service(timeout_sec=5.0):
            for avoid_collisions in (True, False):
                request = GetCartesianPath.Request()
                request.header.frame_id = PLANNING_FRAME
                # ARM_JOINTS only -- naming a gripper joint kills move_group.
                if self._joint_state is not None and all(j in self._joint_state for j in ARM_JOINTS):
                    request.start_state.joint_state.name = ARM_JOINTS
                    request.start_state.joint_state.position = [self._joint_state[j] for j in ARM_JOINTS]
                else:
                    request.start_state.is_diff = True
                request.group_name = GROUP_NAME
                request.link_name = EEF_LINK
                request.waypoints = [_make_pose(xyz, quat_xyzw)]
                request.max_step = CARTESIAN_MAX_STEP
                request.revolute_jump_threshold = CARTESIAN_REVOLUTE_JUMP
                request.avoid_collisions = avoid_collisions
                future = self.cartesian_client.call_async(request)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=15.0)
                response = future.result()
                if response is None:
                    break
                if response.fraction < CARTESIAN_MIN_FRACTION:
                    continue
                travel = self._joint_travel(response.solution)
                if travel > CARTESIAN_MAX_JOINT_TRAVEL:
                    self.node.get_logger().warn(
                        f"[{self.ns}] rejecting the '{name}' straight line: {travel:.1f}rad of joint travel "
                        f"-- that is an IK excursion, not a line"
                    )
                    continue
                self.node.get_logger().info(
                    f"[{self.ns}] linear '{name}' to {_fmt(xyz)}: fraction={response.fraction:.2f} "
                    f"travel={travel:.2f}rad avoid_collisions={avoid_collisions}"
                )
                if not self._execute(name, self._retime(response.solution)):
                    break
                if not verify_tcp or self._tool_reached(f"linear {name}", xyz, quat_xyzw):
                    return True
                # Do NOT fall through to the planners here. The straight line
                # ran, so the tool is already down at the object; every extra
                # planner attempt re-descends onto it from a new direction and
                # risks knocking it away. One rejected grasp cascaded exactly
                # that way and ended with the block flung 1.06m out of reach.
                # Failing here instead sends the caller back through the outer
                # pick retry, which retreats, re-observes and starts clean.
                return False

        if not fallback:
            self.node.get_logger().warn(
                f"[{self.ns}] no usable straight line for '{name}' and no planner fallback allowed here"
            )
            return False
        self.node.get_logger().warn(f"[{self.ns}] no usable straight line for '{name}'; falling back to a planner")
        return self.move_to(name, xyz, quat_xyzw, use_path_constraint=False,
                            pos_tol=GRASP_POSITION_TOLERANCE, ori_tol=GRASP_ORIENTATION_TOLERANCE,
                            verify_tcp=verify_tcp)

    def move_joints(self, name: str, positions) -> bool:
        """Go to a joint configuration directly -- no IK, no pose constraints."""
        self.node.get_logger().info(f"[{self.ns}] -> {name} (joint space)")
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        goal.request.pipeline_id = PIPELINE_OMPL
        goal.request.start_state.is_diff = True
        constraints = Constraints()
        constraints.name = "joint_goal"
        constraints.joint_constraints = [
            JointConstraint(joint_name=joint, position=float(value),
                            tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            for joint, value in zip(ARM_JOINTS, positions)
        ]
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"[{self.ns}] MoveGroup rejected joint goal '{name}'")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.node.get_logger().warn(f"[{self.ns}] joint goal '{name}' failed: code={result.error_code.val}")
        return ok

    def solve_ik(self, xyz, quat_xyzw, seed=None, timeout_sec=2.0):
        """IK for a tool pose, from a fixed seed. Returns joint values or None.

        A 6R arm reaches the same tool pose in several joint configurations
        (elbow up/down, wrist flipped). Asking move_group for a *pose* goal
        lets OMPL land in whichever one its sampling found, and the arm then
        has to do the next move from there: the same 18cm descent measured
        0.4rad of joint travel from a good configuration and 6.8rad from a
        folded one, and the folded ones are what put a link on the floor.
        LMA converges to the solution nearest its seed, so seeding from the
        same configuration every time pins the branch.
        """
        if not self.ik_client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().warn(f"[{self.ns}] compute_ik service unavailable")
            return None
        request = GetPositionIK.Request()
        request.ik_request.group_name = GROUP_NAME
        request.ik_request.ik_link_name = EEF_LINK
        request.ik_request.pose_stamped.header.frame_id = PLANNING_FRAME
        request.ik_request.pose_stamped.pose = _make_pose(xyz, quat_xyzw)
        # ARM_JOINTS only -- see the constant's comment.
        request.ik_request.robot_state.joint_state.name = ARM_JOINTS
        request.ik_request.robot_state.joint_state.position = [
            float(v) for v in (seed if seed is not None else IK_SEED)
        ]
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 1
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        response = future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            code = "no response" if response is None else response.error_code.val
            self.node.get_logger().warn(f"[{self.ns}] IK failed for {_fmt(xyz)}: code={code}")
            return None
        solution = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        if not all(j in solution for j in ARM_JOINTS):
            return None
        return [solution[j] for j in ARM_JOINTS]

    def move_to_seeded(self, name: str, xyz, quat_xyzw=DOWN_ORIENTATION, seed=None,
                       verify_tcp=False) -> bool:
        """Reach a tool pose through a seeded-IK joint goal, else the pose planners.

        The joint goal is what makes the arm's configuration on arrival
        repeatable; the pose-goal path stays as a fallback so a pose the seed
        cannot reach still gets a chance.
        """
        joints = self.solve_ik(xyz, quat_xyzw, seed)
        if joints is not None:
            self.node.get_logger().info(
                f"[{self.ns}] {name} via seeded IK: "
                + " ".join(f"{j}={v:+.2f}" for j, v in zip(ARM_JOINTS, joints))
            )
            if self.move_joints(name, joints) and (not verify_tcp or self._tool_reached(name, xyz, quat_xyzw)):
                return True
            self.node.get_logger().warn(f"[{self.ns}] seeded-IK joint goal for '{name}' failed; trying pose goals")
        return self.move_to(name, xyz, quat_xyzw, verify_tcp=verify_tcp)

    def escape_to_home(self) -> bool:
        """Retract to HOME through the controller, bypassing move_group entirely.

        move_group refuses to plan *anything* from a start state that is in
        collision -- including `move_joints` to the home configuration, which
        is supposed to be the last resort. Once robot B was left touching the
        `robot_a_base` object every request came back PLANNING_FAILED with no
        way out, so the run could only end by restarting the simulator. The
        controller does no collision checking, so it can still retract; HOME is
        the arm straight up, away from the trays and the other robot, which is
        the safest thing to sweep to blind.

        Sends the trajectory to the bridge's FollowJointTrajectory server
        directly (same server MoveIt's execution goes to). Waypoints are dense
        enough for the bridge's per-cycle jump and start-tolerance checks.
        """
        if self._joint_state is None or not all(j in self._joint_state for j in ARM_JOINTS):
            return False
        start = [self._joint_state[j] for j in ARM_JOINTS]
        if not self.escape_client.wait_for_server(timeout_sec=10.0):
            self.node.get_logger().error(f"[{self.ns}] controller action server missing; cannot escape")
            return False
        span = max(abs(h - s) for h, s in zip(HOME_JOINTS, start))
        if span < 1e-3:
            return True
        steps = max(2, int(math.ceil(span / ESCAPE_MAX_STEP_RAD)))
        duration = max(CARTESIAN_MIN_DURATION, span / ESCAPE_JOINT_SPEED)
        self.node.get_logger().warn(
            f"[{self.ns}] escaping to home through the controller: {span:.2f}rad over {duration:.1f}s "
            f"({steps} waypoints, no collision checking)"
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        for k in range(1, steps + 1):
            f = k / steps
            point = JointTrajectoryPoint()
            point.positions = [s + (h - s) * f for h, s in zip(HOME_JOINTS, start)]
            point.velocities = [0.0] * len(ARM_JOINTS)
            t = duration * f
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            goal.trajectory.points.append(point)

        future = self.escape_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"[{self.ns}] controller rejected the escape trajectory")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result
        ok = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not ok:
            self.node.get_logger().error(
                f"[{self.ns}] escape trajectory failed: error_code={result.error_code} {result.error_string}"
            )
        return ok

    def move_to(self, name: str, xyz, quat_xyzw=DOWN_ORIENTATION, use_path_constraint=True,
                pos_tol=POSITION_TOLERANCE, ori_tol=ORIENTATION_TOLERANCE, verify_tcp=False) -> bool:
        """Pilz LIN first (fast, straight-line), fall back to OMPL (with then
        without the path orientation constraint) -- matches the fallback
        sequence observed working in the single-robot demo's logs.

        With `verify_tcp`, the pose is not believed just because move_group
        returned SUCCESS: FK on the measured joint angles has to agree with
        what was asked for. SUCCESS only means the goal *constraints* were
        satisfied, and a constraint is a tolerance region, not a point.
        """
        pose = _make_pose(xyz, quat_xyzw)
        # Deduplicated: with use_path_constraint=False the last two entries are
        # the identical request, and running it twice just burns a planning call.
        # Measured over these runs, the three positions succeed 13/29, 3/16 and
        # 12/12 -- the unconstrained OMPL fallback is what actually gets the arm
        # there, but the constrained attempts do land better paths when they
        # work, so they stay ahead of it.
        attempts = []
        for attempt in (
            (PIPELINE_PILZ, "LIN", use_path_constraint),
            (PIPELINE_OMPL, "", use_path_constraint),
            (PIPELINE_OMPL, "", False),
        ):
            if attempt not in attempts:
                attempts.append(attempt)
        for pipeline_id, planner_id, path_constraint in attempts:
            code = self._move_once(name, pose, True, path_constraint, pipeline_id, planner_id, pos_tol, ori_tol)
            if code != MoveItErrorCodes.SUCCESS:
                continue
            if not verify_tcp or self._tool_reached(name, xyz, quat_xyzw):
                return True
        self.node.get_logger().error(f"[{self.ns}] '{name}' failed on all planner fallbacks")
        return False

    def gripper(self, opening_m: float):
        """Returns the GripperCommand result, or None if the command failed.

        Callers that close on an object check `result.position` against the
        object width -- closing much further than that means the fingers met
        no object and the grasp missed.
        """
        opening_m = max(0.0, min(GRIPPER_MAX_OPEN, opening_m))
        goal = GripperCommand.Goal()
        goal.command.position = opening_m
        goal.command.max_effort = 0.0
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"[{self.ns}] gripper goal rejected")
            return None
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result
        self.node.get_logger().info(
            f"[{self.ns}] gripper: position={result.position:.4f} reached_goal={result.reached_goal} stalled={result.stalled}"
        )
        return result


class RelayController(Node):
    def __init__(self, cycles: int = 1, randomize: bool = False):
        super().__init__("relay_pick_place")
        self.cycles = max(1, cycles)
        self.robot_a = RobotArm(self, "robot_a")
        self.robot_b = RobotArm(self, "robot_b")
        self.object_pose: PoseStamped | None = None
        self.create_subscription(PoseStamped, "/binpicking/object_pose", self._object_pose_cb, 10)
        self.randomize = randomize
        self._episode_pub = self.create_publisher(String, EPISODE_TOPIC, 10)
        self._randomize_pub = self.create_publisher(Empty, RANDOMIZE_TOPIC, 10)
        self._episode_index = 0

    def _mark_episode(self, event: str, **fields):
        """Announce an episode boundary. Fire-and-forget: no logger is required
        for the relay to run, and one that is not listening must not block it."""
        payload = {"event": event, "index": self._episode_index}
        payload.update(fields)
        self._episode_pub.publish(String(data=json.dumps(payload)))
        # The publish is asynchronous; give the logger a moment to act on a
        # boundary before the arm starts moving into (or out of) the episode.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def randomize_block(self):
        """Ask the scene for a new block pose and colour, then wait for it.

        Without this a dataset is every episode with the block in the same spot
        in the same colour, which a policy can solve by memorising the
        coordinate -- it never has to look at the block at all.
        """
        self._randomize_pub.publish(Empty())
        deadline = time.monotonic() + RANDOMIZE_SETTLE_SEC
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        where = self.block_world()
        self.get_logger().info(f"[randomize] block now at {_fmt(where) if where else 'unknown'}")
        return where is not None

    def _object_pose_cb(self, msg: PoseStamped):
        self.object_pose = msg

    # 3s was not enough for DDS discovery when this is started right after
    # the launch file, and the run died before touching the robots.
    def wait_for_object_pose(self, timeout_sec=15.0) -> PoseStamped | None:
        """Wait for a *fresh* reading. Every caller uses this to decide whether
        a grasp/carry/place worked, so a cached pose from before the motion
        would answer the previous question, not the current one."""
        self.object_pose = None
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.object_pose is not None:
                return self.object_pose
        return None

    def block_world(self, timeout_sec=15.0):
        obj = self.wait_for_object_pose(timeout_sec)
        if obj is None:
            return None
        return (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z)

    def wait_until_settled_near(self, world_xy, radius=0.15, stable_for=1.0, timeout_sec=10.0) -> bool:
        """Poll /binpicking/object_pose until it's within `radius` of
        world_xy and hasn't moved much for `stable_for` seconds -- the
        hand-off synchronization signal between robot A's place and robot
        B's pick, instead of a guessed fixed sleep."""
        deadline = time.monotonic() + timeout_sec
        last_pos = None
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.object_pose is None:
                continue
            pos = self.object_pose.pose.position
            dx, dy = pos.x - world_xy[0], pos.y - world_xy[1]
            near = math.hypot(dx, dy) < radius
            if not near:
                stable_since = None
                continue
            if last_pos is not None:
                moved = math.hypot(pos.x - last_pos[0], pos.y - last_pos[1])
                if moved > 0.005:
                    stable_since = None
            if stable_since is None:
                stable_since = time.monotonic()
            last_pos = (pos.x, pos.y)
            if time.monotonic() - stable_since >= stable_for:
                return True
        return False

    def run(self):
        self.get_logger().info("Waiting for move_action/gripper_cmd servers on both robots...")
        if not self.robot_a.wait_for_servers() or not self.robot_b.wait_for_servers():
            self.get_logger().error("Action servers not available -- is dual_binpicking.launch.py up?")
            return False

        obj = self.wait_for_object_pose()
        if obj is None:
            self.get_logger().error("No /binpicking/object_pose received -- is dual_binpicking_scene.py running?")
            return False

        # Frame confusion between the world frame and each robot's local
        # planning frame has been the recurring failure mode here, so state
        # the whole layout up front where a failed run's log will show it.
        self.get_logger().info(
            f"Layout: B_OFFSET={_fmt(B_OFFSET)} | handoff world={_fmt(HANDOFF_WORLD)} "
            f"= A-local {_fmt(HANDOFF_LOCAL_A)} = B-local {_fmt(HANDOFF_LOCAL_B)} | "
            f"B dest B-local=({DEST_X:.3f}, {DEST_Y:.3f})"
        )

        for cycle in range(1, self.cycles + 1):
            self.get_logger().info(f"===== cycle {cycle}/{self.cycles}: forward (source -> handoff -> dest) =====")
            if self.randomize and not self.randomize_block():
                self.get_logger().error("randomization did not produce a block pose")
                return False
            if not self.transfer(self.robot_a, "A", (0.0, 0.0, 0.0), WATCH_A, HANDOFF_LOCAL_A, HANDOFF_WORLD,
                                 task=TASK_A_TO_HANDOFF):
                return False
            if not self.transfer(self.robot_b, "B", B_OFFSET, WATCH_B, (DEST_X, DEST_Y, 0.0), DEST_WORLD,
                                 task=TASK_B_TO_DEST):
                return False

            if cycle == self.cycles:
                break

            # Bring it back so the next cycle starts from the same state --
            # the same two transfers with pick/place ends swapped. Every
            # transfer picks wherever the block actually is (ground-truth
            # pose), so the reverse direction needs no separate code path.
            self.get_logger().info(f"===== cycle {cycle}/{self.cycles}: return (dest -> handoff -> source) =====")
            if not self.transfer(self.robot_b, "B", B_OFFSET, WATCH_B, HANDOFF_LOCAL_B, HANDOFF_WORLD,
                                 task=TASK_B_TO_HANDOFF):
                return False
            if not self.transfer(self.robot_a, "A", (0.0, 0.0, 0.0), WATCH_A, (SRC_X, SRC_Y, 0.0), SOURCE_WORLD,
                                 task=TASK_A_TO_SOURCE):
                return False

        self.get_logger().info(f"Relay complete: {self.cycles} cycle(s) finished.")
        return True

    def _open_and_verify(self, arm, arm_label) -> bool:
        """Open the gripper, insisting the fingers actually moved.

        `result.position` is proportional to the knuckle ANGLE, not the finger
        gap -- 0 is fully open -- so `reached_goal` is the signal to trust
        here. A silently-failed open leaves the arm retreating with the block
        still in its fingers, which then gets dropped somewhere random along
        the way to the next pose.
        """
        for attempt in range(1, GRIPPER_RETRIES + 1):
            result = arm.gripper(GRIPPER_MAX_OPEN)
            if result is None:
                return False
            if result.reached_goal:
                return True
            self.get_logger().warn(
                f"[{arm_label}] gripper did not open on attempt {attempt}/{GRIPPER_RETRIES} "
                f"(knuckle still at {result.position / GRIPPER_MAX_OPEN * 100:.0f}% closed); retrying"
            )
        self.get_logger().error(f"[{arm_label}] gripper would not open after {GRIPPER_RETRIES} attempts")
        return False

    def _close_and_verify(self, arm, arm_label, grasp_xyz, grasp_quat) -> bool:
        """Close the gripper, then confirm the block actually came with it.

        The finger opening at stall is NOT a usable success signal here: a
        genuine grasp of the 42mm cube was observed reporting anywhere from
        28mm to 45mm (the fingers pinch corners and sink into the mesh), so a
        width threshold rejects real grasps. Instead do what the single-robot
        expert does -- lift a few cm and check the block's ground-truth z rose
        with the gripper. That is unambiguous.
        """
        obj_before = self.wait_for_object_pose()
        if obj_before is None:
            return False
        z_before = obj_before.pose.position.z

        # Soft contact first, then squeeze -- see GRASP_CONTACT_TARGET_M.
        if arm.gripper(GRASP_CONTACT_TARGET_M) is None:
            return False
        if arm.gripper(GRASP_CLOSE_TARGET_M) is None:
            return False

        test_lift_xyz = (grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + TEST_LIFT_HEIGHT)
        if not arm.move_linear("test_lift", test_lift_xyz, grasp_quat, verify_tcp=False):
            return False

        obj_after = self.wait_for_object_pose()
        if obj_after is None:
            return False
        z_rise = obj_after.pose.position.z - z_before
        if z_rise < TEST_LIFT_MIN_RISE:
            self.get_logger().error(
                f"[{arm_label}] grasp missed: block rose only {z_rise * 1000:.1f}mm during a "
                f"{TEST_LIFT_HEIGHT * 1000:.0f}mm test lift (need {TEST_LIFT_MIN_RISE * 1000:.0f}mm) "
                f"-- the gripper is not holding it"
            )
            # Leave the hand open so the retry does not drag a half-pinched
            # block along on its way back to the watch pose.
            arm.carrying = False
            self._open_and_verify(arm, arm_label)
            return False
        self.get_logger().info(f"[{arm_label}] grasp confirmed: block rose {z_rise * 1000:.1f}mm on test lift")
        arm.carrying = True
        return True

    def transfer(self, arm, arm_label, arm_offset, watch_pose, place_local, place_world,
                 task: str = "") -> bool:
        """Pick the block wherever it currently is and place it at `place_local`.

        `place_local` is in `arm`'s own planning frame; `arm_offset` is that
        robot's base in world, used to convert the world-frame ground-truth
        block pose into the same frame. Both robots and both directions of
        the relay run through here -- the only thing that differs between
        them is which arm and which target.
        """
        # The whole pick-and-place is the retry unit, not just the pick: a
        # carry that drops or throws the block leaves it somewhere new, and the
        # only sane response is to look again and start over from wherever it
        # landed. The reach check inside _pick stops this from looping
        # pointlessly when the block has ended up outside the arm's workspace.
        for attempt in range(1, PICK_ATTEMPTS + 1):
            if attempt > 1:
                self.get_logger().warn(
                    f"[{arm_label}] retrying ({attempt}/{PICK_ATTEMPTS}) -- re-observing the block first"
                )
            self._episode_index += 1
            self._mark_episode("start", task=task, arm=arm.ns, attempt=attempt)
            grasp = self._pick(arm, arm_label, arm_offset, watch_pose)
            placed = grasp is not None and self._place(
                arm, arm_label, watch_pose, place_local, place_world, *grasp
            )
            # Failed attempts are marked, not hidden: whoever consumes the
            # dataset decides whether to train on them, but they must be able
            # to tell. Silently recording a dropped block as a demonstration is
            # how a policy learns to drop blocks.
            self._mark_episode("end", task=task, arm=arm.ns, attempt=attempt, success=bool(placed))
            if placed:
                return True
        self.get_logger().error(f"[{arm_label}] could not move the block in {PICK_ATTEMPTS} attempts")
        return False

    def _pick(self, arm, arm_label, arm_offset, watch_pose):
        """One pick attempt. Returns (grasp_xyz, grasp_quat) on success, None if
        the block is still on the table afterwards (worth another attempt)."""
        # Get clear of the table and open the hand BEFORE reading the block's
        # pose. Those two moves are exactly when the block is most likely to
        # get nudged (a planner fallback that aborts mid-execution can clip
        # it), and grasping at a pose read before them means closing on empty
        # air where the block used to be -- observed once already. The
        # single-robot expert re-observes here for the same reason.
        for name, step in [
            ("watch", lambda: self._go_to_watch(arm, arm_label, watch_pose)),
            ("open", lambda: self._open_and_verify(arm, arm_label)),
        ]:
            if not step():
                self.get_logger().error(f"[{arm_label}] pick attempt aborted at '{name}'")
                return None

        obj = self.wait_for_object_pose()
        if obj is None:
            self.get_logger().error(f"[{arm_label}] no object pose available")
            return None
        obj_world = (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z)
        obj_local = tuple(w - o for w, o in zip(obj_world, arm_offset))
        self.get_logger().info(f"[{arm_label}] block at {_fmt(obj_world)} world = {_fmt(obj_local)} local")

        # Fail fast and legibly on an out-of-reach block instead of burning
        # ~20s of planner fallbacks to discover there is no IK solution.
        reach = math.hypot(obj_local[0], obj_local[1])
        if reach > MAX_REACH_M:
            self.get_logger().error(
                f"[{arm_label}] block is {reach:.3f}m from this robot's base (max {MAX_REACH_M:.3f}m) "
                f"-- it is not where this stage of the relay expected it"
            )
            return None

        raw_yaw = _yaw_from_quaternion(
            obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w
        )
        yaw = _wrap_to_cube_symmetry(raw_yaw)
        tilt = _tilt_from_quaternion(
            obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w
        )
        self.get_logger().info(
            f"[{arm_label}] block yaw {math.degrees(raw_yaw):+.1f}deg -> grasping at "
            f"{math.degrees(yaw):+.1f}deg (cube symmetry), tilt {math.degrees(tilt):.1f}deg, "
            f"reach {reach:.3f}m"
        )
        grasp_quat = _quaternion_multiply(_yaw_quaternion(yaw), DOWN_ORIENTATION)
        grasp_xyz = (obj_local[0], obj_local[1], obj_local[2] + GRASP_OFFSET_Z + GRASP_FLOOR_MARGIN)
        pre_grasp_xyz = (grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + PRE_GRASP_LIFT)

        steps = [
            ("pre_grasp", lambda: arm.move_to_seeded("pre_grasp", pre_grasp_xyz, grasp_quat)),
            # Straight down from directly above the block -- see move_linear().
            # No planner fallback on the descent: that fallback is what wedged
            # robot B against robot A's base.
            ("grasp", lambda: arm.move_linear("grasp", grasp_xyz, grasp_quat, fallback=False)),
            ("close", lambda: self._close_and_verify(arm, arm_label, grasp_xyz, grasp_quat)),
        ]
        for name, step in steps:
            if not step():
                self.get_logger().error(f"[{arm_label}] pick attempt aborted at '{name}'")
                return None
        return grasp_xyz, grasp_quat

    def _place(self, arm, arm_label, watch_pose, place_local, place_world, grasp_xyz, grasp_quat) -> bool:
        """Carry the held block to `place_local` and let go of it."""
        lift_xyz = (grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + POST_GRASP_LIFT)
        place_xyz = (place_local[0], place_local[1], PLACE_REST_Z + GRASP_OFFSET_Z)
        pre_place_xyz = (place_xyz[0], place_xyz[1], place_xyz[2] + PRE_GRASP_LIFT)
        retreat_xyz = (place_xyz[0], place_xyz[1], place_xyz[2] + POST_GRASP_LIFT)
        self.get_logger().info(
            f"[{arm_label}] place at {_fmt(place_xyz)} local = {_fmt(place_world)} world"
        )

        steps = [
            ("lift", lambda: arm.move_linear("lift", lift_xyz, grasp_quat, verify_tcp=False)),
            ("carrying?", lambda: self._still_carrying(arm_label, "lift")),
            ("pre_place", lambda: arm.move_to_seeded("pre_place", pre_place_xyz, DOWN_ORIENTATION)),
            ("carrying?", lambda: self._still_carrying(arm_label, "pre_place")),
            # No tool-pose check on the place, unlike the grasp. The grasp has to
            # be millimetre-accurate because the pads must straddle a 42mm cube;
            # the place only has to get the block over a 28x23cm tray, and
            # `over_target?` below measures exactly that, on the block itself.
            # Verifying the tool here threw away an otherwise perfect transfer --
            # grasped, carried the whole way, then rejected for being 8.3mm off
            # over a tray with centimetres of margin.
            ("place", lambda: arm.move_linear("place", place_xyz, DOWN_ORIENTATION, verify_tcp=False)),
            ("over_target?", lambda: self._block_over_target(arm_label, place_world)),
            ("release", lambda: self._release(arm, arm_label)),
            # Straight up, so the retreat cannot drag the just-placed block.
            ("retreat", lambda: arm.move_linear("retreat", retreat_xyz, DOWN_ORIENTATION, verify_tcp=False)),
            ("home", lambda: arm.move_to_seeded("watch", watch_pose["position"], watch_pose["orientation"])),
        ]
        for name, step in steps:
            if not step():
                self.get_logger().error(f"[{arm_label}] transfer aborted at '{name}'")
                # Whatever went wrong, the hand is no longer reliably holding
                # anything -- leaving the flag set would crawl through the
                # recovery moves and every later approach too.
                arm.carrying = False
                return False

        if not self.wait_until_settled_near(place_world):
            where = self.block_world()
            self.get_logger().error(
                f"[{arm_label}] block is at {_fmt(where) if where else 'unknown'}, not settled near "
                f"{_fmt(place_world)} -- this stage did not actually deliver it"
            )
            return False
        return True

    def _go_to_watch(self, arm, arm_label, watch_pose) -> bool:
        """Reach the watch pose, unsticking the arm through HOME_JOINTS if the
        pose planners cannot find a way out of where the last failure left it."""
        outside = arm.joints_out_of_bounds()
        if outside:
            # Worth naming explicitly: from here every plan fails instantly with
            # a bare FAILURE and nothing says why, because move_group rejects an
            # out-of-bounds start state before it plans anything.
            self.get_logger().error(
                f"[{arm_label}] joints outside the URDF limits: "
                + ", ".join(f"{j}={v:+.4f}rad" for j, v in outside)
                + " -- no plan can succeed until the simulation is restarted"
            )
            return False
        if arm.move_to_seeded("watch", watch_pose["position"], watch_pose["orientation"]):
            return True

        # Straight up first. A failed grasp leaves the tool down among the tray
        # geometry, and from there free-space planning has nothing valid to
        # sample -- even a joint-space goal to the home configuration came back
        # PLANNING_FAILED. A pure vertical Cartesian lift needs no sampling at
        # all, and getting clear of the trays is usually enough to make the rest
        # plannable again.
        measured = arm.tcp_pose()
        if measured is not None:
            here, here_quat = measured
            lifted = (here[0], here[1], here[2] + RECOVERY_LIFT)
            self.get_logger().warn(f"[{arm_label}] stuck at {_fmt(here)}; lifting straight up to get clear")
            arm.move_linear("recover_lift", lifted, here_quat, verify_tcp=False)
            if arm.move_to_seeded("watch", watch_pose["position"], watch_pose["orientation"]):
                return True

        self.get_logger().warn(
            f"[{arm_label}] still cannot plan to the watch pose; recovering through the home configuration"
        )
        if not arm.move_joints("recover_home", HOME_JOINTS):
            # Planning failed even in joint space, which in practice means
            # move_group considers the start state itself in collision. Go
            # around move_group and retract through the controller.
            if not arm.escape_to_home():
                self.get_logger().error(
                    f"[{arm_label}] joint-space recovery and the blind controller escape both failed "
                    f"-- the arm is genuinely stuck and the simulation has to be restarted"
                )
                return False
        return arm.move_to_seeded("watch", watch_pose["position"], watch_pose["orientation"])

    def _release(self, arm, arm_label) -> bool:
        """Open the hand and stop treating the arm as loaded."""
        opened = self._open_and_verify(arm, arm_label)
        arm.carrying = False
        return opened

    def _still_carrying(self, arm_label, after_step) -> bool:
        """The block has to be up in the air with the tool, not back on a tray."""
        where = self.block_world()
        if where is None:
            self.get_logger().error(f"[{arm_label}] no object pose after '{after_step}'")
            return False
        if where[2] < CARRY_CHECK_MIN_Z:
            self.get_logger().error(
                f"[{arm_label}] block is at {_fmt(where)} after '{after_step}' -- z below "
                f"{CARRY_CHECK_MIN_Z * 1000:.0f}mm means it is no longer in the gripper"
            )
            return False
        return True

    def _block_over_target(self, arm_label, place_world) -> bool:
        """Do not open the hand until the block really is above the target."""
        where = self.block_world()
        if where is None:
            self.get_logger().error(f"[{arm_label}] no object pose before release")
            return False
        offset = math.hypot(where[0] - place_world[0], where[1] - place_world[1])
        if offset > PLACED_CHECK_RADIUS:
            self.get_logger().error(
                f"[{arm_label}] block is {offset * 1000:.0f}mm from the place target in xy "
                f"({_fmt(where)} vs {_fmt(place_world)}) -- not releasing here"
            )
            return False
        self.get_logger().info(f"[{arm_label}] block over target ({offset * 1000:.0f}mm off); releasing")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of forward relays. With >1, the block is carried back to the "
        "source between them so each cycle starts from the same state.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the block's position, yaw and colour at the start of every "
        "cycle (dataset collection -- a fixed block is memorisable).",
    )
    args = parser.parse_args()

    rclpy.init()
    node = RelayController(cycles=args.cycles, randomize=args.randomize)
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
