"""Task-specific reward terms.

`object_ee_distance`, `object_is_lifted`, `object_goal_distance` are the
same reach/lift/goal-track shaping IsaacLab's stock `manipulation.lift` task
uses (see isaaclab_tasks...lift/mdp/rewards.py) -- kept close to that
reference on purpose (it's a well-tested shaping for this class of task).

`place_and_release` is new: the stock lift task's goal is just "hold the
object near a floating target pose" -- it never has to be released. A real
pick-AND-place needs the policy to actually let go once the object is at the
destination, or the reward has no pressure to ever open the gripper. This
term only pays out when the object is (a) inside the destination-goal
tolerance, (b) moving slowly (actually settled, not just passing through),
and (c) the gripper is open -- i.e. genuinely placed, not just parked near
the goal while still held.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from . import grasp_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching toward the object (tanh kernel on EE-object distance)."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object.data.root_pos_w[:, :3] - ee_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse bonus once the object clears `minimal_height` (world Z)."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward tracking the commanded goal position, gated on the object
    actually being lifted first (otherwise the policy could get goal-tracking
    reward for an object that never left the source bin)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def place_and_release(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    velocity_threshold: float,
    gripper_open_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bonus for an actual, settled, released placement -- not just hovering
    the held object near the goal (see module docstring)."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    at_goal = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1) < position_threshold
    settled = torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    gripper_open = knuckle_angle < gripper_open_threshold

    return (at_goal & settled & gripper_open).float()


def holding_at_goal_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    gripper_open_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Anti-hoarding pressure: the dense `object_goal_distance` terms alone
    reward hovering the held object at the goal forever just as much as
    releasing it once does (`place_and_release` only fires on the strictly
    harder at_goal+settled+open condition) -- a policy that finds "hover
    near goal, never open" is a locally-consistent way to farm the dense
    terms while avoiding the (mildly risky, since release can bounce the
    object out of `position_threshold`) act of actually letting go. This
    term makes "at the goal but still gripping" cost something every step,
    tipping the balance toward releasing rather than camping."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    at_goal = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1) < position_threshold
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    gripper_closed = knuckle_angle >= gripper_open_threshold

    return (at_goal & gripper_closed).float()


# ===========================================================================
# Curriculum stages (Reach / GraspLift / Transport / Curriculum). Reuse
# `grasp_state.py` for all grasp/contact/goal-footprint logic so it's
# defined once, not duplicated per stage.
# ===========================================================================


# --- Stage 1: Reach --------------------------------------------------------


def grasp_pose_position_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(distance / std)` from the end-effector to the pre-grasp
    target (object + `grasp_state.PRE_GRASP_OFFSET_Z`), NOT the object
    center -- see `grasp_state.pre_grasp_target_pos_b`."""
    ee_pos_b, _ = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_pos_b = grasp_state.pre_grasp_target_pos_b(env, robot_cfg, object_cfg)
    distance = torch.norm(target_pos_b - ee_pos_b, dim=-1)
    return 1 - torch.tanh(distance / std)


def grasp_pose_orientation_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(angle_error / std)` between the current EE orientation and
    the yaw-aligned top-down grasp orientation (`grasp_state.desired_grasp_quat_b`)."""
    _, ee_quat_b = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_quat_b = grasp_state.desired_grasp_quat_b(env, robot_cfg, object_cfg)
    angle_err = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)
    return 1 - torch.tanh(angle_err / std)


def action_jerk_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action-smoothness penalty: `||a_t - 2*a_{t-1} + a_{t-2}||^2`
    (the change in the step-to-step action delta, i.e. "jerk").

    `mdp.action_rate_l2` (`||a_t - a_{t-1}||^2`) only discourages absolute
    movement -- a policy oscillating back and forth by a fixed amount every
    step pays the same `action_rate_l2` cost regardless of how erratic that
    oscillation is, since each individual step-to-step difference is small.
    This term specifically penalizes that oscillation pattern. Added after
    visible trembling was observed in the ReachGrasp GUI demo (2026-08-04),
    both while approaching and while holding a grasp near the floor.

    The `ActionManager` only exposes the current and one step of action
    history (`.action` / `.prev_action`); this term tracks its own 2-step
    history in `grasp_state`'s per-env scratch buffers (zeroed on reset by
    `events.reset_rb5_pp_state` like every other buffer there)."""
    current = env.action_manager.action
    prev = env.action_manager.prev_action
    prev_prev = grasp_state.get_tensor_buffer(env, "jerk_prev_prev_action", current.shape[1:])

    jerk = current - 2.0 * prev + prev_prev

    def _update():
        prev_prev[:] = prev

    grasp_state.step_once(env, "jerk_prev_prev_action_update", _update)
    return torch.sum(jerk**2, dim=-1)


def floor_contact_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 when either fingertip is in contact with the source-bin floor
    (see `grasp_state.add_floor_contact_sensors` / `any_floor_contact`).

    Replaces an earlier height-based proxy (penalize EE height below a
    guessed threshold) -- measuring real grasp episodes from a trained
    checkpoint (`scripts/measure_grasp_ee_height.py`) showed the "right"
    threshold was genuinely unclear (bilateral contact was observed at EE
    heights up to ~33cm above the floor, well above any reasonable guess at
    a grasp height, likely from the object being knocked/batted around
    rather than cleanly gripped) -- a height proxy risked penalizing
    legitimate grasp descent instead of only the actual failure mode. Real
    contact sensing has no such ambiguity: it only fires when the fingertip
    is genuinely touching the floor."""
    return grasp_state.any_floor_contact(env).float()


def grasp_approach_closing_reward(
    env: ManagerBasedRLEnv,
    proximity_std: float,
    max_knuckle_angle: float = 0.8,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[grasp_state.GRIPPER_PRIMARY_JOINT]),
) -> torch.Tensor:
    """Dense bridge between reach and grasp: `proximity_gate * closing_amount`.

    Neither `grasp_pose_position_reward` (rewards being close, regardless of
    gripper state) nor `bilateral_fingertip_contact_reward` (sparse/binary,
    only pays on actual dual-fingertip contact) gives the policy any gradient
    toward *attempting* a close once it's near the object -- this term fills
    that gap. Deliberately tight `proximity_std` (much tighter than
    `grasp_pose_position_reward`'s 0.3) so it only activates genuinely close
    to the pre-grasp target, not throughout the approach -- otherwise it would
    just reward closing early/far away, the same problem
    `empty_gripper_close_penalty` was trying (and failing) to prevent.

    Deliberately no penalty for closing while far away (unlike the retired
    `empty_gripper_close_penalty`, which suppressed exploration of closing at
    all) -- this term is purely additive pressure toward the correct
    behavior, never punitive. Keep this term's weight well below
    `bilateral_fingertip_contact_reward`/`stable_grasp_reward` in the env
    cfg -- it's meant as a nudge toward attempting a close near the object,
    not a substitute for actually achieving one.
    """
    ee_pos_b, _ = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_pos_b = grasp_state.pre_grasp_target_pos_b(env, robot_cfg, object_cfg)
    distance = torch.norm(target_pos_b - ee_pos_b, dim=-1)
    proximity_gate = 1 - torch.tanh(distance / proximity_std)

    robot: Articulation = env.scene[gripper_cfg.name]
    knuckle_angle = robot.data.joint_pos[:, gripper_cfg.joint_ids].squeeze(-1)
    closing_amount = (knuckle_angle / max_knuckle_angle).clamp(0.0, 1.0)

    return proximity_gate * closing_amount


# --- Stage 2: Grasp and Lift ------------------------------------------------


def bilateral_fingertip_contact_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 when both fingertips are simultaneously in contact with the
    object -- NOT awarded for single-finger contact (spec requirement:
    avoid rewarding a one-sided touch as if it were a grasp)."""
    return grasp_state.bilateral_contact(env).float()


def stable_grasp_reward(env: ManagerBasedRLEnv, min_steps: int = grasp_state.STABLE_GRASP_STEPS) -> torch.Tensor:
    """1.0 once bilateral contact has held for >= `min_steps` consecutive
    steps (see `grasp_state.stable_grasp`) -- rewards a *held* grasp, not a
    single-frame contact flicker."""
    return grasp_state.stable_grasp(env, min_steps).float()


def continuous_lift_reward(
    env: ManagerBasedRLEnv,
    target_lift_height: float,
    source_floor_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous (not sparse-threshold) lift reward, gated on an actual
    grasp: `grasped * clamp((height - floor) / target_height, 0, 1)`. Sparse
    thresholding is what `object_is_lifted` (the legacy task's reward) does;
    this version gives gradient signal throughout the lift instead of only
    at the moment it crosses a fixed height, and -- critically -- can't be
    farmed by just knocking/pushing the object upward without holding it,
    since it's gated on `grasp_state.grasped_object`."""
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2] - source_floor_height
    normalized_height = (height / target_lift_height).clamp(0.0, 1.0)
    return grasp_state.grasped_object(env).float() * normalized_height


def empty_gripper_close_penalty(
    env: ManagerBasedRLEnv,
    gripper_open_threshold: float = grasp_state.DEFAULT_GRIPPER_OPEN_THRESHOLD,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
) -> torch.Tensor:
    """1.0 when the gripper is substantially closed but neither fingertip
    has valid object contact -- i.e. clamping on nothing. A single
    per-step boolean (not a one-step-delay-sensitive edge trigger): closing
    while still approaching (gripper closed, not yet in contact) gets
    penalized every step it's in that state, which is intentionally a soft
    continuous pressure rather than a noisy single-frame event, per the
    spec's "avoid noisy one-step penalties" requirement -- the threshold on
    `gripper_open_threshold` itself is the smoothing (only counts once
    substantially closed, not from the first hint of closing)."""
    robot: Articulation = env.scene[robot_cfg.name]
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    gripper_closed = knuckle_angle >= gripper_open_threshold
    return (gripper_closed & ~grasp_state.bilateral_contact(env)).float()


def object_drop_penalty(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """1.0 the step the object transitions from "previously grasped" to "no
    longer grasped" -- only fires after a real grasp was achieved (tracked
    via `grasp_state`'s sticky `ever_grasped` flag), so an object merely
    sitting ungrasped in the source bin never counts as a "drop" (spec
    requirement)."""
    ever_grasped = grasp_state.set_sticky_flag(env, "ever_grasped", grasp_state.grasped_object(env))
    currently_grasped = grasp_state.grasped_object(env)
    # "was grasped at some point, isn't right now" -- fires every step
    # while ungrasped after having been grasped, not just on the single
    # transition frame, which is fine: it's a per-step penalty like the
    # other terms, weighted low enough (-5.0, applied only while true) that
    # a fast recovery still nets positive overall.
    return (ever_grasped.bool() & ~currently_grasped).float()


# --- Stage 3: Transport -----------------------------------------------------


def maintain_grasp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 while the object remains grasped -- the "don't drop it while
    moving" pressure, separate from `object_drop_penalty`'s one-sided cost."""
    return grasp_state.grasped_object(env).float()


def object_to_goal_position_reward(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    safe_transport_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tracking reward toward the commanded goal, active only while (a) the
    object is actually grasped and (b) above `safe_transport_height` --
    unlike the legacy `object_goal_distance`, which only gates on a lifted
    *height* (so it could reward tracking an object that's merely being
    pushed/dragged, not held)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    above_safe_height = object.data.root_pos_w[:, 2] > safe_transport_height
    gate = grasp_state.grasped_object(env) & above_safe_height
    return gate.float() * (1 - torch.tanh(distance / std))


def object_height_safety_reward(
    env: ManagerBasedRLEnv,
    safe_transport_height: float,
    max_safe_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 while the (grasped) object's height stays within
    [`safe_transport_height`, `max_safe_height`] during transport -- penalizes
    neither too low (risk of clipping the bin wall) nor absurdly high
    (wasted motion / instability), only while actually grasped."""
    object: RigidObject = env.scene[object_cfg.name]
    z = object.data.root_pos_w[:, 2]
    in_band = (z > safe_transport_height) & (z < max_safe_height)
    return grasp_state.grasped_object(env).float() * in_band.float()


# --- Stage: Place (scripted release, mirrors ReachGrasp's scripted close) --


def premature_drop_penalty(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Like `object_drop_penalty`, but excludes the Place stage's own
    scripted release: fires only if contact is lost WHILE
    `actions.AutoReleaseAction` hasn't opened the gripper yet (the
    `released_at_place` sticky flag that action sets on trigger) -- a real
    accidental drop, not the intended end-of-episode release
    `released_and_stable_reward` rewards. Without this distinction, the
    ordinary `object_drop_penalty` would fire on every successful placement
    too, directly fighting the behavior this stage exists to train."""
    ever_grasped = grasp_state.set_sticky_flag(env, "ever_grasped", grasp_state.grasped_object(env, object_cfg))
    currently_grasped = grasp_state.grasped_object(env, object_cfg)
    released_on_purpose = grasp_state.get_counter(env, "released_at_place").bool()
    return (ever_grasped.bool() & ~currently_grasped & ~released_on_purpose).float()


# --- Stage 4: full Curriculum (place + release) -----------------------------


def object_inside_destination_reward(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """1.0 while the object's world XY is inside the destination bin's
    actual inner footprint (`grasp_state.object_inside_destination_xy`,
    reading `bin_geometry.yaml`) -- not just "close to the bin center" by
    Euclidean distance, per the spec's requirement to use the real
    footprint."""
    return grasp_state.object_inside_destination_xy(env, margin).float()


def released_and_stable_reward(
    env: ManagerBasedRLEnv,
    margin: float,
    max_height_above_floor: float,
    velocity_threshold: float,
    gripper_open_threshold: float,
    hold_steps: int,
    dest_floor_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 once `full_place_success` (see below) has held for `hold_steps`
    consecutive steps. Shares its underlying condition function with
    `terminations.full_place_success` (spec requirement: reward and
    termination must use the same success definition, not two
    independently-drifting ones)."""
    condition = full_place_success_condition(
        env, margin, max_height_above_floor, velocity_threshold, gripper_open_threshold, dest_floor_height,
        robot_cfg, object_cfg,
    )
    counter = grasp_state.advance_consecutive_counter(env, "place_success_streak", condition)
    return (counter >= hold_steps).float()


def full_place_success_condition(
    env: ManagerBasedRLEnv,
    margin: float,
    max_height_above_floor: float,
    velocity_threshold: float,
    gripper_open_threshold: float,
    dest_floor_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Shared success condition for Stage 4: object inside the destination
    footprint, near the floor (not still hovering), gripper open, no valid
    grasp contact, and settled (low linear velocity). Defined once here and
    called from both `released_and_stable_reward` (this file) and
    `terminations.full_place_success` -- NOT gripper-angle alone (spec
    requirement) and NOT position-only (also checks release + settle)."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    inside_footprint = grasp_state.object_inside_destination_xy(env, margin, object_cfg)
    near_floor = (object.data.root_pos_w[:, 2] - dest_floor_height) < max_height_above_floor
    settled = torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    gripper_open = knuckle_angle < gripper_open_threshold
    not_grasped = ~grasp_state.grasped_object(env, object_cfg)

    return inside_footprint & near_floor & settled & gripper_open & not_grasped
