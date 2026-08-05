"""Task-specific termination terms.

`time_out` and object-dropped-below-floor are both covered by generic
`isaaclab.envs.mdp` terms already re-exported through this package's
`mdp/__init__.py` (`mdp.time_out`, `mdp.root_height_below_minimum`) -- no
need to reimplement them here. This module only adds the one thing that's
actually task-specific: an early-success termination once the object is
genuinely placed and released (same condition as the `place_and_release`
reward bonus in rewards.py, checked here as a boolean instead of a reward)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from . import grasp_state
from .rewards import full_place_success_condition

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_placed(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    velocity_threshold: float,
    gripper_open_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    at_goal = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1) < position_threshold
    settled = torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    gripper_open = knuckle_angle < gripper_open_threshold

    return at_goal & settled & gripper_open


# ===========================================================================
# Curriculum stage terminations
# ===========================================================================


def reach_success(
    env: ManagerBasedRLEnv,
    position_threshold: float,
    orientation_threshold: float,
    hold_steps: int,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Stage 1 early-success: position AND orientation error both below
    threshold, held for `hold_steps` consecutive control steps."""
    condition = grasp_state.is_near_pregrasp(
        env, position_threshold, orientation_threshold, robot_cfg, object_cfg, ee_frame_cfg
    )
    counter = grasp_state.advance_consecutive_counter(env, "reach_success_streak", condition)
    return counter >= hold_steps


def grasp_lift_success(
    env: ManagerBasedRLEnv,
    target_lift_height: float,
    source_floor_height: float,
    velocity_threshold: float,
    hold_steps: int,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 2 early-success: grasped, above `target_lift_height`, settled
    (low velocity -- not being swung around), held for `hold_steps`."""
    object: RigidObject = env.scene[object_cfg.name]
    height_ok = (object.data.root_pos_w[:, 2] - source_floor_height) > target_lift_height
    settled = torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold
    condition = grasp_state.grasped_object(env) & height_ok & settled

    counter = grasp_state.advance_consecutive_counter(env, "grasp_lift_success_streak", condition)
    return counter >= hold_steps


def grasp_success(
    env: ManagerBasedRLEnv,
    hold_steps: int,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage-Grasp early-success (`grasp_env_cfg.py`): grasped, held for
    `hold_steps` -- no lift-height requirement, unlike `grasp_lift_success`.
    This stage's whole point is isolating "can the policy discover a grasp
    at all" from "can it also lift", so its success condition mirrors that."""
    condition = grasp_state.grasped_object(env, object_cfg)
    counter = grasp_state.advance_consecutive_counter(env, "grasp_success_streak", condition)
    return counter >= hold_steps


def transport_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    velocity_threshold: float,
    hold_steps: int,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage 3 early-success: object near the transport-hover target,
    still grasped, settled, held for `hold_steps`."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    pos_ok = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1) < position_threshold
    settled = torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold
    condition = pos_ok & grasp_state.grasped_object(env) & settled

    counter = grasp_state.advance_consecutive_counter(env, "transport_success_streak", condition)
    return counter >= hold_steps


def full_place_success(
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
    """Stage 4 early-success. Uses the SAME condition function as
    `rewards.released_and_stable_reward` (`rewards.full_place_success_condition`)
    so the reward bonus and the termination can never disagree about what
    "placed" means (spec requirement)."""
    condition = full_place_success_condition(
        env, margin, max_height_above_floor, velocity_threshold, gripper_open_threshold, dest_floor_height,
        robot_cfg, object_cfg,
    )
    counter = grasp_state.advance_consecutive_counter(env, "place_success_streak", condition)
    return counter >= hold_steps
