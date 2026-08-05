"""Task-specific observation terms (adapted from
isaaclab_tasks.manager_based.manipulation.lift.mdp.observations, extended
with object orientation and gripper-openness, since the ROS heuristic
pipeline found object yaw alignment and grasp/gripper state to matter a lot
for this exact gripper -- Manipulator/README2.md §7.13, §7.16, §7.18)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from . import grasp_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3]
    )
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Orientation (w, x, y, z) of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object.data.root_pos_w[:, :3],
        object.data.root_quat_w,
    )
    return object_quat_b


def gripper_opening(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_85_left_knuckle_joint"]),
) -> torch.Tensor:
    """Primary knuckle joint angle (0 = open, 0.8 rad = closed) -- a cheap,
    always-available proxy for "how open is the gripper right now" since we
    don't have a contact/force sensor (mirrors trajectory_bridge.py's own
    documented limitation: knuckle angle is a heuristic, not true contact
    sensing)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids]


# ---------------------------------------------------------------------------
# Reach-stage observations (Stage 1+): end-effector pose, pre-grasp vector,
# grasp-orientation error. Added for the curriculum stages -- the original
# `RB5-PickPlace-JointPos-v0` observation set (above) never exposed EE pose
# directly, only object pose, since the stock lift-task-derived reward
# didn't need it.
# ---------------------------------------------------------------------------


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector (tcp) position in the robot root frame."""
    ee_pos_b, _ = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    return ee_pos_b


def ee_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector (tcp) orientation (wxyz) in the robot root frame."""
    _, ee_quat_b = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    return ee_quat_b


def ee_to_pregrasp_vector(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    offset_z: float = grasp_state.PRE_GRASP_OFFSET_Z,
) -> torch.Tensor:
    """Vector (robot-root frame) from the end-effector to the pre-grasp
    target (object position + `offset_z`) -- NOT the object center itself,
    per the "avoid position-only reaching to the object center" requirement
    (a top-down pre-grasp approach point, not the grasp depth itself)."""
    ee_pos_b, _ = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_pos_b = grasp_state.pre_grasp_target_pos_b(env, robot_cfg, object_cfg, offset_z)
    return target_pos_b - ee_pos_b


def grasp_orientation_error(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Scalar (num_envs, 1) angular error between the current EE orientation
    and the desired top-down grasp orientation (`grasp_state.desired_grasp_quat_b`,
    yaw-aligned to the object) -- `quat_error_magnitude` is numerically
    stable (no gimbal-lock-prone Euler-angle differencing), per the "another
    numerically stable grasp-alignment representation" requirement."""
    _, ee_quat_b = grasp_state.ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_quat_b = grasp_state.desired_grasp_quat_b(env, robot_cfg, object_cfg)
    return math_utils.quat_error_magnitude(ee_quat_b, target_quat_b).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Grasp-stage observations (Stage 2+): contact-based grasp state, object
# velocity, object-relative-to-EE. Contact-based, not gripper-angle-based --
# see `grasp_state.py` module docstring.
# ---------------------------------------------------------------------------


def fingertip_contact_forces_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 2): [left, right] fingertip contact-force magnitudes."""
    left, right = grasp_state.fingertip_contact_forces(env)
    return torch.stack([left, right], dim=-1)


def bilateral_contact_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 1) float 0/1: both fingertips currently in contact."""
    return grasp_state.bilateral_contact(env).float().unsqueeze(-1)


def grasped_object_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 1) float 0/1: `grasp_state.grasped_object` (bilateral
    contact AND object near the grasp center)."""
    return grasp_state.grasped_object(env).float().unsqueeze(-1)


def object_position_relative_to_ee(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object position relative to the end-effector (world-frame vector --
    translation-only, no frame rotation needed since this is just used as a
    proximity/alignment signal)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return object.data.root_pos_w[:, :3] - ee_pos_w


def object_lin_vel(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_lin_vel_w


def object_ang_vel(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_ang_vel_w
