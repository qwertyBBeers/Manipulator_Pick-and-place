"""Reset-event terms specific to the curriculum stages.

`reset_rb5_pp_state` must be included (mode="reset") in every curriculum
stage's `EventCfg` that uses any of `grasp_state.py`'s per-env counters
(`stable_grasp`, drop-tracking, success-hold counters, phase index for
Stage 4) -- otherwise a counter from a previous episode leaks into the next
one for envs that just reset.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from rb5_isaaclab.robots.rb5_850e import GRIPPER_MIMIC_JOINTS, GRIPPER_MIMIC_MULTIPLIERS, GRIPPER_PRIMARY_JOINT

from . import grasp_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_rb5_pp_state(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Zero every `grasp_state.py` per-env scratch counter/flag for the
    envs being reset (bilateral-contact streak, sticky "ever grasped" flag,
    success-hold counters, ...). Safe to call even if some named buffers
    don't exist yet for this stage -- only zeros what's present."""
    if not hasattr(env, "_rb5_pp_state"):
        return
    for key, buf in env._rb5_pp_state.items():
        if key.startswith("__step__"):
            continue
        buf[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Stage 3 (Transport): start from an already-grasped, already-lifted state.
# ---------------------------------------------------------------------------

# Measured via `scripts/solve_holding_pose.py`, not guessed: drives a real
# Jacobian-based differential-IK controller down toward the object until a
# fingertip contact sensor actually fires, closes the gripper, confirms
# `grasp_state.bilateral_contact` holds with real force on both fingertips,
# then reads back the arm's joint config, the gripper's real
# contact-stopped knuckle angle, and the object's actual resting pose. See
# `CURRICULUM_REPORT.md` section 1d for why this replaced an earlier,
# geometrically-wrong version that placed the object at the tcp frame
# (which sits well above the actual fingertip plane, so the object ended up
# inside the gripper's own mechanism).
#
# Known simplification: the object doesn't actually leave the bin floor
# here -- "grasped, still at floor height" rather than "grasped and
# lifted". Exercises Stage 3's reward/termination machinery correctly
# either way; a higher, airborne holding pose is a follow-up improvement.
HOLDING_ARM_JOINT_POS = {
    "base": 0.2174,
    "shoulder": 0.4317,
    "elbow": 2.0918,
    "wrist1": -0.8864,
    "wrist2": 1.5859,
    "wrist3": -0.2439,
}
HOLDING_GRIPPER_KNUCKLE_POS = 0.0762  # ACTUAL settled angle on contact, not a commanded target
# The 5 follower joints' ACTUAL measured settled angles from the same real
# grasp closure -- NOT derived from `GRIPPER_MIMIC_MULTIPLIERS *
# HOLDING_GRIPPER_KNUCKLE_POS`, since a follower stopped early by real
# contact doesn't settle at a fixed ratio of the primary's angle (each of
# the 6 independently-driven gripper joints settles wherever contact stops
# it -- e.g. `left_inner_knuckle` settled at 0.335 rad here, nowhere near
# the `1.0 * 0.0762` the multiplier formula would predict).
HOLDING_GRIPPER_FOLLOWER_POS = {
    "robotiq_85_left_inner_knuckle_joint": 0.3350,
    "robotiq_85_right_inner_knuckle_joint": -0.0122,
    "robotiq_85_right_knuckle_joint": 0.3088,
    "robotiq_85_left_finger_tip_joint": 0.5829,
    "robotiq_85_right_finger_tip_joint": -0.5687,
}
HOLDING_OBJECT_POS_B = (0.5083, -0.0024, 0.0241)
HOLDING_OBJECT_QUAT_B = (0.9995, 0.0147, -0.0214, -0.0196)


def reset_robot_holding_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Stage 3 reset: put the robot in `HOLDING_ARM_JOINT_POS` with the
    gripper at its real contact-stopped angles, and the object at its own
    measured resting pose for that exact configuration -- all measured
    together from one real simulated grasp, so mutually consistent by
    construction. A fixed, pre-measured state, not a live grasp-closure
    simulation -- if per-episode randomization is added on top later, the
    object may need a few settle steps before `bilateral_contact` registers
    (fine: `maintain_grasp_reward` gates on current state, not reset-time).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    joint_names = robot.data.joint_names
    n = len(env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    for name, val in HOLDING_ARM_JOINT_POS.items():
        joint_pos[:, joint_names.index(name)] = val
    joint_pos[:, joint_names.index(GRIPPER_PRIMARY_JOINT)] = HOLDING_GRIPPER_KNUCKLE_POS
    for jname in GRIPPER_MIMIC_JOINTS:
        joint_pos[:, joint_names.index(jname)] = HOLDING_GRIPPER_FOLLOWER_POS[jname]

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Target is the full "close" command (not the measured partial angle) --
    # matches what the gripper action sends to maintain a grip, so the
    # actuator-vs-object pressure is the same real situation this was
    # measured from.
    target = joint_pos.clone()
    target[:, joint_names.index(GRIPPER_PRIMARY_JOINT)] = 0.8
    for jname in GRIPPER_MIMIC_JOINTS:
        target[:, joint_names.index(jname)] = GRIPPER_MIMIC_MULTIPLIERS[jname] * 0.8
    robot.set_joint_position_target(target, env_ids=env_ids)

    object_pos_b = torch.tensor(HOLDING_OBJECT_POS_B, device=env.device).expand(n, 3)
    object_quat_b = torch.tensor(HOLDING_OBJECT_QUAT_B, device=env.device).expand(n, 4)
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]
    import isaaclab.utils.math as math_utils

    object_pos_w, object_quat_w = math_utils.combine_frame_transforms(
        root_pos_w, root_quat_w, object_pos_b, object_quat_b
    )
    object.write_root_pose_to_sim(torch.cat([object_pos_w, object_quat_w], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    reset_rb5_pp_state(env, env_ids)
    # Mark these envs as "already grasped" so the drop-penalty sticky flag
    # (which normally only arms after a real grasp is observed in-episode)
    # is active from step 0 -- otherwise dropping the object in the first
    # few steps of a Transport episode wouldn't be penalized.
    flag = grasp_state.get_counter(env, "ever_grasped")
    flag[env_ids] = 1.0
