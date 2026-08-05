"""Gripper action term with a close-side deadband.

`isaaclab.envs.mdp.BinaryJointPositionActionCfg` treats any non-positive
raw action as "close" -- fine once a policy has learned to output a clearly
positive value to stay open, but a fresh/random policy's near-zero action
(exactly what PPO starts with) snaps the gripper closed on effectively
every step, injecting a large disturbance now that all 6 gripper joints are
independently, fully PD-driven (see `robots/rb5_850e.py`).

`DeadbandBinaryJointPositionAction` only commits to "close" once the raw
action drops meaningfully below zero (`close_threshold`, default -0.5) --
same safety reasoning `config/reach_env_cfg.py` uses for excluding the
gripper action term entirely in Stage 1: default to the safe/open state
unless the policy clearly asks otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from . import grasp_state


class DeadbandBinaryJointPositionAction(BinaryJointPositionAction):
    """Same as `BinaryJointPositionAction`, but "close" only fires below
    `cfg.close_threshold` instead of below 0.0."""

    cfg: "DeadbandBinaryJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        binary_mask = actions < self.cfg.close_threshold
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class DeadbandBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = DeadbandBinaryJointPositionAction

    close_threshold: float = -0.5
    """Raw action must drop below this to command "close" (default: open).
    -0.5 leaves a wide open-side deadband around 0 (PPO's early-training
    action mean) while still letting the policy reliably reach "close" by
    saturating its output negative, same as it already must for "open" to
    reliably win under the un-modified convention's own >=0 threshold."""


class AutoGraspAction(BinaryJointPositionAction):
    """Scripted (non-learned) gripper close/open -- see `AutoGraspActionTermCfg`.

    Every other gripper action term in this package (`BinaryJointPositionAction`,
    `DeadbandBinaryJointPositionAction`) derives its open/close decision from
    the RL policy's own action output -- the policy has to discover *when* to
    close. This term instead derives the decision entirely from env state
    (`grasp_state.is_near_pregrasp` / `grasp_state.grasped_object`), so
    closing at the right moment is guaranteed by construction rather than
    something exploration has to stumble into. Used by `reach_grasp_env_cfg.py`
    to make "reach precisely enough to actually grasp" the thing being
    learned/rewarded, instead of a separate learned close-timing skill on top
    of it -- see that module's docstring for the full rationale.
    """

    cfg: "AutoGraspActionTermCfg"

    @property
    def action_dim(self) -> int:
        # 0, not 1 -- this term takes no input from the policy at all (see
        # class docstring). `ActionManager.process_action` still calls
        # `process_actions` on every registered term each env step
        # regardless of action_dim, passing an empty (num_envs, 0) slice --
        # that's all this term needs, since it reads env state directly.
        return 0

    def process_actions(self, actions: torch.Tensor):
        # `actions` (the empty slice described above) is intentionally
        # ignored. Close on EITHER condition:
        #   - near the pre-grasp target (attempt a grasp), or
        #   - already holding a real grasp (stay closed).
        # The second condition matters on its own: without it, a momentary
        # drift back outside `is_near_pregrasp`'s tight tolerance -- while
        # already holding the object -- would immediately command the
        # gripper open again and drop it.
        near = grasp_state.is_near_pregrasp(self._env, self.cfg.position_threshold, self.cfg.orientation_threshold)
        holding = grasp_state.grasped_object(self._env)
        close_mask = (near | holding).unsqueeze(-1)
        self._raw_actions[:] = close_mask.float()
        self._processed_actions = torch.where(close_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class AutoGraspActionTermCfg(BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = AutoGraspAction

    position_threshold: float = grasp_state.DEFAULT_POSITION_THRESHOLD
    orientation_threshold: float = grasp_state.DEFAULT_ORIENTATION_THRESHOLD


class AutoReleaseAction(BinaryJointPositionAction):
    """Scripted (non-learned) gripper release for the Place stage -- the
    mirror of `AutoGraspAction`, opening instead of closing.

    Every Place-stage episode starts already holding the object (see
    `mdp.events.reset_robot_holding_object`). The gripper stays closed while
    the RL policy carries the object toward the destination, then opens
    automatically once `grasp_state.is_at_place_target` is true -- same
    "remove the timing-discovery problem from the policy" rationale
    `AutoGraspAction` uses, just inverted: a structurally-guaranteed release
    at the right moment instead of a right-moment close.

    Sticky once triggered (mirrors `AutoGraspAction`'s "stay closed once
    holding" -- inverted here to "stay open once released"), so a momentary
    settle-velocity blip right after opening can't reclose the gripper
    around the object it just placed.
    """

    cfg: "AutoReleaseActionTermCfg"

    @property
    def action_dim(self) -> int:
        # 0 -- see AutoGraspAction.action_dim's docstring, same reasoning.
        return 0

    def process_actions(self, actions: torch.Tensor):
        ready = grasp_state.is_at_place_target(
            self._env,
            self.cfg.margin,
            self.cfg.max_height_above_floor,
            self.cfg.velocity_threshold,
            self.cfg.dest_floor_height,
        )
        released = grasp_state.set_sticky_flag(self._env, "released_at_place", ready)
        open_mask = released.bool().unsqueeze(-1)
        self._raw_actions[:] = open_mask.float()
        self._processed_actions = torch.where(open_mask, self._open_command, self._close_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class AutoReleaseActionTermCfg(BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = AutoReleaseAction

    margin: float = grasp_state.DEFAULT_POSITION_THRESHOLD
    max_height_above_floor: float = 0.06
    velocity_threshold: float = grasp_state.DEFAULT_VELOCITY_THRESHOLD
    # No sensible task-agnostic default (depends on bin_geometry.yaml) --
    # each stage's env cfg must supply this explicitly, same as
    # `ArticulationCfg.spawn`-style MISSING fields elsewhere in this repo.
    dest_floor_height: float = MISSING


class EMASmoothedJointPositionAction(JointPositionAction):
    """Same as `JointPositionAction`, but low-pass filters the position
    target before sending it to the actuator:

        smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

    `JointPositionAction.process_actions` recomputes its target fresh from a
    FIXED default pose every control step (`offset + scale * raw_action`) --
    not incremental/delta control, and nothing smooths it across steps. Any
    step-to-step noise in the policy's own output (even the *mean* action,
    not just sampling noise -- `play_spawn_and_wait.py` uses `mean_actions`
    deterministically) goes straight into a PD position target every 20ms.
    Added after visible trembling was observed in the ReachGrasp GUI demo
    (2026-08-04); see `reach_grasp_env_cfg.py`'s RewardsCfg for the fuller
    context (this is one of three changes made together: damping increase,
    this filter, and `rewards.action_jerk_penalty`).

    The blend happens once per control step in `process_actions` (not once
    per physics substep in `apply_actions`, which runs `cfg.decimation`
    times per control step against the same already-blended target) --
    doing it per-substep would double-apply the blend within one control
    step and change the effective time constant.

    `cfg.alpha` can be a single float (same blend for every joint) or a
    dict (per-joint, same `resolve_matching_names_values` pattern the base
    `JointAction` already uses for `scale`/`offset`) -- added 2026-08-04
    after a uniform alpha=0.3 across all 6 arm joints wrecked orientation
    tracking (~51deg error) while barely affecting position: the wrist
    joints (wrist1/2/3) dominate EE *orientation* and need fast correction,
    while base/shoulder/elbow dominate *position* via slow, large-scale
    motion the same lag barely touches. A uniform blend forced both to live
    with whichever tradeoff bit the wrist hardest."""

    cfg: "EMASmoothedJointPositionActionCfg"

    def __init__(self, cfg: "EMASmoothedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        # Seeded from the default pose (same as `self._offset` when
        # `use_default_offset=True`) so the first control step's blend
        # starts from rest, not from zero.
        if isinstance(self._offset, torch.Tensor):
            self._smoothed_actions = self._offset.clone()
        else:
            self._smoothed_actions = torch.full_like(self._processed_actions, self._offset)

        # Resolve per-joint alpha (mirrors JointAction.__init__'s scale/offset
        # dict-resolution above).
        if isinstance(cfg.alpha, (float, int)):
            self._alpha = float(cfg.alpha)
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones(self._num_joints, device=self.device)
            index_list, name_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            if len(index_list) != self._num_joints:
                raise ValueError(
                    f"EMASmoothedJointPositionActionCfg.alpha must cover every joint in {self._joint_names};"
                    f" missing: {set(self._joint_names) - set(name_list)}"
                )
            self._alpha[index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported alpha type: {type(cfg.alpha)}. Supported types are float and dict.")

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._smoothed_actions[:] = (
            self._alpha * self._processed_actions + (1.0 - self._alpha) * self._smoothed_actions
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._smoothed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if isinstance(self._offset, torch.Tensor):
            self._smoothed_actions[ids] = self._offset[ids]
        else:
            self._smoothed_actions[ids] = self._offset


@configclass
class EMASmoothedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = EMASmoothedJointPositionAction

    alpha: float | dict[str, float] = 0.3
    """EMA blend weight for the newest raw target each control step
    (`smoothed = alpha*raw + (1-alpha)*smoothed`). Lower = smoother but
    slower to track a genuinely new target. Either a single float (applied
    to every joint) or a dict (per-joint, resolved by regex/name like
    `JointActionCfg.scale`/`.offset` -- must cover every joint in
    `joint_names`). 0.3 uniform was a first experiment; see
    `EMASmoothedJointPositionAction`'s docstring for why per-joint values
    (lighter smoothing on the wrist) replaced it."""
