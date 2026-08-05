"""Stage: Place with a scripted release probe -- the mirror of
`reach_grasp_env_cfg.py`'s scripted-grasp idea, but for the release side.

Every earlier attempt at a full pick-place task (`curriculum_env_cfg.py`)
had the RL policy control the gripper directly for BOTH grasp and release,
which meant it had to discover two separate timing decisions. ReachGrasp
already removed the grasp-timing half of that problem by scripting the
close (`mdp.AutoGraspActionTermCfg`). This stage does the same for the
release half: the gripper is driven by a scripted, non-learned action term
(`mdp.AutoReleaseActionTermCfg`, 0 policy dimensions) that opens
automatically once the object is physically at the destination
(`grasp_state.is_at_place_target` -- inside the destination footprint, near
the floor, settled), and stays open for the rest of the episode once
triggered. The RL policy only controls the arm (6-dim action space) and
only has one thing left to learn: carry the already-grasped object to the
destination and set it down gently enough that the scripted release
actually counts as a real placement.

Every episode starts already holding the object -- same
`mdp.events.reset_robot_holding_object` reset `transport_env_cfg.py` uses
(this stage is purely about carry + lower + release, not about grasping;
that's ReachGrasp's job). `released_and_stable_reward` / `full_place_success`
(both already shared with Stage 4's Curriculum -- see `rewards.py`'s
`full_place_success_condition`) now measure the REAL physical outcome of
the scripted release attempt rather than something the policy has to
separately decide to attempt, same "judge success by the real outcome"
pattern ReachGrasp uses for the grasp side.

Built on top of `transport_env_cfg.py` (same scene/reset/goal-command
machinery) rather than from scratch -- Transport already solved "carry a
held object toward a destination-relative target"; this stage only needs to
(a) push that target down to the actual resting height instead of a hover
point, and (b) swap the learned gripper action for the scripted release.

Registered as `RB5-PickPlace-Place-JointPos-v0`.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import (
    ARM_JOINT_NAMES,
    GRIPPER_ALL_JOINT_NAMES,
    GRIPPER_CLOSE_COMMAND_EXPR,
    GRIPPER_OPEN_COMMAND_EXPR,
    GRIPPER_PRIMARY_JOINT,
)

from .. import mdp
from ..bin_geometry import load_bin_geometry
from .curriculum_env_cfg import (
    DEST_INSIDE_MARGIN_TRAIN,
    PLACE_HOLD_STEPS,
    PLACE_MAX_HEIGHT_ABOVE_FLOOR,
    PLACE_VELOCITY_THRESHOLD,
)
from .grasp_lift_env_cfg import ARM_ACTION_SCALE
from .reach_env_cfg import OBJECT_HALF_HEIGHT
from .transport_env_cfg import RB5PickPlaceTransportEnvCfg

SOURCE_BIN, DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_DST_FLOOR_Z = DEST_BIN.center[2]

# Neuters `object_to_goal_position_reward`'s built-in "above safe transport
# height" gate -- Transport never needs the object to descend below that
# band, but Place's whole point is descending all the way to the
# destination floor, so the gate must always pass here. Reusing the
# existing function with an always-true threshold rather than writing an
# ungated variant.
_ALWAYS_SAFE_HEIGHT = _SRC_FLOOR_Z - 1.0


@configclass
class ActionsCfg:
    """6-dim: arm only. Gripper release is fully scripted
    (`AutoReleaseActionTermCfg`, 0 policy dims) -- see module docstring."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True
    )
    auto_release_action: mdp.AutoReleaseActionTermCfg = mdp.AutoReleaseActionTermCfg(
        asset_name="robot",
        joint_names=GRIPPER_ALL_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSE_COMMAND_EXPR,
        margin=DEST_INSIDE_MARGIN_TRAIN,
        max_height_above_floor=PLACE_MAX_HEIGHT_ABOVE_FLOOR,
        velocity_threshold=PLACE_VELOCITY_THRESHOLD,
        dest_floor_height=_DST_FLOOR_Z,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_orientation = ObsTerm(func=mdp.ee_orientation_in_robot_root_frame)
        object_position_rel_ee = ObsTerm(func=mdp.object_position_relative_to_ee)
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel)
        bilateral_contact = ObsTerm(func=mdp.bilateral_contact_flag)
        grasped_object = ObsTerm(func=mdp.grasped_object_flag)
        gripper_opening = ObsTerm(
            func=mdp.gripper_opening, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    maintain_grasp_reward = RewTerm(func=mdp.maintain_grasp_reward, weight=1.0)
    object_to_goal_position_reward = RewTerm(
        func=mdp.object_to_goal_position_reward,
        params={"std": 0.3, "command_name": "object_pose", "safe_transport_height": _ALWAYS_SAFE_HEIGHT},
        weight=4.0,
    )
    fine_goal_position_reward = RewTerm(
        func=mdp.object_to_goal_position_reward,
        params={"std": 0.05, "command_name": "object_pose", "safe_transport_height": _ALWAYS_SAFE_HEIGHT},
        weight=2.0,
    )
    # Ground-truth outcome of the scripted release attempt -- see module
    # docstring. Dominant weight, same role ReachGrasp gives
    # `stable_grasp_reward`.
    released_and_stable_reward = RewTerm(
        func=mdp.released_and_stable_reward,
        params={
            "margin": DEST_INSIDE_MARGIN_TRAIN,
            "max_height_above_floor": PLACE_MAX_HEIGHT_ABOVE_FLOOR,
            "velocity_threshold": PLACE_VELOCITY_THRESHOLD,
            "gripper_open_threshold": 0.4,
            "hold_steps": PLACE_HOLD_STEPS,
            "dest_floor_height": _DST_FLOOR_Z,
            "robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
        weight=4.0,
    )
    premature_drop_penalty = RewTerm(func=mdp.premature_drop_penalty, weight=-5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SRC_FLOOR_Z - 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    # Same shared success condition Curriculum's Stage 4 uses -- see
    # `rewards.full_place_success_condition`'s docstring (reward and
    # termination must never disagree about what "placed" means).
    full_place_success = DoneTerm(
        func=mdp.full_place_success,
        params={
            "margin": DEST_INSIDE_MARGIN_TRAIN,
            "max_height_above_floor": PLACE_MAX_HEIGHT_ABOVE_FLOOR,
            "velocity_threshold": PLACE_VELOCITY_THRESHOLD,
            "gripper_open_threshold": 0.4,
            "hold_steps": PLACE_HOLD_STEPS,
            "dest_floor_height": _DST_FLOOR_Z,
            "robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
    )


@configclass
class EventCfg:
    reset_holding = EventTerm(func=mdp.reset_robot_holding_object, mode="reset")
    # Same validated-best setup the 2026-08-04/05 ReachGrasp ablation landed
    # on (see reach_grasp_env_cfg.py / robots/rb5_850e.py's ARM_DAMPING) --
    # PD gain randomization alone was the single best individual change of
    # that ablation, baked in directly here rather than re-running the same
    # ablation for a new stage.
    randomize_arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RB5PickPlacePlaceEnvCfg(RB5PickPlaceTransportEnvCfg):
    """Inherits Transport's scene/commands/reset wholesale (via subclassing
    + `super().__post_init__()`) -- actions, observations, rewards,
    terminations, and events are overridden. The commands' Z-range is also
    overridden (below) to target the real resting height instead of
    Transport's hover point."""

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Push the transport goal down from Transport's hover point to the
        # actual destination resting height -- this stage has to finish the
        # job Transport stops short of.
        self.commands.object_pose.ranges.pos_z = (
            _DST_FLOOR_Z + OBJECT_HALF_HEIGHT,
            _DST_FLOOR_Z + OBJECT_HALF_HEIGHT,
        )
        # Same GPU-utilization scale-up GraspLift/ReachGrasp use, not
        # Transport's original num_envs=2048 default.
        self.scene.num_envs = 8192
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        # Longer than Transport's 6.0s -- this stage also has to descend to
        # the floor and hold PLACE_HOLD_STEPS steady, not just arrive at a
        # hover point.
        self.episode_length_s = 8.0


@configclass
class RB5PickPlacePlaceEnvCfg_PLAY(RB5PickPlacePlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
