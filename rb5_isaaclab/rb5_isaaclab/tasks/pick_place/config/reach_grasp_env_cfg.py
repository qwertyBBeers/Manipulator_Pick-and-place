"""Stage: Reach with a scripted grasp probe -- makes real grasp success the
judge of reach precision, instead of a separate learned grasp-timing skill.

Every earlier attempt at Grasp/GraspLift had the RL policy control the
gripper directly (`DeadbandBinaryJointPositionActionCfg`), which meant it had
to *discover* the right moment to close -- a sparse/binary reward
(`bilateral_fingertip_contact_reward`) that repeatedly failed to ever fire
before exploration (`Policy / Standard deviation`) collapsed, across three
separate reward-tuning rounds (see `grasp_lift_env_cfg.py`'s RewardsCfg
history and `grasp_env_cfg.py`'s docstring).

This stage removes that discovery problem structurally rather than trying to
shape around it further: the gripper is driven by a scripted, non-learned
action term (`mdp.AutoGraspActionTermCfg`, 0 policy dimensions) that closes
automatically once the end-effector is within `grasp_state.is_near_pregrasp`'s
tolerance (and stays closed for as long as a real grasp holds, even if the EE
drifts slightly back out of that tolerance afterward). The RL policy only
controls the arm (6-dim action space, down from 7) and only has one thing
left to learn: reach precisely enough that the scripted close actually
grasps. `bilateral_fingertip_contact_reward`/`stable_grasp_reward` -- both
kept from GraspLift/Grasp -- now measure a REAL physical outcome of that
scripted attempt rather than something the policy has to separately decide to
attempt, which is exactly the "judge reach's success by whether grasping
actually succeeded" signal requested for this stage: `is_near_pregrasp`'s
geometric thresholds are a proxy (tuned by guesswork -- see
`grasp_state.PRE_GRASP_OFFSET_Z`'s own docstring), but bilateral contact is
ground truth.

No `empty_gripper_close_penalty`, no `grasp_approach_closing_reward` (the
dense reach->grasp bridge added for `grasp_env_cfg.py`) -- both existed only
to influence a *learned* close decision, which doesn't exist in this stage.

Once this stage reliably achieves real grasps, the plan is to measure the
resulting robot+object state (a genuine physical grasp, not IK-guessed) and
use it to seed a Lift+Place-only RL stage's reset -- same idea as
`grasp_env_cfg.py`'s docstring described, just reached via a route that
doesn't depend on Grasp-stage RL ever having worked.

--- 2026-08-04 anti-trembling ablation -----------------------------------
The 2026-08-03_20-33-44 run (fixed learning rate, no scheduler) was the
first to achieve real grasps (bilateral contact climbing to ~0.78 by the end
of training, still rising). GUI playback of that checkpoint showed visible
trembling and occasional drops near the floor, which motivated five changes
all at once (arm damping, EMA action smoothing, a jerk penalty, a floor
contact penalty, PD gain randomization). That combination broke orientation
tracking badly (~51deg error) and a later attempt to fix it (per-joint
damping/alpha) diverged outright, then a `kl_threshold` safety fix stalled
learning entirely -- three failed retrains without ever isolating which
single change (or interaction) was actually responsible.

The flags below restart from that known-good baseline and re-add each
change ONE AT A TIME, training and checking TensorBoard after each, instead
of stacking all five again. All `False` = the exact baseline config.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import (
    ARM_JOINT_NAMES,
    GRIPPER_ALL_JOINT_NAMES,
    GRIPPER_CLOSE_COMMAND_EXPR,
    GRIPPER_OPEN_COMMAND_EXPR,
)

from .. import mdp
from ..bin_geometry import load_bin_geometry
from ..mdp.grasp_state import add_floor_contact_sensors
from .grasp_lift_env_cfg import ARM_ACTION_SCALE, RB5PickPlaceGraspLiftEnvCfg

SOURCE_BIN, _DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]

GRASP_HOLD_STEPS = 8  # consecutive steps `grasped_object` must hold for early success

# --- Ablation flags (2026-08-04) -- see module docstring. Exactly one
# should be True per experiment; all False reproduces the known-good
# 2026-08-03_20-33-44 baseline. (Arm damping is toggled separately, via
# `robots.rb5_850e.ARM_DAMPING`, since it isn't part of this file.)
USE_EMA_SMOOTHING = False  # CONFIRMED CULPRIT: alone, reproduced the orientation
# collapse (0.0098 final, bilateral_contact/stable_grasp both exactly 0 for
# all 100000 steps) -- matches the wrist-lag hypothesis. Off for run 3+.
USE_JERK_PENALTY = False  # CONFIRMED PROBLEMATIC: alone, position reward
# dropped to 0.18 (below the random-action baseline of 0.27) and total
# reward to 1.4 (worst of the ablation series) -- seems to over-suppress
# movement generally, not just orientation. Off for run 4+.
USE_FLOOR_CONTACT_PENALTY = False  # CONFIRMED PROBLEMATIC: alone, training
# crashed hard around step 29000 (position/orientation/total reward all
# collapsed near 0, coinciding with a floor_contact spike at step 36000
# suggesting a hard floor collision) then partially recovered but never
# achieved real grasping (total reward plateaued ~3.4-4.8, well below run
# 1's 21.1). Off for run 5+.
USE_PD_GAIN_RANDOMIZATION = True  # CONFIRMED BEST, FINAL: alone, total reward
# 33.1 (peak 50.7) -- session best by a wide margin. Tried combining with
# ARM_DAMPING=2000 (also individually good, 21.1 alone) for a 2026-08-05
# 150000-step retrain -- that combination was WORSE than either alone
# (total reward 6.5, orientation collapsed to 0.001, grasp success ~never
# fired). See robots/rb5_850e.py's ARM_DAMPING comment for the root-cause
# analysis (randomized gain range shifts up with the base damping, and the
# policy can't observe which per-env gain it landed on). Final config:
# ARM_DAMPING back at its original 1000.0, PD randomization alone.

if USE_EMA_SMOOTHING:
    _ARM_ACTION_CFG = mdp.EMASmoothedJointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True, alpha=0.3
    )
else:
    _ARM_ACTION_CFG = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ActionsCfg:
    """6-dim: arm only. Gripper is fully scripted (`AutoGraspActionTermCfg`,
    0 policy dims) -- see module docstring."""

    arm_action = _ARM_ACTION_CFG
    auto_gripper_action: mdp.AutoGraspActionTermCfg = mdp.AutoGraspActionTermCfg(
        asset_name="robot",
        joint_names=GRIPPER_ALL_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSE_COMMAND_EXPR,
    )


@configclass
class RewardsCfg:
    grasp_pose_position_reward = RewTerm(func=mdp.grasp_pose_position_reward, params={"std": 0.3}, weight=1.0)
    grasp_pose_orientation_reward = RewTerm(func=mdp.grasp_pose_orientation_reward, params={"std": 0.35}, weight=0.5)
    # Ground-truth outcome of the scripted grasp attempt -- see module
    # docstring. Same weights as `grasp_env_cfg.py`'s Grasp stage.
    bilateral_fingertip_contact_reward = RewTerm(func=mdp.bilateral_fingertip_contact_reward, weight=2.0)
    stable_grasp_reward = RewTerm(func=mdp.stable_grasp_reward, weight=4.0)
    object_drop_penalty = RewTerm(func=mdp.object_drop_penalty, weight=-5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})
    if USE_JERK_PENALTY:
        # -0.00005 (not the initial -0.0002 that reward_diagnostic.py flagged
        # as too large relative to task reward under random actions).
        action_jerk = RewTerm(func=mdp.action_jerk_penalty, weight=-0.00005)
    if USE_FLOOR_CONTACT_PENALTY:
        floor_contact = RewTerm(func=mdp.floor_contact_penalty, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SRC_FLOOR_Z - 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    grasp_success = DoneTerm(func=mdp.grasp_success, params={"hold_steps": GRASP_HOLD_STEPS})


@configclass
class EventCfg:
    """Same as GraspLift's `EventCfg` (reset_all / reset_rb5_pp_state /
    reset_object_position), plus PD gain randomization when
    `USE_PD_GAIN_RANDOMIZATION` is on. `mode="startup"`, not `"reset"` --
    IsaacLab's own `randomize_actuator_gains` docstring warns it uses CPU
    tensor writes for implicit actuators and recommends startup-only use;
    at num_envs=8192 doing this every reset would add a CPU sync point on
    an already compute-bound task."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_rb5_pp_state = EventTerm(func=mdp.reset_rb5_pp_state, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    if USE_PD_GAIN_RANDOMIZATION:
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
class RB5PickPlaceReachGraspEnvCfg(RB5PickPlaceGraspLiftEnvCfg):
    """Inherits GraspLift's scene/observations/episode-length wholesale (via
    subclassing + `super().__post_init__()`) -- actions, rewards,
    terminations, and events are overridden."""

    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        if USE_FLOOR_CONTACT_PENALTY:
            # GraspLift's __post_init__ already adds fingertip-vs-object
            # sensors (add_grasp_contact_sensors); this adds the
            # fingertip-vs-floor pair mdp.floor_contact_penalty needs.
            add_floor_contact_sensors(self.scene)


@configclass
class RB5PickPlaceReachGraspEnvCfg_PLAY(RB5PickPlaceReachGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
