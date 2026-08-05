"""Stage: Grasp only -- isolates grasp discovery from lift.

Objective: reach the cube from `FAR_START_JOINT_POS` and achieve a stable
bilateral grasp. No lift required.

Split out from GraspLift (2026-07-30) after two rounds of reward-weight
tuning alone failed to make `bilateral_fingertip_contact_reward` ever fire
across a full 100k-step run (see PICK_PLACE_COMPLETION_REPORT.md).
TensorBoard showed the actual mechanism: `empty_gripper_close_penalty`
fires on almost every early random "close" attempt (it doesn't check
proximity to the object, only "closed AND no contact"), so the policy
learns "never close" -- a dense, easy, always-available negative signal --
long before it could ever discover "close near the object" -- a sparse
signal that requires reach to already be mastered first. By 75% through
training, `Policy / Standard deviation` had collapsed from ~1.0 to ~0.06:
exploration was gone before grasping was ever found.

Two changes from GraspLift's reward composition, both aimed at that same
root cause:
  - `empty_gripper_close_penalty` dropped entirely -- removes the
    easiest-to-learn negative signal that was crowding out exploration of
    closing at all.
  - No lift-related reward terms -- one fewer skill to discover before any
    positive reward for a correct close exists.
This stage's agent config (`agents/skrl_ppo_cfg_grasp.yaml`) also raises
`entropy_loss_scale` to slow the exploration collapse itself, independent
of the reward-shape fix.

Once this stage reliably achieves stable grasps (verified via
`evaluate_policy.py`, not reward curves alone), the plan is to measure the
resulting robot+object state from a real successful grasp (same
"solve, don't guess" approach `solve_holding_pose.py` used) and use that to
build a proper already-grasped reset for a follow-up Lift-only stage --
which would also give Transport's currently-broken reset a validated
alternative to its own IK-based measurement.

Scene/actions/observations/events are identical to GraspLift's -- reused
directly via subclassing rather than redefined; only rewards, terminations,
and episode length differ.

Registered as `RB5-PickPlace-Grasp-JointPos-v0`.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import GRIPPER_PRIMARY_JOINT

from .. import mdp
from ..bin_geometry import load_bin_geometry
from .grasp_lift_env_cfg import RB5PickPlaceGraspLiftEnvCfg

SOURCE_BIN, _DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]

GRASP_HOLD_STEPS = 8  # consecutive steps `grasped_object` must hold for early success


@configclass
class RewardsCfg:
    grasp_pose_position_reward = RewTerm(func=mdp.grasp_pose_position_reward, params={"std": 0.3}, weight=1.0)
    grasp_pose_orientation_reward = RewTerm(func=mdp.grasp_pose_orientation_reward, params={"std": 0.35}, weight=0.5)
    # Bridges reach -> grasp: dense reward for closing the gripper once
    # genuinely close to the object (proximity_std=0.05, much tighter than
    # grasp_pose_position_reward's 0.3, so it only activates near the actual
    # grasp point). Weight kept well below bilateral_fingertip_contact_reward/
    # stable_grasp_reward -- a nudge toward attempting a close, not a
    # substitute for actually achieving one. See rewards.py docstring for why
    # this was added: bilateral_fingertip_contact_reward is pure sparse/binary
    # and gave the policy no gradient toward ever trying to close near the
    # object.
    grasp_approach_closing_reward = RewTerm(
        func=mdp.grasp_approach_closing_reward,
        # `gripper_cfg` must be explicit here (not left to the function's
        # default arg) -- IsaacLab's manager only auto-resolves SceneEntityCfg
        # instances it finds in `term_cfg.params`, not ones only present as
        # Python default parameter values. Left implicit, `joint_ids` stays
        # unresolved and indexes ALL 12 robot joints instead of just the
        # knuckle -- confirmed via smoke test (shape mismatch: 4 vs 12).
        params={
            "proximity_std": 0.05,
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
        weight=0.5,
    )
    bilateral_fingertip_contact_reward = RewTerm(func=mdp.bilateral_fingertip_contact_reward, weight=2.0)
    # 4.0, not GraspLift's reweighted 1.5 -- there's no lift reward here to
    # rebalance against, so this can pay its original full rate for holding
    # a stable grasp.
    stable_grasp_reward = RewTerm(func=mdp.stable_grasp_reward, weight=4.0)
    # No `empty_gripper_close_penalty`, no lift terms -- see module docstring.
    object_drop_penalty = RewTerm(func=mdp.object_drop_penalty, weight=-5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SRC_FLOOR_Z - 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    grasp_success = DoneTerm(func=mdp.grasp_success, params={"hold_steps": GRASP_HOLD_STEPS})


@configclass
class RB5PickPlaceGraspEnvCfg(RB5PickPlaceGraspLiftEnvCfg):
    """Inherits GraspLift's scene/actions/observations/events wholesale
    (via subclassing + `super().__post_init__()`) -- only rewards,
    terminations, and episode length are overridden."""

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # No `__post_init__` override anymore -- inherits GraspLift's
    # `episode_length_s = 8.0` unchanged. A first attempt shortened this to
    # 6.0 ("no lift phase to budget for"), but that assumption was wrong:
    # `grasp_pose_position_reward` only converged to ~0.44 at 6.0s (vs
    # GraspLift's ~0.85 at 8.0s) -- reach itself needs close to the full
    # 8 seconds regardless of whether a lift follows, and cutting the
    # episode short left too few near-object steps for any grasp attempt to
    # ever coincide with being close enough. `bilateral_fingertip_contact_reward`
    # was exactly 0.0 across all 100 logged points of a full 100k-step run
    # as a direct result. See PICK_PLACE_COMPLETION_REPORT.md for the fuller
    # history.


@configclass
class RB5PickPlaceGraspEnvCfg_PLAY(RB5PickPlaceGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
