"""Stage 2 of the PPO curriculum: Grasp and Lift.

Objective: approach the (near-fixed-position) cube, grasp it with both
fingertips, and lift it -- no destination-bin transport yet. Builds on
Stage 1's reach shaping (kept as a lower-weight guidance term) plus real
contact-sensor-based grasp detection.

Registered as `RB5-PickPlace-GraspLift-JointPos-v0` (reset difficulty:
`RB5PickPlaceGraspLiftEnvCfg` = "Normal" per spec -- small ±0.02m object
position randomization, arm still starts near the pre-grasp pose;
`RB5PickPlaceGraspLiftEnvCfg_Easy` = fixed object position, for initial
debugging runs before Normal is attempted).
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import (
    ARM_JOINT_NAMES,
    GRIPPER_ALL_JOINT_NAMES,
    GRIPPER_CLOSE_COMMAND_EXPR,
    GRIPPER_OPEN_COMMAND_EXPR,
    GRIPPER_PRIMARY_JOINT,
    RB5_850E_ROBOTIQ_CFG,
)

from .. import mdp
from ..bin_geometry import load_bin_geometry, xy_sample_range
from ..mdp.grasp_state import add_grasp_contact_sensors
from .reach_env_cfg import ARM_ACTION_SCALE, FAR_START_JOINT_POS, FIXED_OBJECT_POS, OBJECT_HALF_HEIGHT, OBJECT_MASS, OBJECT_SIZE

SOURCE_BIN, _DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_SRC_RANGE = xy_sample_range(SOURCE_BIN)

TARGET_LIFT_HEIGHT = 0.10  # m above source floor -- spec's 0.08-0.12m range
INITIAL_LIFT_HEIGHT = 0.02  # m -- see RewardsCfg.initial_lift_bonus below
NORMAL_RESET_XY_RANGE = 0.02  # m, +-, per spec's "Normal reset" definition


@configclass
class GraspLiftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=FIXED_OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.75, 0.1)),
        ),
    )

    source_bin_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SourceBinFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(SOURCE_BIN.center[0], SOURCE_BIN.center[1], _SRC_FLOOR_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(SOURCE_BIN.inner_size[0], SOURCE_BIN.inner_size[1], SOURCE_BIN.wall_thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(50.0, 50.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True
    )
    gripper_action: mdp.DeadbandBinaryJointPositionActionCfg = mdp.DeadbandBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_ALL_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSE_COMMAND_EXPR,
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
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel)
        left_right_contact_force = ObsTerm(func=mdp.fingertip_contact_forces_obs, clip=(0.0, 20.0))
        bilateral_contact = ObsTerm(func=mdp.bilateral_contact_flag)
        grasped_object = ObsTerm(func=mdp.grasped_object_flag)
        gripper_opening = ObsTerm(
            func=mdp.gripper_opening, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_rb5_pp_state = EventTerm(func=mdp.reset_rb5_pp_state, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    # std=0.3, not the original 0.10 -- the arm now starts at `FAR_START_JOINT_POS`
    # (~0.9m from the target, same as Reach), not already hovering above the
    # object; 0.10 saturates `1 - tanh(distance/std)` to ~0 almost everywhere
    # over that approach distance, same reasoning `reach_env_cfg.py` used.
    grasp_pose_position_reward = RewTerm(func=mdp.grasp_pose_position_reward, params={"std": 0.3}, weight=1.0)
    grasp_pose_orientation_reward = RewTerm(func=mdp.grasp_pose_orientation_reward, params={"std": 0.35}, weight=0.5)
    # Round 1 (halved bilateral_contact to 1.0) over-corrected: TensorBoard's
    # per-term Episode_Reward curves showed `bilateral_fingertip_contact_reward`
    # essentially never fired (nonzero at 27/100 logged points, max value
    # 0.00025 -- noise-level) for the ENTIRE 100k-step run, while
    # `empty_gripper_close_penalty` fired almost every episode. With contact's
    # payout halved but the close-attempt penalty unchanged, closing the
    # gripper became net-negative in expectation before the policy ever
    # learned good timing, so it converged to never closing at all --
    # `grasp_pose_position/orientation_reward` alone (no grasp needed)
    # already explains ~100% of the achieved total reward. Restored to 2.0;
    # `empty_gripper_close_penalty` (below) reduced instead, since that
    # penalty -- not `stable_grasp_reward` -- was what was actually
    # suppressing exploration of closing the gripper at all.
    bilateral_fingertip_contact_reward = RewTerm(func=mdp.bilateral_fingertip_contact_reward, weight=2.0)
    stable_grasp_reward = RewTerm(func=mdp.stable_grasp_reward, weight=1.5)
    # NEW: steep low-threshold companion to `continuous_lift_reward` below --
    # ramps 0 -> weight over just the first `INITIAL_LIFT_HEIGHT` (2cm), so
    # the very first millimeters of an attempted lift pay off fast, instead
    # of the barely-perceptible gradient a single 0-10cm-wide term gives
    # near height=0. Same function, different target height -- no new reward
    # code, matches the "reuse existing conditions" rule.
    initial_lift_bonus = RewTerm(
        func=mdp.continuous_lift_reward,
        params={"target_lift_height": INITIAL_LIFT_HEIGHT, "source_floor_height": _SRC_FLOOR_Z},
        weight=6.0,
    )
    # Doubled from 4.0 -- with the two changes above, a full lift now nets
    # ~18/step vs ~4/step for freezing at floor height (was ~11.5 vs ~7.5).
    continuous_lift_reward = RewTerm(
        func=mdp.continuous_lift_reward,
        params={"target_lift_height": TARGET_LIFT_HEIGHT, "source_floor_height": _SRC_FLOOR_Z},
        weight=8.0,
    )
    # `robot_cfg` must be explicit here (not left to the mdp function's own
    # default arg value) for the manager to auto-resolve joint_names ->
    # joint_ids -- omitting it causes a shape mismatch (12 vs num_envs).
    # -0.2 -> -0.05: with round 1's data showing this fired almost every
    # episode while the policy was still learning to time a close attempt,
    # the unchanged -0.2 penalty made "try closing, miss" a clear net loss
    # before contact was ever reliably discovered -- reduced so mistimed
    # attempts during early exploration cost less than the (now-restored)
    # 2.0 payout for a successful one.
    empty_gripper_close_penalty = RewTerm(
        func=mdp.empty_gripper_close_penalty,
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])},
        weight=-0.05,
    )
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
    grasp_lift_success = DoneTerm(
        func=mdp.grasp_lift_success,
        params={
            "target_lift_height": TARGET_LIFT_HEIGHT,
            "source_floor_height": _SRC_FLOOR_Z,
            "velocity_threshold": 0.1,
            "hold_steps": 8,
        },
    )


@configclass
class RB5PickPlaceGraspLiftEnvCfg(ManagerBasedRLEnvCfg):
    """"Normal" reset difficulty: object XY randomized +-2cm around the
    fixed source-bin-center position (spec: yaw randomization deferred
    until fixed-position grasping succeeds -- not enabled here)."""

    scene: GraspLiftSceneCfg = GraspLiftSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.robot = RB5_850E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Start away from the object (same validated pose Reach uses), not
        # already hovering at the pre-grasp target -- otherwise this stage's
        # own reach/grasp-pose shaping has nothing left to learn.
        self.scene.robot.init_state.joint_pos.update(FAR_START_JOINT_POS)
        add_grasp_contact_sensors(self.scene)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/tcp", name="end_effector")],
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-NORMAL_RESET_XY_RANGE, NORMAL_RESET_XY_RANGE),
            "y": (-NORMAL_RESET_XY_RANGE, NORMAL_RESET_XY_RANGE),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.decimation = 2
        # 8.0, not 6.0 -- the episode now has to cover the full approach
        # from `FAR_START_JOINT_POS` (~0.9m away) before grasp/lift even
        # begins, not just grasp/lift from an already-arrived pose.
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # 64*1024, not 16*1024 -- scaled 4x for the num_envs=8192 GPU-utilization
        # run (2026-07-30): this buffer was originally sized for the
        # num_envs=2048 default and isn't auto-derived from scene.num_envs
        # (the CLI --num_envs override happens after this __post_init__
        # runs), so it needs bumping by hand when running at a higher scale.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class RB5PickPlaceGraspLiftEnvCfg_Easy(RB5PickPlaceGraspLiftEnvCfg):
    """Fixed object position (no randomization) -- for initial debugging
    before attempting the "Normal" +-2cm variant above."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0),
        }


@configclass
class RB5PickPlaceGraspLiftEnvCfg_PLAY(RB5PickPlaceGraspLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
