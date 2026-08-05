"""Stage 1 of the PPO curriculum: Reach.

Objective: move the end-effector to a valid pre-grasp pose above a
(initially fixed-position) cube, gripper held open. No grasping, lifting,
or placing yet -- this stage exists purely to get a policy that reliably
reaches the right pose before layering on the (much harder to explore into)
grasp/lift/place stages.

Registered as `RB5-PickPlace-Reach-JointPos-v0`.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import ARM_JOINT_NAMES, RB5_850E_ROBOTIQ_CFG

from .. import mdp
from ..bin_geometry import load_bin_geometry, xy_sample_range
from ..mdp.grasp_state import PRE_GRASP_OFFSET_Z, STABLE_GRASP_STEPS

SOURCE_BIN, _DEST_BIN = load_bin_geometry()
OBJECT_SIZE = 0.042
OBJECT_HALF_HEIGHT = OBJECT_SIZE / 2.0
OBJECT_MASS = 0.10
_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_SRC_RANGE = xy_sample_range(SOURCE_BIN)

# Fixed cube spawn position (Stage 1: "single fixed cube position", no XY/yaw
# randomization) -- the source bin center, matching what `pre_grasp_target`
# is computed relative to elsewhere in this package.
FIXED_OBJECT_POS = (SOURCE_BIN.center[0], SOURCE_BIN.center[1], _SRC_FLOOR_Z + OBJECT_HALF_HEIGHT)

# Reduced from the legacy task's 0.5 -- large per-step position deltas make
# early random-exploration PPO overshoot wildly and rarely land near a
# useful pose; the spec calls for 0.15 as the initial safer value.
ARM_ACTION_SCALE = 0.15

# A genuine "away from the bin" arm pose (tcp world ~(-0.033, -0.111, 0.828),
# not in collision with anything) -- NOT the shared pre-grasp default in
# `RB5_850E_ROBOTIQ_CFG`. Named here (not just inlined in `__post_init__`
# below) so GraspLift/Curriculum can reuse the exact same validated start
# pose instead of starting directly above the object: with the shared
# pre-grasp default, those later stages' own `grasp_pose_position_reward`
# term had nothing left to learn -- the episode already started at the
# target.
FAR_START_JOINT_POS = {"base": 0.0, "shoulder": -1.0, "elbow": 1.6, "wrist1": -0.6, "wrist2": 1.57, "wrist3": 0.0}


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Robot + a single fixed-position cube + source bin floor (visual
    reference only) + ground/light. No destination bin -- Stage 1 never
    transports anywhere."""

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
    """6-dim: arm only, no gripper action term. `BinaryJointAction`'s
    convention treats a raw zero action as "close", not "hold" -- excluding
    the term entirely is the safe way to guarantee the gripper stays open
    for this stage."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_orientation = ObsTerm(func=mdp.ee_orientation_in_robot_root_frame)
        ee_to_pregrasp_vector = ObsTerm(func=mdp.ee_to_pregrasp_vector)
        grasp_orientation_error = ObsTerm(func=mdp.grasp_orientation_error)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Disabled for curriculum stages 1-4 until the full task
            # succeeds reliably (spec requirement) -- re-enable once
            # Stage 4 is solving reliably without noise.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_rb5_pp_state = EventTerm(func=mdp.reset_rb5_pp_state, mode="reset")
    # No object-position randomization event at all in the base Reach cfg
    # (spec: "single fixed cube position ... disable object XY and yaw
    # randomization initially") -- `ReachEnvCfg_Randomized` below adds it
    # back in as an opt-in variant once fixed-position reaching works.


@configclass
class RewardsCfg:
    # std=0.3 (not the original Lift task's 0.10): this stage starts ~0.9m
    # from the target, and 1-tanh(distance/std) saturates to ~0 almost
    # everywhere at std=0.10, giving no gradient over most of the approach.
    grasp_pose_position_reward = RewTerm(func=mdp.grasp_pose_position_reward, params={"std": 0.3}, weight=1.0)
    grasp_pose_orientation_reward = RewTerm(
        func=mdp.grasp_pose_orientation_reward, params={"std": 0.35}, weight=0.5
    )
    # Reduced 10x/5x from the initially-suggested -0.0005/-0.0001: under
    # random actions those dominated the (saturated) task reward by
    # 20-30x -- the "smoothness penalty drowns out task signal" failure
    # mode -- measured via `reward_diagnostic.py`.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.00002, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reach_success = DoneTerm(
        func=mdp.reach_success,
        params={"position_threshold": 0.03, "orientation_threshold": 0.20, "hold_steps": 8},
    )


@configclass
class RB5PickPlaceReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.robot = RB5_850E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # `RB5_850E_ROBOTIQ_CFG`'s shared default init pose is the pre-grasp
        # pose -- correct for Stages 2-4, but wrong here: if the robot
        # starts at the reach target there's nothing left to learn (`reach_success`
        # fired almost immediately under zero action). Override just this
        # stage's start pose to a genuine "away from the bin" one (tcp world
        # ~(-0.033, -0.111, 0.828), not in collision with anything).
        self.scene.robot.init_state.joint_pos.update(FAR_START_JOINT_POS)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/tcp", name="end_effector")],
        )

        self.decimation = 2
        self.episode_length_s = 4.0  # shorter horizon than the full task -- reach alone shouldn't need 8s
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class RB5PickPlaceReachEnvCfg_PLAY(RB5PickPlaceReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
