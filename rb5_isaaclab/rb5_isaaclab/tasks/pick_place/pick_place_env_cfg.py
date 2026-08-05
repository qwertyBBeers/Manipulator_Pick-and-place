"""Base pick-and-place task config (robot/action-term-agnostic -- filled in
by config/joint_pos_env_cfg.py or config/ik_rel_env_cfg.py).

Structure mirrors IsaacLab's own `manipulation.lift` task
(isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg) -- see that
file for the pattern this is deliberately kept close to. The differences
from that stock lift task:

  * Source/destination bin placement is read from the SAME
    `bin_geometry.yaml` the ROS/MoveIt pipeline uses (bin_geometry.py in
    this directory), not new hardcoded numbers.
  * The object's reset spawn range covers the *source* bin footprint, and
    the goal command's range covers the *destination* bin footprint --
    stock lift just samples one floating 3D target, this samples two bin
    footprints, closer to an actual bin-picking task.
  * An extra `place_and_release` reward + `object_placed` termination
    require the gripper to actually open once the object is at the goal
    (see mdp/rewards.py module docstring for why).
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
# 관절 로봇, 강체 물체, 조형물 asset import.
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
# Manager-based RL 환경 구성 import.
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

# rb5 model import
from rb5_isaaclab.robots.rb5_850e import GRIPPER_PRIMARY_JOINT

from . import mdp
from .bin_geometry import load_bin_geometry, xy_sample_range

# bin yaml에서 source/destination bin 정보 로드
SOURCE_BIN, DEST_BIN = load_bin_geometry()

# Single-object simplification, matching the ROS pipeline's own documented
# Phase-1 limitation (README2.md §7.6: /binpicking/object_pose only ever
# tracks one fixed prim) -- and matching binpicking_scene.py's cube size
# (0.042m / 0.10kg DynamicCuboid) so sim-to-real-pipeline behavior stays
# comparable.
OBJECT_SIZE = 0.042
OBJECT_HALF_HEIGHT = OBJECT_SIZE / 2.0
OBJECT_MASS = 0.10

_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_DST_FLOOR_Z = DEST_BIN.center[2]
_SRC_RANGE = xy_sample_range(SOURCE_BIN)
_DST_RANGE = xy_sample_range(DEST_BIN)


##
# Scene definition
##


@configclass
#Simulation의 물체들을 모두 정의. 
class RB5ObjectSceneCfg(InteractiveSceneCfg):
    """Robot + source/destination bin floors (visual only, real geometry
    comes from the USD's own collision meshes / the ground plane -- unlike
    the ROS/MoveIt pipeline, IsaacLab doesn't need a separately-published
    MoveIt CollisionObject) + one pickable cube."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    object: RigidObjectCfg = RigidObjectCfg(
        # 병렬 환경 마다의 object prim path이 달라야 하므로, {ENV_REGEX_NS}를 포함한 prim_path를 사용.
        prim_path="{ENV_REGEX_NS}/Object",
        # 초기 상태
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(SOURCE_BIN.center[0], SOURCE_BIN.center[1], _SRC_FLOOR_Z + OBJECT_HALF_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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

    dest_bin_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DestBinFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(DEST_BIN.center[0], DEST_BIN.center[1], _DST_FLOOR_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(DEST_BIN.inner_size[0], DEST_BIN.inner_size[1], DEST_BIN.wall_thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.55)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # A plain procedural plane instead of IsaacLab's GroundPlaneCfg default
    # (which references a Nucleus-hosted USD, unreachable in this
    # no-Nucleus-server environment) -- keeps this task fully self-contained
    # / offline, consistent with the object and bin floors above.
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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings (Markov Decision Process)
##

# 달성 목표 값 setting
@configclass
class CommandsCfg:
    """Placement goal: a point inside the destination bin's footprint, at
    resting height -- NOT a floating 3D target like the stock lift task,
    since this is "place into that bin", not "hold at that point in
    space"."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="tcp",
        resampling_time_range=(5.0, 5.0),
        # False, not the stock lift task's True: the debug-vis marker
        # (goal_pose_visualizer) spawns from a Nucleus-hosted USD
        # (frame_prim.usd), and this machine has no reachable Nucleus
        # server -- with debug_vis=True, env creation hangs on a ~300s
        # server-availability timeout before failing outright. Not needed
        # for headless training; re-enable for interactive GUI debugging on
        # a machine with Nucleus access.
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=_DST_RANGE["x"],
            pos_y=_DST_RANGE["y"],
            pos_z=(_DST_FLOOR_Z + OBJECT_HALF_HEIGHT, _DST_FLOOR_Z + OBJECT_HALF_HEIGHT),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Arm + gripper action terms -- concrete term picked by
    config/joint_pos_env_cfg.py (default) or config/ik_rel_env_cfg.py."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        # Explicit params (not the mdp function's own default arg value) so
        # the manager actually resolves joint_names -> joint_ids -- see
        # RewardsCfg.place_and_release's comment in this same file for why.
        # Without this the term silently returns all 12 joints' positions
        # instead of the primary knuckle's alone.
        gripper_opening = ObsTerm(
            func=mdp.gripper_opening, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": _SRC_RANGE["x"], "y": _SRC_RANGE["y"], "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": _SRC_FLOOR_Z + 0.04}, weight=15.0
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": _SRC_FLOOR_Z + 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": _SRC_FLOOR_Z + 0.04, "command_name": "object_pose"},
        weight=5.0,
    )
    place_and_release = RewTerm(
        func=mdp.place_and_release,
        params={
            "command_name": "object_pose",
            "position_threshold": 0.03,
            "velocity_threshold": 0.05,
            "gripper_open_threshold": 0.4,
            # Explicit, not relying on the mdp function's own default
            # SceneEntityCfg(...) value: only params passed here get
            # auto-resolved (joint_names -> joint_ids) by the manager base
            # (see manager_base.py's _prepare_terms, which iterates
            # term_cfg.params, not a function's Python-level defaults) --
            # without this, joint_ids silently stays slice(None) (all 12
            # joints) instead of just the primary knuckle.
            "robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
        weight=25.0,
    )
    # Starts at weight 0.0 (see CurriculumCfg -- ramped in after step 5000)
    # so early training, which naturally spends a lot of time hovering near
    # the goal while it's still learning to grasp/lift at all, isn't
    # punished for something it hasn't learned to fix yet.
    holding_at_goal_penalty = RewTerm(
        func=mdp.holding_at_goal_penalty,
        params={
            "command_name": "object_pose",
            "position_threshold": 0.03,
            "gripper_open_threshold": 0.4,
            "robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
        weight=0.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SRC_FLOOR_Z - 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    object_placed = DoneTerm(
        func=mdp.object_placed,
        params={
            "command_name": "object_pose",
            "position_threshold": 0.03,
            "velocity_threshold": 0.05,
            "gripper_open_threshold": 0.4,
            # See RewardsCfg.place_and_release's comment -- must be
            # explicit here to get auto-resolved.
            "robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
        },
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
    holding_at_goal_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "holding_at_goal_penalty", "weight": -2.0, "num_steps": 5000},
    )


##
# Environment configuration
##


@configclass
class RB5PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: RB5ObjectSceneCfg = RB5ObjectSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
