"""Stage 4 of the PPO curriculum: the full task (reach -> grasp -> lift ->
transport -> lower -> release -> stabilize), built entirely from the same
`mdp/grasp_state.py` / `mdp/rewards.py` functions the earlier stages use.

Phase-awareness note: rather than a single discrete phase-index variable
gating which reward terms are "active", every reward term below is
individually gated on its own physical precondition (e.g.
`continuous_lift_reward` needs `grasped_object`, `object_to_goal_position_reward`
needs grasped+above-safe-height, `released_and_stable_reward` needs the full
place-success condition). This achieves the same practical goal the spec's
phase-index idea is after -- no reward for being in the "wrong phase", and
nothing irreversible/exploitable after a drop, since every gate is
recomputed fresh from the CURRENT physical state each step, not a sticky
phase counter -- without adding a separate phase-tracking state variable.
An explicit phase-index observation (0=approach..5=released/stable) is a
reasonable follow-up enhancement, not implemented here (see the curriculum
report's "known simplifications").

Registered as `RB5-PickPlace-Curriculum-JointPos-v0`.
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
from .grasp_lift_env_cfg import INITIAL_LIFT_HEIGHT, TARGET_LIFT_HEIGHT
from .reach_env_cfg import ARM_ACTION_SCALE, FAR_START_JOINT_POS, FIXED_OBJECT_POS, OBJECT_HALF_HEIGHT, OBJECT_MASS, OBJECT_SIZE
from .transport_env_cfg import MAX_SAFE_HEIGHT, SAFE_TRANSPORT_HEIGHT, TRANSPORT_HOVER_OFFSET

SOURCE_BIN, DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_DST_FLOOR_Z = DEST_BIN.center[2]
_DST_RANGE = xy_sample_range(DEST_BIN)

DEST_INSIDE_MARGIN_TRAIN = 0.03  # spec: 0.02-0.04m training margin (looser than final eval)
PLACE_MAX_HEIGHT_ABOVE_FLOOR = 0.06
PLACE_VELOCITY_THRESHOLD = 0.05
PLACE_HOLD_STEPS = 15  # spec: 10-20 steps


@configclass
class CurriculumSceneCfg(InteractiveSceneCfg):
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
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="tcp",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(DEST_BIN.center[0], DEST_BIN.center[0]),
            pos_y=(DEST_BIN.center[1], DEST_BIN.center[1]),
            pos_z=(_DST_FLOOR_Z + OBJECT_HALF_HEIGHT, _DST_FLOOR_Z + OBJECT_HALF_HEIGHT),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


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
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel)
        bilateral_contact = ObsTerm(func=mdp.bilateral_contact_flag)
        grasped_object = ObsTerm(func=mdp.grasped_object_flag)
        gripper_opening = ObsTerm(
            func=mdp.gripper_opening, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Spec: no observation corruption / domain randomization in the
            # initial curriculum training configuration.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_rb5_pp_state = EventTerm(func=mdp.reset_rb5_pp_state, mode="reset")
    # Spec: "Initially use: Fixed cube position ... Small or no object
    # randomization" -- zero range for now. `CurriculumEnvCfg_Randomized`
    # (a later, opt-in subclass) would widen this once the fixed-pose task
    # is solved reliably.
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
    # std=0.3 -- see grasp_lift_env_cfg.py's identical comment: the arm now
    # starts at `FAR_START_JOINT_POS`, not already at the pre-grasp target.
    grasp_pose_position_reward = RewTerm(func=mdp.grasp_pose_position_reward, params={"std": 0.3}, weight=1.0)
    grasp_pose_orientation_reward = RewTerm(func=mdp.grasp_pose_orientation_reward, params={"std": 0.35}, weight=0.5)
    # Reweighted -- see grasp_lift_env_cfg.py's identical comment
    # (PICK_PLACE_COMPLETION_REPORT.md section E.1: "grasp and freeze" local
    # optimum, confirmed at 100/100 episodes never lifting despite high reward).
    bilateral_fingertip_contact_reward = RewTerm(func=mdp.bilateral_fingertip_contact_reward, weight=1.0)
    stable_grasp_reward = RewTerm(func=mdp.stable_grasp_reward, weight=1.5)
    initial_lift_bonus = RewTerm(
        func=mdp.continuous_lift_reward,
        params={"target_lift_height": INITIAL_LIFT_HEIGHT, "source_floor_height": _SRC_FLOOR_Z},
        weight=6.0,
    )
    continuous_lift_reward = RewTerm(
        func=mdp.continuous_lift_reward,
        params={"target_lift_height": TARGET_LIFT_HEIGHT, "source_floor_height": _SRC_FLOOR_Z},
        weight=8.0,
    )
    object_to_goal_position_reward = RewTerm(
        func=mdp.object_to_goal_position_reward,
        params={"std": 0.3, "command_name": "object_pose", "safe_transport_height": SAFE_TRANSPORT_HEIGHT},
        weight=4.0,
    )
    object_inside_destination_reward = RewTerm(
        func=mdp.object_inside_destination_reward, params={"margin": DEST_INSIDE_MARGIN_TRAIN}, weight=8.0
    )
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
        weight=15.0,
    )
    empty_gripper_close_penalty = RewTerm(
        func=mdp.empty_gripper_close_penalty,
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])},
        weight=-0.2,
    )
    object_drop_penalty = RewTerm(func=mdp.object_drop_penalty, weight=-5.0)
    # Spec: do NOT use the previous -0.1 action-rate/joint-velocity weights;
    # keep these fixed (no curriculum ramp) for the initial successful-
    # training experiments -- unlike the legacy task's CurriculumCfg.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _SRC_FLOOR_Z - 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
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
class RB5PickPlaceCurriculumEnvCfg(ManagerBasedRLEnvCfg):
    scene: CurriculumSceneCfg = CurriculumSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # No CurriculumCfg (no reward-weight ramping) -- spec: keep weights
    # fixed for the initial successful-training experiments.

    def __post_init__(self):
        self.scene.robot = RB5_850E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Start away from the object -- see grasp_lift_env_cfg.py's identical
        # comment. The full task now has to reach, not just grasp/lift/
        # transport/place from an already-arrived pose.
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

        self.decimation = 2
        # 10.0, not 8.0 -- the episode must now also cover the initial
        # ~0.9m approach (same as GraspLift's episode_length_s bump) on top
        # of grasp/lift/transport/place.
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class RB5PickPlaceCurriculumEnvCfg_PLAY(RB5PickPlaceCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
