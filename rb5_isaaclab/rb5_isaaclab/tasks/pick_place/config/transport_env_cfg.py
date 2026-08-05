"""Stage 3 of the PPO curriculum: Transport.

Objective: move an ALREADY grasped and lifted cube to a position above the
destination bin. Starts every episode from a fixed "holding" state (see
`mdp/events.py::reset_robot_holding_object`) rather than from the source
bin -- this stage is purely about the transport motion + not dropping the
object, not about grasping.

Registered as `RB5-PickPlace-Transport-JointPos-v0`.
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
from .reach_env_cfg import ARM_ACTION_SCALE, OBJECT_HALF_HEIGHT, OBJECT_MASS, OBJECT_SIZE

SOURCE_BIN, DEST_BIN = load_bin_geometry()
_SRC_FLOOR_Z = SOURCE_BIN.center[2]
_DST_FLOOR_Z = DEST_BIN.center[2]
_DST_RANGE = xy_sample_range(DEST_BIN)

# How high above the destination-bin center the transport goal sits --
# spec's 0.08-0.15m range.
TRANSPORT_HOVER_OFFSET = 0.12
SAFE_TRANSPORT_HEIGHT = _SRC_FLOOR_Z + 0.06  # object must stay above this while "in transport"
MAX_SAFE_HEIGHT = _DST_FLOOR_Z + TRANSPORT_HOVER_OFFSET + 0.15


@configclass
class TransportSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # init_state here is irrelevant -- `reset_robot_holding_object`
    # overwrites the object's pose every episode (mode="reset").
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.15), rot=(1.0, 0.0, 0.0, 0.0)),
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
    """Transport goal: a hover point above the destination bin center, NOT
    the final resting pose -- Stage 3 doesn't place/release, only carries
    the object to above the bin (spec requirement)."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="tcp",
        resampling_time_range=(1e6, 1e6),  # effectively fixed for the whole (short) episode
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(DEST_BIN.center[0], DEST_BIN.center[0]),
            pos_y=(DEST_BIN.center[1], DEST_BIN.center[1]),
            pos_z=(_DST_FLOOR_Z + TRANSPORT_HOVER_OFFSET, _DST_FLOOR_Z + TRANSPORT_HOVER_OFFSET),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=ARM_ACTION_SCALE, use_default_offset=True
    )
    # close_threshold=0.5, inverted from the other stages' -0.5: this
    # stage's reset starts every episode with the gripper already closed
    # around the object, so the safe default for a near-zero action is to
    # keep holding, not release -- opening requires the action to clearly
    # say so (> 0.5).
    gripper_action: mdp.DeadbandBinaryJointPositionActionCfg = mdp.DeadbandBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_ALL_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSE_COMMAND_EXPR,
        close_threshold=0.5,
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
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_holding = EventTerm(func=mdp.reset_robot_holding_object, mode="reset")


@configclass
class RewardsCfg:
    maintain_grasp_reward = RewTerm(func=mdp.maintain_grasp_reward, weight=2.0)
    # Carried forward from Stage 2 at a lower weight (same pattern GraspLift
    # uses for Stage 1's reach shaping) so grasp-quality behavior isn't
    # forgotten while learning to transport. `bilateral_fingertip_contact_reward`/
    # `continuous_lift_reward`/`grasp_pose_position_reward` aren't copied --
    # redundant with or superseded by this stage's own terms.
    stable_grasp_reward = RewTerm(func=mdp.stable_grasp_reward, weight=1.0)
    empty_gripper_close_penalty = RewTerm(
        func=mdp.empty_gripper_close_penalty,
        params={"robot_cfg": SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])},
        weight=-0.2,
    )
    object_to_goal_position_reward = RewTerm(
        func=mdp.object_to_goal_position_reward,
        params={"std": 0.3, "command_name": "object_pose", "safe_transport_height": SAFE_TRANSPORT_HEIGHT},
        weight=4.0,
    )
    fine_goal_position_reward = RewTerm(
        func=mdp.object_to_goal_position_reward,
        params={"std": 0.05, "command_name": "object_pose", "safe_transport_height": SAFE_TRANSPORT_HEIGHT},
        weight=2.0,
    )
    object_height_safety_reward = RewTerm(
        func=mdp.object_height_safety_reward,
        params={"safe_transport_height": SAFE_TRANSPORT_HEIGHT, "max_safe_height": MAX_SAFE_HEIGHT},
        weight=1.0,
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
    transport_success = DoneTerm(
        func=mdp.transport_success,
        params={"command_name": "object_pose", "position_threshold": 0.05, "velocity_threshold": 0.1, "hold_steps": 8},
    )


@configclass
class RB5PickPlaceTransportEnvCfg(ManagerBasedRLEnvCfg):
    scene: TransportSceneCfg = TransportSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.robot = RB5_850E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        self.episode_length_s = 6.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class RB5PickPlaceTransportEnvCfg_PLAY(RB5PickPlaceTransportEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
