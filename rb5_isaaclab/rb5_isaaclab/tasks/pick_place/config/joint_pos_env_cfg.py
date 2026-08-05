"""Joint-position-control variant -- the default/recommended config for
initial training runs (simpler than IK: no DifferentialIKController
dependency, action space is a direct delta on the 6 arm joints)."""

from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import (
    ARM_JOINT_NAMES,
    GRIPPER_ALL_JOINT_NAMES,
    GRIPPER_CLOSE_COMMAND_EXPR,
    GRIPPER_OPEN_COMMAND_EXPR,
    RB5_850E_ROBOTIQ_CFG,
)

from .. import mdp
from ..pick_place_env_cfg import RB5PickPlaceEnvCfg


@configclass
class RB5PickPlaceJointPosEnvCfg(RB5PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RB5_850E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5, use_default_offset=True
        )
        # All 6 gripper joints are independently PD-driven (see
        # robots/rb5_850e.py module docstring -- no PhysX mimic constraint),
        # so the policy's single open/close action must command all 6 at
        # once here, same as the ROS pipeline's trajectory_bridge.py does.
        self.actions.gripper_action = mdp.DeadbandBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=GRIPPER_ALL_JOINT_NAMES,
            open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
            close_command_expr=GRIPPER_CLOSE_COMMAND_EXPR,
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tcp",
                    name="end_effector",
                ),
            ],
        )


@configclass
class RB5PickPlaceJointPosEnvCfg_PLAY(RB5PickPlaceJointPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class RB5PickPlaceJointPosEnvCfg_SAC(RB5PickPlaceJointPosEnvCfg):
    """Small-scale variant for the SAC comparison run -- off-policy replay
    buffers don't pair well with thousands of parallel envs the way PPO's
    on-policy rollouts do (no IsaacLab example task runs SAC at
    num_envs=2048+; see rb5_isaaclab/README.md)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
