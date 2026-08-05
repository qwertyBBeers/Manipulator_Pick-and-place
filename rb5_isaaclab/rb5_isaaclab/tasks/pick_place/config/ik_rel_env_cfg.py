"""Differential-IK, relative-pose-delta control variant (secondary/optional
-- see rb5_isaaclab/README.md for when to reach for this over the default
joint-position config). Mirrors IsaacLab's own
manipulation/lift/config/franka/ik_rel_env_cfg.py pattern: inherit the
joint-pos config (keeps gripper action / scene / ee_frame), swap only
`scene.robot` (stiffer PD for tighter IK tracking) and `actions.arm_action`
(task-space delta instead of joint-space delta)."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from rb5_isaaclab.robots.rb5_850e import ARM_JOINT_NAMES, RB5_850E_ROBOTIQ_CFG

from . import joint_pos_env_cfg

RB5_850E_ROBOTIQ_HIGH_PD_CFG = RB5_850E_ROBOTIQ_CFG.copy()
RB5_850E_ROBOTIQ_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
RB5_850E_ROBOTIQ_HIGH_PD_CFG.actuators["rb5_arm"].stiffness = 20000.0
RB5_850E_ROBOTIQ_HIGH_PD_CFG.actuators["rb5_arm"].damping = 2000.0


@configclass
class RB5PickPlaceIkRelEnvCfg(joint_pos_env_cfg.RB5PickPlaceJointPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RB5_850E_ROBOTIQ_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            body_name="tcp",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            # "tcp" is already a real, correctly-placed body (see
            # rb5_isaac's URDF tcp_joint) -- no body_offset needed, unlike
            # Franka's fictitious panda_hand-plus-offset end-effector.
        )
