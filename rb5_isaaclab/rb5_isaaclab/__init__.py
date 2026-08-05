"""RB5-850E + Robotiq 2F-85 IsaacLab pick-and-place tasks.

Importing this package registers its gym environment IDs (mirrors
IsaacLab's own `isaaclab_tasks.manager_based.manipulation.lift.config.franka`
registration pattern -- see that file for the reference this was copied
from).
"""

import gymnasium as gym

from .tasks.pick_place.config import agents

_JOINT_POS_ENTRY = "rb5_isaaclab.tasks.pick_place.config.joint_pos_env_cfg"
_IK_REL_ENTRY = "rb5_isaaclab.tasks.pick_place.config.ik_rel_env_cfg"
_REACH_ENTRY = "rb5_isaaclab.tasks.pick_place.config.reach_env_cfg"
_GRASP_ENTRY = "rb5_isaaclab.tasks.pick_place.config.grasp_env_cfg"
_REACH_GRASP_ENTRY = "rb5_isaaclab.tasks.pick_place.config.reach_grasp_env_cfg"
_GRASP_LIFT_ENTRY = "rb5_isaaclab.tasks.pick_place.config.grasp_lift_env_cfg"
_TRANSPORT_ENTRY = "rb5_isaaclab.tasks.pick_place.config.transport_env_cfg"
_PLACE_ENTRY = "rb5_isaaclab.tasks.pick_place.config.place_env_cfg"
_CURRICULUM_ENTRY = "rb5_isaaclab.tasks.pick_place.config.curriculum_env_cfg"

##
# Joint Position Control (default/recommended)
##

gym.register(
    id="RB5-PickPlace-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_JOINT_POS_ENTRY}:RB5PickPlaceJointPosEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_JOINT_POS_ENTRY}:RB5PickPlaceJointPosEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# Small-scale variant for the SAC comparison run (README.md: off-policy
# replay doesn't pair well with thousands of parallel envs) -- also carries
# the PPO entry point so the same low-env-count scene can be used for an
# apples-to-apples small-scale PPO-vs-SAC comparison if wanted.
gym.register(
    id="RB5-PickPlace-JointPos-SAC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_JOINT_POS_ENTRY}:RB5PickPlaceJointPosEnvCfg_SAC",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# PPO curriculum stages (see rb5_isaaclab/TRAINING_CONFIG.md / the
# curriculum implementation report for why: the original single-shot
# `RB5-PickPlace-JointPos-v0` task never trained a working policy --
# degenerate "open gripper, lay arm back" behavior even after a real
# 100k-timestep/2048-env run). Each stage is independently trainable/
# playable; `RB5-PickPlace-JointPos-v0` above is left registered unchanged
# so old checkpoints/commands still resolve.
##

gym.register(
    id="RB5-PickPlace-Reach-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_REACH_ENTRY}:RB5PickPlaceReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-Reach-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_REACH_ENTRY}:RB5PickPlaceReachEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-Grasp-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_GRASP_ENTRY}:RB5PickPlaceGraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_grasp.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-Grasp-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_GRASP_ENTRY}:RB5PickPlaceGraspEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_grasp.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-ReachGrasp-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_REACH_GRASP_ENTRY}:RB5PickPlaceReachGraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_reach_grasp.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-ReachGrasp-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_REACH_GRASP_ENTRY}:RB5PickPlaceReachGraspEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_reach_grasp.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-GraspLift-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_GRASP_LIFT_ENTRY}:RB5PickPlaceGraspLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # Same hyperparameters, only `experiment.experiment_name` differs
        # (gives this stage's TensorBoard/checkpoint runs a readable name) --
        # select with `train.py --agent skrl_ppo_cfg_entry_point` (the key
        # name itself must keep "ppo" in it so train.py's own
        # `algorithm = ...split("skrl_")[-1]` derivation still resolves to
        # "ppo", preserving the existing `<date>_ppo_torch_...` folder
        # convention instead of renaming it).
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_grasplift.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-GraspLift-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_GRASP_LIFT_ENTRY}:RB5PickPlaceGraspLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_grasplift.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-Transport-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_TRANSPORT_ENTRY}:RB5PickPlaceTransportEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-Transport-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_TRANSPORT_ENTRY}:RB5PickPlaceTransportEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-Place-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_PLACE_ENTRY}:RB5PickPlacePlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_place.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-Place-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_PLACE_ENTRY}:RB5PickPlacePlaceEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_place.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="RB5-PickPlace-Curriculum-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_CURRICULUM_ENTRY}:RB5PickPlaceCurriculumEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_curriculum.yaml",
    },
    disable_env_checker=True,
)
gym.register(
    id="RB5-PickPlace-Curriculum-JointPos-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_CURRICULUM_ENTRY}:RB5PickPlaceCurriculumEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_curriculum.yaml",
    },
    disable_env_checker=True,
)

##
# Differential IK, relative pose control (secondary/optional)
##

gym.register(
    id="RB5-PickPlace-IKRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{_IK_REL_ENTRY}:RB5PickPlaceIkRelEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
