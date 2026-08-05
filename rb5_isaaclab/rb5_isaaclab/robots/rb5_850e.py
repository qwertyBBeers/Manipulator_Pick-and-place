"""RB5-850E + Robotiq 2F-85 ArticulationCfg.

Gains are ported from the ROS/Isaac-Sim-4.1 pipeline's `binpicking_scene.py`
(same PhysX `drive:angular:physics:*` attributes, no unit conversion
needed), tuned there per `Manipulator/README2.md` §7.14-§7.28.

Gripper coupling: all 6 gripper joints are driven independently, each with
its own real PD actuator, matching `binpicking_scene.py`. An earlier
version used a real PhysX mimic-joint constraint for the 5 follower joints
instead (baked in from the URDF's `<mimic>` tags at USD-conversion time),
but that permanently deadlocked the primary knuckle joint outside its own
limit range -- 5 simultaneous bilateral constraints on one primary DOF is
an over-constrained system this PhysX/IsaacLab version can't resolve. See
`CURRICULUM_REPORT.md` section 1d for the full investigation.
`GRIPPER_MIMIC_JOINTS`'s commanded targets are scaled by
`GRIPPER_MIMIC_MULTIPLIERS` in each stage's `ActionsCfg.gripper_action`
instead of a physics constraint.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RB5_850E_USD_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "assets", "usd", "rb5_850e_robotiq.usd")
)

ARM_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
# Kept for future per-joint tuning (EMA alpha, split actuator damping) --
# proximal joints dominate EE *position*, wrist joints dominate
# *orientation*. Not currently used for a joint-group split: see
# ARM_DAMPING's comment below -- the 2026-08-04 anti-trembling ablation
# reverted to a single uniform actuator group per variable being tested one
# at a time, starting from the known-good baseline.
ARM_PROXIMAL_JOINT_NAMES = ["base", "shoulder", "elbow"]
ARM_WRIST_JOINT_NAMES = ["wrist1", "wrist2", "wrist3"]

# 2026-08-04/05 anti-trembling ablation, final conclusion: back to the
# original 1000.0 baseline. Damping alone (2000.0) was individually good
# (total reward 21.1) and PD gain randomization alone was individually the
# session's best (33.1, peak 50.7) -- but COMBINING them (2026-08-05
# 13:37 run, 150000 steps) was worse than either alone (total reward 6.5,
# orientation reward collapsed 0.35->0.001, grasp success ~never fired).
# Root cause: `randomize_actuator_gains` scales damping multiplicatively
# (0.8-1.2x) around whatever ARM_DAMPING is -- at base=1000 that's an
# 800-1200 range (run 5, worked); at base=2000 the SAME multiplier shifts
# the whole range to 1600-2400, and since the policy has no observation of
# which per-env gain it landed on, it has to find one control strategy
# robust across that entire range. Wrist-joint orientation control (already
# shown sensitive to damping/responsiveness by the EMA-smoothing ablation)
# apparently isn't robust across a 1600-2400 range the way it is across a
# single fixed 2000 (run 1) or an 800-1200 range (run 5) -- the policy gave
# up on it and retreated to a lower-variance "nail position, ignore
# orientation" strategy. PD gain randomization alone, at the ORIGINAL
# damping baseline, remains the validated best config.
ARM_DAMPING = 1000.0

GRIPPER_PRIMARY_JOINT = "robotiq_85_left_knuckle_joint"
# The 5 joints the URDF's <mimic> tags reference (assets/urdf/rb5_850e_robotiq_mimic.urdf).
GRIPPER_MIMIC_JOINTS = [
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
# Multiplier applied to the primary joint's own commanded target to get each
# follower's target -- same values as the URDF's <mimic multiplier="..."/>
# tags and rb5_isaac/rb5_isaac/trajectory_bridge.py's GRIPPER_MIMIC array.
GRIPPER_MIMIC_MULTIPLIERS = {
    "robotiq_85_right_knuckle_joint": -1.0,
    "robotiq_85_left_inner_knuckle_joint": 1.0,
    "robotiq_85_right_inner_knuckle_joint": -1.0,
    "robotiq_85_left_finger_tip_joint": -1.0,
    "robotiq_85_right_finger_tip_joint": 1.0,
}
GRIPPER_ALL_JOINT_NAMES = [GRIPPER_PRIMARY_JOINT, *GRIPPER_MIMIC_JOINTS]
# open/close command dicts for BinaryJointPositionActionCfg(joint_names=GRIPPER_ALL_JOINT_NAMES, ...).
GRIPPER_OPEN_COMMAND_EXPR = {j: 0.0 for j in GRIPPER_ALL_JOINT_NAMES}
GRIPPER_CLOSE_COMMAND_EXPR = {GRIPPER_PRIMARY_JOINT: 0.8, **{j: 0.8 * m for j, m in GRIPPER_MIMIC_MULTIPLIERS.items()}}

FINGERTIP_LINK_NAMES = ["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"]

RB5_850E_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RB5_850E_USD_PATH,
        # Required by any ContactSensorCfg targeting this articulation
        # (fingertip-vs-object sensors, see grasp_state.add_grasp_contact_sensors) --
        # env creation fails without it ("no bodies with contact reporter API").
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # Solved via scripts/solve_pregrasp_pose.py (Differential-IK convergence
        # to a pre-grasp target above the source bin, top-down grasp orientation) --
        # converged to tcp (0.517, -0.0003, 0.121) against a (0.51, 0.0, 0.10)
        # target, 2.2cm/~7deg residual.
        joint_pos={
            "base": 0.2193,
            "shoulder": 0.4545,
            "elbow": 2.1273,
            "wrist1": -1.0018,
            "wrist2": 1.5497,
            "wrist3": -0.2261,
            GRIPPER_PRIMARY_JOINT: 0.0,  # open
            **{j: 0.0 for j in GRIPPER_MIMIC_JOINTS},
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # stiffness/damping ratio kept from binpicking_scene.py's tuning;
        # effort_limit_sim bounded down from that pipeline's ~infinite 1e9
        # to a physically plausible industrial-arm torque.
        "rb5_arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness=10000.0,
            damping=ARM_DAMPING,
            effort_limit_sim=150.0,
        ),
        # All 6 gripper joints share one actuator group -- mechanically each
        # is the same lightweight linkage member, just actuated in the
        # opposite or same direction per GRIPPER_MIMIC_MULTIPLIERS.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_PRIMARY_JOINT, *GRIPPER_MIMIC_JOINTS],
            stiffness=7000.0,
            damping=350.0,
            effort_limit_sim=80.0,
        ),
    },
)
"""RB5-850E arm + Robotiq 2F-85 gripper, all 6 gripper joints independently
PD-driven (see module docstring), with friction/actuator gains ported from
the ROS/Isaac-Sim-4.1 pipeline's live-tuned values."""
