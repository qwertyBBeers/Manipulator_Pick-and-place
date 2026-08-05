"""Solves (measures, not guesses) Stage 3 (Transport)'s "already grasped and
holding" reset state for `mdp/events.py::reset_robot_holding_object`.

Performs a REAL simulated grasp using `RB5-PickPlace-GraspLift-JointPos-v0`'s
own object/bin/contact-sensor scene: drives the tcp down to the object with
a real Jacobian-based differential-IK controller (same class IsaacLab's own
`scripts/tutorials/05_controllers/run_diff_ik.py` uses standalone, driven
here through the *existing* joint-position action term by inverting its
`target = default_joint_pos + scale * raw_action` formula -- no new IK
action-term registration needed), closes the gripper, and verifies
`grasp_state.bilateral_contact` actually fires. Reads off the resulting,
physically-consistent state -- arm joint config, the gripper's real
contact-stopped knuckle angle, and the object's actual resting pose --
rather than deriving the object's placement from an assumed offset (see
`CURRICULUM_REPORT.md` section 1d for why the previous tcp-offset approach
was wrong).

Usage:
    <IsaacLab repo>/isaaclab.sh -p solve_holding_pose.py --headless \
        --result_file /path/to/result.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--descend_steps", type=int, default=150)
parser.add_argument("--close_steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import rb5_isaaclab  # noqa: F401
import isaaclab_tasks  # noqa: F401
import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

from rb5_isaaclab.robots.rb5_850e import ARM_JOINT_NAMES, GRIPPER_PRIMARY_JOINT
from rb5_isaaclab.tasks.pick_place.config.grasp_lift_env_cfg import RB5PickPlaceGraspLiftEnvCfg_Easy
from rb5_isaaclab.tasks.pick_place.mdp import grasp_state

_LOG_LINES = []


def _log(msg):
    print(msg)
    _LOG_LINES.append(str(msg))


def _flush():
    if not args_cli.result_file:
        return
    with open(args_cli.result_file, "w") as f:
        f.write("\n".join(_LOG_LINES))
        f.flush()
        os.fsync(f.fileno())


def main():
    cfg = RB5PickPlaceGraspLiftEnvCfg_Easy()
    cfg.scene.num_envs = 1
    env = gym.make("RB5-PickPlace-GraspLift-JointPos-v0", cfg=cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    object_ = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    device = env.unwrapped.device
    joint_names = robot.data.joint_names

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["tcp"])
    robot_entity_cfg.resolve(env.unwrapped.scene)
    arm_ids = robot_entity_cfg.joint_ids
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=device)

    arm_scale = env.unwrapped.action_manager.get_term("arm_action").cfg.scale
    default_arm_pos = robot.data.default_joint_pos[:, arm_ids].clone()

    action = torch.zeros(env.unwrapped.action_space.shape, device=device)

    def current_ee_pos_quat_b():
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def servo_step(gripper_cmd):
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids]
        ee_pos_b, ee_quat_b = current_ee_pos_quat_b()
        joint_pos = robot.data.joint_pos[:, arm_ids]
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        raw_action = (joint_pos_des - default_arm_pos) / arm_scale
        action[:, :6] = raw_action.clamp(-1.0, 1.0)
        action[:, 6] = gripper_cmd
        env.step(action)

    def hold_step(gripper_cmd):
        """Like `servo_step` but doesn't re-run the IK solve -- reuses the
        last `action[:, :6]`. Re-solving IK every step during close/settle
        injected small persistent jitter (object speed held ~0.5 m/s
        instead of damping to 0); freezing the arm command once contact is
        found removes that noise source."""
        action[:, 6] = gripper_cmd
        env.step(action)

    # Target XY/orientation: the object's own position + grasp orientation.
    # Target Z is probed incrementally rather than jumped to in one step --
    # the tcp frame sits well above the fingertip-pad plane, so driving
    # straight to the object's center height drove the gripper into the bin
    # floor and stalled short. Instead: start at the pre-grasp height and
    # lower a small step at a time, confirming convergence before going
    # lower; stop the instant it stalls or a fingertip contact force
    # appears. That stopping point is used as-is.
    target_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_.data.root_pos_w[:, :3]
    )
    target_quat_b = grasp_state.desired_grasp_quat_b(env.unwrapped)

    from rb5_isaaclab.tasks.pick_place.mdp.grasp_state import PRE_GRASP_OFFSET_Z

    z_offset = PRE_GRASP_OFFSET_Z
    z_step = 0.005
    settle_steps = 15
    min_z_offset = 0.0  # never target below the object's own center height

    while z_offset >= min_z_offset:
        probe_target = target_pos_b.clone()
        probe_target[:, 2] += z_offset
        ik_controller.set_command(torch.cat([probe_target, target_quat_b], dim=-1))
        for _ in range(settle_steps):
            servo_step(gripper_cmd=1.0)  # keep open while descending
        ee_pos_b, _ = current_ee_pos_quat_b()
        pos_err = torch.norm(probe_target - ee_pos_b, dim=-1).item()
        left_f, right_f = grasp_state.fingertip_contact_forces(env.unwrapped)
        touching = (left_f[0].item() > 0.1) or (right_f[0].item() > 0.1)
        _log(
            f"[descend probe] z_offset={z_offset:.3f} pos_err={pos_err:.4f} "
            f"left_force={left_f[0].item():.3f} right_force={right_f[0].item():.3f}"
        )
        _flush()
        if touching:
            # Debounce: a single-instant force reading can be a transient
            # brush during the IK transient, not real settled contact --
            # hold position a bit longer and re-check before trusting it.
            for _ in range(10):
                hold_step(gripper_cmd=1.0)
            left_f2, right_f2 = grasp_state.fingertip_contact_forces(env.unwrapped)
            touching2 = (left_f2[0].item() > 0.1) or (right_f2[0].item() > 0.1)
            _log(f"[descend probe] re-check after settle: left_force={left_f2[0].item():.3f} right_force={right_f2[0].item():.3f} touching={touching2}")
            if touching2:
                _log("[descend probe] stopping: contact confirmed after debounce")
                break
            _log("[descend probe] transient contact cleared, continuing descent")
        elif pos_err > 0.02:
            _log("[descend probe] stopping: arm stalled short of target")
            break
        z_offset -= z_step

    for step in range(args_cli.descend_steps):
        servo_step(gripper_cmd=1.0)  # final settle at the stopping height
        if step % 20 == 0 or step == args_cli.descend_steps - 1:
            ee_pos_b, _ = current_ee_pos_quat_b()
            pos_err = torch.norm(ik_controller.ee_pos_des - ee_pos_b, dim=-1).item()
            _log(f"[settle] step={step} pos_err={pos_err:.4f}")
            _flush()

    servo_step(gripper_cmd=1.0)  # one last IK solve to lock in action[:, :6], then freeze the arm
    bilateral_ok = False
    for step in range(args_cli.close_steps):
        hold_step(gripper_cmd=-1.0)  # close, arm command frozen (see hold_step docstring)
        if step % 10 == 0 or step == args_cli.close_steps - 1:
            bc = grasp_state.bilateral_contact(env.unwrapped)[0].item()
            left_f, right_f = grasp_state.fingertip_contact_forces(env.unwrapped)
            obj_speed = torch.norm(object_.data.root_lin_vel_w[0]).item()
            _log(
                f"[close] step={step} bilateral_contact={bc} "
                f"left_force={left_f[0].item():.3f} right_force={right_f[0].item():.3f} "
                f"knuckle={robot.data.joint_pos[0, joint_names.index(GRIPPER_PRIMARY_JOINT)].item():.4f} "
                f"obj_speed={obj_speed:.4f}"
            )
            bilateral_ok = bilateral_ok or bool(bc)
            _flush()

    _log(f"raw object velocity vector before lift: {object_.data.root_lin_vel_w[0].tolist()}")

    # Lift phase: `HOLDING_ARM_JOINT_POS`/`HOLDING_OBJECT_POS_B` are meant to
    # represent Stage 3 (Transport)'s "already grasped AND LIFTED" starting
    # state (see events.py's own pre-existing docstring note that the
    # previous version approximated this with the pre-grasp height instead
    # of a real lift) -- actively raise the tcp back up by
    # `PRE_GRASP_OFFSET_Z` while keeping the gripper closed, so the final
    # measurement reflects the object actually held up in the air, not just
    # resting/sliding on the bin floor with fingers touching it.
    lift_target = target_pos_b.clone()
    lift_target[:, 2] += PRE_GRASP_OFFSET_Z
    ik_controller.set_command(torch.cat([lift_target, target_quat_b], dim=-1))
    for step in range(80):
        servo_step(gripper_cmd=-1.0)
        if step % 20 == 0 or step == 79:
            bc = grasp_state.bilateral_contact(env.unwrapped)[0].item()
            obj_z = object_.data.root_pos_w[0, 2].item()
            _log(f"[lift] step={step} bilateral_contact={bc} object_z={obj_z:.4f}")
            _flush()

    # Extra settle phase: freeze the arm command and wait for the object's
    # velocity to actually damp out before trusting any of the final
    # numbers -- `bilateral_contact` being True at a single instant doesn't
    # mean the grasp is *stable* (the object could still be slipping/
    # swinging). First attempt at this (closing without lifting) showed
    # object speed pinned around 0.5 m/s indefinitely -- worth re-checking
    # whether that was "never left the floor" (fixed by the lift phase
    # above) or a genuine unstable grasp.
    _log("--- settle phase (checking object velocity actually damps out) ---")
    for step in range(100):
        hold_step(gripper_cmd=-1.0)
        if step % 20 == 0 or step == 99:
            obj_speed = torch.norm(object_.data.root_lin_vel_w[0]).item()
            bc = grasp_state.bilateral_contact(env.unwrapped)[0].item()
            _log(f"[settle-final] step={step} obj_speed={obj_speed:.4f} bilateral_contact={bc}")
            _flush()
    _log(f"raw object velocity vector at end: {object_.data.root_lin_vel_w[0].tolist()}")

    _log(f"--- FINAL STATE (bilateral_contact ever True during close: {bilateral_ok}) ---")
    joint_pos = robot.data.joint_pos[0]
    for name, val in zip(joint_names, joint_pos.tolist()):
        _log(f"  joint {name}: {val:.4f}")

    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_.data.root_pos_w[:, :3], object_.data.root_quat_w
    )
    arm_vals = ", ".join(f'"{n}": {joint_pos[joint_names.index(n)].item():.4f}' for n in ARM_JOINT_NAMES)
    _log(f"HOLDING_ARM_JOINT_POS = {{{arm_vals}}}")
    _log(
        f"HOLDING_GRIPPER_KNUCKLE_POS = {joint_pos[joint_names.index(GRIPPER_PRIMARY_JOINT)].item():.4f}"
        "  # ACTUAL settled angle, not commanded target"
    )
    _log(f"HOLDING_OBJECT_POS_B = {tuple(round(v, 4) for v in object_pos_b[0].tolist())}")
    _log(f"HOLDING_OBJECT_QUAT_B = {tuple(round(v, 4) for v in object_quat_b[0].tolist())}")
    obj_vel = torch.norm(object_.data.root_lin_vel_w[0]).item()
    _log(f"object linear speed at end: {obj_vel:.4f} m/s (should be small/settled)")
    _flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        _log(f"EXCEPTION: {e}")
        _log(traceback.format_exc())
        _flush()
        raise
    finally:
        simulation_app.close()
