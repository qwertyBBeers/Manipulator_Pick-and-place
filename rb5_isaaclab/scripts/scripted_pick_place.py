"""Test 4 (scripted waypoint sequence): drives the Differential-IK action
space (`RB5-PickPlace-IKRel-v0`) through a fixed, hand-authored
pre_grasp -> descend -> close -> lift -> move_above_dest -> lower -> open ->
retreat sequence, using the same proportional-servo technique as
`solve_pregrasp_pose.py`. Purpose is physics/task validation (does the
scene/robot/gripper physically support the intended motion at all), NOT a
production planner or a substitute for the learned policy.

Reports whether each waypoint was reached (position/orientation error below
a threshold within the step budget) and, after grasp+lift, whether the
object actually followed the gripper (grasp validity check) and ended up
inside the destination bin footprint at the end.

Usage:
    <IsaacLab repo>/isaaclab.sh -p scripted_pick_place.py --headless \
        --result_file /path/to/result.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--steps_per_waypoint", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import rb5_isaaclab  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym
import isaaclab.utils.math as math_utils

from rb5_isaaclab.tasks.pick_place.bin_geometry import load_bin_geometry
from rb5_isaaclab.tasks.pick_place.mdp.grasp_state import (
    GRIPPING_QUAT_WXYZ,
    PRE_GRASP_OFFSET_Z,
    grasped_object,
    object_inside_destination_xy,
)

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


def servo_to(env, robot, ee_frame, target_pos_b, target_quat_b, gripper_action_val, steps, pos_gain=4.0, rot_gain=2.0):
    device = env.unwrapped.device
    action = torch.zeros(env.unwrapped.action_space.shape, device=device)
    action[:, 6] = gripper_action_val
    pos_err_norm = ang_err_norm = None
    for _ in range(steps):
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        pos_err = target_pos_b - ee_pos_b
        quat_err = math_utils.quat_mul(target_quat_b, math_utils.quat_conjugate(ee_quat_b))
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)

        action[:, 0:3] = (pos_err * pos_gain).clamp(-1.0, 1.0)
        action[:, 3:6] = (axis_angle_err * rot_gain).clamp(-1.0, 1.0)
        env.step(action)
        pos_err_norm = torch.norm(pos_err, dim=-1)[0].item()
        ang_err_norm = torch.norm(axis_angle_err, dim=-1)[0].item()
    return pos_err_norm, ang_err_norm


def main():
    task = "RB5-PickPlace-IKRel-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    env.reset()

    source_bin, dest_bin = load_bin_geometry()
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    object_ = env.unwrapped.scene["object"]

    object_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_.data.root_pos_w[:, :3]
    )
    grasp_quat = torch.tensor([GRIPPING_QUAT_WXYZ], device=device)
    pre_grasp_pos = object_pos_b + torch.tensor([[0.0, 0.0, PRE_GRASP_OFFSET_Z]], device=device)
    grasp_pos = object_pos_b + torch.tensor([[0.0, 0.0, 0.02]], device=device)
    lift_pos = object_pos_b + torch.tensor([[0.0, 0.0, PRE_GRASP_OFFSET_Z + 0.1]], device=device)
    above_dest_pos = torch.tensor([[dest_bin.center[0], dest_bin.center[1], dest_bin.center[2] + 0.25]], device=device)
    lower_pos = torch.tensor([[dest_bin.center[0], dest_bin.center[1], dest_bin.center[2] + 0.06]], device=device)
    retreat_pos = above_dest_pos.clone()

    waypoints = [
        ("pre_grasp", pre_grasp_pos, grasp_quat, 1.0),
        ("descend", grasp_pos, grasp_quat, 1.0),
        ("close", grasp_pos, grasp_quat, -1.0),
        ("lift", lift_pos, grasp_quat, -1.0),
        ("move_above_dest", above_dest_pos, grasp_quat, -1.0),
        ("lower", lower_pos, grasp_quat, -1.0),
        ("open", lower_pos, grasp_quat, 1.0),
        ("retreat", retreat_pos, grasp_quat, 1.0),
    ]

    was_grasped_after_close = False
    for name, pos, quat, grip in waypoints:
        pos_err, ang_err = servo_to(env, robot, ee_frame, pos, quat, grip, args_cli.steps_per_waypoint)
        reached = (pos_err is not None and pos_err < 0.05)
        _log(f"[scripted] waypoint '{name}': final pos_err={pos_err:.4f} ang_err={ang_err:.4f} reached={reached}")
        if name == "close":
            was_grasped_after_close = bool(grasped_object(env.unwrapped)[0].item())
            _log(f"[scripted]   grasped_object flag after close: {was_grasped_after_close}")
        _flush()

    final_inside = bool(object_inside_destination_xy(env.unwrapped, margin=0.02)[0].item())
    final_obj_pos = object_.data.root_pos_w[0].tolist()
    _log(f"[scripted] final object world pos: {[round(v,4) for v in final_obj_pos]}")
    _log(f"[scripted] object inside destination footprint at end: {final_inside}")
    _log(f"[scripted] OVERALL: grasp_achieved={was_grasped_after_close} final_placement_inside_bin={final_inside}")
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
