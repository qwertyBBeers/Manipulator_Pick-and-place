"""Initial-pose diagnostic: spawns a single environment, resets it to the
configured default pose (`robots/rb5_850e.py`'s `init_state.joint_pos`),
takes no policy actions, and reports joint names/positions and the
end-effector (tcp) pose so the previously-unverified initial pose guess can
actually be checked against something (collision, reachability, joint
limits) instead of just assumed.

Also reports the same for a "gripper closed" variant of the same pose
(same arm joint_pos, gripper closed) -- used to get a real measured
`ee_frame` position for Stage 3 (Transport)'s "start already grasped" reset,
which otherwise has no way to know where the tcp will be for a given joint
config without actually asking the simulator (see
`mdp/events.py::reset_robot_holding_object`'s docstring).

Usage:
    <IsaacLab repo>/isaaclab.sh -p diagnose_initial_pose.py --headless \
        --result_file /path/to/result.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import rb5_isaaclab  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym


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


def _report(env, label):
    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0].tolist()
    _log(f"--- {label} ---")
    for name, pos in zip(joint_names, joint_pos):
        _log(f"  joint {name}: {pos:.4f} rad")
    ee_pos_w = ee_frame.data.target_pos_w[0, 0].tolist()
    ee_quat_w = ee_frame.data.target_quat_w[0, 0].tolist()
    _log(f"  ee(tcp) world pos: {[round(v, 4) for v in ee_pos_w]}")
    _log(f"  ee(tcp) world quat(wxyz): {[round(v, 4) for v in ee_quat_w]}")
    robot_root_pos = robot.data.root_pos_w[0].tolist()
    robot_root_quat = robot.data.root_quat_w[0].tolist()
    _log(f"  robot root world pos: {[round(v, 4) for v in robot_root_pos]}")
    _log(f"  robot root world quat(wxyz): {[round(v, 4) for v in robot_root_quat]}")
    joint_limits = robot.data.soft_joint_pos_limits[0].tolist()
    for name, pos, (lo, hi) in zip(joint_names, joint_pos, joint_limits):
        if name in ("robotiq_85_left_knuckle_joint",) or "wrist" in name or name in ("shoulder", "elbow", "base"):
            margin_lo, margin_hi = pos - lo, hi - pos
            if min(margin_lo, margin_hi) < 0.05:
                _log(f"  WARNING: joint {name} is within 0.05 rad of its limit ({lo:.3f}, {hi:.3f})")
    _flush()


def main():
    task = "RB5-PickPlace-JointPos-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)

    env.reset()
    for _ in range(5):
        env.step(torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device))
    _report(env, "default init pose (gripper open, after 5 zero-action steps to settle)")

    # Close the gripper via the actual action interface (not a raw joint
    # write, which the action manager's own position-target push would
    # immediately fight/overwrite on the next env.step() anyway) -- per
    # `BinaryJointAction`'s own convention (isaaclab/envs/mdp/actions/
    # binary_joint_actions.py): positive action = open, NEGATIVE = close.
    close_action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    close_action[:, -1] = -1.0
    for _ in range(10):
        env.step(close_action)
    _report(env, "same arm pose, gripper CLOSED (action=-1 on gripper dim), after 10 settle steps -- for Stage 3 reset reference")

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
