"""Uses the already-built Differential-IK action space (`config/ik_rel_env_cfg.py`)
to *solve* (not guess) a joint configuration whose tcp sits at the pre-grasp
target above the source bin, with the verified top-down grasp orientation
(`mdp/grasp_state.py::GRIPPING_QUAT_WXYZ`). Drives the relative-IK action
space with a simple proportional servo toward the target each step until it
converges, then reports the resulting joint positions -- this becomes the
new candidate `init_state.joint_pos` for `robots/rb5_850e.py` (replacing the
previous unverified guess, which `diagnose_initial_pose.py` showed lands the
tcp at world (-0.033, -0.111, 0.828) -- nowhere near the source bin at
world (0.51, 0.0, ~0)).

Usage:
    <IsaacLab repo>/isaaclab.sh -p solve_pregrasp_pose.py --headless \
        --result_file /path/to/result.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--steps", type=int, default=150)
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
from rb5_isaaclab.tasks.pick_place.mdp.grasp_state import GRIPPING_QUAT_WXYZ, PRE_GRASP_OFFSET_Z

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
    task = "RB5-PickPlace-IKRel-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    env.reset()

    source_bin, _ = load_bin_geometry()
    device = env.unwrapped.device
    target_pos_b = torch.tensor(
        [[source_bin.center[0], source_bin.center[1], source_bin.center[2] + PRE_GRASP_OFFSET_Z]], device=device
    )
    # Object yaw = 0 assumed for this solve (fixed pose, no rotation) --
    # matches Stage 1's fixed-cube-position config.
    target_quat_b = torch.tensor([GRIPPING_QUAT_WXYZ], device=device)

    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]

    action = torch.zeros(env.unwrapped.action_space.shape, device=device)
    for step in range(args_cli.steps):
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        pos_err = target_pos_b - ee_pos_b
        quat_err = math_utils.quat_mul(target_quat_b, math_utils.quat_conjugate(ee_quat_b))
        axis_angle_err = math_utils.axis_angle_from_quat(quat_err)

        # Relative-IK action: (dpos[3], drot_axis_angle[3], gripper[1]).
        # Proportional gains chosen conservatively (this is an offline
        # convergence solve, not a real-time controller -- no need to be
        # aggressive) and clipped to +-1 (the action space's expected
        # range, scaled by 0.5 inside the action term itself).
        act = torch.zeros_like(action)
        act[:, 0:3] = (pos_err * 4.0).clamp(-1.0, 1.0)
        act[:, 3:6] = (axis_angle_err * 2.0).clamp(-1.0, 1.0)
        obs, rew, terminated, truncated, info = env.step(act)

        if step % 25 == 0 or step == args_cli.steps - 1:
            pos_err_norm = torch.norm(pos_err, dim=-1).item()
            ang_err_norm = torch.norm(axis_angle_err, dim=-1).item()
            _log(f"step {step}: pos_err={pos_err_norm:.4f} ang_err={ang_err_norm:.4f}")
            _flush()

    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0].tolist()
    _log("--- converged joint config ---")
    for name, pos in zip(joint_names, joint_pos):
        _log(f"  {name}: {pos:.4f}")
    final_ee_pos_b = ee_pos_b[0].tolist()
    final_ee_quat_b = ee_quat_b[0].tolist()
    _log(f"final ee pos (robot-root frame): {[round(v,4) for v in final_ee_pos_b]}")
    _log(f"final ee quat wxyz (robot-root frame): {[round(v,4) for v in final_ee_quat_b]}")
    _log(f"target was: pos={target_pos_b[0].tolist()} quat={target_quat_b[0].tolist()}")
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
