"""Minimal end-to-end sanity check: build the env, step it with random
actions for a few steps, confirm no NaN/crash. Does NOT verify the task is
learnable -- only that the config/asset/mdp wiring is not broken.

Usage (from the isaaclab conda env, PYTHONPATH cleared of any ROS
workspace):
    <IsaacLab repo>/isaaclab.sh -p scripts/smoke_test.py --num_envs 4 --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="rb5_isaaclab smoke test")
parser.add_argument("--task", type=str, default="RB5-PickPlace-JointPos-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument(
    "--result_file",
    type=str,
    default=None,
    help="If set, also write a plain-text result summary here (fsync'd) -- "
    "Omniverse Kit has been observed to swallow buffered stdout on process "
    "exit in headless mode, so a normal print()-only script can silently "
    "lose its final status even on success.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import rb5_isaaclab  # noqa: F401  (registers gym task IDs on import)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

_LOG_LINES = []


def _log(msg):
    print(msg)
    _LOG_LINES.append(str(msg))


def _flush_result_file():
    if not args_cli.result_file:
        return
    with open(args_cli.result_file, "w") as f:
        f.write("\n".join(_LOG_LINES))
        f.flush()
        os.fsync(f.fileno())


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    _log(f"[smoke_test] task={args_cli.task} num_envs={args_cli.num_envs}")
    _log(f"[smoke_test] observation_space={env.observation_space}")
    _log(f"[smoke_test] action_space={env.action_space}")
    _flush_result_file()

    obs, _ = env.reset()
    _assert_finite("reset.obs", obs)
    _log("[smoke_test] reset OK")
    _flush_result_file()

    for step in range(args_cli.steps):
        actions = torch.rand(env.action_space.shape, device=env.unwrapped.device) * 2.0 - 1.0
        obs, rew, terminated, truncated, info = env.step(actions)
        _assert_finite(f"step{step}.obs", obs)
        _assert_finite(f"step{step}.rew", rew)
        if step % 5 == 0:
            _log(f"[smoke_test] step {step} OK")
            _flush_result_file()

    env.close()
    _log(f"[smoke_test] OK -- no NaN/Inf, no crash, over {args_cli.steps} random-action steps")
    _flush_result_file()


def _assert_finite(label, value):
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_finite(f"{label}.{k}", v)
        return
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"[smoke_test] non-finite values in {label}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        _log(f"[smoke_test] EXCEPTION: {e}")
        _log(traceback.format_exc())
        _flush_result_file()
        raise
    finally:
        simulation_app.close()
