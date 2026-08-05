"""Measures the end-effector's real world height (above the source-bin
floor) during actual successful grasps from a trained checkpoint -- used to
ground `reach_grasp_env_cfg.py`'s `FLOOR_CLEARANCE_MIN_HEIGHT` in measured
data instead of a guess from the TCP-to-fingertip-pad offset docstring
comment (which was explicitly flagged as unvalidated).

Runs the checkpoint deterministically (`mean_actions`, same as
`play_spawn_and_wait.py`) and records EE height at every env-step where
`grasp_state.bilateral_contact` is true, separately from steps where the
stricter `grasp_state.grasped_object` also holds.

Usage:
    <IsaacLab repo>/isaaclab.sh -p measure_grasp_ee_height.py \
        --task RB5-PickPlace-ReachGrasp-JointPos-v0 --checkpoint <path> \
        --num_envs 256 --steps 400 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import rb5_isaaclab  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import gymnasium as gym


def main():
    from rb5_isaaclab.tasks.pick_place.mdp import grasp_state
    from rb5_isaaclab.tasks.pick_place.bin_geometry import load_bin_geometry

    source_bin, _ = load_bin_geometry()
    floor_z = source_bin.center[2]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    import skrl  # noqa: F401
    from skrl.utils.runner.torch import Runner
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(args_cli.checkpoint)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    def policy_fn(obs_dict):
        outputs = runner.agent.act(obs_dict["policy"], None, timestep=0, timesteps=0)
        return outputs[-1].get("mean_actions", outputs[0])

    bilateral_heights = []
    grasped_heights = []

    for step in range(args_cli.steps):
        action = policy_fn(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        ee_frame = env.unwrapped.scene["ee_frame"]
        height = (ee_frame.data.target_pos_w[..., 0, 2] - floor_z).detach().cpu()

        bilateral = grasp_state.bilateral_contact(env.unwrapped).cpu()
        grasped = grasp_state.grasped_object(env.unwrapped).cpu()

        if bilateral.any():
            bilateral_heights.append(height[bilateral])
        if grasped.any():
            grasped_heights.append(height[grasped])

    print(f"[measure] task={args_cli.task} steps={args_cli.steps} num_envs={args_cli.num_envs}")
    for label, chunks in [("bilateral_contact", bilateral_heights), ("grasped_object", grasped_heights)]:
        if not chunks:
            print(f"[measure] {label}: never triggered in {args_cli.steps} steps -- no data")
            continue
        vals = torch.cat(chunks)
        print(
            f"[measure] {label}: n={vals.numel()} "
            f"min={vals.min().item():.4f} p5={vals.quantile(0.05).item():.4f} "
            f"mean={vals.mean().item():.4f} p95={vals.quantile(0.95).item():.4f} "
            f"max={vals.max().item():.4f} (meters above floor)"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
