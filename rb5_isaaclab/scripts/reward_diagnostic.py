"""Per-term reward-scale diagnostic (spec: detect whether smoothness
penalties dominate all positive task rewards, or any single term's scale
is wildly out of line with the others).

For each active reward term, reports mean/std/min/max/nonzero-activation-
rate of the RAW (unweighted) term value, and the weighted mean contribution
(raw_mean * weight) -- the weighted numbers are what actually matters for
"is this term drowned out", the raw numbers are what matters for "is this
term's own scale sane" (e.g. a tanh-bounded term living in [0,1] vs an L2
penalty that can be much larger).

Supports 4 action sources:
    --policy zero      : all-zero actions
    --policy random     : uniform random actions in [-1, 1]
    --policy scripted   : `scripted_pick_place.py`'s waypoint controller (if
                           the task has a compatible action space)
    --policy checkpoint  : a trained skrl checkpoint (--checkpoint path)

Usage:
    <IsaacLab repo>/isaaclab.sh -p reward_diagnostic.py \
        --task RB5-PickPlace-Reach-JointPos-v0 --policy random \
        --num_envs 64 --steps 200 --headless --result_file /path/to/result.txt
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--policy", type=str, default="random", choices=["zero", "random", "checkpoint"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    reward_manager = env.unwrapped.reward_manager
    term_names = reward_manager.active_terms
    weights = [cfg.weight for cfg in reward_manager._term_cfgs]
    raw_values = {name: [] for name in term_names}

    # Physical-condition gate activation rates -- separate from the reward
    # terms above: a gated reward term's own nonzero% already conflates "is
    # the gate open" with "how large is the continuous part while it's
    # open"; these report the gate itself. Probed once (try/except) since
    # not every stage has contact sensors (see `grasp_state.py`).
    from rb5_isaaclab.tasks.pick_place.mdp import grasp_state

    gate_fns = {
        "bilateral_contact": grasp_state.bilateral_contact,
        "stable_grasp": grasp_state.stable_grasp,
        "grasped_object": grasp_state.grasped_object,
        "is_gripper_open": grasp_state.is_gripper_open,
    }
    gate_hits = {}
    gate_total = {}
    for name, fn in gate_fns.items():
        try:
            val = fn(env.unwrapped)
            assert val.shape == (args_cli.num_envs,)
            gate_hits[name] = 0.0
            gate_total[name] = 0
        except Exception as e:
            _log(f"[reward_diagnostic] gate '{name}' unsupported for this task ({type(e).__name__}: {e})")

    policy_fn = None
    if args_cli.policy == "checkpoint":
        import skrl  # noqa: F401
        from skrl.utils.runner.torch import Runner
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        import yaml

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(args_cli.checkpoint)
        if hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")
        else:
            runner.agent.enable_training_mode(False, apply_to_models=True)

        def policy_fn(obs_dict):
            # `env` here is the RAW gym env (not `SkrlVecEnvWrapper`-wrapped,
            # since this script also needs direct, un-wrapped `env.step()`
            # access to `reward_manager`/`env.unwrapped`) -- so `obs_dict` is
            # still the manager-based env's native `{"policy": Tensor}` dict.
            # skrl's `agent.act()` expects the plain tensor.
            outputs = runner.agent.act(obs_dict["policy"], None, timestep=0, timesteps=0)
            return outputs[-1].get("mean_actions", outputs[0])

    for step in range(args_cli.steps):
        if args_cli.policy == "zero":
            action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        elif args_cli.policy == "random":
            action = torch.rand(env.unwrapped.action_space.shape, device=env.unwrapped.device) * 2.0 - 1.0
        else:
            action = policy_fn(obs)

        obs, rew, terminated, truncated, info = env.step(action)
        # `_step_reward` is (num_envs, num_terms) UNWEIGHTED*dt... actually
        # weighted-but-not-yet-summed per-term contribution (see
        # RewardManager.compute) -- divide by weight*dt to recover the raw
        # term value for the "is this term's own scale sane" comparison.
        step_reward = reward_manager._step_reward  # (num_envs, num_terms)
        dt = env.unwrapped.step_dt
        for i, name in enumerate(term_names):
            w = weights[i]
            denom = w * dt if abs(w) > 1e-12 else 1.0
            raw = (step_reward[:, i] / denom).detach().cpu()
            raw_values[name].append(raw)

        for name in gate_hits:
            hit = gate_fns[name](env.unwrapped)
            gate_hits[name] += hit.float().sum().item()
            gate_total[name] += hit.numel()

    _log(f"[reward_diagnostic] task={args_cli.task} policy={args_cli.policy} steps={args_cli.steps} num_envs={args_cli.num_envs}")
    _log(f"{'term':<38} {'weight':>9} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'nonzero%':>9} {'pos%':>7} {'neg%':>7} {'w*mean':>10}")
    for i, name in enumerate(term_names):
        vals = torch.cat(raw_values[name])
        w = weights[i]
        mean, std, mn, mx = vals.mean().item(), vals.std().item(), vals.min().item(), vals.max().item()
        nonzero_pct = (vals.abs() > 1e-8).float().mean().item() * 100
        # Split by sign of the RAW term value -- for an unsigned (e.g. tanh-kernel
        # or 0/1 flag) term this collapses to nonzero%/0%; for a penalty term
        # written as a positive count (e.g. `object_drop_penalty`) whose weight is
        # negative, "positive%" is still the term's own activation rate, not the
        # signed contribution to reward (that's what `w*mean` already reports).
        pos_pct = (vals > 1e-8).float().mean().item() * 100
        neg_pct = (vals < -1e-8).float().mean().item() * 100
        weighted_mean = mean * w
        _log(f"{name:<38} {w:>9.5f} {mean:>9.4f} {std:>9.4f} {mn:>9.4f} {mx:>9.4f} {nonzero_pct:>8.1f}% {pos_pct:>6.1f}% {neg_pct:>6.1f}% {weighted_mean:>10.5f}")

    if gate_hits:
        _log("\nPhysical-condition gate activation rates (fraction of env-steps true):")
        for name, hits in gate_hits.items():
            rate = 100.0 * hits / gate_total[name] if gate_total[name] else 0.0
            _log(f"  {name:<25} {rate:>6.1f}%")

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
