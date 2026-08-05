"""Deterministic policy evaluator: loads a trained skrl checkpoint, runs it
against a task with NO learning updates, and measures actual physical
task-success metrics (not just reward) over many episodes.

Why this exists: reward curves confirm a policy is optimizing *something*,
not that the something is the real task. This script re-derives every
milestone (reached pre-grasp, bilateral contact, stable grasp, lifted,
transported, placed, released, dropped, ...) from the SAME physical-state
functions `mdp/grasp_state.py` and `mdp/rewards.py` already use for
shaping/termination, so the evaluation can never silently disagree with what
the environment itself considers "grasped" or "placed".

Vectorized-episode-counting note: `ManagerBasedRLEnv.step()` auto-resets any
env that just terminated *before* returning -- the observation/scene state
you read immediately after `step()` for a just-reset env already belongs to
the NEXT episode. Two different fixes are used here depending on the signal:
  - Milestone termination terms (`reach_success`, `grasp_lift_success`,
    `transport_success`, `full_place_success`) are read via
    `termination_manager.get_term(name)`, which IsaacLab computes and caches
    *before* the internal auto-reset -- safe to read post-`step()` regardless
    of whether that env just reset.
  - Non-termination physical predicates (bilateral contact, grasped, lifted,
    ...) are only OR'd into an episode's sticky "ever_*" flags for envs that
    did NOT reset this step. In practice this only risks losing the single
    exact frame a sustained condition (contact, stable grasp, ...) was true
    on the very last step of an episode; `STABLE_GRASP_STEPS`-style hold
    requirements mean the sticky flag was almost always already set several
    steps earlier. Documented here rather than hidden.

Usage:
    <IsaacLab repo>/isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/evaluate_policy.py \
        --task RB5-PickPlace-Curriculum-JointPos-Play-v0 \
        --checkpoint <checkpoint-path> \
        --num_envs 64 --num_episodes 500 --seed 42 --headless
"""

import argparse
import csv
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained rb5_isaaclab checkpoint against real task-success metrics.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--result_dir", type=str, default=None, help="Defaults to logs/evaluation/<task>/<timestamp>/")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.managers import SceneEntityCfg

import rb5_isaaclab  # noqa: F401 (registers gym task IDs)
from rb5_isaaclab.robots.rb5_850e import GRIPPER_PRIMARY_JOINT
from rb5_isaaclab.tasks.pick_place.bin_geometry import load_bin_geometry
from rb5_isaaclab.tasks.pick_place.mdp import grasp_state

# Reused, not re-derived -- same constants the stage env cfgs themselves use
# (spec requirement: reuse existing configuration values, don't duplicate
# thresholds independently). Imported lazily/defensively since not every
# checkpoint's task necessarily has all of these modules wired the same way.
try:
    from rb5_isaaclab.tasks.pick_place.config.grasp_lift_env_cfg import TARGET_LIFT_HEIGHT
except Exception:
    TARGET_LIFT_HEIGHT = 0.05
try:
    from rb5_isaaclab.tasks.pick_place.config.transport_env_cfg import SAFE_TRANSPORT_HEIGHT
except Exception:
    SAFE_TRANSPORT_HEIGHT = None  # resolved from bin geometry below if still None
try:
    from rb5_isaaclab.tasks.pick_place.config.curriculum_env_cfg import DEST_INSIDE_MARGIN_TRAIN
except Exception:
    DEST_INSIDE_MARGIN_TRAIN = 0.03

SOURCE_BIN, DEST_BIN = load_bin_geometry()
SOURCE_FLOOR_Z = SOURCE_BIN.center[2]
if SAFE_TRANSPORT_HEIGHT is None:
    SAFE_TRANSPORT_HEIGHT = SOURCE_FLOOR_Z + 0.06
NEAR_DEST_RADIUS = 0.15  # loose "in the vicinity of the destination bin" radius
DEST_MARGIN_EVAL = min(DEST_INSIDE_MARGIN_TRAIN, 0.02)  # tighter than the training margin, per spec

# ---------------------------------------------------------------------------
# Per-step physical-state predicates. Every one of these is a thin call into
# `grasp_state.py` -- no thresholds are redefined here.
# ---------------------------------------------------------------------------
def build_physical_metrics(gripper_robot_cfg: SceneEntityCfg) -> dict:
    """`gripper_robot_cfg` must already be `.resolve(scene)`d -- the default
    `SceneEntityCfg` param baked into `grasp_state.is_gripper_open`/
    `is_object_released` is only ever resolved automatically when a manager
    parses it from a *Cfg term; called directly like this, `.joint_ids`
    would still be unresolved."""
    return {
        "near_pregrasp": lambda env: grasp_state.is_near_pregrasp(env),
        "bilateral_contact": lambda env: grasp_state.bilateral_contact(env),
        "stable_grasp": lambda env: grasp_state.stable_grasp(env),
        "grasped_object": lambda env: grasp_state.grasped_object(env),
        "object_lifted": lambda env: grasp_state.is_object_lifted(env, TARGET_LIFT_HEIGHT, SOURCE_FLOOR_Z),
        "above_safe_height": lambda env: grasp_state.is_object_above_safe_height(env, SAFE_TRANSPORT_HEIGHT),
        "near_destination": lambda env: grasp_state.is_object_near_destination(env, NEAR_DEST_RADIUS),
        "inside_destination": lambda env: grasp_state.object_inside_destination_xy(env, DEST_MARGIN_EVAL),
        "gripper_open": lambda env: grasp_state.is_gripper_open(env, robot_cfg=gripper_robot_cfg),
        "object_released": lambda env: grasp_state.is_object_released(env, robot_cfg=gripper_robot_cfg),
        "object_stable": lambda env: grasp_state.is_object_stable(env),
    }


# Milestone termination terms -- read via `get_term()`, safe against the
# auto-reset timing issue described in the module docstring. Only some
# stages register some of these (e.g. Reach has `reach_success` but not
# `full_place_success`); missing ones are probed away below.
TERMINATION_METRICS = ["reach_success", "grasp_lift_success", "transport_success", "full_place_success"]


def probe_capabilities(core_env, physical_metrics: dict) -> tuple[dict, dict]:
    """Call every metric once against the just-reset env; anything that
    raises (missing contact sensor, missing command term, ...) is recorded
    as unsupported for this task instead of crashing later mid-run."""
    physical_ok, term_ok = {}, {}
    for name, fn in physical_metrics.items():
        try:
            val = fn(core_env)
            assert val.shape == (core_env.num_envs,)
            physical_ok[name] = True
        except Exception as e:
            physical_ok[name] = False
            print(f"[evaluate_policy] metric '{name}' unsupported for this task ({type(e).__name__}: {e})")
    for name in TERMINATION_METRICS:
        try:
            core_env.termination_manager.get_term(name)
            term_ok[name] = True
        except KeyError:
            term_ok[name] = False
    return physical_ok, term_ok


def classify_failure(ep: dict) -> str:
    """Deepest-milestone-reached failure classification (spec section:
    "Required failure-stage classification"). An approximation, not a
    formally exhaustive state machine -- adjust categories as new failure
    modes are observed."""
    if ep["full_pick_place_success"]:
        return "success"
    if ep["ever_inside_destination"] and ep["ever_object_released"]:
        return "stability_failure"  # released inside the bin but never settled+held
    if ep["ever_inside_destination"]:
        return "release_failure"  # reached the bin, never opened the gripper
    if ep["ever_grasped"] and (ep["ever_above_safe_height"] or ep["ever_near_destination"]):
        if ep["drop_count"] > 0:
            return "drop_during_transport"
        return "transport_failure"
    if ep["ever_lifted"] and ep["ever_grasped"]:
        return "transport_failure"  # lifted but never made meaningful transport progress
    if ep["ever_stable_grasp"]:
        if ep["drop_count"] > 0:
            return "drop_during_lift"
        return "lift_failure"
    if ep["ever_bilateral_contact"]:
        return "grasp_instability"
    if ep["ever_near_pregrasp"]:
        return "grasp_failure"
    return "reach_failure"


def main():
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    core_env = raw_env.unwrapped
    wrapped_env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    print("[evaluate_policy] constructing skrl Runner...", flush=True)
    runner = Runner(wrapped_env, agent_cfg)
    print(f"[evaluate_policy] loading checkpoint: {args_cli.checkpoint}", flush=True)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    print("[evaluate_policy] checkpoint loaded", flush=True)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False, apply_to_models=True)

    num_envs = core_env.num_envs
    device = core_env.device

    gripper_robot_cfg = SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT])
    gripper_robot_cfg.resolve(core_env.scene)
    physical_metrics = build_physical_metrics(gripper_robot_cfg)

    obs, _ = wrapped_env.reset()
    physical_ok, term_ok = probe_capabilities(core_env, physical_metrics)
    print(f"[evaluate_policy] task={args_cli.task} num_envs={num_envs} "
          f"target_episodes={args_cli.num_episodes} max_episode_length={core_env.max_episode_length}")

    STICKY_NAMES = [
        "ever_near_pregrasp", "ever_bilateral_contact", "ever_stable_grasp", "ever_grasped",
        "ever_lifted", "ever_above_safe_height", "ever_near_destination", "ever_inside_destination",
        "ever_object_released", "ever_released_and_stable",
    ]
    sticky = {name: torch.zeros(num_envs, dtype=torch.bool, device=device) for name in STICKY_NAMES}
    term_sticky = {name: torch.zeros(num_envs, dtype=torch.bool, device=device) for name in TERMINATION_METRICS}
    ep_reward = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    drop_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)

    results: list[dict] = []
    max_total_steps = int(args_cli.num_episodes / max(1, num_envs) * core_env.max_episode_length * 3) + 200
    total_steps = 0
    start_time = time.time()

    while len(results) < args_cli.num_episodes and total_steps < max_total_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rew, terminated, truncated, info = wrapped_env.step(actions)
        total_steps += 1

        rew_t = rew.reshape(-1).to(device)
        terminated_t = terminated.reshape(-1).to(device).bool()
        truncated_t = truncated.reshape(-1).to(device).bool()
        done_this_step = terminated_t | truncated_t
        active_mask = ~done_this_step

        ep_reward += rew_t
        ep_len += 1

        # Non-termination physical predicates -- only trusted for envs that
        # didn't just auto-reset this step (see module docstring).
        if active_mask.any():
            if physical_ok["near_pregrasp"]:
                sticky["ever_near_pregrasp"][active_mask] |= physical_metrics["near_pregrasp"](core_env)[active_mask]
            if physical_ok["bilateral_contact"]:
                sticky["ever_bilateral_contact"][active_mask] |= physical_metrics["bilateral_contact"](core_env)[active_mask]
            if physical_ok["stable_grasp"]:
                sticky["ever_stable_grasp"][active_mask] |= physical_metrics["stable_grasp"](core_env)[active_mask]
            if physical_ok["grasped_object"]:
                currently_grasped = physical_metrics["grasped_object"](core_env)
                sticky["ever_grasped"][active_mask] |= currently_grasped[active_mask]
                just_dropped = prev_grasped & ~currently_grasped & sticky["ever_grasped"] & active_mask
                drop_count[just_dropped] += 1
                prev_grasped[active_mask] = currently_grasped[active_mask]
            if physical_ok["object_lifted"]:
                sticky["ever_lifted"][active_mask] |= physical_metrics["object_lifted"](core_env)[active_mask]
            if physical_ok["above_safe_height"]:
                sticky["ever_above_safe_height"][active_mask] |= physical_metrics["above_safe_height"](core_env)[active_mask]
            if physical_ok["near_destination"]:
                sticky["ever_near_destination"][active_mask] |= physical_metrics["near_destination"](core_env)[active_mask]
            if physical_ok["inside_destination"]:
                sticky["ever_inside_destination"][active_mask] |= physical_metrics["inside_destination"](core_env)[active_mask]
            if physical_ok["object_released"]:
                released_now = physical_metrics["object_released"](core_env)
                # Gated on `ever_grasped` -- an object that was never picked
                # up is trivially "gripper open, not grasped, stationary"
                # from step 0, which is NOT a release (same reasoning
                # `rewards.object_drop_penalty` already uses for "drop").
                released_now_real = released_now & sticky["ever_grasped"]
                sticky["ever_object_released"][active_mask] |= released_now_real[active_mask]
                if physical_ok["object_stable"]:
                    stable_now = physical_metrics["object_stable"](core_env)
                    sticky["ever_released_and_stable"][active_mask] |= (released_now_real & stable_now)[active_mask]

        # Termination-term milestones -- safe to read for ALL envs, done or not.
        for name in TERMINATION_METRICS:
            if term_ok[name]:
                term_sticky[name] |= core_env.termination_manager.get_term(name)

        if done_this_step.any():
            done_idx = done_this_step.nonzero(as_tuple=True)[0]
            for i in done_idx.tolist():
                if len(results) >= args_cli.num_episodes:
                    break
                # Fallback (for stages without a `full_place_success` termination
                # term, e.g. GraspLift/Transport in isolation) mirrors
                # `rewards.full_place_success_condition`'s own AND-of-conditions:
                # grasped at some point, actually inside the destination
                # footprint, AND released+settled there -- NOT released+stable
                # alone (trivially true for an object that was simply never
                # picked up in the first place).
                full_success = bool(term_sticky["full_place_success"][i]) or bool(
                    sticky["ever_grasped"][i] and sticky["ever_inside_destination"][i]
                    and sticky["ever_released_and_stable"][i]
                )
                ep = {
                    "reached_pregrasp": bool(sticky["ever_near_pregrasp"][i] or term_sticky["reach_success"][i]),
                    "bilateral_contact_achieved": bool(sticky["ever_bilateral_contact"][i]),
                    "stable_grasp_achieved": bool(sticky["ever_stable_grasp"][i] or term_sticky["grasp_lift_success"][i]),
                    "object_lifted": bool(sticky["ever_lifted"][i] or term_sticky["grasp_lift_success"][i]),
                    "transport_started": bool(sticky["ever_grasped"][i] and sticky["ever_above_safe_height"][i]),
                    "object_reached_destination": bool(sticky["ever_near_destination"][i] or term_sticky["transport_success"][i]),
                    "object_inside_destination": bool(sticky["ever_inside_destination"][i]),
                    "gripper_released": bool(sticky["ever_object_released"][i]),
                    "object_released_and_stable": bool(sticky["ever_released_and_stable"][i]),
                    "full_pick_place_success": full_success,
                    "object_dropped": bool(drop_count[i] > 0),
                    "drop_count": int(drop_count[i].item()),
                    "episode_length_steps": int(ep_len[i].item()),
                    "episode_reward": float(ep_reward[i].item()),
                    # internal-only fields consumed by classify_failure, stripped before CSV/JSON
                    "ever_grasped": bool(sticky["ever_grasped"][i]),
                    "ever_above_safe_height": bool(sticky["ever_above_safe_height"][i]),
                    "ever_near_destination": bool(sticky["ever_near_destination"][i]),
                    "ever_inside_destination": bool(sticky["ever_inside_destination"][i]),
                    "ever_object_released": bool(sticky["ever_object_released"][i]),
                    "ever_stable_grasp": bool(sticky["ever_stable_grasp"][i] or term_sticky["grasp_lift_success"][i]),
                    "ever_lifted": bool(sticky["ever_lifted"][i] or term_sticky["grasp_lift_success"][i]),
                    "ever_bilateral_contact": bool(sticky["ever_bilateral_contact"][i]),
                    "ever_near_pregrasp": bool(sticky["ever_near_pregrasp"][i] or term_sticky["reach_success"][i]),
                }
                ep["failure_stage"] = classify_failure(ep)
                results.append(ep)

            # reset per-env trackers for envs that just completed an episode
            for name in sticky:
                sticky[name][done_this_step] = False
            for name in term_sticky:
                term_sticky[name][done_this_step] = False
            ep_reward[done_this_step] = 0.0
            ep_len[done_this_step] = 0
            drop_count[done_this_step] = 0
            prev_grasped[done_this_step] = False

        if total_steps % 200 == 0:
            print(f"[evaluate_policy] step {total_steps}: {len(results)}/{args_cli.num_episodes} episodes "
                  f"({time.time() - start_time:.0f}s elapsed)")

    if total_steps >= max_total_steps:
        print(f"[evaluate_policy] WARNING: hit the step safety cap ({max_total_steps}) with only "
              f"{len(results)}/{args_cli.num_episodes} episodes collected -- some envs may never be terminating.")

    write_results(results, physical_ok, term_ok)
    env_close_start = time.time()
    raw_env.close()
    print(f"[evaluate_policy] done in {time.time() - start_time:.0f}s ({len(results)} episodes, env close {time.time()-env_close_start:.1f}s)")


def write_results(results: list[dict], physical_ok: dict, term_ok: dict):
    if args_cli.result_dir:
        out_dir = args_cli.result_dir
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join("logs", "evaluation", args_cli.task, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    internal_only = {"ever_grasped", "ever_above_safe_height", "ever_near_destination", "ever_inside_destination",
                      "ever_object_released", "ever_stable_grasp", "ever_lifted", "ever_bilateral_contact",
                      "ever_near_pregrasp"}
    csv_fields = [k for k in (results[0].keys() if results else []) if k not in internal_only]

    csv_path = os.path.join(out_dir, "episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for ep in results:
            writer.writerow({k: ep[k] for k in csv_fields})

    n = len(results)
    def rate(key):
        return (sum(1 for e in results if e[key]) / n) if n else 0.0
    def mean(key, filter_key=None):
        vals = [e[key] for e in results if (filter_key is None or e[filter_key])]
        return (sum(vals) / len(vals)) if vals else 0.0

    failure_dist = {}
    for e in results:
        failure_dist[e["failure_stage"]] = failure_dist.get(e["failure_stage"], 0) + 1

    summary = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "num_episodes_requested": args_cli.num_episodes,
        "num_episodes_evaluated": n,
        "supported_physical_metrics": physical_ok,
        "supported_termination_metrics": term_ok,
        "rates": {
            "full_success_rate": rate("full_pick_place_success"),
            "reach_rate": rate("reached_pregrasp"),
            "bilateral_contact_rate": rate("bilateral_contact_achieved"),
            "stable_grasp_rate": rate("stable_grasp_achieved"),
            "lift_rate": rate("object_lifted"),
            "transport_started_rate": rate("transport_started"),
            "destination_reach_rate": rate("object_reached_destination"),
            "inside_destination_rate": rate("object_inside_destination"),
            "release_rate": rate("gripper_released"),
            "released_and_stable_rate": rate("object_released_and_stable"),
            "drop_rate": rate("object_dropped"),
        },
        "mean_episode_length": mean("episode_length_steps"),
        "mean_successful_episode_length": mean("episode_length_steps", filter_key="full_pick_place_success"),
        "mean_reward": mean("episode_reward"),
        "failure_stage_distribution": failure_dist,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[evaluate_policy] ==== {args_cli.task} ====")
    print(f"Episodes evaluated: {n}")
    print(f"Reached pre-grasp:    {sum(1 for e in results if e['reached_pregrasp']):>5} / {n}")
    print(f"Bilateral contact:    {sum(1 for e in results if e['bilateral_contact_achieved']):>5} / {n}")
    print(f"Stable grasp:         {sum(1 for e in results if e['stable_grasp_achieved']):>5} / {n}")
    print(f"Lifted object:        {sum(1 for e in results if e['object_lifted']):>5} / {n}")
    print(f"Transport started:    {sum(1 for e in results if e['transport_started']):>5} / {n}")
    print(f"Reached destination:  {sum(1 for e in results if e['object_reached_destination']):>5} / {n}")
    print(f"Inside destination:   {sum(1 for e in results if e['object_inside_destination']):>5} / {n}")
    print(f"Released and stable:  {sum(1 for e in results if e['object_released_and_stable']):>5} / {n}")
    print(f"Full success:         {sum(1 for e in results if e['full_pick_place_success']):>5} / {n}")
    print(f"Drop rate: {rate('object_dropped')*100:.1f}%   Mean reward: {summary['mean_reward']:.2f}   "
          f"Mean episode length: {summary['mean_episode_length']:.1f}")
    print("Failure-stage distribution:")
    for stage, count in sorted(failure_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<25} {count:>5} ({100.0*count/n:.1f}%)")
    print(f"\nUnsupported metrics for this task: "
          f"{[k for k, v in physical_ok.items() if not v] + [k for k, v in term_ok.items() if not v]}")
    print(f"[evaluate_policy] wrote {csv_path}")
    print(f"[evaluate_policy] wrote {json_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
