"""Consolidated sanity-test harness (spec Tests 1-3; Test 4/scripted-waypoint
is in `scripted_pick_place.py`). Each `--mode` is a standalone check that
prints/logs enough to answer "did this behave as expected" without needing a
full training run.

Usage:
    <IsaacLab repo>/isaaclab.sh -p sanity_test.py --mode zero \
        --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 4 --headless \
        --result_file /path/to/result.txt
    ... --mode joint_mapping ...
    ... --mode gripper_contact --task RB5-PickPlace-GraspLift-JointPos-v0 ...
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["zero", "joint_mapping", "gripper_contact"])
parser.add_argument("--task", type=str, default="RB5-PickPlace-Reach-JointPos-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50)
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


def _make_env(num_envs):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    return gym.make(args_cli.task, cfg=env_cfg)


def test_zero(env):
    """Test 1: zero actions for a full episode -- robot/object should
    stay (roughly) stable, no NaN/explosion, reward terms printed."""
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    object_ = env.unwrapped.scene["object"]
    start_joint_pos = robot.data.joint_pos.clone()
    start_obj_pos = object_.data.root_pos_w.clone()

    zero_action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    max_joint_vel = 0.0
    for step in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(zero_action)
        if not torch.isfinite(rew).all():
            _log(f"FAIL: non-finite reward at step {step}")
            return
        max_joint_vel = max(max_joint_vel, robot.data.joint_vel.abs().max().item())

    end_joint_pos = robot.data.joint_pos.clone()
    end_obj_pos = object_.data.root_pos_w.clone()
    joint_drift = (end_joint_pos - start_joint_pos).abs().max().item()
    obj_drift = torch.norm(end_obj_pos - start_obj_pos, dim=-1).max().item()

    _log(f"[zero] max |joint drift| over {args_cli.steps} steps: {joint_drift:.4f} rad")
    _log(f"[zero] max object drift: {obj_drift:.4f} m")
    _log(f"[zero] max observed joint velocity: {max_joint_vel:.4f} rad/s")
    _log(f"[zero] final per-term reward (env 0): {info}")
    if joint_drift > 0.3:
        _log("WARNING: significant joint drift under zero action -- actuator gains or action convention issue?")


def test_joint_mapping(env):
    """Test 2: nudge each action dim +/- and report which joint moves."""
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names
    n_actions = env.unwrapped.action_space.shape[-1]

    for dim in range(n_actions):
        for sign, label in [(1.0, "+"), (-1.0, "-")]:
            env.reset()
            before = robot.data.joint_pos[0].clone()
            action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
            action[:, dim] = sign
            for _ in range(10):
                env.step(action)
            after = robot.data.joint_pos[0]
            delta = (after - before)
            moved_idx = torch.argmax(delta.abs()).item()
            _log(
                f"[joint_mapping] action[{dim}]={label}1.0 -> largest joint delta: "
                f"{joint_names[moved_idx]} ({delta[moved_idx].item():+.4f} rad), "
                f"all deltas: {[round(v, 4) for v in delta.tolist()]}"
            )
    _flush()


def test_gripper_contact(env):
    """Test 3: open/close with and without an object between the fingers,
    report finger joint positions and (if the task has contact sensors)
    contact values + computed grasp state."""
    import rb5_isaaclab.tasks.pick_place.mdp.grasp_state as gs

    has_contact = hasattr(env.unwrapped.scene, "sensors") and gs.LEFT_CONTACT_SENSOR_NAME in env.unwrapped.scene.sensors

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)

    for label, gripper_action in [("open", 1.0), ("close", -1.0)]:
        action[:, -1] = gripper_action
        for _ in range(20):
            env.step(action)
        knuckle_idx = robot.data.joint_names.index("robotiq_85_left_knuckle_joint")
        knuckle_pos = robot.data.joint_pos[0, knuckle_idx].item()
        _log(f"[gripper_contact] after {label} (20 steps): left_knuckle={knuckle_pos:.4f} rad")
        if has_contact:
            left, right = gs.fingertip_contact_forces(env.unwrapped)
            grasped = gs.grasped_object(env.unwrapped)
            _log(
                f"[gripper_contact]   left_force={left[0].item():.3f}N right_force={right[0].item():.3f}N "
                f"grasped_flag={bool(grasped[0].item())}"
            )
        else:
            _log("[gripper_contact]   (no contact sensors on this task)")
    _flush()


def main():
    env = _make_env(args_cli.num_envs)
    _log(f"[sanity_test] task={args_cli.task} num_envs={args_cli.num_envs} mode={args_cli.mode}")
    _log(f"[sanity_test] observation_space={env.observation_space}")
    _log(f"[sanity_test] action_space={env.action_space}")
    _flush()

    if args_cli.mode == "zero":
        test_zero(env)
    elif args_cli.mode == "joint_mapping":
        test_joint_mapping(env)
    elif args_cli.mode == "gripper_contact":
        test_gripper_contact(env)

    _log("[sanity_test] DONE")
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
