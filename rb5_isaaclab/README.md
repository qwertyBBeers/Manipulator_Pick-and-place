# rb5_isaaclab — RB5-850E + Robotiq 2F-85 pick-and-place via IsaacLab RL

IsaacLab (manager-based RL) task for the same robot/gripper used by the
ROS2/MoveIt heuristic pipeline in `../rb5_binpicking` and `../rb5_isaac`,
trained with PPO (primary, via [skrl](https://skrl.readthedocs.io)) and
optionally SAC (small-scale comparison — see "Why PPO, not SAC" below).

This is a separate, non-ROS runtime: pure Python + PyTorch + IsaacLab,
installed into the `isaaclab` conda environment. It does **not** touch or
depend on the ROS colcon workspace at build/run time — it only *reads*
`../rb5_binpicking/config/bin_geometry.yaml` at task-config import time, so
both pipelines always agree on where the source/destination bins are.

## Prerequisites

- The `isaaclab` conda environment (already set up on this machine: IsaacLab
  0.54.3, Isaac Sim 5.1.0, skrl 2.0.0, rsl-rl-lib 5.0.1, PyTorch 2.7+cu128).
- An IsaacLab source checkout for the `isaaclab.sh` launcher and
  `scripts/tools/convert_urdf.py` / `scripts/reinforcement_learning/skrl/*.py`.
  Defaults to `/home/hh/asl_ws/vla_project/IsaacLab`; override with
  `ISAACLAB_REPO=/path/to/IsaacLab` if needed.

**Important gotcha**: this machine's shell profile sets `PYTHONPATH` to a
long list of ROS colcon `install/*/site-packages` paths, including
`.../install/isaaclab*` — a *different*, ROS-wrapped package that happens to
share the import name `isaaclab` with the real one and will silently shadow
it. Always run:
```bash
conda activate isaaclab
unset PYTHONPATH
```
before anything below. (This is the same class of dual-workspace shadowing
issue documented for the ROS side in `Manipulator/README2.md`.)

## 1. Convert the URDF to USD

```bash
conda activate isaaclab && unset PYTHONPATH
cd rb5_isaaclab
./scripts/convert_robot_to_usd.sh          # headless
./scripts/convert_robot_to_usd.sh --gui    # opens the result for inspection
```

This converts `assets/urdf/rb5_850e_robotiq_mimic.urdf` — a copy of
`../rb5_isaac/urdf/rb5_with_tools.urdf` with `<mimic>` tags added to the
gripper's 5 passive joints (the original ROS URDF is untouched; the
multipliers are copied from the already-validated
`rb5_isaac/rb5_isaac/trajectory_bridge.py` `GRIPPER_MIMIC` array, not
re-derived). Output goes to `assets/usd/rb5_850e_robotiq.usd` (gitignored).

**The `<mimic>` tags themselves are no longer used at the physics level**
(despite the filename) — an earlier version of this pipeline imported with
`parse_mimic=True` to bake them into a real PhysX `PhysxMimicJointAPI`
constraint per follower joint, but that was abandoned after extensive live
debugging: even after fixing three real, independently-verified importer
bugs (inverted gearing sign, empty `referenceJoint` target, un-authored
`referenceJointAxis` defaulting to the wrong axis), 5 simultaneous
bilateral mimic constraints all referencing the same one primary DOF
permanently deadlocked the primary knuckle joint outside its own
joint-limit range — an over-constrained system this PhysX/IsaacLab
version's GPU pipeline can't resolve, not something fixable by more gain
tuning. `convert_robot_to_usd.py` now imports with `parse_mimic=False`
(drops every `<mimic>` tag) so all 6 gripper joints come in as independent
revolute joints, and `robots/rb5_850e.py` gives each of them a real PD
actuator. The URDF's gearing values live on as
`GRIPPER_MIMIC_MULTIPLIERS` in that same file, now used to scale each
follower's *commanded target* (see `mdp/actions.py`) instead of a physics
constraint — see `CURRICULUM_REPORT.md` section 1d for the full debugging
trail (importer bugs, the deadlock diagnosis, and the fix). The post-export
verification step now confirms the *opposite* of what it originally did:
that all 6 gripper joints are independent (zero `PhysxMimicJointAPI` prims)
with finite authored joint limits.

**Note**: `convert_robot_to_usd.py` does the import directly via
`omni.kit.commands` rather than calling IsaacLab's own
`scripts/tools/convert_urdf.py` CLI tool, mainly for the non-standard
mesh-path-rewriting and fingertip-friction-material steps that tool doesn't
do (see the script's own docstring).

## 2. Install the package

```bash
conda activate isaaclab && unset PYTHONPATH
pip install -e rb5_isaaclab/
```

## 3. Smoke test

```bash
conda activate isaaclab && unset PYTHONPATH
cd <IsaacLab repo>
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/smoke_test.py --num_envs 4 --headless
```

Builds the env, steps it with random actions for 50 steps, checks for
NaN/Inf and crashes. This is a wiring check only — it says nothing about
whether the task is learnable.

## 4. Train

**Use this repo's own `scripts/train.py`, not IsaacLab's
`scripts/reinforcement_learning/skrl/train.py` directly** — IsaacLab's stock
script only ever does `import isaaclab_tasks`, so it has no way to know
`RB5-PickPlace-*` exists (confirmed: running it directly fails with
`gymnasium.error.NameNotFound: Environment 'RB5-PickPlace-JointPos' doesn't
exist.`). `scripts/train.py`/`scripts/play.py` here are byte-for-byte forks
of IsaacLab's own scripts with exactly one added line
(`import rb5_isaaclab  # noqa: F401`) at the same
`# PLACEHOLDER: Extension template` insertion point IsaacLab's own external-
project tooling uses — not a hack, this is IsaacLab's documented mechanism
for out-of-tree task packages. All CLI flags are identical to the upstream
scripts.

PPO (main path — recommended for the first real run):
```bash
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/train.py \
  --task RB5-PickPlace-JointPos-v0 --num_envs 512 --headless
```
Start with a modest `--num_envs` (this repo's env cfg defaults to 2048) to
confirm stability/throughput on your GPU before scaling up — a 24GB RTX
4090 should comfortably handle 2048-4096 for this scene (no cameras, small
number of rigid bodies), but always verify rather than assuming.

SAC (small-scale comparison):
```bash
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/train.py \
  --task RB5-PickPlace-JointPos-SAC-v0 --agent skrl_sac_cfg_entry_point --headless
```
(`RB5-PickPlace-JointPos-SAC-v0` fixes `num_envs=64` in its env cfg;
`--agent skrl_sac_cfg_entry_point` is required because the stock
`--algorithm` flag's choices don't include SAC — see below.)

Differential-IK (secondary control mode):
```bash
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/train.py \
  --task RB5-PickPlace-IKRel-v0 --num_envs 512 --headless
```

TensorBoard (works for either run — skrl writes automatically, no extra
code needed; `experiment.directory: "rb5_pick_place"` in both agent yamls
controls this path):
```bash
tensorboard --logdir <IsaacLab repo>/logs/skrl/rb5_pick_place
```

### Why PPO, not SAC, as the main path

IsaacLab's *own* bundled manipulation tasks — every single one, checked
across the whole `isaaclab_tasks` tree — only ship PPO agent configs. The
`skrl` training script's `--algorithm` flag doesn't even list SAC as a
choice. The reason is architectural, not a gap someone forgot to fill:
off-policy algorithms with a replay buffer (SAC, TD3, DDPG) don't pair well
with IsaacLab's thousands-of-parallel-envs-per-GPU-tensor-step model — each
step already generates enormous on-policy-sized batches, which is exactly
what PPO's on-policy rollout wants and exactly what makes a modest-sized
replay buffer's gradient-to-data ratio assumptions awkward. `skrl_sac_cfg.yaml`
in this repo is a **hand-written** config (based directly on skrl 2.0's
`SAC_CFG` dataclass and `Runner` model-role requirements, not on any
IsaacLab template — there wasn't one) for a **small** (`num_envs=64`)
comparison run. Treat its hyperparameters as an untuned starting point, not
a validated config the way the PPO one is.

## 5. Play / evaluate a checkpoint

Use this repo's `scripts/play.py` (same reasoning as `train.py` above —
IsaacLab's own `play.py` doesn't know about `RB5-PickPlace-*` either):
```bash
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/play.py \
  --task RB5-PickPlace-JointPos-Play-v0 --checkpoint /path/to/checkpoint.pt
```
Needs a real X display (e.g. `DISPLAY=:1` — not `--headless`) to actually
show the Isaac Sim GUI window.

`scripts/play_spawn_and_wait.py` is the same thing, but pauses right after
spawning/resetting the scene (robot held at its reset pose) and only starts
running the policy once a trigger file appears:
```bash
./isaaclab.sh -p <path-to>/rb5_isaaclab/scripts/play_spawn_and_wait.py \
  --task RB5-PickPlace-Curriculum-JointPos-Play-v0 --checkpoint /path/to/checkpoint.pt \
  --num_envs 1 --real-time --trigger_file /path/to/go.trigger
# later, from anywhere:
touch /path/to/go.trigger
```

## Task design notes

- **Single object, one size** (0.042m cube, 0.10kg — matches
  `binpicking_scene.py`'s `DynamicCuboid`), matching the ROS pipeline's own
  documented Phase-1 simplification (`README2.md` §7.6: perception only
  ever tracks one object). Multi-object / varied-geometry is a natural next
  step, not attempted here.
- **Bin geometry comes from `bin_geometry.yaml`**, not hardcoded here —
  object spawn range covers the source bin footprint, the placement-goal
  command range covers the destination bin footprint (margin: 0.03m from
  the inner wall, reusing the ROS pipeline's already-tuned
  `destination_wall_clearance` constant rather than inventing a new one).
- **`place_and_release` reward + `object_placed` termination**: the stock
  IsaacLab `lift` task's goal is just "hold the object near a floating
  target" — it's never required to let go. This task adds a bonus/success
  condition that only pays out once the object is at the goal, settled
  (low velocity), *and* the gripper has actually opened — otherwise there's
  no training pressure to ever release, which isn't really "place."
- **Gripper action space is a single open/close scalar** even though all 6
  gripper joints are independently PD-driven (see
  `rb5_isaaclab/robots/rb5_850e.py`'s module docstring — an earlier version
  of this repo used a real PhysX `<mimic>`-joint constraint for the other 5
  joints instead, abandoned after it caused an unresolvable actuator
  deadlock; see `CURRICULUM_REPORT.md` section 1d for the full story). The
  single scalar still applies to all 6 joints at once via
  `mdp/actions.py::DeadbandBinaryJointPositionActionCfg`'s
  `open_command_expr`/`close_command_expr` (each joint's target scaled by
  `GRIPPER_MIMIC_MULTIPLIERS`), so there's still nothing for the policy to
  separately control per-joint — just not because of a physics-level
  constraint anymore.

## Known gaps / things to verify before trusting a real training run

This section covers the original single-shot `RB5-PickPlace-JointPos-v0`
task. **For the current 4-stage curriculum** (`RB5-PickPlace-{Reach,
GraspLift,Transport,Curriculum}-JointPos-v0`, the recommended path — see
"PPO curriculum stages" below), see `CURRICULUM_REPORT.md` instead — it's
the up-to-date, actively-maintained account of what's verified, what's
fixed, what's still open, and (section 8) the results of the first full
training run (2026-07-28, all 4 stages, ~4h total).

- **Arm `init_state.joint_pos`**: was an unverified guess originally;
  **now solved (not guessed)** via `scripts/solve_pregrasp_pose.py`
  (Differential-IK convergence to a real pre-grasp target above the source
  bin) — see `robots/rb5_850e.py`'s `init_state` comment for the measured
  residual error.
- **`effort_limit_sim` values are physically-plausible placeholders**, not
  pulled from an RB5-850E datasheet (the ROS pipeline's own values, 1e6/1e9,
  are explicitly non-physical "may as well be infinite" numbers, so there
  was nothing to port for this specific field) — tune if joints appear
  unable to move the arm/grip firmly enough, or suspiciously overpowered.
- **Mimic-joint conversion was verified, then abandoned entirely** (not the
  same thing as "still in use and now verified" — see step 1 above and
  `CURRICULUM_REPORT.md` section 1d): the real PhysX mimic-joint constraint
  approach passed static USD verification but turned out to permanently
  deadlock the gripper at runtime regardless. Replaced with 6 independently
  PD-driven joints, verified moving smoothly toward commanded targets.
- **Learning-curve/success-rate**: the full curriculum has now completed a
  real training run for all 4 stages (`CURRICULUM_REPORT.md` section 8) —
  reward curves confirm learning happened (especially Curriculum, whose
  worst-case reward converged close to its mean by the end). Actual
  success-*rate* per stage is still unmeasured (skrl's runner doesn't log
  it to tensorboard) — would need a dedicated eval pass reading the
  `Episode_Termination/*` info dict, e.g. extending `play.py` the way
  `sanity_test.py` already does.
