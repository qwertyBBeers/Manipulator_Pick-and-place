# Pick-and-Place Completion Report

Status as of this pass: a real, physical-state-based evaluation pipeline now
exists and has been run against all four existing curriculum checkpoints.
It found a specific, reproducible failure point -- **the grasp-to-lift
transition** -- that reward curves alone (see `CURRICULUM_REPORT.md`) never
surfaced. Two of the four checkpoints (GraspLift, Curriculum) will need
retraining after this pass, both because of the finding itself and because
the start-pose fix below changes their reset distribution. Nothing here
retrains a policy yet -- this is the diagnosis, per the "diagnose before
tuning" ordering the task asked for.

## A. Code changes

**New files**
- `scripts/evaluate_policy.py` -- deterministic policy evaluator. Loads a
  checkpoint, runs `--num_episodes` episodes with no learning updates, and
  derives every milestone (reached pre-grasp, bilateral contact, stable
  grasp, lifted, transport started, reached/inside destination, released,
  released-and-stable, full success, dropped) from the exact same
  `mdp/grasp_state.py` functions the environment itself uses for
  reward/termination -- not a re-implementation. Handles vectorized
  async episode completion correctly (see the module's own docstring for
  the auto-reset timing subtlety this requires). Writes
  `logs/evaluation/<task>/<timestamp>/{episodes.csv,summary.json}`.

**Modified files**
- `rb5_isaaclab/tasks/pick_place/mdp/grasp_state.py` -- added reusable
  physical-state predicates (`is_near_pregrasp`, `is_object_lifted`,
  `is_object_above_safe_height`, `is_object_near_destination`,
  `is_gripper_open`, `is_object_released`, `is_object_stable`) so
  `evaluate_policy.py` reads off the same definitions as the reward/
  termination code, not a duplicate. `terminations.reach_success` now
  delegates to `is_near_pregrasp` (pure extraction, no behavior change --
  verified via the smoke tests in section B).
- `scripts/reward_diagnostic.py` -- added `pos%`/`neg%` activation-rate
  columns per reward term, and a "physical-condition gate activation rate"
  section (bilateral contact / stable grasp / grasped / gripper-open rates,
  independent of any one reward term's own scale). Also fixed a real,
  pre-existing bug: `--policy checkpoint` passed the raw `{"policy": Tensor}`
  observation dict straight into skrl's `agent.act()`, which expects a plain
  tensor -- crashed every checkpoint-mode invocation (`TypeError: unsupported
  operand type(s) for -: 'dict' and 'Tensor'`, confirmed via a real run in
  this pass, see section E). This looks like the checkpoint mode had never
  actually been exercised before.
- `config/reach_env_cfg.py` -- extracted the inline far-start joint pose
  into a named constant `FAR_START_JOINT_POS` (behavior unchanged) so
  GraspLift/Curriculum can reuse it.
- `config/grasp_lift_env_cfg.py`, `config/curriculum_env_cfg.py` -- **start-pose
  fix** (explicit request this pass): both stages now start from
  `FAR_START_JOINT_POS` (same pose Reach uses) instead of the shared
  pre-grasp default. Previously the arm started *already at* the pre-grasp
  target, which meant `grasp_pose_position_reward` had nothing left to
  learn and the "reach" part of these stages was never actually exercised.
  Adjusted `grasp_pose_position_reward`'s `std` 0.10 -> 0.3 (same reasoning
  `reach_env_cfg.py` already used: a tight std saturates `1-tanh(d/std)` to
  ~0 over a ~0.9m approach) and `episode_length_s` (GraspLift 6.0 -> 8.0,
  Curriculum 8.0 -> 10.0) to give the added approach distance room. **This
  invalidates the existing GraspLift and Curriculum checkpoints' reset
  distribution -- both need retraining before they can be re-evaluated
  meaningfully.** Not randomized (single fixed far pose), per the request.

No task IDs were removed or renamed; no reward/termination *semantics*
changed for Transport or Reach.

## B. Evaluation results (existing checkpoints, unmodified reset semantics)

500-episode runs were not run this pass (100-episode diagnostic pass only,
per the task's own phase ordering: "100 episodes for the first diagnostic
pass"). All four used `--num_envs 64 --num_episodes 100 --seed 42
--headless` against the checkpoint listed.

| Stage | Checkpoint | Reach | Bilateral contact | Stable grasp | Lifted | Full success | Mean reward | Dominant failure |
|---|---|---|---|---|---|---|---|---|
| Reach | `2026-07-28_15-34-45/best_agent.pt` | 0/100 | n/a | n/a | n/a | n/a | 3.56 | `reach_failure` 100% |
| GraspLift | `2026-07-28_16-13-21/best_agent.pt` | 96/100 | 95/100 | 95/100 | 28/100 | 0/100 | 36.24 | `lift_failure` 95% |
| Transport | `2026-07-28_17-14-48/best_agent.pt` | 100/100 | 1/100 | 0/100 | 0/100 | 0/100 | -30.69 | `grasp_failure` 99% |
| Curriculum | `2026-07-28_18-17-00/best_agent.pt` | 100/100 | 100/100 | 100/100 | 0/100 | 0/100 | 53.37 | `lift_failure` 100% |

Full per-episode CSVs and JSON summaries at
`logs/evaluation/<task>/<timestamp>/` under the IsaacLab repo (paths printed
by each run, e.g.
`logs/evaluation/RB5-PickPlace-Curriculum-JointPos-Play-v0/2026-07-29_15-28-44/`).

**Reward is not aligned with physical success for two of the four
checkpoints, confirmed directly (not inferred):**

- **Curriculum**: mean reward 53.37 (matches the training run's converged
  ~56.1, see `CURRICULUM_REPORT.md` section 8), yet the object is *never*
  lifted 10cm in 100 deterministic episodes, despite reliably (100/100)
  achieving bilateral contact and holding a stable grasp. The reward is high
  because `bilateral_fingertip_contact_reward` (weight 2.0) +
  `stable_grasp_reward` (weight 4.0) + `grasp_pose_position/orientation_reward`
  (weights 1.0/0.5) can all be farmed indefinitely just by reaching and
  holding, without ever paying the `continuous_lift_reward` (weight 4.0)
  cost of actually gaining height. Classic reward-shaping local optimum:
  "grasp and hold forever" is a locally-consistent, high-reward policy that
  never does the harder thing.
- **GraspLift** shows the same pattern less severely: grasps reliably
  (95/100) but only actually clears the 10cm lift target in 28/100 episodes.
- **Transport**, evaluated from its own designed reset (starts "already
  holding" via `mdp/events.py::reset_robot_holding_object`, no reach/grasp
  phase), essentially never registers real bilateral contact (1/100) even
  though the reset explicitly writes the gripper to its measured
  contact-stopped joint angles. This is the clearest, most direct evidence
  yet for the exact concern the task spec raised about Transport's reset:
  whatever contact exists right after the state write does not hold up
  under the policy's own subsequent actions -- consistent with
  `CURRICULUM_REPORT.md`'s already-documented "grasped, not actually
  lifted off the floor" simplification, but now shown to fail even the
  *grasped* half, not just the *lifted* half.
- **Reach**: reward improved substantially during training (`CURRICULUM_REPORT.md`
  section 8: 0.02 -> 3.28), and this pass's mean reward (3.56) is consistent
  with that -- but the strict `reach_success` condition (0.03m position /
  0.20 rad orientation, held 8 consecutive steps) is met in 0/100
  deterministic episodes. The policy gets close but not reliably inside the
  tight success band.

## C. Transport reset report

**Not fixed this pass** -- root-cause investigation only, via the new
evaluation tooling (this pass did not re-run `solve_holding_pose.py` or
retune any physical parameters; that's the next concrete step, not
completed here).

- **Old (still current) reset semantics**: `reset_robot_holding_object`
  writes the arm to `HOLDING_ARM_JOINT_POS`, the gripper to its measured
  contact-stopped angles (`HOLDING_GRIPPER_KNUCKLE_POS` /
  `HOLDING_GRIPPER_FOLLOWER_POS`), and the object to its measured resting
  pose relative to that grasp -- all measured together from one real
  simulated grasp via `scripts/solve_holding_pose.py`. Documented known
  simplification: the object never actually leaves the source-bin floor
  (`HOLDING_OBJECT_POS_B` sits at the floor height that grasp closure
  converged to), so this was already known to not satisfy the task's
  Phase-6 requirement (`object_height > source_bin_floor_height + 0.05m`).
- **New finding this pass**: it's worse than "not lifted" -- under the
  trained Transport policy's own actions, real bilateral contact (measured
  via the fingertip contact-force sensors, not gripper angle) is present in
  only 1/100 deterministic episodes. Either the measured contact-stopped
  state doesn't produce genuine sustained contact once physics settles
  post-reset, or the trained policy's arm motion breaks whatever marginal
  contact existed almost immediately. Not yet distinguished -- the next
  diagnostic step is a zero-action hold test (spec section 6's "Required
  validation": run `--num_envs 64`, log max joint/object velocity and
  bilateral-contact retention rate over the first ~50 steps with the
  gripper action held at "close" and the arm action at zero) to isolate
  "does the reset itself produce contact" from "does the policy destroy
  it".
- **Not yet done**: re-running `solve_holding_pose.py` with an actual lift
  phase (raise the arm after closing, not just close-and-hold at floor
  height), re-measuring `HOLDING_ARM_JOINT_POS`/`HOLDING_GRIPPER_*_POS`/
  `HOLDING_OBJECT_POS_B` from that lifted state, and validating retention
  over a longer hold per spec section 6.

## D. Hierarchical playback report

**Not implemented this pass.** The Reach-stage checkpoint's own strict
success rate is 0/100 (section B), and Curriculum's checkpoint never lifts
the object at all -- chaining Reach's policy into Curriculum's right now
would just hand off into a stage that can't proceed past its own grasp
regardless of where it starts. Per the task's own Phase 9/10 ordering
("retrain only after the evaluation identifies a specific failure" /
"decide between single-policy and hierarchical completion" comes *after*
a working baseline), building the hand-off script now would produce a
demo that fails for reasons unrelated to the hand-off mechanism itself and
wouldn't answer the real question (single-policy vs. hierarchical). Deferred
until GraspLift/Curriculum are retrained past the lift-failure local optimum
found in section B.

## E. Remaining issues

**Blocking** (must be resolved before a visual full-pipeline demo means
anything):
1. GraspLift and Curriculum policies don't reliably lift the object despite
   high reward -- reward-shaping local optimum (section B). Needs either
   reward reweighting (e.g. raise `continuous_lift_reward`'s relative
   weight, or reduce `stable_grasp_reward`'s standalone payout so holding
   alone isn't as profitable) or a retrain with the same weights now that
   the start-pose fix also changes the reset distribution -- these two
   should probably be addressed together in one retrain rather than two
   sequential ones.
2. Transport's reset doesn't produce a real, retained grasp under the
   trained policy (section C) -- needs the zero-action isolation test, then
   likely a re-measured *actually-lifted* holding pose, before Transport can
   be meaningfully retrained or evaluated again.
3. GraspLift and Curriculum checkpoints are now stale relative to their own
   env cfgs (start-pose change, section A) -- old checkpoints will
   under-perform their old numbers for reasons unrelated to the lift-failure
   finding above; re-evaluating them further isn't useful until retrained.

**Important, non-blocking**:
- `reach_success`'s strict threshold (0.03m/0.20rad) is never met by the
  current Reach checkpoint even though downstream stages don't currently
  consume Reach's output directly (each stage has its own independent
  reset) -- matters once/if a hierarchical hand-off is revisited.
- Evaluator's documented approximation: non-termination physical predicates
  (bilateral contact, grasped, ...) are not trusted on the exact final
  frame of an episode that terminates via time-out, to avoid the
  auto-reset contamination bug described in `evaluate_policy.py`'s module
  docstring. Given `STABLE_GRASP_STEPS`-style hold requirements, this is
  very unlikely to have flipped any of the results in section B, but it's
  a known, documented limitation of the tool, not a proven non-issue.

**Future extensions** (explicitly not started): task-phase observation
signal, grasp-slip signal, hierarchical policy handoff with recovery
behavior, and the temporal/Transformer comparison plan are all unstarted,
per the task's own explicit ordering (baseline first).

**A sequencing mistake worth flagging explicitly** (rule 12: report
disproved hypotheses, not just successes): the `reward_diagnostic.py` run
in section F below was executed *after* the start-pose code change in
section A, against the *old* (pre-fix) Curriculum checkpoint. Its 0%
bilateral-contact/stable-grasp/grasped-object gate rates do **not** explain
section B's original "high reward, never lifts" finding (that data was
collected before the code change, against the checkpoint's actual training
conditions) -- they instead show, incidentally, that the old checkpoint has
zero transfer to the new far-start reset, which is expected (it never
practiced reaching from that distance) and is separate evidence for the
same conclusion (E.3: this checkpoint must be retrained, not re-evaluated).
Section B's reward-shaping explanation stands on the reward-term *weights*
themselves (2.0 + 4.0 + 1.0 + 0.5 = 7.5 max instantaneous reward available
just for reach+grasp+hold, vs. `continuous_lift_reward`'s 4.0 that
additionally requires real height gain) rather than on this diagnostic run.

## F. Reward-term diagnostic (old checkpoint against the new far-start reset)

`reward_diagnostic.py --task RB5-PickPlace-Curriculum-JointPos-Play-v0
--policy checkpoint --checkpoint .../2026-07-28_18-17-00.../best_agent.pt
--num_envs 64 --steps 300`, run after the start-pose fix (see the flagged
sequencing note in section E). Only the two reach-shaping terms ever fire;
every grasp/lift/place term is exactly zero across all 64 envs x 300 steps:

```
term                                weight     mean     nonzero%   pos%   w*mean
grasp_pose_position_reward          1.00000   0.3038      100.0%  100.0%  0.30383
grasp_pose_orientation_reward       0.50000   0.0958      100.0%  100.0%  0.04791
bilateral_fingertip_contact_reward  2.00000   0.0000        0.0%    0.0%  0.00000
stable_grasp_reward                 4.00000   0.0000        0.0%    0.0%  0.00000
continuous_lift_reward              4.00000   0.0000        0.0%    0.0%  0.00000
object_inside_destination_reward    8.00000   0.0000        0.0%    0.0%  0.00000
released_and_stable_reward         15.00000   0.0000        0.0%    0.0%  0.00000
empty_gripper_close_penalty        -0.20000  43.3333       86.7%   86.7% -8.66667

Physical-condition gate activation rates:
  bilateral_contact  0.0%   stable_grasp  0.0%   grasped_object  0.0%
```

Reading this correctly (per the sequencing note above): `grasp_pose_position_reward`
averaging 0.30 (out of a max of 1.0) shows the policy makes *some* progress
toward the object from the new far pose within the episode, but never
closes the distance enough for real contact -- expected, since this
checkpoint was trained exclusively from a pose where that distance was
already zero. `empty_gripper_close_penalty` firing 86.7% of the time (the
policy closes the gripper on empty air, substantially closed but with no
object contact) is itself informative for the *eventual* retrain: the
policy's grasp-timing behavior was tuned to a context ("I am already at the
target, close now") that no longer matches when a real approach phase
exists first -- worth re-checking after retraining that this penalty rate
drops once the policy learns to sequence approach-then-close again.

This diagnostic pass on a *matching* (pre-fix) checkpoint against its own
training conditions was not re-run this pass (would require checking out
the pre-fix config) -- the reward-weight-structure explanation in section B
does not depend on it.

## G. Commands to reproduce

```bash
conda activate isaaclab
unset PYTHONPATH
cd /home/hh/asl_ws/vla_project/IsaacLab

# Evaluate any checkpoint against real task-success metrics
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/evaluate_policy.py \
  --task RB5-PickPlace-Curriculum-JointPos-Play-v0 \
  --checkpoint logs/skrl/rb5_pick_place/2026-07-28_18-17-00_ppo_torch/checkpoints/best_agent.pt \
  --num_envs 64 --num_episodes 100 --seed 42 --headless

# Per-reward-term + physical-gate activation diagnostics against a checkpoint
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/reward_diagnostic.py \
  --task RB5-PickPlace-Curriculum-JointPos-Play-v0 --policy checkpoint \
  --checkpoint logs/skrl/rb5_pick_place/2026-07-28_18-17-00_ppo_torch/checkpoints/best_agent.pt \
  --num_envs 64 --steps 300

# Smoke-test the start-pose-fixed GraspLift/Curriculum configs
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/smoke_test.py \
  --task RB5-PickPlace-GraspLift-JointPos-v0 --num_envs 8 --steps 30
```

## Next recommended step

Retrain GraspLift and Curriculum with the start-pose fix already applied
(section A), after first addressing the reward-shaping imbalance in section
E.1 (otherwise the same lift-failure local optimum is likely to just
re-emerge with a longer approach on top of it). Re-evaluate with
`evaluate_policy.py` before declaring either stage fixed -- reward curves
alone already proved unreliable for this exact failure mode.
