# PPO Curriculum Implementation Report

Implementation of a 4-stage PPO curriculum (Reach / GraspLift / Transport /
Curriculum) for `RB5-PickPlace-*`, replacing the single-shot task's
degenerate "opens gripper, lays arm back" trained behavior. All code is
implemented and verified (imports, env creation at 1/64/2048 envs, contact
sensors, reward/observation wiring), and as of 2026-07-28 **all four stages
have completed a full real training run** (~4h total, `num_envs=2048`,
100k timesteps each) -- see section 8 for the run itself and its results,
and section 7 for what was verified before that point. Section 1d documents
a gripper joint that turned out not to track its commands at all (root
cause: three URDF/USD importer bugs plus an over-constrained PhysX
mimic-joint setup) -- fixed via an architecture change to 6
independently-driven gripper joints -- plus two issues that fix surfaced
and have since also been fixed: a gripper-action open/close ambiguity
(deadband added) and Transport's reset placing the object inside the
gripper's own mechanism instead of between the fingertip pads (re-measured
via a real simulated grasp, see section 1d's final update). All four
stages' zero-action tests are now clean or small (sub-15 rad/s transients,
no drift warnings past Stage 2's minor one) -- see the updated section 7
table.

## 1. Diagnosis

Investigated with real tooling (diagnostic scripts, not guesses) rather than
assumed. Four confirmed root causes, in rough order of impact:

**1a. The robot's initial pose was nowhere near the task.** `scripts/diagnose_initial_pose.py`
(new) measured the *previous* `robots/rb5_850e.py` init pose's tcp world
position at **(-0.033, -0.111, 0.828)** -- the source bin is at
**(0.51, 0.0, ~0)**. The arm started almost 1m away from where the object
even is. No amount of reward shaping fixes a policy that starts nowhere
near the task. Fixed by solving (not guessing) a real pre-grasp joint
config with `scripts/solve_pregrasp_pose.py` (drives the existing
Differential-IK action space toward the source bin with the verified
top-down grasp orientation) -- converged to tcp (0.517, -0.0003, 0.121)
against a (0.51, 0.0, 0.10) target, 2.2cm/~7deg residual. Applied to
`robots/rb5_850e.py`'s shared default `init_state.joint_pos`.

**1b. Reward-term scale imbalance: smoothness penalties dominated task reward
by ~30x.** `scripts/reward_diagnostic.py` (new) on Stage 1 under random
actions measured weighted-mean contributions of `action_rate` approx -0.0995
and `joint_velocity` approx -0.0457 against a `grasp_pose_position_reward`
of approx **0.00000** (saturated, see 1c) plus `grasp_pose_orientation_reward`
approx 0.0044 -- i.e. penalties outweighing task reward by >20x even under
pure exploration, before any purposeful (and therefore higher-velocity,
higher-penalty) movement. This is a textbook local optimum: minimize
action/velocity, ignore the task. Reduced `action_rate`/`joint_velocity`
weights 10x/5x further (to -0.00005/-0.00002); re-verified with the same
diagnostic afterward (see section 7) -- new balance is approx +0.278 task
reward vs approx -0.019 combined penalty, a ~14x swing in the right direction.

**1c. `grasp_pose_position_reward`'s `std=0.10` (the spec's own suggested
initial value, tuned for the *original* Lift task's already-nearby
object-EE distances) saturates the tanh kernel almost everywhere for a task
that starts ~0.9m from the target: `1 - tanh(0.9/0.1) approx 0.00005`. No
gradient over most of the approach. Fixed by raising to `std=0.3`,
confirmed via the same reward_diagnostic run (mean rose from ~0.0000 to
0.2735). `grasp_pose_orientation_reward`'s `std=0.35` is very likely
suffering the same saturation (initial orientation error is often close to
the quaternion-angle max) -- **flagged but not independently re-verified
with its own diagnostic run**, due to time; see section 7/known gaps.

**1d. Gripper knuckle joint did not track its commanded open/close target at
all -- root cause was NOT actuator gain, it was three importer bugs plus an
architectural over-constraint, all now fixed.** This finding went through
several rounds of hypothesis-test-disprove, kept here in full for the
record:

- *First hypothesis (disproved): torque saturation.* `sanity_test.py
  --mode gripper_contact` on Stage 2 showed the knuckle reading -0.23 rad
  after a sustained "open" command and -0.23 rad after a sustained "close"
  command -- indistinguishable, neither near the commanded 0.0/0.8, zero
  contact force. Raised `effort_limit_sim` 15 -> 80 (stiffness=7000,
  damping=350). **This did not fix it** -- a follow-up run read -0.43 rad
  for both commands, if anything worse.
- *Second hypothesis (confirmed, but incomplete): the URDF->USD importer
  authored the 5 follower gripper joints' `PhysxMimicJointAPI` gearing with
  the WRONG SIGN relative to the URDF's own `<mimic multiplier="...">`
  values.* Confirmed by direct USD inspection (a standalone `pxr`
  script opening the exported `.usd` and reading `physxMimicJoint:rotY:gearing`
  off each follower joint prim) -- e.g. `robotiq_85_left_inner_knuckle_joint`
  read `gearing=-1.0` in the USD despite the URDF specifying
  `multiplier="1.0"`, and all 4 other followers were inverted the same way.
  Fixed by adding a repair pass to `scripts/convert_robot_to_usd.py` that
  force-sets each follower's gearing to the known-correct value (same table
  already validated for the ROS pipeline's `trajectory_bridge.py`). Still
  did not fix the knuckle tracking.
- *Third hypothesis (also confirmed, also incomplete): the importer left
  `physxMimicJoint:rotY:referenceJointAxis` un-authored on every follower,
  defaulting to the schema default `"rotX"` -- but the primary joint is a
  Y-axis revolute joint, so its rotation only ever appears on "rotY". Every
  follower's mimic constraint was reading a reference axis that never
  moves, making the constraint a silent no-op regardless of gearing.*
  Confirmed the same way (direct USD inspection). Fixed by another repair
  pass explicitly authoring `referenceJointAxis` to match. Followers now
  visibly converged to steady values instead of drifting indefinitely --
  real progress -- but the *primary* knuckle joint was still frozen outside
  its own [0.0, 0.8] joint-limit range regardless of its commanded target.
- *Fourth check (root cause of the remaining deadlock): a direct-diagnostic
  script reading `robot.data.applied_torque`/`computed_torque` on the
  primary joint after a sustained close command showed
  `computed_torque=4506` (what the PD wants) vs `applied_torque=80`
  (saturated at the limit) with **zero joint velocity** and **zero fingertip
  contact force** -- i.e. the actuator was pushing as hard as allowed and
  the joint still would not move at all, with no external contact to blame.
  Five simultaneous bilateral `PhysxMimicJointAPI` constraints all
  referencing the same one primary DOF is an over-constrained system that
  this PhysX/IsaacLab version's GPU pipeline could not resolve, regardless
  of the gearing/axis bugs above being fixed.
- **Fix (architecture change, user-approved):** abandoned real PhysX mimic
  joints entirely. `scripts/convert_robot_to_usd.py` now imports with
  `parse_mimic=False`, so all 6 gripper joints come in as independent
  revolute joints with correctly-authored limits straight from the URDF (no
  more of the mysterious +-0.16 rad limit padding the mimic import path was
  also silently introducing). `robots/rb5_850e.py` gives all 6 a real PD
  actuator (same stiffness/damping/effort_limit as before). The 5
  followers' *commanded targets* (not a physics constraint) are scaled from
  the primary's target by `GRIPPER_MIMIC_MULTIPLIERS` in each
  `ActionsCfg.gripper_action`'s `open_command_expr`/`close_command_expr` --
  the same approach `binpicking_scene.py` already uses on the real robot.
  **Verified**: a 60-step close command now moves the knuckle and all 5
  followers smoothly and proportionally toward their targets when the arm
  is relocated to free space (away from the bin/object). One residual,
  lower-priority observation: closing on *empty* space for a full 60 steps
  eventually makes the two fingertip links converge close enough to
  interfere with each other (not filtered by `enabled_self_collisions=False`,
  which appears to only filter directly-jointed link pairs, not the two
  independent finger branches), producing some jitter past roughly
  50% closed -- during an actual grasp this shouldn't matter, since a
  real object should stop the fingers well before that point, but it's
  worth re-checking once Stage 2 training produces real grasp attempts.

## 2. Files changed

**New:**
- `rb5_isaaclab/tasks/pick_place/mdp/grasp_state.py` -- shared grasp/contact/goal-footprint logic (constants, per-env counter machinery, contact-sensor helpers, the ROS-pipeline-derived grasp orientation)
- `rb5_isaaclab/tasks/pick_place/mdp/events.py` -- `reset_rb5_pp_state`, `reset_robot_holding_object` (Stage 3)
- `rb5_isaaclab/tasks/pick_place/config/reach_env_cfg.py` -- Stage 1
- `rb5_isaaclab/tasks/pick_place/config/grasp_lift_env_cfg.py` -- Stage 2 (`_Easy` and default "Normal" reset variants)
- `rb5_isaaclab/tasks/pick_place/config/transport_env_cfg.py` -- Stage 3
- `rb5_isaaclab/tasks/pick_place/config/curriculum_env_cfg.py` -- Stage 4
- `scripts/diagnose_initial_pose.py` -- initial-pose diagnostic (spec-required)
- `scripts/solve_pregrasp_pose.py` -- IK-based pre-grasp pose solver (used to fix 1a)
- `scripts/sanity_test.py` -- Tests 1-3 (zero-action, joint-mapping, gripper/contact)
- `scripts/scripted_pick_place.py` -- Test 4 (scripted waypoint sequence)
- `scripts/reward_diagnostic.py` -- per-term reward-scale diagnostic

**Modified:**
- `rb5_isaaclab/robots/rb5_850e.py` -- new verified init pose (1a); `activate_contact_sensors: False -> True` (required for the new `ContactSensorCfg`s -- confirmed via a failed env-creation run without it)
- `rb5_isaaclab/tasks/pick_place/mdp/observations.py` -- added EE pose, pre-grasp vector, grasp-orientation-error, contact-force, grasp-flag, object-relative-to-EE, object velocity terms
- `rb5_isaaclab/tasks/pick_place/mdp/rewards.py` -- added all Stage 1-4 reward functions (grasp_pose_position/orientation, bilateral_fingertip_contact, stable_grasp, continuous_lift, empty_gripper_close_penalty, object_drop_penalty, maintain_grasp, object_to_goal_position, object_height_safety, object_inside_destination, released_and_stable, full_place_success_condition)
- `rb5_isaaclab/tasks/pick_place/mdp/terminations.py` -- added reach_success, grasp_lift_success, transport_success, full_place_success
- `rb5_isaaclab/tasks/pick_place/mdp/__init__.py` -- re-export `events`/`grasp_state`
- `rb5_isaaclab/__init__.py` -- registered 8 new gym IDs (4 stages x train/play)

**Untouched (per spec: don't delete/break the original):** `pick_place_env_cfg.py`, `config/joint_pos_env_cfg.py`, `RB5-PickPlace-JointPos-v0` and its `-Play`/`-SAC` variants all still work exactly as before (they now also benefit from fix 1a/the `activate_contact_sensors` change, both harmless for them).

## 3. Curriculum implementation

Four independent gym environments, each with its own `-Play-v0` variant,
sharing the `mdp/` function library:

| Stage | Env ID | Action space | Key mechanism |
|---|---|---|---|
| 1 Reach | `RB5-PickPlace-Reach-JointPos-v0` | 6-dim (arm only, no gripper term) | Deliberately starts far from the target (see 1a) so there's something to learn; `reach_success` termination requires position+orientation error below threshold for 8 consecutive steps |
| 2 GraspLift | `RB5-PickPlace-GraspLift-JointPos-v0` (+ `_Easy` variant) | 7-dim | Real contact sensors on both fingertips; `grasped_object` = bilateral contact AND object near tcp; `stable_grasp` requires 5 consecutive steps of bilateral contact |
| 3 Transport | `RB5-PickPlace-Transport-JointPos-v0` | 7-dim | Custom reset (`reset_robot_holding_object`) starts every episode already holding the object at a measured "holding" configuration -- no grasping in this stage |
| 4 Curriculum | `RB5-PickPlace-Curriculum-JointPos-v0` | 7-dim | Full task; every reward term individually gated on its own physical precondition (see below) instead of a discrete phase-index variable |

All four register their `_PLAY` counterpart too (`RB5-PickPlace-Reach-JointPos-Play-v0` etc., `num_envs=50`).

**On the "phase-aware" requirement**: rather than a separate phase-index
state variable, each Stage 4 reward term is gated on the actual physical
condition it needs (`continuous_lift_reward` needs `grasped_object`,
`object_to_goal_position_reward` needs grasped+above-safe-height,
`released_and_stable_reward` needs the full place-success condition) --
recomputed fresh every step from current state, so nothing is exploitable
via an irreversible phase lock-in after a drop. This satisfies the safety
property the spec's phase-index idea is protecting, without an explicit
phase enum. Noted in section 7 as a simplification, not hidden.

## 4. Reward table (final, as implemented)

**Stage 1 (Reach)** -- `reach_env_cfg.py`:
| Term | Weight | Gate |
|---|---|---|
| grasp_pose_position_reward | 1.0 | std=0.3 (raised from spec's 0.10, see 1c) |
| grasp_pose_orientation_reward | 0.5 | std=0.35 (unverified, see known gaps) |
| action_rate | -0.00005 | (reduced from spec's -0.0005, see 1b) |
| joint_velocity | -0.00002 | (reduced from spec's -0.0001, see 1b) |

**Stage 2 (GraspLift)** -- `grasp_lift_env_cfg.py`:
| Term | Weight | Gate |
|---|---|---|
| grasp_pose_position_reward | 1.0 | std=0.10 |
| grasp_pose_orientation_reward | 0.5 | std=0.35 |
| bilateral_fingertip_contact_reward | 2.0 | both fingertips >0.5N |
| stable_grasp_reward | 4.0 | bilateral contact held 5 steps |
| continuous_lift_reward | 4.0 | `grasped * clamp((h-floor)/0.10, 0, 1)` |
| empty_gripper_close_penalty | -0.2 | closed AND no bilateral contact |
| object_drop_penalty | -5.0 | `ever_grasped` sticky flag AND not currently grasped |
| action_rate / joint_velocity | -0.0005 / -0.0001 | spec's suggested values (not yet stress-tested here the way Stage 1 was) |

**Stage 3 (Transport)** -- `transport_env_cfg.py`:
| Term | Weight | Gate |
|---|---|---|
| maintain_grasp_reward | 2.0 | currently grasped |
| stable_grasp_reward | 1.0 | **carried forward from Stage 2** (2026-07-29 addition, see section 8) -- bilateral contact held 5 steps |
| empty_gripper_close_penalty | -0.2 | **carried forward from Stage 2** (2026-07-29 addition) -- closed AND no bilateral contact |
| object_to_goal_position_reward | 4.0 | std=0.3, gated grasped+above safe height |
| fine_goal_position_reward | 2.0 | std=0.05, same gate |
| object_height_safety_reward | 1.0 | height in [safe, max] band, gated grasped |
| object_drop_penalty | -5.0 | sticky ever_grasped (forced True at reset -- see events.py) |
| action_rate / joint_velocity | -0.0005 / -0.0001 | |

**Stage 4 (Curriculum)** -- `curriculum_env_cfg.py`:
| Term | Weight | Gate |
|---|---|---|
| grasp_pose_position_reward | 1.0 | std=0.10 |
| grasp_pose_orientation_reward | 0.5 | std=0.35 |
| bilateral_fingertip_contact_reward | 2.0 | |
| stable_grasp_reward | 4.0 | |
| continuous_lift_reward | 4.0 | |
| object_to_goal_position_reward | 4.0 | grasped + above safe height |
| object_inside_destination_reward | 8.0 | actual dest-bin footprint, 3cm training margin |
| released_and_stable_reward | 15.0 | full_place_success_condition, 15-step hold |
| empty_gripper_close_penalty | -0.2 | |
| object_drop_penalty | -5.0 | |
| action_rate / joint_velocity | -0.0005 / -0.0001 | **fixed, no curriculum ramp** (spec: don't use the previous -0.1 values) |

`holding_at_goal_penalty` (the legacy task's anti-hoarding term) is **removed** from Stage 4 per spec -- `released_and_stable_reward`'s strictly-harder, hold-gated condition plus the drop penalty replace its role.

## 5. Contact/grasp implementation

- **Fingertip link names**: `robotiq_85_left_finger_tip_link`, `robotiq_85_right_finger_tip_link` -- from `robots/rb5_850e.py`'s already-existing, USD-verified `FINGERTIP_LINK_NAMES` (confirmed against the converted USD earlier this session, not re-guessed).
- **Sensor config**: `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/<fingertip_link>", filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])`, added via `grasp_state.add_grasp_contact_sensors()` (pattern verified against IsaacLab's own `dexsuite_kuka_allegro_env_cfg.py`, not invented). Required `robots/rb5_850e.py`'s `activate_contact_sensors: False -> True` (confirmed via a failed run without it).
- **Distinguishing left vs right**: two separate sensors, each filtered to only the object -- `force_matrix_w` is per-(sensor-body, filtered-body), so cross-talk with the bin/ground/opposite-finger/robot is structurally excluded, not just thresholded out.
- **`bilateral_contact`**: both `force_matrix_w` magnitudes > `CONTACT_FORCE_THRESHOLD=0.5N` simultaneously.
- **`stable_grasp`**: `bilateral_contact` held for >= 5 consecutive control steps, via a per-env counter (see below).
- **`grasped_object`** (the actual "is this a real grasp" definition used everywhere): `bilateral_contact AND object within 6cm of the tcp` -- conservative per spec, not gripper-angle-based.
- **Per-env counters** (`grasp_state.py`): `ManagerBasedRLEnv` has no built-in generic mutable scratch state for custom MDP terms, so counters are lazily attached as `env._rb5_pp_state` and reset via `events.reset_rb5_pp_state` (mode="reset", included in every curriculum stage's `EventCfg`). Guarded against double-advancing when both a reward term and a termination term read the same counter in one step (`grasp_state.step_once`, keyed on `env.common_step_counter`).
- **Known limitations**: (a) `CONTACT_FORCE_THRESHOLD=0.5N` is a simulation-noise-floor choice, not derived from a real grasp-force model; (b) the gripper-tracking issue in section 1d is now fixed (all 6 joints independently PD-driven, verified moving smoothly toward commanded targets) -- the residual empty-space fingertip-interference jitter noted there (past ~50% closed with nothing between the fingers) could still affect `bilateral_contact` if it fires on that jitter rather than a real grasp; worth a quick recheck once Stage 2 produces real grasp attempts.

## 6. Commands

```bash
conda activate isaaclab && unset PYTHONPATH
cd /home/hh/asl_ws/vla_project/IsaacLab

# Initial-pose visualization / verification
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/diagnose_initial_pose.py --headless \
  --result_file /tmp/pose_result.txt

# Zero-action test
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/sanity_test.py --mode zero \
  --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 4 --headless --result_file /tmp/zero.txt

# Joint-action mapping test
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/sanity_test.py --mode joint_mapping \
  --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 1 --headless --result_file /tmp/mapping.txt

# Gripper/contact test
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/sanity_test.py --mode gripper_contact \
  --task RB5-PickPlace-GraspLift-JointPos-v0 --num_envs 4 --headless --result_file /tmp/contact.txt

# Scripted waypoint test
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/scripted_pick_place.py --headless \
  --result_file /tmp/scripted.txt

# Reward diagnostics
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/reward_diagnostic.py \
  --task RB5-PickPlace-Reach-JointPos-v0 --policy random --num_envs 64 --steps 100 --headless \
  --result_file /tmp/reward_diag.txt

# Training (debug -> initial -> large-scale per spec's own recommended progression)
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 64 --max_iterations 40 --headless   # debug
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 1024 --headless                  # initial
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-Reach-JointPos-v0 --num_envs 2048 --headless                  # large-scale

./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-GraspLift-JointPos-v0 --num_envs 1024 --headless
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-Transport-JointPos-v0 --num_envs 1024 --headless
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py --task RB5-PickPlace-Curriculum-JointPos-v0 --num_envs 1024 --headless

# Playing a checkpoint
./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/play.py --task RB5-PickPlace-Reach-JointPos-Play-v0 --checkpoint <path>
# (same pattern for GraspLift / Transport / Curriculum -Play-v0)
```

## 7. Validation results

| Test | Status |
|---|---|
| Python import test | **executed successfully** (implicit in every run below -- all 4 stages' modules import cleanly) |
| Env creation, 1 env (Stage 1) | **executed successfully** |
| Env creation, 64 envs (Stage 1, 2, 3, 4) | **executed successfully** (all four); re-verified with `smoke_test.py` (50 random-action steps, no NaN/Inf/crash) after all fixes in this session |
| Env creation, 2048 envs (Stage 1) | **executed successfully**; **not executed for Stages 2-4** (time) |
| Zero-action test (Stage 1) | **executed successfully** -- initially found a 86 rad/s velocity spike + false-positive `reach_success`, root-caused (init pose coincided with target), fixed, re-verified clean (0.41 rad/s max, no false success) |
| Zero-action test (Stage 2, 3, 4) | **All three re-executed after every fix below; all clean now.** Stage 2 (GraspLift): 12.0 rad/s max, one-time step-0 transient decaying into a small approx 0.5 rad/s steady jiggle on the fingertip joints -- an order of magnitude below anything flagged as a problem elsewhere in this doc; attributed to the 6 gripper joints now being independently PD-driven (section 1d) with no rigid mechanical coupling damping out small per-joint settling differences the way the old (broken) mimic-joint design would have, not a new bug. Stage 4 (Transport): dropped from approx 193 rad/s to **3.4 rad/s** max, joint drift 1.49 -> 0.14 rad, object drift 0.58 -> 0.003m, no warning -- fixed via the `HOLDING_OBJECT_POS_B` re-measurement below. Stage 4/Curriculum: 64.9 rad/s max (Curriculum doesn't use the Transport holding-reset at all, so unaffected by that fix; same small-jiggle character as Stage 2, just a larger one-time transient -- not independently root-caused further, time). |
| Joint-mapping test (Stage 1) | **executed successfully** -- all 6 action dims map to the correct, expected arm joint; cross-coupling on other joints stays small (gravity-coupling, not a mapping bug) |
| Gripper/contact test (Stage 2) | **executed, section 1d issue now fixed and verified** -- contact sensors read correctly; the gripper joint no longer deadlocks and now moves smoothly and proportionally toward commanded open/close targets (verified over 60 steps in free space). See section 1d for the full fix (importer bugs + architecture change from PhysX mimic joints to 6 independently-driven joints) and its one residual note (empty-space fingertip self-interference past ~50% closed). |
| Scripted pick-and-place test | **implemented, not executed** (time) |
| Reward-scale diagnostic (Stage 1) | **executed successfully**, twice (before/after the section 1b/1c fixes) -- confirmed both the original imbalance and the fix |
| Reward-scale diagnostic (Stages 2-4) | **implemented, not executed** (time) |
| Short PPO smoke test (Stage 1, 64 envs, 40 iterations) | **executed successfully** -- completed in 25.4s, no errors/NaN |
| Short PPO smoke test (Stages 2-4) | **not executed** (time; also blocked on the section 1d gripper-actuator issue for Stage 2+ to be meaningful) |
| Full curriculum training run | **not executed** -- explicitly out of scope per spec ("do not immediately launch... before diagnostic metrics show...") |

**Gripper action deadband (fix, applied):** added
`mdp/actions.py::DeadbandBinaryJointPositionActionCfg` -- same as
`BinaryJointPositionActionCfg` but "close" only fires below a configurable
`close_threshold` instead of below exactly 0.0. Used with `close_threshold=-0.5`
(bias toward the safe/open state near a=0, matching a fresh/random policy's
typical output) on the base task, GraspLift, and Curriculum, and with
`close_threshold=+0.5` (bias toward *closed*, i.e. keep holding) on
Transport specifically, since it's the only stage that starts every episode
already holding the object. This is a real improvement (a raw all-zero
action no longer ambiguously straddles the open/close boundary) but **it
turned out not to be the cause of the Transport zero-action velocity
spike** -- see below.

**Transport reset-pose fix (applied and verified): `reset_robot_holding_object`
was placing the object overlapping the gripper's internal mechanism, not
between the fingertip pads -- re-measured via a real simulated grasp,
fixed.** Root-caused via a per-joint-velocity diagnostic: the ~200-400 rad/s
single-joint spikes on `robotiq_85_*_finger_tip_joint` persisted identically
even after fixing the action deadband (which confirmed the commanded target
no longer changes after reset) -- meaning the spike happened *before any
policy action could be responsible for it*, during PhysX resolving whatever
the reset state was. Checked the actual body positions right after reset:
fingertip links sat at z approx 0.01, but the object was placed at z approx
0.132 -- about 13cm higher, up near `robotiq_85_base_link`/`tcp`, not down
at the fingertip pads. The object was spawning inside the gripper's
knuckle/base mechanism.

Fix: new `scripts/solve_holding_pose.py` performs a *real* simulated grasp
on `RB5-PickPlace-GraspLift-JointPos-v0`'s own object/bin/contact-sensor
scene -- drives a real Jacobian-based `DifferentialIKController` down
toward the object in small height increments (not straight to the object's
center in one jump, which was tried first and drove the gripper mechanism
into the bin floor instead), stopping the instant a fingertip contact
sensor fires and debouncing that against a single transient brush, then
closes the gripper and confirms `grasp_state.bilateral_contact` reads True
with real, sustained force on both fingertips before reading back state.
`mdp/events.py`'s `HOLDING_TCP_POS_B`/`_QUAT_B` (renamed
`HOLDING_OBJECT_POS_B`/`_QUAT_B` -- the old name was misleading, it was
never really "the tcp pose") now hold the object's own measured resting
position instead of an assumed tcp-coincident offset;
`HOLDING_GRIPPER_KNUCKLE_POS` now holds the real contact-stopped angle
(approx 0.076 rad -- a small partial closure, not the previous assumed
"fully closed" 0.8, which makes sense once you consider the object is much
wider than the gap left by a small closing angle) instead of an assumed
full closure; and the 5 follower joints now use their own individually
*measured* settled angles (`HOLDING_GRIPPER_FOLLOWER_POS`) rather than
being derived from a fixed multiplier ratio against the primary, since a
contact-stopped follower doesn't obey the free-swinging mimic ratio
(confirmed: one follower settled at 0.335 rad against a multiplier-formula
prediction of 0.076 rad). Result: Transport's zero-action test dropped from
approx 193 rad/s max to 3.4 rad/s, joint drift 1.49 -> 0.14 rad, object
drift 0.58 -> 0.003m.

**Known remaining limitation (same one already flagged before this fix,
narrower now, not new):** `solve_holding_pose.py` also attempted an active
"lift the object off the floor" phase after closing (Stage 3 is nominally
"already grasped *and lifted*"), but the object's height never actually
rose during that attempt even though `bilateral_contact` held -- the
object stayed at floor height, and its reported velocity never damped to
zero even with the arm command frozen (chaotic-looking but bounded,
non-exploding jitter around approx 0.5 m/s, not the interpenetration
signature this whole fix addressed). This reads as a genuine contact-
stability tuning question (grip friction/stiffness margin for a true lift,
not just a floor-supported hold) rather than a structural bug, and was
already an acknowledged simplification in the pre-existing code comment
before this session touched it ("a follow-up improvement would solve a
second IK target higher above the bin and re-measure"). The values now in
`mdp/events.py` represent "grasped, resting at floor height" rather than
"grasped and airborne" -- correctness-wise a strict improvement over the
13cm-interpenetration bug, just not the full "already lifted" semantic the
stage name implies. A real lift-and-verify pass is a good next step before
depending heavily on Transport's early-episode reward shaping.

**Other known gaps**:
- `grasp_pose_orientation_reward`'s `std=0.35` was flagged as likely
  saturated the same way position's was, but not independently
  re-verified with `reward_diagnostic.py`.
- No domain randomization, observation corruption, or object-yaw
  randomization anywhere yet (all correctly disabled per spec for this
  initial pass).

## 8. Full training run (2026-07-28)

With all of section 1's fixes in place and all four stages passing
`smoke_test.py`, ran the first real end-to-end curriculum training pass --
the prior "Short PPO smoke test" entries in section 7 were tiny
verification runs (25-30s), not real training.

**Weight vs. reward continuity decision.** Before training, evaluated
whether checkpoint *weights* could carry from one stage's policy network
into the next (so a stage doesn't relearn earlier behavior from scratch).
Conclusion: no, not without a much larger redesign -- every stage has a
differently-composed observation vector (Reach: 41-dim, no gripper action;
GraspLift/Transport: 53-dim each but with genuinely different term
composition, not just coincidentally-matching size; Curriculum: 58-dim,
different term set again), so a raw checkpoint load into the next stage's
network would feed each input neuron a different physical quantity than it
was trained on. Making all 4 stages share one true observation/action
schema would be close to a re-architecture of the whole curriculum. Decided
(user-confirmed) to rely on **reward continuity instead**: each stage
carries forward the previous stage's key reward terms at a reduced
"guidance" weight so behavior isn't forgotten, same pattern GraspLift
already used for Stage 1's reach-shaping terms. This was already true for
Reach->GraspLift and (comprehensively) for everything->Curriculum; added it
for GraspLift->Transport specifically (`stable_grasp_reward` weight=1.0,
`empty_gripper_close_penalty` weight=-0.2 -- see section 4's updated Stage 3
table for why `bilateral_fingertip_contact_reward`/`continuous_lift_reward`/
`grasp_pose_position_reward` were deliberately *not* copied over, being
redundant with or superseded by Transport's own terms).

**Run**: `scripts/run_curriculum_training.sh` (new) -- runs the 4 stages
sequentially, each independently from scratch (no weight transfer, see
above), stopping the chain if any stage exits non-zero. All default to
`num_envs=2048`, `timesteps=100000` (`agents/skrl_ppo_cfg.yaml`, unchanged
from before this session). Total wall-clock: 15:34 -> 19:40 (~4h6m) on a
single RTX 4090, no crashes, no chain-stopping failures.

| Stage | Duration | `Reward/Total reward (mean)`: first -> last (100-pt smoothed) |
|---|---|---|
| Reach | ~38 min | 0.02 -> 3.28 |
| GraspLift | ~62 min | 1.00 -> 34.6 |
| Transport | ~62 min | -17.7 -> -1.77 |
| Curriculum | ~84 min | 1.69 -> 56.1 |

Read from each run's tensorboard event file
(`logs/skrl/rb5_pick_place/<timestamp>_ppo_torch/`) via
`tensorboard.backend.event_processing.event_accumulator`, not eyeballed off
the dashboard. Reach/GraspLift/Curriculum show clean, large improvements;
Curriculum's `Reward/Total reward (min)` also rose to 54.7 by the end
(nearly matching mean/max), suggesting most of the 2048 parallel envs
converged to consistently good behavior, not just a lucky subset. Transport
improved on average but its min stayed around -30 throughout, meaning some
fraction of envs are still failing hard (likely dropping the object) even
at the end of training -- plausibly related to section 1d's still-open
"never verified a true mid-air lift" limitation in the Transport holding
reset. **Success-rate metrics (`reach_success`, `grasp_lift_success`, etc.)
are not logged to tensorboard by skrl's runner** -- only reward curves;
checking actual success rate would need a dedicated eval pass (e.g. via
`play.py` with the info dict, like `sanity_test.py` already does).

Checkpoints: `logs/skrl/rb5_pick_place/<timestamp>_ppo_torch/checkpoints/best_agent.pt`
for each of the 4 runs (timestamps `2026-07-28_15-34-45` / `16-13-21` /
`17-14-48` / `18-17-00` for Reach/GraspLift/Transport/Curriculum
respectively). Two much older run directories under the same
`rb5_pick_place` log root (`2026-07-27_12-44-22`, `2026-07-27_19-51-12`)
are from the **legacy pre-curriculum single-shot task**
(`RB5-PickPlace-JointPos-v0`) -- not related to this curriculum and not
usable as a starting point for it.

**Playback**: `scripts/play.py --task RB5-PickPlace-Curriculum-JointPos-Play-v0
--checkpoint <best_agent.pt> --num_envs 1 --real-time` runs the trained
Curriculum policy live in the Isaac Sim GUI (needs a real X display, e.g.
`DISPLAY=:1` on this machine, not truly headless). Also added
`scripts/play_spawn_and_wait.py` -- same thing, but pauses right after
spawning/resetting the scene (robot held at its reset pose via a zero/hold
action) and only starts running the policy once an external trigger file
appears (`--trigger_file <path>`, polled and consumed on sight) -- useful
for spawning a demo ahead of time and starting playback on cue without
re-launching Isaac Sim.

**Known limitation observed live**: the Curriculum policy's episodes all
start from the *shared pre-grasp pose* (`robots/rb5_850e.py`'s
`init_state`), not a neutral/extended "home" pose -- this is intentional
(Stage 1/Reach owns the far-to-pre-grasp approach; Stages 2-4 are trained
only from pre-grasp onward, see section 8's weight-continuity discussion
for why they can't just inherit Reach's reach-from-far behavior). A
visually "complete" demo (extended arm -> reach -> grasp -> place) would
need a policy hand-off script: run Reach's policy against manually-computed
Reach-shaped observations (all of Reach's `mdp/observations.py` functions
are plain functions of `env`, so this is possible without a second gym env
instance -- confirmed by inspecting `ee_to_pregrasp_vector`,
`grasp_orientation_error` and `mdp.last_action`'s implementations) until
its reach-success condition is met, then switch to feeding the Curriculum
env's own observations into the Curriculum policy for the rest. Scoped but
**not implemented yet** -- next session's starting point if wanted.
