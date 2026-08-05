#!/usr/bin/env bash
# Retrains GraspLift then Curriculum after two changes documented in
# PICK_PLACE_COMPLETION_REPORT.md:
#   1. Both stages now start from FAR_START_JOINT_POS (not already at the
#      pre-grasp pose) -- invalidates the old checkpoints' reset assumptions.
#   2. Reward reweighting to break the "grasp and freeze, never lift" local
#      optimum found via evaluate_policy.py (100/100 Curriculum episodes
#      never lifted despite high reward) -- see section E.1.
# Reach and Transport are NOT retrained here: Reach's checkpoint is
# unaffected by these changes, and Transport's reset is separately still
# broken (see report section C) -- retraining it now would just burn GPU
# hours on a stage that can't produce a meaningful checkpoint yet.
#
# Each run uses the `skrl_ppo_cfg_{grasplift,curriculum}.yaml` agent config
# variant (same hyperparameters as skrl_ppo_cfg.yaml, only
# `experiment.experiment_name` differs) via `--agent skrl_ppo_cfg_entry_point`,
# so each run's TensorBoard/checkpoint folder is
# `<date>_<time>_ppo_torch_PPO(GraspLift)` / `..._PPO(Curriculum)` instead of
# an undistinguishable bare timestamp.
#
# Usage:
#   nohup ./run_retrain_grasplift_curriculum.sh > /path/to/retrain.log 2>&1 &
source /home/hh/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab
unset PYTHONPATH

cd /home/hh/asl_ws/vla_project/IsaacLab

STAGES=(
  "RB5-PickPlace-GraspLift-JointPos-v0"
  "RB5-PickPlace-Curriculum-JointPos-v0"
)

for task in "${STAGES[@]}"; do
    echo "=================================================================="
    echo "[$(date -Iseconds)] Starting training: ${task}"
    echo "=================================================================="
    ./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py \
        --task "${task}" --num_envs 2048 --headless \
        --agent skrl_ppo_cfg_entry_point
    status=$?
    if [ $status -ne 0 ]; then
        echo "[$(date -Iseconds)] Training FAILED for ${task} (exit ${status}) -- stopping chain."
        exit $status
    fi
    echo "[$(date -Iseconds)] Finished training: ${task}"
done

echo "[$(date -Iseconds)] GraspLift + Curriculum retraining finished successfully."
