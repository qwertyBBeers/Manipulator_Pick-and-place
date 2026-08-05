#!/usr/bin/env bash
# Runs the 4-stage PPO curriculum sequentially: Reach -> GraspLift ->
# Transport -> Curriculum. Each stage trains independently from scratch
# (see CURRICULUM_REPORT.md for why checkpoint weight-transfer between
# stages isn't meaningful here -- each stage's observation composition
# differs) at this repo's own configured defaults (num_envs=2048,
# timesteps=100000 per skrl_ppo_cfg.yaml). Reward continuity across stages
# (not weight continuity) is what's carrying "don't forget the previous
# stage" -- see grasp_lift_env_cfg.py's and transport_env_cfg.py's
# RewardsCfg for the carried-forward guidance terms.
#
# Stops the chain (does not proceed to the next stage) if any stage exits
# non-zero, so a crashed early stage doesn't silently burn GPU-hours on
# later stages built on top of a broken one.
#
# Usage:
#   nohup ./run_curriculum_training.sh > /path/to/curriculum_training.log 2>&1 &
source /home/hh/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab
unset PYTHONPATH

cd /home/hh/asl_ws/vla_project/IsaacLab

STAGES=(
  "RB5-PickPlace-Reach-JointPos-v0"
  "RB5-PickPlace-GraspLift-JointPos-v0"
  "RB5-PickPlace-Transport-JointPos-v0"
  "RB5-PickPlace-Curriculum-JointPos-v0"
)

for task in "${STAGES[@]}"; do
    echo "=================================================================="
    echo "[$(date -Iseconds)] Starting training: ${task}"
    echo "=================================================================="
    ./isaaclab.sh -p /home/hh/asl_ws/Manipulator/rb5_isaaclab/scripts/train.py \
        --task "${task}" --num_envs 2048 --headless
    status=$?
    if [ $status -ne 0 ]; then
        echo "[$(date -Iseconds)] Training FAILED for ${task} (exit ${status}) -- stopping chain."
        exit $status
    fi
    echo "[$(date -Iseconds)] Finished training: ${task}"
done

echo "[$(date -Iseconds)] All 4 curriculum stages finished successfully."
