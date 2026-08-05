# Checkpoints

Only the best-validated policy is committed here (others are left in the
IsaacLab log tree, which isn't part of this repo).

## `reach_grasp_best_agent.pt`

- Task: `RB5-PickPlace-ReachGrasp-JointPos-v0`
- Run: `2026-08-05_16-28-09_ppo_torch_PPO(ReachGrasp)`, 150000 timesteps, `num_envs=8192`
- Config: `robots/rb5_850e.py` `ARM_DAMPING=1000.0` (original) + PD gain
  randomization only (`reach_grasp_env_cfg.py`'s `USE_PD_GAIN_RANDOMIZATION=True`,
  all other ablation flags `False`) -- see `README2.md` §9 for how this was
  arrived at (a damping-increase variant tested individually well but made
  things worse when combined with the randomization).
- Result: total reward 43.98 (peak 50.08), bilateral fingertip contact and
  stable-grasp reward both firing consistently, orientation reward stable
  at ~0.25 (no collapse), object-drop penalty at 0.

Play it back:
```bash
conda activate isaaclab && unset PYTHONPATH
cd <IsaacLab repo>
./isaaclab.sh -p <path>/rb5_isaaclab/scripts/play.py \
  --task RB5-PickPlace-ReachGrasp-JointPos-Play-v0 \
  --checkpoint <path>/rb5_isaaclab/checkpoints/reach_grasp_best_agent.pt
```
