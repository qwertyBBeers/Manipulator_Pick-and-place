# Deprecated / superseded files

Moved here (not deleted — this repo's `.git` directory turned out to be
empty/non-functional, so a normal `git rm` would not have been recoverable
via history the way it usually is) during the `rb5_isaaclab` package
addition and cleanup pass. Each was confirmed unreferenced by any launch
file, `setup.py` entry point, or other source file before moving.

- `rb5_binpicking/lab_envs/` — the old, entirely-commented-out "Phase 2"
  IsaacLab environment skeleton. Superseded by the real, working
  `../rb5_isaaclab/` package.
- `rb5_binpicking/scripts/action_adapter.py`,
  `rb5_binpicking/config/rb5_action_space.yaml` — unwired Phase-2
  scaffolding for a future GR00T/VLA action adapter (see `README.md`'s
  roadmap section); never imported or launched by anything.
- `rb5_isaac/scripts/01_convert_urdf_to_usd.py`,
  `rb5_isaac/scripts/02_isaac_rb5_scene.py`,
  `rb5_isaac/launch/moveit_isaac.launch.py` — an earlier iteration of the
  Isaac Sim scene/launch setup, superseded by
  `rb5_binpicking/scripts/binpicking_scene.py` and
  `rb5_binpicking/launch/binpicking.launch.py`.
