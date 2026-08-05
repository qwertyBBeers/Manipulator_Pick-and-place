#!/usr/bin/env bash
# Converts rb5_850e_robotiq_mimic.urdf -> USD via convert_robot_to_usd.py
# (a direct omni.kit.commands import, NOT IsaacLab's convert_urdf.py CLI --
# see that script's module docstring for why: IsaacLab's UrdfConverterCfg
# wrapper has an inverted default for mimic-joint preservation, verified
# against Isaac Sim's own importer test suite).
#
# Usage:
#   ./convert_robot_to_usd.sh [--gui]
#
# Requires the `isaaclab` conda environment and an IsaacLab source checkout
# (defaults to /home/hh/asl_ws/vla_project/IsaacLab; override with
# ISAACLAB_REPO=/path/to/IsaacLab).
set -euo pipefail

ISAACLAB_REPO="${ISAACLAB_REPO:-/home/hh/asl_ws/vla_project/IsaacLab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${ISAACLAB_REPO}" ]]; then
  echo "[ERROR] IsaacLab repo not found at ${ISAACLAB_REPO} (set ISAACLAB_REPO=...)" >&2
  exit 1
fi

"${ISAACLAB_REPO}/isaaclab.sh" -p "${SCRIPT_DIR}/convert_robot_to_usd.py" "$@"
