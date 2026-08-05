#!/usr/bin/env python3
"""Convert recorded RB5-850E episodes into a LeRobotDataset.

No hardware/teleop is involved (unlike `lerobot-record`) -- this follows the
same pattern as lerobot's own `examples/port_datasets/port_droid.py`:
LeRobotDataset.create() -> add_frame() per timestep -> save_episode() per
episode -> finalize().

Run inside the `lerobot` conda env:
    conda activate lerobot
    python convert_isaac_episodes_to_lerobot.py --repo-id <hf_user>/rb5_pick_place --fps 30

`iter_episodes` reads the on-disk layout written by
../ros_bridge/episode_logger.py: `<source_dir>/episode_XXXX/{state.npz,
images_color.npy, task.txt}`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot_robot_rb5.config_rb5 import DEFAULT_JOINT_NAMES

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        # 6 arm joints + 1 gripper
        "shape": (len(DEFAULT_JOINT_NAMES) + 1,),
        "names": [*DEFAULT_JOINT_NAMES, "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (len(DEFAULT_JOINT_NAMES) + 1,),
        "names": [*DEFAULT_JOINT_NAMES, "gripper"],
    },
    "observation.images.color": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}


@dataclass
class Frame:
    state: np.ndarray  # (7,) float32: 6 joints + gripper, current observed position
    action: np.ndarray  # (7,) float32: 6 joints + gripper, commanded target
    image_color: np.ndarray  # (480, 640, 3) uint8


@dataclass
class Episode:
    frames: list[Frame]
    task: str  # language instruction, e.g. "pick up the cube and place it in the bin"


def iter_episodes(source_dir: Path) -> Iterator[Episode]:
    """Yield episodes recorded by ../ros_bridge/episode_logger.py."""
    for episode_dir in sorted(source_dir.glob("episode_*")):
        npz = np.load(episode_dir / "state.npz")
        images = np.load(episode_dir / "images_color.npy")
        task = (episode_dir / "task.txt").read_text().strip()
        frames = [
            Frame(state=npz["state"][i], action=npz["action"][i], image_color=images[i])
            for i in range(len(npz["state"]))
        ]
        yield Episode(frames=frames, task=task)


def build_dataset(episodes: Iterator[Episode], repo_id: str, fps: int, root: Path | None = None) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=root,
        robot_type="rb5_850e",
        use_videos=True,
    )
    for episode in episodes:
        for frame in episode.frames:
            dataset.add_frame(
                {
                    "observation.state": frame.state,
                    "action": frame.action,
                    "observation.images.color": frame.image_color,
                    "task": episode.task,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory of recorded episodes")
    parser.add_argument("--repo-id", required=True, help="e.g. <hf_user>/rb5_pick_place")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--root", type=Path, default=None, help="Local output dir; defaults to HF cache")
    args = parser.parse_args()

    build_dataset(iter_episodes(args.source_dir), repo_id=args.repo_id, fps=args.fps, root=args.root)


if __name__ == "__main__":
    main()
