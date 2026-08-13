#!/usr/bin/env python3
"""Convert two-robot relay episodes into a LeRobotDataset for VLA fine-tuning.

A second converter rather than an extension of
convert_isaac_episodes_to_lerobot.py: that one reads the single-robot layout
(`episode_XXXX/images_color.npy`, one camera, every episode a demonstration).
The relay logger writes a different shape -- two cameras as per-frame JPEGs,
and successes and failures in separate trees -- and conflating the two would
make it easy to train on failures by accident.

Input layout (dual_robot/dual_episode_logger.py):
    <root>/relay_inst<N>/{success,failure}/episode_XXXX/
        state.npz          state (T,7) action (T,7) timestamps (T,)
        images_scene/000000.jpg ...     640x480, front overview
        images_wrist/000000.jpg ...     320x240, the acting arm's wrist
        task.txt  meta.json

Defaults to successes only. Failures are recorded deliberately and are useful
(they show what a dropped block looks like), but they are not demonstrations of
the task, so including them takes an explicit --include-failures.

Run inside the `lerobot` conda env:
    conda activate lerobot
    python convert_relay_episodes_to_lerobot.py \\
        --source-root /mnt/hdd/relay_datasets \\
        --repo-id <hf_user>/rb5_relay \\
        --root /mnt/hdd/relay_datasets/lerobot_relay
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets import LeRobotDataset

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
SCENE_SHAPE = (480, 640, 3)
WRIST_SHAPE = (240, 320, 3)

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES) + 1,),
        "names": [*JOINT_NAMES, "gripper"],
    },
    "action": {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES) + 1,),
        "names": [*JOINT_NAMES, "gripper"],
    },
    "observation.images.scene": {
        "dtype": "video",
        "shape": SCENE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": WRIST_SHAPE,
        "names": ["height", "width", "channels"],
    },
}


@dataclass
class Episode:
    path: Path
    task: str
    meta: dict

# episode 에서 state, action, scene image, wrist image 을 yield 하는 generator
def iter_episodes(source_root: Path, include_failures: bool, min_frames: int) -> Iterator[Episode]:
    trees = ["success"] + (["failure"] if include_failures else [])
    for inst_dir in sorted(source_root.glob("relay_inst*")):
        for tree in trees:
            for episode_dir in sorted((inst_dir / tree).glob("episode_*")):
                meta_path = episode_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                # An episode is only usable if the frame count, the state array
                # and both image directories agree. They can disagree when a
                # run was killed mid-save.
                n = meta.get("frames", 0)
                if n < min_frames:
                    continue
                scene = sorted((episode_dir / "images_scene").glob("*.jpg"))
                wrist = sorted((episode_dir / "images_wrist").glob("*.jpg"))
                if len(scene) != n or len(wrist) != n:
                    print(f"  skipping {episode_dir}: {n} frames in meta, "
                          f"{len(scene)} scene / {len(wrist)} wrist images")
                    continue
                yield Episode(path=episode_dir, task=(episode_dir / "task.txt").read_text().strip(), meta=meta)


def load_frames(episode: Episode):
    npz = np.load(episode.path / "state.npz")
    state, action = npz["state"], npz["action"]
    scene = sorted((episode.path / "images_scene").glob("*.jpg"))
    wrist = sorted((episode.path / "images_wrist").glob("*.jpg"))
    for i in range(len(state)):
        yield (
            state[i].astype(np.float32),
            action[i].astype(np.float32),
            np.asarray(Image.open(scene[i]).convert("RGB")),
            np.asarray(Image.open(wrist[i]).convert("RGB")),
        )


def median_fps(source_root: Path) -> float:
    """Frame rate implied by the recorded timestamps.

    Do not guess this: the frames are paced by the wrist camera, whose rate
    depends on the simulator's real-time factor, so it is neither the camera's
    nominal rate nor a round number. LeRobot uses fps to build the time index,
    and a wrong value silently mislabels how fast the demonstrations are.
    """
    deltas = []
    for npz_path in sorted(source_root.glob("relay_inst*/success/episode_*/state.npz"))[:50]:
        try:
            t = np.load(npz_path)["timestamps"]
        except Exception:
            continue
        if len(t) > 2:
            deltas.append(float(np.median(np.diff(t))))
    return 1.0 / float(np.median(deltas)) if deltas else 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/hdd/relay_datasets"))
    parser.add_argument("--repo-id", required=True, help="e.g. <hf_user>/rb5_relay")
    parser.add_argument("--root", type=Path, default=None, help="Local output dir; defaults to the HF cache")
    parser.add_argument("--fps", type=int, default=None, help="Default: measured from the recorded timestamps")
    parser.add_argument("--include-failures", action="store_true",
                        help="Also convert the failure/ tree (not demonstrations of the task)")
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many episodes")
    args = parser.parse_args()

    fps = args.fps or round(median_fps(args.source_root))
    print(f"fps: {fps}")

    episodes = list(iter_episodes(args.source_root, args.include_failures, args.min_frames))
    if args.limit:
        episodes = episodes[: args.limit]
    n_ok = sum(1 for e in episodes if e.meta.get("success"))
    print(f"{len(episodes)} episodes ({n_ok} success, {len(episodes) - n_ok} failure)")
    if not episodes:
        return

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=FEATURES,
        root=args.root,
        robot_type="rb5_850e",
        use_videos=True,
    )
    for i, episode in enumerate(episodes, 1):
        for state, action, scene_img, wrist_img in load_frames(episode):
            dataset.add_frame({
                "observation.state": state,
                "action": action,
                "observation.images.scene": scene_img,
                "observation.images.wrist": wrist_img,
                "task": episode.task,
            })
        dataset.save_episode()
        print(f"  [{i}/{len(episodes)}] {episode.path.parent.parent.name}/{episode.path.name} "
              f"{episode.meta['frames']} frames")
    dataset.finalize()
    print(f"done -> {args.root or 'HF cache'}")


if __name__ == "__main__":
    main()
