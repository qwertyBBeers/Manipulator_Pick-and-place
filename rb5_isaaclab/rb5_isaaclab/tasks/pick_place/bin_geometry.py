"""Reads the SAME `bin_geometry.yaml` the ROS/MoveIt pipeline uses
(`rb5_binpicking/config/bin_geometry.yaml`), so this IsaacLab task and the
ROS heuristic pipeline never disagree about where the source/destination
bins are -- one YAML, not two hand-copied numbers (Manipulator/README2.md
documents multiple bugs caused by exactly this kind of duplication drifting
out of sync, e.g. §7.22/§7.23).

This is a plain filesystem read (no ROS environment/AMENT_PREFIX_PATH
needed) since `rb5_isaaclab` is not a ROS package and doesn't source a ROS
workspace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

_BIN_GEOMETRY_YAML = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "rb5_binpicking", "config", "bin_geometry.yaml",
    )
)


@dataclass(frozen=True)
class BinSpec:
    center: tuple[float, float, float]
    inner_size: tuple[float, float, float]
    wall_thickness: float


def load_bin_geometry(path: str | None = None) -> tuple[BinSpec, BinSpec]:
    path = path or _BIN_GEOMETRY_YAML
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"bin_geometry.yaml not found at {path} -- rb5_isaaclab expects to be checked "
            "out as a sibling of rb5_binpicking inside the same Manipulator repo."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    src, dst = data["source_bin"], data["destination_bin"]
    return (
        BinSpec(tuple(src["center"]), tuple(src["inner_size"]), float(src["wall_thickness"])),
        BinSpec(tuple(dst["center"]), tuple(dst["inner_size"]), float(dst["wall_thickness"])),
    )


def xy_sample_range(spec: BinSpec, margin: float = 0.03) -> dict[str, tuple[float, float]]:
    """(x, y) sampling half-ranges around `spec.center`, kept `margin` away
    from the inner wall face -- reuses the ROS pipeline's
    `destination_wall_clearance` value (0.03m) as the default margin rather
    than inventing a new constant."""
    cx, cy, _ = spec.center
    w, d, _ = spec.inner_size
    ix = w / 2.0 - spec.wall_thickness - margin
    iy = d / 2.0 - spec.wall_thickness - margin
    if ix <= 0 or iy <= 0:
        raise ValueError(f"bin inner_size {spec.inner_size} too small for margin={margin}")
    return {"x": (cx - ix, cx + ix), "y": (cy - iy, cy + iy)}
