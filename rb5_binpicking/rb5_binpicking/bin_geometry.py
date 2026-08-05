"""Bin geometry loader — single source of truth is config/bin_geometry.yaml.

Used by both the MoveIt pick-place expert (moveit_pick_place.py) and the
scene collision-object publisher (scripts/scene_setup.py) so MoveIt's
planning-scene geometry always matches config/bin_geometry.yaml.
binpicking_scene.py (which runs inside Isaac Sim's separate Python
interpreter) intentionally does NOT import this module — see the plain YAML
reader duplicated there — but it reads the exact same YAML file, so all three
consumers stay in sync by construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

DEFAULT_FRAME = "link0"


@dataclass(frozen=True)
class BinSpec:
    """One bin's geometry, as loaded from bin_geometry.yaml."""

    frame_id: str
    center: tuple  # (x, y, floor_z)
    inner_size: tuple  # (width, depth, height)
    wall_thickness: float


def bin_geometry_yaml_path() -> str:
    """Resolve config/bin_geometry.yaml: prefer the installed share path, fall
    back to the source-tree path for an uninstalled development checkout."""
    try:
        installed = os.path.join(
            get_package_share_directory("rb5_binpicking"), "config", "bin_geometry.yaml"
        )
        if os.path.isfile(installed):
            return installed
    except PackageNotFoundError:
        pass
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config", "bin_geometry.yaml")
    )


def _parse_bin_spec(data: dict, key: str) -> BinSpec:
    try:
        section = data[key]
        center = tuple(float(v) for v in section["center"])
        inner_size = tuple(float(v) for v in section["inner_size"])
        if len(center) != 3 or len(inner_size) != 3:
            raise ValueError("center and inner_size must each have 3 elements")
        return BinSpec(
            frame_id=str(section["frame_id"]),
            center=center,
            inner_size=inner_size,
            wall_thickness=float(section["wall_thickness"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"bin_geometry.yaml: section '{key}' is missing or malformed ({exc}). "
            "Refusing to fall back to unrelated hardcoded defaults."
        ) from exc


def load_bin_geometry(path: str | None = None) -> tuple[BinSpec, BinSpec]:
    """Load (source_bin, destination_bin) from bin_geometry.yaml.

    Raises RuntimeError with a clear message if the file is missing or
    malformed — callers must not silently substitute unrelated defaults.
    """
    path = path or bin_geometry_yaml_path()
    if not os.path.isfile(path):
        raise RuntimeError(
            f"bin_geometry.yaml not found at {path}. This is the single source "
            "of truth for bin geometry; there is no hardcoded fallback."
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return _parse_bin_spec(data, "source_bin"), _parse_bin_spec(data, "destination_bin")


def make_box_collision_object(
    object_id: str,
    xyz: Iterable[float],
    dims: Iterable[float],
    frame_id: str = DEFAULT_FRAME,
) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.operation = CollisionObject.ADD

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(v) for v in dims]

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in xyz]
    pose.orientation.w = 1.0

    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    return obj


WALL_SAFETY_MARGIN = 0.0  # m, see make_bin_collision_objects() docstring.
# Was 0.02 (README2.md §7.17) but move_group logs (§7.21) caught it causing
# a *worse* failure mode than the one it was meant to fix: the arm's forearm
# (link2) legitimately needs to pass close to source_bin_wall_xn to reach
# into the bin, and padding that wall outward turned an already-tight-but-
# valid resting/approach configuration into a "Start state appears to be in
# collision" -- which blocks planning outright, rather than just being a
# transient graze during transit. Reverted to 0 (true wall size, no padding)
# until a more targeted fix (e.g. constraining the approach path instead of
# inflating the obstacle) is worked out.


def make_bin_collision_objects(
    prefix: str, spec: BinSpec, safety_margin: float = WALL_SAFETY_MARGIN
) -> list[CollisionObject]:
    """Five-piece open-top bin: floor + 4 walls, interior left free.

    `safety_margin` grows each wall OUTWARD only (the interior-facing
    surface stays exactly at the true wall position, so reachable interior
    space is unchanged) -- observed in practice (README2.md §7.17) that the
    elbow/forearm links (not the gripper) repeatedly grazed bin walls with
    near-zero clearance during free-space transit, which is consistent with
    planning right up to the true wall surface with no buffer. This gives
    the planner a small buffer to actually stay clear of, instead of only
    being required to not overlap the wall's exact geometry.
    """
    cx, cy, base_z = spec.center
    width, depth, height = spec.inner_size
    wall_t = spec.wall_thickness
    m = safety_margin

    specs = [
        ("floor", [cx, cy, base_z + wall_t / 2.0], [width, depth, wall_t]),
        (
            "wall_xp",
            [cx + width / 2.0 - wall_t / 2.0 + m / 2.0, cy, base_z + height / 2.0],
            [wall_t + m, depth, height],
        ),
        (
            "wall_xn",
            [cx - width / 2.0 + wall_t / 2.0 - m / 2.0, cy, base_z + height / 2.0],
            [wall_t + m, depth, height],
        ),
        (
            "wall_yp",
            [cx, cy + depth / 2.0 - wall_t / 2.0 + m / 2.0, base_z + height / 2.0],
            [width, wall_t + m, height],
        ),
        (
            "wall_yn",
            [cx, cy - depth / 2.0 + wall_t / 2.0 - m / 2.0, base_z + height / 2.0],
            [width, wall_t + m, height],
        ),
    ]
    return [
        make_box_collision_object(f"{prefix}_{name}", xyz, dims, frame_id=spec.frame_id)
        for name, xyz, dims in specs
    ]


def make_all_bin_collision_objects(
    path: str | None = None,
) -> list[CollisionObject]:
    source_bin, dest_bin = load_bin_geometry(path)
    objects = []
    objects.extend(make_bin_collision_objects("source_bin", source_bin))
    objects.extend(make_bin_collision_objects("dest_bin", dest_bin))
    return objects
