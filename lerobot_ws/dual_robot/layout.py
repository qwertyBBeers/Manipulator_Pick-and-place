"""Shared world layout for the two-robot relay demo.

Imported by BOTH dual_binpicking_scene.py (runs under Isaac Sim's python) and
relay_pick_place.py (runs under ROS2 Humble's python). Previously each file
recomputed B_OFFSET/HANDOFF_WORLD from bin_geometry.yaml on its own, which
meant changing the layout in one place silently desynced the other -- the
controller would aim at a handoff point the scene hadn't put a tray at.
Everything positional lives here now.

All values are plain tuples (no numpy) so both interpreters agree.

Frames: robot A's base is the world origin, so "world" == "robot A local".
Robot B's base sits at B_OFFSET, so a world point P is (P - B_OFFSET) in
robot B's local frame -- which is what B's move_group plans in.
"""

import os

import yaml


def _find_bin_geometry_yaml() -> str:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
        if not prefix:
            continue
        candidate = os.path.join(prefix, "share", "rb5_binpicking", "config", "bin_geometry.yaml")
        if os.path.isfile(candidate):
            return candidate
    candidate = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "rb5_binpicking", "config", "bin_geometry.yaml")
    )
    return candidate if os.path.isfile(candidate) else ""


def load_bin_geometry():
    """(src_center, src_inner_size, wall_thickness, dest_center, dest_inner_size)."""
    path = _find_bin_geometry_yaml()
    if not path:
        raise RuntimeError("bin_geometry.yaml not found -- source the Manipulator install first")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    src, dst = data["source_bin"], data["destination_bin"]
    return (
        tuple(float(v) for v in src["center"]),
        tuple(float(v) for v in src["inner_size"]),
        float(src["wall_thickness"]),
        tuple(float(v) for v in dst["center"]),
        tuple(float(v) for v in dst["inner_size"]),
    )


(
    (SRC_X, SRC_Y, SRC_Z), (SRC_W, SRC_D, SRC_H), WALL_T,
    (DEST_X, DEST_Y, DEST_Z), (DEST_W, DEST_D, DEST_H),
) = load_bin_geometry()

# Robot B's base, in world. Direction is the original "B-local source lands on
# A-local dest" vector; magnitude was opened up to ~1m of base-to-base
# separation because at the original ~0.5m the two arms visually overlapped.
_B_DIR = (-0.23, 0.44, 0.0)
_B_DIR_LEN = 0.496
# 1.0 -> 0.8. The handoff tray is the midpoint of the two bins, so base
# separation sets how far each arm has to reach for it: at 1.0m that was 0.67m,
# vs the 0.51m robot A uses for its own bin. Robot A grasped reliably at 0.51m
# while robot B kept missing the identical relative geometry at 0.67m -- with
# the tool measured at a deterministic 3.2mm and 0.00deg off, so it was neither
# IK variance nor tilt, just the arm being near the edge of its useful envelope.
# 0.8m puts the handoff at ~0.60m for both arms. Separation only has to stay
# large enough that the two arms do not visually overlap (0.5m was too close).
BASE_SEPARATION_M = 0.8
B_OFFSET = tuple(c * (BASE_SEPARATION_M / _B_DIR_LEN) for c in _B_DIR)

# Where each robot's own bin ends up in world coordinates. Each robot reaches
# its own bin using the plain bin_geometry.yaml numbers in its local frame
# (0.51,0) / (0.28,0.44) -- the exact reach problem the single-robot demo
# already proved solvable.
SOURCE_WORLD = (SRC_X, SRC_Y, 0.0)                                   # robot A picks from here
DEST_WORLD = tuple(B_OFFSET[i] + (DEST_X, DEST_Y, 0.0)[i] for i in range(3))  # robot B places here

# Handoff placement is set by REACH, not by splitting the distance between the
# bins. The midpoint of the two bins put it 0.612m from A and 0.596m from B,
# and reach turned out to be what actually governs grasp reliability here:
# picks at 0.38-0.50m succeed, picks at 0.56m failed four attempts in a row,
# and B missing repeatedly at the handoff was the same effect all along -- with
# the tool measured arriving a deterministic ~3mm and 0.00deg off, so it is the
# arm being extended, not an accuracy problem. Being inside the 850mm nominal
# reach is not the same as being in the part of the envelope where the drives
# hold the tool steady enough to close on a 42mm cube.
#
# So put the handoff on the intersection of two circles of HANDOFF_REACH_M
# about the two bases: both arms then work at exactly the distance A uses for
# its own bin, which is the one reach this demo has repeatedly proven.
HANDOFF_REACH_M = 0.51


def _equidistant_point(a, b, radius):
    """The point at `radius` from both `a` and `b`, on the +x side of their axis."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    span = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    if span > 2.0 * radius:
        raise ValueError(
            f"bases are {span:.3f}m apart; no point is {radius}m from both "
            f"(need base separation <= {2 * radius:.3f}m)"
        )
    mid = ((ax + bx) / 2.0, (ay + by) / 2.0)
    half_chord = (radius ** 2 - (span / 2.0) ** 2) ** 0.5
    ux, uy = (bx - ax) / span, (by - ay) / span
    px, py = uy, -ux                      # perpendicular to the base-to-base axis
    candidates = [
        (mid[0] + s * half_chord * px, mid[1] + s * half_chord * py) for s in (1.0, -1.0)
    ]
    # Both arms work on the +x side; the mirrored solution sits behind them.
    best = max(candidates, key=lambda c: c[0])
    return (best[0], best[1], 0.0)


HANDOFF_WORLD = _equidistant_point(ROBOT_BASES_A := (0.0, 0.0, 0.0), B_OFFSET, HANDOFF_REACH_M)

# Same point expressed in each robot's own planning frame.
HANDOFF_LOCAL_A = HANDOFF_WORLD                                       # A's base is the origin
HANDOFF_LOCAL_B = tuple(HANDOFF_WORLD[i] - B_OFFSET[i] for i in range(3))

# Where each robot's base sits in world. "robot A local" == world.
ROBOT_BASES = {"robot_a": (0.0, 0.0, 0.0), "robot_b": B_OFFSET}

# The three physical trays: (id, world center (cx, cy, floor_z), inner
# (width, depth, height), wall_thickness). dual_binpicking_scene.py builds the
# Isaac FixedCuboid walls from this and dual_scene_setup.py builds the matching
# MoveIt collision objects from it, so what the planner avoids and what the
# arm can actually hit are the same list -- previously the planning scene came
# from bin_geometry.yaml directly and knew nothing about the handoff tray.
TRAYS = [
    ("source_bin",   SOURCE_WORLD,  (SRC_W,  SRC_D,  SRC_H),  WALL_T),
    ("handoff_tray", HANDOFF_WORLD, (DEST_W, DEST_D, DEST_H), WALL_T),
    ("dest_bin",     DEST_WORLD,    (DEST_W, DEST_D, DEST_H), WALL_T),
]


def to_local(world_xyz, ns: str):
    """A world point in `ns`'s own planning (link0) frame."""
    base = ROBOT_BASES[ns]
    return tuple(world_xyz[i] - base[i] for i in range(3))
