#!/usr/bin/env python3
"""Publishes this scene's real obstacles into ONE robot's MoveIt planning scene.

Replaces rb5_binpicking/scripts/scene_setup.py for the two-robot relay. That
node was wrong here in two independent ways:

  1. It publishes to the *absolute* topic "/collision_object", while
     move_group's planning scene monitor subscribes to the *relative*
     "collision_object" -- which under PushRosNamespace is
     "/robot_a/collision_object". So nothing it published ever reached either
     move_group: both planners were running against a completely empty world.
  2. Even if it had arrived, it describes the single-robot layout straight out
     of bin_geometry.yaml -- a source bin at (0.51, 0) and a destination bin at
     (0.28, 0.44) in *link0*. In this scene robot B's bin is at (0.28, 0.44) in
     *B's* frame, robot A has no bin there at all, and the handoff tray that
     both arms reach into does not exist in that file.

Here each robot gets the same three physical trays, each expressed in that
robot's own link0 frame, plus a box for the other robot's base pedestal. The
geometry comes from layout.TRAYS, which dual_binpicking_scene.py also builds
the Isaac cuboids from, so the planner's world and the physical world are the
same list by construction.

Run one instance per robot, inside that robot's namespace:
  python3 dual_scene_setup.py --robot robot_a --ros-args -r __ns:=/robot_a
"""

import argparse
import os
import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import ROBOT_BASES, TRAYS, to_local  # noqa: E402

# Both robots plan in their own link0, which each robot's own static
# world->link0 transform makes coincident with that robot's base.
PLANNING_FRAME = "link0"
REPUBLISH_PERIOD_SEC = 2.0

# The other arm's fixed pedestal. Only the base is modelled: it is the one part
# of the other robot whose pose is known without subscribing to its joint
# states, and at 1.0m base separation vs 0.85m reach it is the only part an arm
# could drive into while the other is parked. The moving links are kept apart
# by sequencing instead (relay_pick_place.py runs one arm at a time).
OTHER_BASE_SIZE = (0.30, 0.30, 0.30)

# The ground. Isaac has a real ground plane; MoveIt did not, so nothing stopped
# a plan from sweeping the elbow or the wrist through the floor -- the planner
# reported SUCCESS, the arm hit the actual ground, tracking error blew up and
# the move came back CONTROL_FAILED (-4) or TIMED_OUT (-6). Visible as parts of
# the arm scraping along the floor. This is also why the Cartesian solver kept
# rejecting clean descents: with avoid_collisions=True it had to contort into
# another IK branch (3.6rad of travel) to dodge tray walls it could only
# approach from below, because below was free as far as it knew.
#
# Top face sits GROUND_CLEARANCE below z=0 rather than at it: link0 rests
# exactly on z=0, and a ground box flush with it would put every start state in
# collision, which makes move_group refuse to plan anything at all.
GROUND_SIZE = (4.0, 4.0, 0.02)
GROUND_CLEARANCE = 0.01


def _box(object_id: str, xyz, dims) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = PLANNING_FRAME
    obj.id = object_id
    obj.operation = CollisionObject.ADD

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [float(v) for v in dims]

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.w = 1.0

    obj.primitives = [primitive]
    obj.primitive_poses = [pose]
    return obj


def _tray_objects(object_id: str, center_local, inner_size, wall_t) -> list:
    """Five-piece open-top tray (floor + 4 walls), matching the FixedCuboid
    layout dual_binpicking_scene.py builds from the same TRAYS entry."""
    cx, cy, floor_z = center_local
    w, d, h = inner_size
    specs = [
        ("floor",   [cx, cy, floor_z + wall_t / 2.0],           [w, d, wall_t]),
        ("wall_xp", [cx + w / 2.0 - wall_t / 2.0, cy, floor_z + h / 2.0], [wall_t, d, h]),
        ("wall_xn", [cx - w / 2.0 + wall_t / 2.0, cy, floor_z + h / 2.0], [wall_t, d, h]),
        ("wall_yp", [cx, cy + d / 2.0 - wall_t / 2.0, floor_z + h / 2.0], [w, wall_t, h]),
        ("wall_yn", [cx, cy - d / 2.0 + wall_t / 2.0, floor_z + h / 2.0], [w, wall_t, h]),
    ]
    return [_box(f"{object_id}_{name}", xyz, dims) for name, xyz, dims in specs]


def build_objects(ns: str) -> list:
    objects = []
    for tray_id, center_world, inner_size, wall_t in TRAYS:
        objects.extend(_tray_objects(tray_id, to_local(center_world, ns), inner_size, wall_t))

    other_ns = "robot_b" if ns == "robot_a" else "robot_a"
    ox, oy, _ = to_local(ROBOT_BASES[other_ns], ns)
    objects.append(_box(f"{other_ns}_base", (ox, oy, OTHER_BASE_SIZE[2] / 2.0), OTHER_BASE_SIZE))

    # Ground is flat and level everywhere, so the same local box works for both
    # robots without going through to_local().
    objects.append(_box("ground", (0.0, 0.0, -GROUND_CLEARANCE - GROUND_SIZE[2] / 2.0), GROUND_SIZE))
    return objects


class DualSceneSetup(Node):
    def __init__(self, ns: str):
        super().__init__("dual_scene_setup")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        # Relative name on purpose -- see module docstring point (1).
        self.collision_pub = self.create_publisher(CollisionObject, "collision_object", 10)
        self.objects = build_objects(ns)
        self.timer = self.create_timer(REPUBLISH_PERIOD_SEC, self._publish_once)
        self.get_logger().info(
            f"[{ns}] publishing {len(self.objects)} collision objects to "
            f"{self.collision_pub.topic_name} every {REPUBLISH_PERIOD_SEC:.1f}s: "
            + ", ".join(o.id for o in self.objects)
        )

    def _publish_once(self):
        stamp = self.get_clock().now().to_msg()
        for obj in self.objects:
            obj.header.stamp = stamp
            self.collision_pub.publish(obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=sorted(ROBOT_BASES))
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = DualSceneSetup(args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
