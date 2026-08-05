#!/usr/bin/env python3
"""Publishes the source/destination bin geometry to MoveIt's planning scene
so it shows up in RViz (Motion Planning display) as soon as binpicking.launch.py
starts, without needing to run the pick-place expert first.

/collision_object is not latched, so this keeps republishing at a low rate
for late joiners (move_group's planning scene monitor, RViz).
"""

import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject

from rb5_binpicking.bin_geometry import make_all_bin_collision_objects

REPUBLISH_PERIOD_SEC = 2.0


class SceneSetup(Node):
    def __init__(self):
        super().__init__("scene_setup")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)
        self.objects = make_all_bin_collision_objects()
        self.timer = self.create_timer(REPUBLISH_PERIOD_SEC, self._publish_once)
        self.get_logger().info(
            f"Publishing {len(self.objects)} bin collision objects to /collision_object "
            f"every {REPUBLISH_PERIOD_SEC:.1f}s"
        )

    def _publish_once(self):
        stamp = self.get_clock().now().to_msg()
        for obj in self.objects:
            obj.header.stamp = stamp
            self.collision_pub.publish(obj)


def main(args=None):
    rclpy.init(args=args)
    node = SceneSetup()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
