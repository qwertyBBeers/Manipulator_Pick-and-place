#!/usr/bin/env python3
"""Convert a depth image into PointCloud2 for MoveIt OctoMap updates."""

from __future__ import annotations
from typing import Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

DEFAULT_FX = 390.0
DEFAULT_FY = 390.0
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("points_topic", "/camera/depth/points")
        self.declare_parameter("pointcloud_frame", "camera_depth_points_frame")
        self.declare_parameter("stride", 4)
        self.declare_parameter("max_range", 1.2)
        self.declare_parameter("min_range", 0.05)
        self.declare_parameter("fallback_fx", DEFAULT_FX)
        self.declare_parameter("fallback_fy", DEFAULT_FY)
        self.declare_parameter("fallback_cx", DEFAULT_CX)
        self.declare_parameter("fallback_cy", DEFAULT_CY)

        self.camera_info: CameraInfo | None = None
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self._depth_cb,
            10,
        )
        self.points_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("points_topic").value,
            10,
        )
        self.get_logger().info(
            "DepthToPointCloud publishing "
            f"{self.get_parameter('points_topic').value} from "
            f"{self.get_parameter('depth_topic').value}"
        )

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def _intrinsics(self, image: Image) -> Tuple[float, float, float, float]:
        if self.camera_info is not None:
            k = self.camera_info.k
            if k[0] > 0.0 and k[4] > 0.0:
                return float(k[0]), float(k[4]), float(k[2]), float(k[5])
        return (
            float(self.get_parameter("fallback_fx").value),
            float(self.get_parameter("fallback_fy").value),
            float(self.get_parameter("fallback_cx").value),
            float(self.get_parameter("fallback_cy").value),
        )

    def _depth_array(self, image: Image) -> np.ndarray:
        if image.encoding in ("32FC1", "32FC"):
            dtype = np.float32
            scale = 1.0
        elif image.encoding in ("16UC1", "mono16"):
            dtype = np.uint16
            scale = 0.001
        else:
            raise ValueError(f"Unsupported depth encoding: {image.encoding}")

        raw = np.frombuffer(image.data, dtype=dtype)
        if image.step:
            row_items = image.step // np.dtype(dtype).itemsize
            raw = raw.reshape((image.height, row_items))[:, : image.width]
        else:
            raw = raw.reshape((image.height, image.width))
        return raw.astype(np.float32, copy=False) * scale

    def _depth_cb(self, image: Image):
        try:
            depth = self._depth_array(image)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            return

        stride = max(1, int(self.get_parameter("stride").value))
        min_range = float(self.get_parameter("min_range").value)
        max_range = float(self.get_parameter("max_range").value)
        fx, fy, cx, cy = self._intrinsics(image)

        sampled = depth[::stride, ::stride]
        v_idx, u_idx = np.indices(sampled.shape)
        u = (u_idx * stride).astype(np.float32)
        v = (v_idx * stride).astype(np.float32)
        z = sampled

        valid = np.isfinite(z) & (z > min_range) & (z < max_range)
        if not np.any(valid):
            return

        z = z[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        points = np.column_stack((x, y, z)).astype(np.float32, copy=False)

        cloud = PointCloud2()
        cloud.header = image.header
        pointcloud_frame = str(self.get_parameter("pointcloud_frame").value)
        if pointcloud_frame:
            cloud.header.frame_id = pointcloud_frame
        elif not cloud.header.frame_id:
            cloud.header.frame_id = "camera_depth_optical_frame"
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.tobytes()
        self.points_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
