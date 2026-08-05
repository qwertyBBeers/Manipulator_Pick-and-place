#!/usr/bin/env python3
"""Action adapter for RB5 bin-picking demonstration policies.

Initial action format:
    [dx, dy, dz, gripper]

Translation components are normalized to [-1, 1] and scaled by 3 cm. The
gripper component is interpreted as a binary open/close command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import PoseStamped


@dataclass
class RB5ActionAdapter:
    """Convert between normalized 4D policy actions and EEF pose targets."""

    translation_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.03, 0.03], dtype=float)
    )
    workspace_min: np.ndarray = field(
        default_factory=lambda: np.array([0.25, -0.45, 0.05], dtype=float)
    )
    workspace_max: np.ndarray = field(
        default_factory=lambda: np.array([0.85, 0.45, 0.70], dtype=float)
    )
    open_value: float = 0.085
    close_value: float = 0.0
    threshold: float = 0.5

    def apply_4d_action(
        self,
        current_pose: PoseStamped,
        action: np.ndarray,
    ) -> tuple[PoseStamped, float]:
        """Apply a normalized 4D action to the current EEF pose.

        Args:
            current_pose: Current end-effector pose.
            action: Array-like [dx, dy, dz, gripper]. Translation values are
                clipped to [-1, 1].

        Returns:
            A tuple of (target_pose, gripper_value). The target pose preserves
            the input orientation.
        """
        action = self._as_action(action)
        delta = action[:3] * self.translation_scale

        target_pose = PoseStamped()
        target_pose.header = current_pose.header
        target_pose.pose.orientation = current_pose.pose.orientation

        current_xyz = np.array(
            [
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z,
            ],
            dtype=float,
        )
        target_xyz = np.clip(
            current_xyz + delta,
            self.workspace_min,
            self.workspace_max,
        )
        target_pose.pose.position.x = float(target_xyz[0])
        target_pose.pose.position.y = float(target_xyz[1])
        target_pose.pose.position.z = float(target_xyz[2])

        gripper_value = (
            self.close_value if float(action[3]) > self.threshold else self.open_value
        )
        return target_pose, gripper_value

    def delta_from_poses_4d(
        self,
        pose_now: PoseStamped,
        pose_next: PoseStamped,
        gripper_value: float,
    ) -> np.ndarray:
        """Compute the normalized 4D action between two EEF poses."""
        now = np.array(
            [
                pose_now.pose.position.x,
                pose_now.pose.position.y,
                pose_now.pose.position.z,
            ],
            dtype=float,
        )
        nxt = np.array(
            [
                pose_next.pose.position.x,
                pose_next.pose.position.y,
                pose_next.pose.position.z,
            ],
            dtype=float,
        )

        delta_norm = (nxt - now) / self.translation_scale
        delta_norm = np.clip(delta_norm, -1.0, 1.0)

        gripper_action = 1.0 if gripper_value <= self.close_value else 0.0
        return np.array(
            [delta_norm[0], delta_norm[1], delta_norm[2], gripper_action],
            dtype=float,
        )

    @staticmethod
    def _as_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action.shape}")
        action = action.copy()
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        return action
