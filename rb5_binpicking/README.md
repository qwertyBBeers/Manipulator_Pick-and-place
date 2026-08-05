# RB5 Bin Picking

> **Note (cleanup pass, see `../README2.md`):** `lab_envs/`, `scripts/action_adapter.py`,
> and `config/rb5_action_space.yaml` mentioned below were unwired Phase-2
> scaffolding and have been moved to `../deprecated/`. The real, working
> IsaacLab RL package is `../rb5_isaaclab/` — see its own README.

`rb5_binpicking` is the bin-picking integration package for the RB5-850E manipulator.
It reuses the RB5 Isaac Sim bridge from `rb5_isaac` and the arm MoveIt2 config from
`rbpodo_moveit_config`, then adds a source bin, destination bin, objects, and a
gripper-mounted D435i-style color/depth camera.

## Package Roles

```text
Manipulator/
├── rbpodo_ros2/
│   ├── rbpodo_description/        # RB robot URDF and meshes
│   ├── rbpodo_moveit_config/      # MoveIt2 SRDF, kinematics, RViz config
│   ├── rbpodo_bringup/            # ros2_control controller config
│   ├── rbpodo_hardware/           # real robot hardware interface
│   └── rbpodo_msgs/               # RB messages/actions/services
├── rb5_isaac/
│   ├── urdf/rb5_with_tools.urdf   # RB5 + Robotiq + RealSense URDF
│   └── rb5_isaac/trajectory_bridge.py
└── rb5_binpicking/
    ├── scripts/binpicking_scene.py
    ├── launch/binpicking.launch.py
    └── lab_envs/rb5_binpicking_env.py
```

`rb5_binpicking` is intentionally a thin integration layer:

- Isaac Sim scene: `scripts/binpicking_scene.py`
- MoveIt2/RViz launch: `launch/binpicking.launch.py`
- Isaac Lab skeleton: `lab_envs/rb5_binpicking_env.py`
- Policy action adapter: `scripts/action_adapter.py`
- Hardcoded MoveIt2 expert: `scripts/moveit_pick_place.py`
- Action-space config: `config/rb5_action_space.yaml`

## ROS Topic Contract

Isaac Sim publishes:

```text
/joint_states
/tf
/clock
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_rect_raw
/camera/depth/camera_info
/binpicking/object_pose
```

Isaac Sim subscribes:

```text
/isaac_joint_commands
/gripper_joint_commands
```

`trajectory_bridge` exposes MoveIt-compatible actions and forwards commands to Isaac Sim:

```text
/joint_trajectory_controller/follow_joint_trajectory
/gripper_controller/gripper_cmd
```

## Build

From the workspace root:

```bash
cd ~/asl_ws/Manipulator
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  rbpodo_description rbpodo_moveit_config rb5_isaac rb5_binpicking
source install/setup.bash
```

Use `--symlink-install` while developing so launch files and scripts update without a
full reinstall. Rebuild after changing Python package metadata or installed data files.

## Run

Terminal 1: start the Isaac Sim bin-picking scene.

```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/Manipulator/install/setup.bash
~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_binpicking/scripts/binpicking_scene.py
```

Wait for:

```text
[OK] ROS2 OmniGraph built
[INFO] D435i color + depth cameras initialized
[OK] Bin picking scene running.
```

Terminal 2: start MoveIt2, RViz2, trajectory bridge, and image viewers.

```bash
source /opt/ros/humble/setup.bash
source ~/asl_ws/Manipulator/install/setup.bash
ros2 launch rb5_binpicking binpicking.launch.py
```

The launch defaults to `use_sim_time:=true`. This is required because Isaac Sim
publishes `/clock` and timestamps `/joint_states` in simulation time.

## RViz and MoveIt Notes

`binpicking.launch.py` uses `rb5_isaac/urdf/rb5_with_tools.urdf` for the mounted
Robotiq gripper and RealSense camera geometry.

- `move_group` and RViz MotionPlanning use a MoveIt-safe copy of that URDF where
  Robotiq finger joints are fixed, so the gripper/camera are included as collision
  bodies without adding extra active planning joints.
- `robot_state_publisher` uses the original full URDF so TF still follows the
  Isaac Sim gripper/camera model.
- `config/rb5_binpicking_with_tools.srdf` keeps the arm planning group as
  `mainpulation: link0 -> tcp` and disables only internal tool self-collisions.

Normal MoveIt startup includes:

```text
[move_group.move_group]: You can start planning now!
[trajectory_bridge]: TrajectoryBridge ready.
```

## Camera Configuration

The scene creates separate color and depth camera prims under `camera_link`:

```python
CAMERA_DEPTH_PATH = _cam_link + "/depth_camera"
CAMERA_COLOR_PATH = _cam_link + "/color_camera"
```

The current orientation is:

```python
RotateY(-90)
RotateZ(90)
```

This makes the camera look along the gripper approach direction and rolls the image to
the corrected upright orientation. If the camera appears wrong after edits, fully close
and restart Isaac Sim; camera prim transforms are created when the scene script starts.

## Quick Checks

Check Isaac Sim publishing joint states:

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
```

Check camera topics:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_rect_raw
ros2 topic echo /camera/depth/camera_info --once
ros2 topic hz /camera/depth/points
```

`depth_to_pointcloud.py` converts `/camera/depth/image_rect_raw` into
`/camera/depth/points`. MoveIt consumes that point cloud through
`config/sensors_3d.yaml` using `PointCloudOctomapUpdater`, giving `move_group` a
live OctoMap-style occupancy view for collision checking while the scripted
expert still uses `/binpicking/object_pose` for the current target.

The point cloud is published in `camera_depth_points_frame`, a child of
`camera_depth_optical_frame`. Its rotation is identical to the optical frame, but
the launch exposes translation offsets for calibration:

```bash
ros2 launch rb5_binpicking binpicking.launch.py \
  pcd_offset_x:=0.035 pcd_offset_y:=0.11 pcd_offset_z:=0.0
```

Use these offsets when the cloud direction is correct but the cloud is shifted in
RViz. The optical-frame convention is `x=right`, `y=down`, `z=forward`, so a cloud
that appears vertically too high usually needs a small positive `pcd_offset_y`,
while distance/depth mismatch is adjusted with `pcd_offset_z`.

Check the current target object pose:

```bash
ros2 topic echo /binpicking/object_pose --once
```

Check MoveIt action bridge:

```bash
ros2 action list | grep joint_trajectory
ros2 action list | grep gripper
```

## Troubleshooting

### RViz opens but MotionPlanning is not usable

Make sure Isaac Sim is already running and `/clock` is being published. Keep
`use_sim_time:=true` unless running without Isaac Sim.

```bash
ros2 topic echo /clock --once
ros2 param get /move_group use_sim_time
ros2 param get /rviz2 use_sim_time
```

### Robot model appears but gripper/camera TF is missing

Confirm `robot_state_publisher` loaded the full URDF:

```bash
ros2 run tf2_ros tf2_echo link6 camera_color_optical_frame
ros2 run tf2_ros tf2_echo link6 camera_depth_optical_frame
```

### Camera image is old or still rotated incorrectly

Close Isaac Sim completely and restart `binpicking_scene.py`. The camera prims are
defined at startup, so changing the script does not affect an already-running scene.

## Demonstration Data Path

The first demonstration interface uses a 4D normalized action:

```text
[dx, dy, dz, gripper]
```

Translation values are normalized to `[-1, 1]` and scaled by 3 cm:

```text
dx_real = dx * 0.03
dy_real = dy * 0.03
dz_real = dz * 0.03
```

The current workspace and gripper thresholds are stored in:

```text
config/rb5_action_space.yaml
```

`scripts/action_adapter.py` provides `RB5ActionAdapter` for converting between
policy actions and target EEF poses. The initial adapter preserves the current EEF
orientation and only controls XYZ translation plus binary gripper open/close.

## Hardcoded Expert

`scripts/moveit_pick_place.py` is a first scripted expert for generating simple
demonstrations. It uses the MoveIt2 `MoveGroup` action directly. The expert first
waits for `/binpicking/object_pose`; when available, it builds dynamic pick
waypoints from the object position. If the topic is not available, it falls back to
the saved-pose sequence. At startup it also publishes source/destination bin walls
to MoveIt's planning scene via `/collision_object`, so MoveIt can avoid planning
through the bins.

Confirmed MoveIt names from `rbpodo_moveit_config`:

```text
planning group: mainpulation
base link:      link0
eef link:       tcp
arm joints:     base shoulder elbow wrist1 wrist2 wrist3
```

Run the full stack first:

```bash
# Terminal 1
source /opt/ros/humble/setup.bash
source ~/asl_ws/Manipulator/install/setup.bash
~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_binpicking/scripts/binpicking_scene.py

# Terminal 2
source /opt/ros/humble/setup.bash
source ~/asl_ws/Manipulator/install/setup.bash
ros2 launch rb5_binpicking binpicking.launch.py
```

Then run the expert:

```bash
# Terminal 3
source /opt/ros/humble/setup.bash
source ~/asl_ws/Manipulator/install/setup.bash
ros2 run rb5_binpicking moveit_pick_place.py
```

Direct execution is also supported:

```bash
python3 ~/asl_ws/Manipulator/rb5_binpicking/scripts/moveit_pick_place.py
```

Dynamic pick behavior:

```text
dynamic_grasp     = object_xyz + [grasp_offset_x, grasp_offset_y, grasp_offset_z]
dynamic_pre_grasp = dynamic_grasp + [0, 0, pre_grasp_lift]
dynamic_lift      = dynamic_grasp + [0, 0, post_grasp_lift]
```

The grasp orientation currently reuses the saved `gripping` orientation. The place
side still uses the saved `second_box` and `end` poses.

Default dynamic pick parameters:

```text
grasp_offset_x: 0.00
grasp_offset_y: 0.00
grasp_offset_z: 0.16
pre_grasp_lift: 0.18
post_grasp_lift: 0.22
position_tolerance: 0.02
orientation_tolerance: 0.20
use_path_orientation_constraint: false
```

Tune them at runtime:

```bash
ros2 run rb5_binpicking moveit_pick_place.py --ros-args \
  -p grasp_offset_x:=0.0 \
  -p grasp_offset_y:=0.0 \
  -p grasp_offset_z:=0.14 \
  -p position_tolerance:=0.03 \
  -p orientation_tolerance:=0.30
```

While carrying the object, the expert keeps the saved `gripping` orientation for
`dynamic_pre_place`, `dynamic_place`, and `dynamic_retreat`. This avoids unnecessary
wrist rotation near the destination bin.

MoveIt constraints are controlled in `scripts/moveit_pick_place.py`:

- `position_tolerance`: final TCP position sphere radius, in meters.
- `orientation_tolerance`: final TCP orientation tolerance, in radians.
- `use_path_orientation_constraint`: optional path-wide TCP orientation lock.

The default now keeps path orientation constraints disabled. This is closer to the
first working behavior and avoids over-constraining OMPL while tuning pick/place
waypoints. If path constraints are re-enabled later and planning fails, loosen
`orientation_tolerance` first.

Planning-scene constraints:

```text
/collision_object
  source_bin_floor, source_bin_wall_xp, source_bin_wall_xn, source_bin_wall_yp, source_bin_wall_yn
  dest_bin_floor,   dest_bin_wall_xp,   dest_bin_wall_xn,   dest_bin_wall_yp,   dest_bin_wall_yn
```

### Commands do not move the robot

Confirm the bridge and Isaac Sim topics are connected:

```bash
ros2 topic hz /isaac_joint_commands
ros2 node list | grep trajectory_bridge
ros2 action info /joint_trajectory_controller/follow_joint_trajectory
```

## Current Scene Contents

- RB5-850E arm imported from `rb5_isaac/urdf/rb5_with_tools.urdf`
- Robotiq 2F-85 gripper
- D435i-style color and depth cameras
- Source bin with procedural objects
- Destination bin
- Target object pose publisher for `/World/Objects/Cube0`
- Optional YCB cracker box USD if found in the Isaac Sim replicator cache

## Next Work Items

- Add camera info publication if downstream perception needs calibrated intrinsics.
- Add object pose/segmentation outputs for perception and grasp planning.
- Add a pick-and-place script that uses MoveIt planning plus gripper actions.
- Expand `lab_envs/rb5_binpicking_env.py` into a full Isaac Lab training environment.
