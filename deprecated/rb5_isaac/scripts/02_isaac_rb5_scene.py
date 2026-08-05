"""
Isaac Sim scene for RB5-850E with gripper + depth camera.

Publishes:   /joint_states, /tf, /clock
             /camera/color/image_raw, /camera/color/camera_info
             /camera/depth/image_rect_raw, /camera/depth/camera_info
Subscribes:  /isaac_joint_commands      (arm — from trajectory_bridge)
             /gripper_joint_commands    (gripper — from trajectory_bridge)

Run:
  source /opt/ros/humble/setup.bash
  source ~/asl_ws/install/setup.bash
  ~/isaacsim/python.sh scripts/02_isaac_rb5_scene.py
"""

import os, sys
import numpy as np

# Derive ROS_PACKAGE_PATH from AMENT_PREFIX_PATH so the URDF importer can
# resolve package:// URIs (e.g. package://rbpodo_description/meshes/...)
if not os.environ.get("ROS_PACKAGE_PATH"):
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    if ament:
        ros_pkg = ":".join(p + "/share" for p in ament.split(":") if p)
        os.environ["ROS_PACKAGE_PATH"] = ros_pkg
        sys.stderr.write(f"[INFO] ROS_PACKAGE_PATH auto-set from AMENT_PREFIX_PATH\n")
    else:
        sys.stderr.write("[WARN] AMENT_PREFIX_PATH not set — source ROS workspace first!\n")

from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(CONFIG)

import carb
import omni
import omni.usd
import omni.kit.commands
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, viewports
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

# ── URDF: rb5_850e + gripper (left/right_finger) + depth camera ─────────────
URDF_PATH = os.path.expanduser(
    "~/asl_ws/Manipulator/rb5_isaac/urdf/rb5_with_tools.urdf"
)

extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

if not os.path.exists(URDF_PATH):
    carb.log_error(f"URDF not found: {URDF_PATH}")
    simulation_app.close()
    sys.exit(1)

simulation_context = SimulationContext(stage_units_in_meters=1.0)
viewports.set_camera_view(eye=np.array([1.5, 1.5, 1.2]), target=np.array([0.0, 0.0, 0.5]))

# ── Import URDF ──────────────────────────────────────────────────────────────
_, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints    = False
import_config.convex_decomp         = False
import_config.import_inertia_tensor = True
import_config.fix_base              = True
import_config.distance_scale        = 1.0
import_config.make_instanceable     = False

status, stage_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    get_articulation_root=True,
)

if not status:
    carb.log_error("URDF import failed")
    simulation_app.close()
    sys.exit(1)

robot_prim_path        = "/" + str(stage_path).strip("/").split("/")[0]  # /rb5_850e
articulation_root_path = robot_prim_path + "/link0"                       # /rb5_850e/link0
sys.stderr.write(f"[INFO] robot prim:        {robot_prim_path}\n")
sys.stderr.write(f"[INFO] articulation root: {articulation_root_path}\n")

# ── Move ArticulationRootAPI from root_joint → link0 ────────────────────────
from pxr import PhysxSchema

_stage           = omni.usd.get_context().get_stage()
_root_joint_prim = _stage.GetPrimAtPath(robot_prim_path + "/root_joint")
_link0_prim      = _stage.GetPrimAtPath(articulation_root_path)

for api_cls in [UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI]:
    if _root_joint_prim.HasAPI(api_cls):
        _root_joint_prim.RemoveAPI(api_cls)

UsdPhysics.ArticulationRootAPI.Apply(_link0_prim)
_physx_art = PhysxSchema.PhysxArticulationAPI.Apply(_link0_prim)
_physx_art.CreateEnabledSelfCollisionsAttr().Set(False)
sys.stderr.write("[INFO] ArticulationRootAPI moved to link0\n")

# ── Joint drive properties ───────────────────────────────────────────────────
# Arm joints: high stiffness for position tracking
ARM_JOINTS = [
    ("link0", "base"), ("link1", "shoulder"), ("link2", "elbow"),
    ("link3", "wrist1"), ("link4", "wrist2"), ("link5", "wrist3"),
]
for link, jnt in ARM_JOINTS:
    p = _stage.GetPrimAtPath(f"{robot_prim_path}/{link}/{jnt}")
    if p.IsValid():
        p.GetAttribute("drive:angular:physics:stiffness").Set(10000.0)
        p.GetAttribute("drive:angular:physics:damping").Set(1000.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1e9)

# Robotiq 2F-85: all joints are regular revolute (mimic removed from URDF).
# Joint prim stored under PARENT link in Isaac Sim.
# Target 0.0 = open state for all joints.
ROBOTIQ_JOINTS = [
    ("robotiq_85_base_link",         "robotiq_85_left_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_right_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_left_inner_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_right_inner_knuckle_joint"),
    ("robotiq_85_left_finger_link",  "robotiq_85_left_finger_tip_joint"),
    ("robotiq_85_right_finger_link", "robotiq_85_right_finger_tip_joint"),
]
for link, jnt in ROBOTIQ_JOINTS:
    p = _stage.GetPrimAtPath(f"{robot_prim_path}/{link}/{jnt}")
    if not p.IsValid():
        sys.stderr.write(f"[WARN] Robotiq joint prim NOT FOUND: {robot_prim_path}/{link}/{jnt}\n")
        continue
    stiff = p.GetAttribute("drive:angular:physics:stiffness")
    if not stiff or not str(stiff.GetTypeName()):
        sys.stderr.write(f"[WARN] Robotiq joint has no drive attrs (unexpected): {jnt}\n")
        continue
    stiff.Set(5000.0)
    p.GetAttribute("drive:angular:physics:damping").Set(200.0)
    p.GetAttribute("drive:angular:physics:maxForce").Set(1e6)
    tgt = p.GetAttribute("drive:angular:physics:targetPosition")
    if tgt and str(tgt.GetTypeName()):
        tgt.Set(0.0)
    sys.stderr.write(f"[INFO] Robotiq drive set → open (0.0 rad): {jnt}\n")

sys.stderr.write("[INFO] Joint drives configured\n")
simulation_app.update()

# ── Physics scene & environment ──────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/physicsScene").IsValid():
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage, planePath="/GroundPlane", axis="Z",
    size=5.0, position=Gf.Vec3f(0, 0, 0), color=Gf.Vec3f(0.3),
)
UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight")).CreateIntensityAttr(1000)

simulation_app.update()
simulation_context.initialize_physics()
simulation_context.play()
simulation_app.update()

# ── RealSense D435i Camera sensor ────────────────────────────────────────────
# camera_link is the D435i body (rpy=0,0,0 → no rotation issues).
# We create UsdGeom.Camera prims under camera_link so the camera orient is 0,0,0.
# Avoid optical_frame prims (they carry rpy="-1.5708 0 -1.5708" internally).

def _find_prim(*candidates):
    for c in candidates:
        if _stage.GetPrimAtPath(c).IsValid():
            return c
    return candidates[-1]

_cam_link = _find_prim(
    f"{robot_prim_path}/camera_link",
    f"{robot_prim_path}/camera_bottom_screw_frame/camera_link",
)
sys.stderr.write(f"[INFO] Camera link prim: {_cam_link}\n")

# Both depth and color cameras created at camera_link (orient = 0,0,0)
CAMERA_DEPTH_PATH = _cam_link + "/depth_camera"
CAMERA_COLOR_PATH = _cam_link + "/color_camera"
UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_DEPTH_PATH))
UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_COLOR_PATH))

from omni.isaac.sensor import Camera as IsaacCamera

camera = IsaacCamera(
    prim_path=CAMERA_DEPTH_PATH,
    frequency=30,
    resolution=(640, 480),
)
camera.initialize()
camera.add_distance_to_image_plane_to_frame()
sys.stderr.write(f"[INFO] D435i depth camera: {CAMERA_DEPTH_PATH}\n")
sys.stderr.write(f"[INFO] D435i color camera: {CAMERA_COLOR_PATH}\n")

simulation_app.update()

# ── OmniGraph: ROS2 bridge ───────────────────────────────────────────────────
try:
    og.Controller.edit(
        {"graph_path": "/ROS2Graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent",         "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime",            "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2Context",            "omni.isaac.ros2_bridge.ROS2Context"),
                # Arm
                ("PublishJointState",      "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                ("SubscribeJointState",    "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                # Gripper
                ("SubscribeGripper",       "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("GripperController",      "omni.isaac.core_nodes.IsaacArticulationController"),
                # Clock & TF
                ("PublishClock",           "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ("PublishTF",              "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                # Camera
                ("CreateRenderProduct",    "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRGB",        "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("CameraHelperDepth",      "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                # Impulse triggers
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeGripper.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "GripperController.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishTF.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "CreateRenderProduct.inputs:execIn"),
                # Context
                ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
                ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ("ROS2Context.outputs:context", "PublishTF.inputs:context"),
                ("ROS2Context.outputs:context", "SubscribeGripper.inputs:context"),
                ("ROS2Context.outputs:context", "CameraHelperRGB.inputs:context"),
                ("ROS2Context.outputs:context", "CameraHelperDepth.inputs:context"),
                # Timestamps
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                # Arm control
                ("SubscribeJointState.outputs:jointNames",      "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",   "ArticulationController.inputs:effortCommand"),
                # Gripper control
                ("SubscribeGripper.outputs:jointNames",      "GripperController.inputs:jointNames"),
                ("SubscribeGripper.outputs:positionCommand", "GripperController.inputs:positionCommand"),
                ("SubscribeGripper.outputs:velocityCommand", "GripperController.inputs:velocityCommand"),
                # Camera render
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperRGB.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
            ],
            og.Controller.Keys.SET_VALUES: [
                # Arm
                ("ArticulationController.inputs:robotPath", articulation_root_path),
                ("PublishJointState.inputs:topicName",      "joint_states"),
                ("SubscribeJointState.inputs:topicName",    "isaac_joint_commands"),
                ("PublishJointState.inputs:targetPrim",     [usdrt.Sdf.Path(articulation_root_path)]),
                # Gripper
                ("GripperController.inputs:robotPath",   articulation_root_path),
                ("SubscribeGripper.inputs:topicName",    "gripper_joint_commands"),
                # TF
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(robot_prim_path)]),
                # Camera — D435i depth stream
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_DEPTH_PATH)]),
                ("CreateRenderProduct.inputs:width",  640),
                ("CreateRenderProduct.inputs:height", 480),
                ("CameraHelperRGB.inputs:topicName",  "camera/color/image_raw"),
                ("CameraHelperRGB.inputs:frameId",    "camera_color_optical_frame"),
                ("CameraHelperRGB.inputs:type",       "rgb"),
                ("CameraHelperDepth.inputs:topicName", "camera/depth/image_rect_raw"),
                ("CameraHelperDepth.inputs:frameId",   "camera_depth_optical_frame"),
                ("CameraHelperDepth.inputs:type",      "depth"),
            ],
        },
    )
    sys.stderr.write("[OK] ROS2 OmniGraph built.\n")
except Exception as e:
    carb.log_error(f"OmniGraph failed: {e}")
    import traceback; traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

sys.stderr.write("[OK] Isaac Sim running.\n")
sys.stderr.write("     Arm:     /joint_states  /isaac_joint_commands\n")
sys.stderr.write("     Gripper: /gripper_joint_commands\n")
sys.stderr.write("     Camera:  /camera/color/image_raw\n")
sys.stderr.write("              /camera/depth/image_rect_raw\n")
sys.stderr.write("     TF/Clock: /tf  /clock\n")

while simulation_app.is_running():
    simulation_context.step(render=True)
    og.Controller.set(
        og.Controller.attribute("/ROS2Graph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()
