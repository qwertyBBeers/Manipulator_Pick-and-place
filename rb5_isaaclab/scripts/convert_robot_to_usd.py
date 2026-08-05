"""Converts rb5_850e_robotiq_mimic.urdf -> a single standalone USD file,
bypassing IsaacLab's `scripts/tools/convert_urdf.py` / `UrdfConverterCfg`
wrapper.

Why not the IsaacLab CLI tool: this script needs the same `omni.kit.commands`
URDF import path `binpicking_scene.py` (the working ROS/Isaac-Sim-4.1
pipeline) already uses, for the same non-standard mesh/friction handling
below (rewriting `package://` URIs, baking a high-friction PhysicsMaterial
onto the fingertip links) that the IsaacLab CLI tool doesn't do.

Gripper coupling: the URDF (see assets/urdf/rb5_850e_robotiq_mimic.urdf)
still carries `<mimic>` tags on the gripper's 5 passive joints, but this
script imports with `import_config.parse_mimic = False`, which -- despite
the confusingly-inverted name -- DROPS them, leaving 6 independent,
individually-actuated revolute joints. An earlier version used
`parse_mimic = True` to bake a real PhysX `PhysxMimicJointAPI` constraint
per follower joint instead, abandoned after it permanently deadlocked the
primary knuckle joint outside its own joint-limit range (5 simultaneous
bilateral mimic constraints on one primary DOF is an over-constrained
system this PhysX/IsaacLab version can't resolve -- see
`CURRICULUM_REPORT.md` section 1d for the full investigation).
`robots/rb5_850e.py` now drives all 6 gripper joints with their own real PD
actuator, and the 5 follower joints' *commanded targets* (not a physics
constraint) are scaled by the same gearing table the mimic tags describe --
see `tasks/pick_place/config/*_env_cfg.py`'s `ActionsCfg.gripper_action`.

Usage:
    <IsaacLab repo>/isaaclab.sh -p convert_robot_to_usd.py [--gui]
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", default=False, help="Keep the app open after conversion to inspect the result.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.gui

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")
simulation_app.update()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_URDF_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "assets", "urdf", "rb5_850e_robotiq_mimic.urdf"))
USD_OUT = os.path.normpath(os.path.join(_THIS_DIR, "..", "assets", "usd", "rb5_850e_robotiq.usd"))

# The URDF's arm links reference meshes via `package://rbpodo_description/...`
# (a ROS package URI); this importer, unlike IsaacLab's convert_urdf.py CLI,
# doesn't resolve `package://` and silently drops any mesh it can't find --
# rewrite to the resolved absolute path in a throwaway copy before importing
# (the tracked source URDF keeps the portable `package://` form).
_RBPODO_DESCRIPTION_SHARE = "/home/hh/asl_ws/Manipulator/install/rbpodo_description/share/rbpodo_description"
if not os.path.isdir(os.path.join(_RBPODO_DESCRIPTION_SHARE, "meshes", "rb5_850e")):
    raise RuntimeError(
        f"rbpodo_description mesh dir not found at {_RBPODO_DESCRIPTION_SHARE} -- "
        "update _RBPODO_DESCRIPTION_SHARE (e.g. after a colcon rebuild changes the install layout)."
    )
with open(_SRC_URDF_PATH) as f:
    _urdf_text = f.read()
_urdf_text = _urdf_text.replace("package://rbpodo_description/", f"file://{_RBPODO_DESCRIPTION_SHARE}/")
URDF_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "assets", "usd", "_rb5_850e_robotiq_mimic_resolved.urdf"))
os.makedirs(os.path.dirname(URDF_PATH), exist_ok=True)
with open(URDF_PATH, "w") as f:
    f.write(_urdf_text)

RESULT_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "assets", "usd", "_convert_result.txt"))
lines = [f"URDF_PATH={URDF_PATH}", f"USD_OUT={USD_OUT}"]

try:
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0
    import_config.parse_mimic = False  # see module docstring

    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=import_config,
        get_articulation_root=True,
    )
    lines.append(f"import status={status} stage_path={stage_path}")
    if not status:
        raise RuntimeError("URDFParseAndImportFile returned status=False")

    simulation_app.update()

    import math

    from pxr import PhysxSchema, Sdf, UsdPhysics

    PRIMARY_JOINT = "robotiq_85_left_knuckle_joint"
    # (joint_name, lower_rad, upper_rad) from the URDF's own <limit> tags,
    # used to repair any joint the importer left with no authored limits.
    follower_joints = {
        "robotiq_85_right_knuckle_joint": (-0.8, 0.0),
        "robotiq_85_left_inner_knuckle_joint": (0.0, 0.8),
        "robotiq_85_right_inner_knuckle_joint": (-0.8, 0.0),
        "robotiq_85_left_finger_tip_joint": (-0.8, 0.0),
        "robotiq_85_right_finger_tip_joint": (0.0, 0.8),
    }

    stage = omni.usd.get_context().get_stage()

    # PhysX requires a finite joint limit for a revolute joint; the importer
    # has been observed to occasionally leave one follower with none.
    limits_repaired = []
    for prim in stage.Traverse():
        if prim.GetName() not in follower_joints:
            continue
        rj = UsdPhysics.RevoluteJoint(prim)
        low_attr, up_attr = rj.GetLowerLimitAttr(), rj.GetUpperLimitAttr()
        lower_rad, upper_rad = follower_joints[prim.GetName()]
        if not (low_attr and low_attr.HasAuthoredValue()) or not (up_attr and up_attr.HasAuthoredValue()):
            rj.CreateLowerLimitAttr().Set(math.degrees(lower_rad))
            rj.CreateUpperLimitAttr().Set(math.degrees(upper_rad))
            limits_repaired.append(str(prim.GetPath()))
    if limits_repaired:
        lines.append(f"REPAIRED missing joint limits on: {limits_repaired}")
    else:
        lines.append("No joint-limit repairs needed.")

    # parse_mimic=False should mean no PhysxMimicJointAPI at all -- verify
    # rather than silently relying on it.
    stray_mimic = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(PhysxSchema.PhysxMimicJointAPI)]
    if stray_mimic:
        raise RuntimeError(f"expected zero PhysxMimicJointAPI prims with parse_mimic=False, found: {stray_mimic}")
    lines.append("Confirmed no PhysxMimicJointAPI prims present (parse_mimic=False) -- gripper joints are independent.")

    # Fingertip friction: the URDF's <gazebo>/ODE <surface><friction> tags
    # are silently ignored by every USD/PhysX importer path, so bind a
    # high-friction material directly (same values as binpicking_scene.py's
    # `_apply_high_friction()`), baked into the USD once instead of redone
    # at every sim launch.
    from pxr import UsdPhysics, UsdShade

    FINGERTIP_LINKS = ["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"]
    friction_bound = []
    for prim in stage.Traverse():
        if prim.GetName() not in FINGERTIP_LINKS:
            continue
        material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("HighFrictionMaterial"))
        mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(1.2)
        mat_api.CreateDynamicFrictionAttr().Set(1.0)
        mat_api.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr().Set("max")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        friction_bound.append(str(prim.GetPath()))
    lines.append(f"high-friction material bound to: {friction_bound}")
    # Each fingertip link legitimately appears under multiple internal
    # reference roots (main tree + /visuals/ + /colliders/ libraries), so
    # >= 2x len(FINGERTIP_LINKS) bindings is expected, not a bug.
    if len(friction_bound) < len(FINGERTIP_LINKS):
        lines.append(f"WARNING: expected at least {len(FINGERTIP_LINKS)} fingertip link prims, bound {len(friction_bound)}")

    os.makedirs(os.path.dirname(USD_OUT), exist_ok=True)
    stage.GetRootLayer().Export(USD_OUT)
    lines.append(f"exported to {USD_OUT}")

    # Re-open the exported file fresh (not the live edited stage) to verify
    # what actually got written to disk: every gripper joint present,
    # independent (no mimic API), with a finite authored limit.
    from pxr import Usd

    check_stage = Usd.Stage.Open(USD_OUT)
    all_gripper_joints = {PRIMARY_JOINT, *follower_joints.keys()}
    found = set()
    for prim in check_stage.Traverse():
        if prim.GetName() not in all_gripper_joints:
            continue
        if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI):
            raise RuntimeError(f"{prim.GetPath()} unexpectedly has a PhysxMimicJointAPI on disk")
        rj = UsdPhysics.RevoluteJoint(prim)
        low_attr, up_attr = rj.GetLowerLimitAttr(), rj.GetUpperLimitAttr()
        if not (low_attr and low_attr.HasAuthoredValue() and up_attr and up_attr.HasAuthoredValue()):
            raise RuntimeError(f"{prim.GetPath()} has no authored joint limit on disk")
        found.add(prim.GetName())
        lines.append(f"VERIFIED (post-export) independent joint: {prim.GetPath()} limit_deg=({low_attr.Get()}, {up_attr.Get()})")
    missing = all_gripper_joints - found
    if missing:
        raise RuntimeError(f"gripper joints missing from exported USD: {missing}")
    lines.append(f"all {len(all_gripper_joints)} gripper joints verified independent on disk.")

except Exception as e:
    import traceback

    lines.append("EXCEPTION: " + str(e))
    lines.append(traceback.format_exc())

with open(RESULT_PATH, "w") as f:
    f.write("\n".join(lines))
    f.flush()
    os.fsync(f.fileno())

print("\n".join(lines))

if args_cli.gui:
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
