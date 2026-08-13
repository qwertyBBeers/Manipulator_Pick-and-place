"""Isaac Sim scene: two RB5-850E manipulators relaying a single block.

Adapted from rb5_binpicking/scripts/binpicking_scene.py (that file is left
untouched -- this is a new, separate scene for a two-robot handoff test).
Differences from the original:
  - Two robots instead of one, each imported from the URDF independently
    (same import call run twice, not a USD-level copy of one into the
    other). An earlier version built robot B via Sdf.CopySpec + a
    translating parent Xform, which was fast to write but left root_joint's
    fixed-base anchor referencing something inconsistent with the copy's
    new location -- PhysX logged "found a joint with disjointed body
    transforms" for robot B and the articulation never stabilized. Importing
    fresh for each robot (then translating that robot's own root prim
    directly, the standard Isaac Sim placement pattern) avoids that class of
    bug entirely: each robot's root_joint is created already consistent with
    wherever its own prim tree ends up.
  - Robot B sits at world offset B_OFFSET (~1m from A). Each robot reaches
    its own bin using the plain bin_geometry.yaml numbers in its own local
    frame, so both solve a reach problem the single-robot demo already
    proved solvable. The handoff tray sits midway between the two bins.
    All of that positional layout lives in layout.py, shared with
    relay_pick_place.py so scene and controller cannot disagree.
  - One block only (no procedural clutter, no YCB asset).
  - No cameras / no ROS2 camera bridge (the heuristic controllers use
    ground-truth /binpicking/object_pose only, same as moveit_pick_place.py).
  - Two independent ROS2 OmniGraph bridges, topics namespaced "robot_a/..."
    and "robot_b/...".
  - Three physical trays instead of two bins: source (robot A picks from),
    handoff (A places here, B picks up from here), dest (B places here).

Topics (per robot, "robot_a" example):
  /robot_a/joint_states  /robot_a/isaac_joint_commands  /robot_a/gripper_joint_commands
  /tf  /clock  (shared)
  /binpicking/object_pose  (shared -- one block, one ground-truth pose)

Run:
  source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
  ~/isaacsim/python.sh ~/asl_ws/Manipulator/lerobot_ws/dual_robot/dual_binpicking_scene.py
"""

import math, os, random, sys, time
import numpy as np

if not os.environ.get("ROS_PACKAGE_PATH"):
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    if ament:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(p + "/share" for p in ament.split(":") if p)

from isaacsim import SimulationApp

# Headless for parallel collection: N instances means N GUI windows competing
# for the same GPU for a view nobody is watching. Camera render products still
# work headless -- they are separate render products, not the viewport -- so
# the recorded images are unaffected.
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": os.environ.get("ISAAC_HEADLESS", "0") == "1",
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(CONFIG)

import carb
import omni
import omni.usd
import omni.kit.commands
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import viewports
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

URDF_PATH = os.path.expanduser("~/asl_ws/Manipulator/rb5_isaac/urdf/rb5_with_tools.urdf")

# Revolute joint stop, in DEGREES (USD angular limits are degrees, unlike every
# other angle in this file). Must be EXACTLY the URDF's +/-3.14rad, not a margin
# inside it: MoveIt plans against the URDF bound, so a stop even slightly tighter
# means MoveIt can order a position the simulated joint physically cannot reach.
# At 178deg that showed up as the controller aborting with the arm parked a
# steady 0.0324rad from its goal and refusing to close the gap -- which is
# precisely the 3.14 - 3.106 = 0.034rad difference. Equally it must not be
# looser, or joints drift outside MoveIt's model and every plan fails instantly
# (the original reason for setting a limit at all -- the base reached -4.4rad).
JOINT_LIMIT_DEG = math.degrees(3.14)

# Contact friction at the grasp. The block kept sliding out of a confirmed
# grasp partway through the carry while the knuckle angle held steady, i.e.
# the pads never let go -- the cube slid between them. Friction combine mode
# is "max" on both materials, so the effective pair coefficient is the larger
# of the two; the gripper's numbers therefore have to exceed the block's to
# matter at all. Overridable from the environment so a value can be swept
# without editing the scene.
GRIPPER_STATIC_FRICTION = float(os.environ.get("GRIPPER_MU_S", "3.0"))
GRIPPER_DYNAMIC_FRICTION = float(os.environ.get("GRIPPER_MU_D", "2.6"))
BLOCK_STATIC_FRICTION = float(os.environ.get("BLOCK_MU_S", "1.5"))
BLOCK_DYNAMIC_FRICTION = float(os.environ.get("BLOCK_MU_D", "1.2"))
# Links that can carry load against the cube once the gripper closes -- the
# tips make first contact, but the finger and inner-knuckle pads share the
# grip once it is closed down onto a 42mm cube.
GRIPPER_CONTACT_LINKS = (
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
)

# All positional layout (bin geometry, robot B placement, handoff point) comes
# from layout.py, shared with relay_pick_place.py so the scene and the
# controller cannot disagree about where anything is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import (  # noqa: E402
    SRC_X, SRC_Y, SRC_Z, SRC_H, SRC_W, SRC_D, WALL_T,
    B_OFFSET, SOURCE_WORLD, DEST_WORLD, HANDOFF_WORLD, TRAYS,
)

sys.stderr.write(
    f"[INFO] robot B world offset: {B_OFFSET}  "
    f"(base-to-base dist from A: {np.linalg.norm(B_OFFSET):.3f}m)\n"
)

extensions_enabled = False
from omni.isaac.core.utils import extensions
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

if not os.path.exists(URDF_PATH):
    carb.log_error(f"URDF not found: {URDF_PATH}")
    simulation_app.close(); sys.exit(1)

simulation_context = SimulationContext(stage_units_in_meters=1.0)

viewports.set_camera_view(
    eye=np.array([-1.2, -0.8, 1.6]),
    target=np.array([(SOURCE_WORLD[0] + HANDOFF_WORLD[0] + DEST_WORLD[0]) / 3,
                     (SOURCE_WORLD[1] + HANDOFF_WORLD[1] + DEST_WORLD[1]) / 3, 0.3]),
)

# ── Robot import + tune, called once per robot (identical to
#    binpicking_scene.py's single-robot logic; see module docstring for why
#    each robot gets its own fresh import instead of one being a USD copy). ──
_stage = omni.usd.get_context().get_stage()

GRIPPER_SOFT_FOLLOWER_JOINTS = {
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
}
GRIPPER_FOLLOWER_JOINTS = GRIPPER_SOFT_FOLLOWER_JOINTS | {
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
}


def _import_robot(label: str, dest_path: str, offset_xyz) -> str:
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints    = False
    import_config.convex_decomp         = False
    import_config.import_inertia_tensor = True
    import_config.fix_base              = True
    import_config.distance_scale        = 1.0
    import_config.make_instanceable     = False

    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=URDF_PATH, import_config=import_config, get_articulation_root=True,
    )
    if not status:
        carb.log_error(f"URDF import failed for {label}"); simulation_app.close(); sys.exit(1)

    robot_path = "/" + str(stage_path).strip("/").split("/")[0]
    if robot_path != dest_path:
        # Importer reuses the same default prim name every call (from the
        # URDF's <robot name=...>) -- rename this instance so the two robots
        # don't collide at the same stage path.
        ok = omni.kit.commands.execute("MovePrim", path_from=robot_path, path_to=dest_path)[0]
        if not ok:
            carb.log_error(f"Failed to move {robot_path} -> {dest_path} for {label}")
            simulation_app.close(); sys.exit(1)
        robot_path = dest_path
    sys.stderr.write(f"[INFO] {label}: {robot_path}\n")

    if any(offset_xyz):
        # The importer already put a translate op in this prim's
        # xformOpOrder (AddTranslateOp() errored on that; XformCommonAPI
        # silently no-op'd instead of raising -- both robots stayed at the
        # same visual position). Find that existing op and set its value
        # directly instead of going through either wrapper.
        target = Gf.Vec3d(float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2]))
        xformable = UsdGeom.Xformable(_stage.GetPrimAtPath(robot_path))
        translate_op = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate_op is not None:
            translate_op.Set(target)
        else:
            xformable.AddTranslateOp().Set(target)

        # fix_base=True makes the importer synthesize a root_joint pinning
        # link0 to the world. Its world-side anchor (physics:localPos0) is
        # baked as an ABSOLUTE position at import time and does NOT follow
        # the prim when it's translated afterwards -- so without this, PhysX
        # sees link0 sitting `offset` away from where its own fixed joint
        # says it should be, warns "found a joint with disjointed body
        # transforms", and yanks the robot back at t=0 (observed: robot B
        # spawning then flying off). Shift the anchor by the same offset.
        root_joint = _stage.GetPrimAtPath(robot_path + "/root_joint")
        if root_joint.IsValid():
            local_pos0 = root_joint.GetAttribute("physics:localPos0")
            if local_pos0 and local_pos0.HasAuthoredValue():
                local_pos0.Set(Gf.Vec3f(local_pos0.Get()) + Gf.Vec3f(target))
            else:
                local_pos0 = root_joint.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f)
                local_pos0.Set(Gf.Vec3f(target))
            sys.stderr.write(f"[INFO] {label} root_joint localPos0 -> {local_pos0.Get()}\n")

        world_pos = UsdGeom.Xformable(_stage.GetPrimAtPath(robot_path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        sys.stderr.write(f"[INFO] {label} placed at {tuple(offset_xyz)} (verified world pos: {tuple(world_pos)})\n")

    for api_cls in [UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI]:
        rj = _stage.GetPrimAtPath(robot_path + "/root_joint")
        if rj.IsValid() and rj.HasAPI(api_cls):
            rj.RemoveAPI(api_cls)

    _link0 = _stage.GetPrimAtPath(robot_path + "/link0")
    UsdPhysics.ArticulationRootAPI.Apply(_link0)
    _physx_art = PhysxSchema.PhysxArticulationAPI.Apply(_link0)
    _physx_art.CreateEnabledSelfCollisionsAttr().Set(False)
    # Default solver iteration counts leave the drives visibly under-converged
    # on this 6-DoF chain: with a plain position hold and nothing else
    # publishing, `base` settled ~0.035rad short of target and the other
    # joints ~0.008rad, which is enough for the trajectory bridge's
    # goal_tolerance check to abort every longer motion. More position
    # iterations is the standard fix for articulation drive accuracy.
    _physx_art.CreateSolverPositionIterationCountAttr().Set(64)
    _physx_art.CreateSolverVelocityIterationCountAttr().Set(8)

    # Arm drives. A silently-skipped joint here leaves the default (weak)
    # drive in place, which shows up much later as the arm settling a few
    # degrees short of every goal and the trajectory bridge aborting with
    # "goal not settled" -- so treat a missing joint prim as fatal rather
    # than continuing with a half-configured robot.
    missing = []
    for link, jnt in [
        ("link0", "base"), ("link1", "shoulder"), ("link2", "elbow"),
        ("link3", "wrist1"), ("link4", "wrist2"), ("link5", "wrist3"),
    ]:
        p = _stage.GetPrimAtPath(f"{robot_path}/{link}/{jnt}")
        if not p.IsValid():
            missing.append(f"{robot_path}/{link}/{jnt}")
            continue
        p.GetAttribute("drive:angular:physics:stiffness").Set(10000.0)
        p.GetAttribute("drive:angular:physics:damping").Set(1000.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1e9)
        # Hard rotation limits. Nothing was enforcing these, and over a long
        # session the base accumulated turn after turn until /joint_states
        # reported -4.4049rad -- outside the URDF's own +/-3.14 bound. From that
        # moment every plan for that arm fails instantly with a bare FAILURE,
        # because move_group's start state is out of bounds, and the arm looks
        # mysteriously "stuck" with no other symptom. Clamped slightly inside
        # the URDF limit so a joint resting exactly on the stop is still a
        # legal state for the planner.
        #
        # Create*Attr, not GetAttribute().Set(): the limit attributes are not
        # authored by the URDF importer, and Set() on a non-existent attribute
        # is a silent no-op -- the first version of this looked applied and the
        # base still wandered to -4.0964rad.
        revolute = UsdPhysics.RevoluteJoint(p)
        revolute.CreateLowerLimitAttr().Set(-JOINT_LIMIT_DEG)
        revolute.CreateUpperLimitAttr().Set(JOINT_LIMIT_DEG)
        # Compare with a tolerance: USD stores these as float32, so an exact
        # equality test against the float64 we wrote always fails
        # (179.9087476710785 comes back as 179.90875244140625) and would abort
        # the whole scene over a rounding difference.
        limit_readback = (revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get())
        if (limit_readback[0] is None or limit_readback[1] is None
                or abs(limit_readback[0] + JOINT_LIMIT_DEG) > 1e-3
                or abs(limit_readback[1] - JOINT_LIMIT_DEG) > 1e-3):
            missing.append(f"{robot_path}/{link}/{jnt} (limit readback={limit_readback})")
        # Read back rather than trusting the Set(): a missing/!authored drive
        # attribute here would leave a weak default drive and show up much
        # later as "goal not settled" aborts, which is expensive to trace.
        readback = p.GetAttribute("drive:angular:physics:stiffness").Get()
        if readback != 10000.0:
            missing.append(f"{robot_path}/{link}/{jnt} (stiffness readback={readback})")
        # maxForce matters as much as stiffness: this URDF declares
        # effort="10" (N.m) on every joint, and if that tiny limit survives
        # anywhere in the drive the joint creeps away under load no matter how
        # stiff it is. Report what actually stuck.
        sys.stderr.write(
            f"[CHECK] {label} {jnt}: stiffness={readback} "
            f"damping={p.GetAttribute('drive:angular:physics:damping').Get()} "
            f"maxForce={p.GetAttribute('drive:angular:physics:maxForce').Get()} "
            f"physxMaxForce={p.GetAttribute('physxJoint:maxJointVelocity').Get()}\n"
        )
    if missing:
        carb.log_error(f"{label}: arm drive setup failed: {missing}")
        sys.stderr.write(f"[ERROR] {label}: arm drive setup failed: {missing}\n")
        simulation_app.close(); sys.exit(1)
    sys.stderr.write(f"[OK] {label} 6 arm joint drives set + verified (stiffness=10000)\n")

    # Gripper drive tuning -- see binpicking_scene.py for the full rationale
    # (soft finger-tip followers so the primary knuckle's contact stall isn't
    # fought by followers forcing through resistance).
    for link, jnt in [
        ("robotiq_85_base_link",         "robotiq_85_left_knuckle_joint"),
        ("robotiq_85_base_link",         "robotiq_85_right_knuckle_joint"),
        ("robotiq_85_base_link",         "robotiq_85_left_inner_knuckle_joint"),
        ("robotiq_85_base_link",         "robotiq_85_right_inner_knuckle_joint"),
        ("robotiq_85_left_finger_link",  "robotiq_85_left_finger_tip_joint"),
        ("robotiq_85_right_finger_link", "robotiq_85_right_finger_tip_joint"),
    ]:
        p = _stage.GetPrimAtPath(f"{robot_path}/{link}/{jnt}")
        if not p.IsValid():
            continue
        stiff = p.GetAttribute("drive:angular:physics:stiffness")
        if not stiff or not str(stiff.GetTypeName()):
            continue
        if jnt in GRIPPER_SOFT_FOLLOWER_JOINTS:
            # Stiffness stays soft (600) on purpose -- see the comment above:
            # a stiff finger tip fights the knuckle's contact stall and the
            # gripper stops closing on the object properly.
            #
            # NOTE: damping was raised to 900 to fight the carry slip, then put
            # back -- the slip was fixed by carrying slower and lower instead
            # (see POST_GRASP_LIFT in relay_pick_place.py), and raising it here
            # could not be shown to help while it does risk the fingertips
            # lagging during the close. Kept as a comment because the reasoning
            # is still the right one if the slip ever comes back:
            #
            # Damping is a different lever from stiffness here. The two levers do different jobs here.
            # Stiffness is a restoring force against *displacement*, which is
            # what fights the knuckle; damping resists *velocity*, which is what
            # actually happens when the carried block tries to pivot the pads
            # during a swing. The block was slipping out of confirmed grasps
            # mid-transit on trajectories that executed cleanly (pos_err
            # ~0.003rad, no aborts) with the grip command verifiably held
            # (knuckle steady at 0.5854rad for 65s) -- i.e. the pads were being
            # rotated open by inertia. This damps that without changing the
            # quasi-static closing behaviour the soft stiffness was chosen for.
            stiff.Set(600.0)
            p.GetAttribute("drive:angular:physics:damping").Set(120.0)
            p.GetAttribute("drive:angular:physics:maxForce").Set(1.5e4)
        elif jnt in GRIPPER_FOLLOWER_JOINTS:
            stiff.Set(7000.0)
            p.GetAttribute("drive:angular:physics:damping").Set(350.0)
            p.GetAttribute("drive:angular:physics:maxForce").Set(1e6)
        else:
            stiff.Set(7000.0)
            p.GetAttribute("drive:angular:physics:damping").Set(350.0)
            p.GetAttribute("drive:angular:physics:maxForce").Set(1e6)
        tgt = p.GetAttribute("drive:angular:physics:targetPosition")
        if tgt and str(tgt.GetTypeName()):
            tgt.Set(0.0)

    sys.stderr.write(f"[OK] {label} joint drives configured\n")

    # Every link the block can touch while held, not just the two tips: the
    # cube also rides against the finger and inner-knuckle pads once the
    # gripper closes past ~30mm.
    for _fname in GRIPPER_CONTACT_LINKS:
        _apply_high_friction(
            _stage, f"{robot_path}/{_fname}",
            static_friction=GRIPPER_STATIC_FRICTION,
            dynamic_friction=GRIPPER_DYNAMIC_FRICTION,
            label=f"{label} {_fname}",
        )
    sys.stderr.write(f"[OK] {label} friction materials bound\n")
    simulation_app.update()

    return robot_path


def _apply_high_friction(stage, link_prim_path, static_friction=1.2, dynamic_friction=1.0, label=""):
    """Bind a high-friction physics material to a link and to its colliders.

    Binding on the link prim alone relies on USD binding *inheritance*, which
    only reaches a collider while nothing closer to that collider overrides
    it -- and the URDF importer authors its own default physics material
    directly on the collision prims it creates. A closer binding always wins,
    so the link-level bind can look perfectly applied while PhysX keeps using
    the importer's default coefficients at the one contact that matters. Same
    failure shape as Set()-on-an-unauthored-attribute earlier in this file:
    no error, no effect. So bind on each collider too, then read back what
    PhysX will actually resolve and report it.
    """
    link_prim = stage.GetPrimAtPath(link_prim_path)
    if not link_prim.IsValid():
        sys.stderr.write(f"[ERROR] friction: no prim at {link_prim_path}\n")
        return False
    material = UsdShade.Material.Define(stage, Sdf.Path(link_prim_path + "/HighFrictionMaterial"))
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(0.0)
    # "max" on both sides of a contact => the pair uses the larger coefficient,
    # so raising the gripper's numbers above the block's does move the pair
    # friction (with "average" or "min" it largely would not).
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr().Set("max")

    targets = [link_prim]
    targets += [p for p in Usd.PrimRange(link_prim)
                if p != link_prim and p.HasAPI(UsdPhysics.CollisionAPI)]
    for prim in targets:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    colliders = [p for p in targets if p.HasAPI(UsdPhysics.CollisionAPI)]
    bad = []
    for prim in colliders:
        try:
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
            bound_path = str(bound.GetPath()) if bound else "<none>"
        except Exception as exc:  # USD version differences in this call only
            bound_path = f"<readback failed: {exc}>"
        if bound_path != str(material.GetPath()):
            bad.append(f"{prim.GetPath()} -> {bound_path}")
    tag = label or link_prim_path
    if bad:
        carb.log_error(f"friction not resolved on {tag}: {bad}")
        sys.stderr.write(f"[ERROR] friction not resolved on {tag}: {bad}\n")
        return False
    sys.stderr.write(
        f"[CHECK] {tag}: mu_s={static_friction} mu_d={dynamic_friction} "
        f"on {len(colliders)} collider(s)\n"
    )
    return True


robot_a_path = _import_robot("robot A", "/RobotA", (0.0, 0.0, 0.0))
robot_b_path = _import_robot("robot B", "/RobotB", tuple(B_OFFSET))

# ── Physics scene, ground, lights ───────────────────────────────────────────
if not _stage.GetPrimAtPath("/physicsScene").IsValid():
    sc = UsdPhysics.Scene.Define(_stage, Sdf.Path("/physicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)

omni.kit.commands.execute(
    "AddGroundPlaneCommand", stage=_stage, planePath="/GroundPlane",
    axis="Z", size=6.0, position=Gf.Vec3f(0, 0, 0), color=Gf.Vec3f(0.45),
)
UsdLux.DomeLight.Define(_stage, "/DomeLight").CreateIntensityAttr(1000)

# ── Three trays: source (A picks), handoff (A places / B picks), dest (B places) ──
_src_c = np.array([0.14, 0.14, 0.16])
_handoff_c = np.array([0.35, 0.35, 0.60])
_dest_c = np.array([0.55, 0.48, 0.28])


def _fixed(path, pos, w, d, h, color):
    FixedCuboid(prim_path=path, position=np.array(pos), scale=np.array([w, d, h]), color=color)


def _tray(prefix, cx, cy, cz, w, d, h, wall_t, color):
    _fixed(f"{prefix}/Floor",  [cx, cy, cz + wall_t / 2],                    w,      d,      wall_t, color)
    _fixed(f"{prefix}/WallXp", [cx + w / 2 - wall_t / 2, cy, cz + h / 2],   wall_t, d,      h,      color)
    _fixed(f"{prefix}/WallXn", [cx - w / 2 + wall_t / 2, cy, cz + h / 2],   wall_t, d,      h,      color)
    _fixed(f"{prefix}/WallYp", [cx, cy + d / 2 - wall_t / 2, cz + h / 2],   w,      wall_t, h,      color)
    _fixed(f"{prefix}/WallYn", [cx, cy - d / 2 + wall_t / 2, cz + h / 2],   w,      wall_t, h,      color)


# layout.TRAYS is the shared list; dual_scene_setup.py builds MoveIt collision
# objects from the same entries, so the planner's world and this one match.
_TRAY_PRIM = {"source_bin": "/World/SourceBin", "handoff_tray": "/World/HandoffTray", "dest_bin": "/World/DestBin"}
_TRAY_COLOR = {"source_bin": _src_c, "handoff_tray": _handoff_c, "dest_bin": _dest_c}
for _tray_id, (_cx, _cy, _cz), (_w, _d, _h), _wt in TRAYS:
    _tray(_TRAY_PRIM[_tray_id], _cx, _cy, _cz, _w, _d, _h, _wt, _TRAY_COLOR[_tray_id])

sys.stderr.write("[OK] source (dark) / handoff (blue) / dest (beige) trays created\n")

# ── One block, centered in the source bin ───────────────────────────────────
TARGET_OBJECT_PATH = "/World/Objects/Block0"
OBJECT_POSE_TOPIC = "/binpicking/object_pose"
OBJECT_RESET_TOPIC = "/binpicking/reset_object"
OBJECT_PLACE_TOPIC = "/binpicking/place_object"
# Domain randomization for VLA data collection. A dataset recorded with the
# block always at the same spot in the same colour teaches the policy the
# coordinate, not the task -- it can solve every training episode without ever
# looking at the block. Randomizing both is what forces the visual grounding.
OBJECT_RANDOMIZE_TOPIC = "/binpicking/randomize_object"   # std_msgs/Empty
# Keep the block clear of the tray walls: half the block plus the wall slab plus
# a little, so a randomized spawn never starts interpenetrating a wall (PhysX
# resolves that by launching it).
RANDOM_PLACE_MARGIN = 0.030
# Radial cap on a randomized spawn, measured from robot A's base. 0.62 was too
# generous: a spawn at 0.559m failed four consecutive grasps while spawns at
# 0.377/0.440/0.498m succeeded, which is the same reach effect that made robot B
# unreliable at the old 0.596m handoff. Cap just above the 0.51m that is
# repeatedly proven rather than at the arm's nominal envelope -- randomized
# episodes that cannot be picked are not data, they are noise.
RANDOM_MAX_REACH_M = 0.53
# Hues a policy should learn to treat as "the block". Deliberately spread across
# the wheel rather than shades of one colour, and none of them the grey of the
# floor/trays or the white of the arms.
OBJECT_COLORS = [
    (0.90, 0.75, 0.10),   # yellow (the original)
    (0.85, 0.20, 0.15),   # red
    (0.15, 0.45, 0.85),   # blue
    (0.20, 0.70, 0.30),   # green
    (0.85, 0.45, 0.10),   # orange
    (0.60, 0.25, 0.75),   # purple
    (0.15, 0.70, 0.70),   # teal
    (0.90, 0.55, 0.70),   # pink
]
OBJECT_POSE_HZ = 10.0
BLOCK_HALF = 0.021
# Resting height on any tray: the trays stand on the ground, so the block sits
# one wall thickness (the floor slab) plus its own half-height up.
BLOCK_REST_Z = WALL_T + BLOCK_HALF
BLOCK_SPAWN_POS = np.array([SRC_X, SRC_Y, SRC_Z + SRC_H + 0.05])

_block = DynamicCuboid(
    prim_path=TARGET_OBJECT_PATH,
    position=BLOCK_SPAWN_POS,
    scale=np.array([0.042, 0.042, 0.042]),
    color=np.array([0.90, 0.75, 0.10]),
    mass=0.10,
)
# The gripper links got a high-friction material at import; the block was left
# on Isaac's default, so the pair friction at the one contact that matters was
# whatever the default resolves to. Without this the cube slipped out of a
# confirmed grasp partway through the 0.5m carry to the handoff tray, and the
# only way to hold it was to squeeze hard enough to squirt it out on contact.
_apply_high_friction(
    _stage, TARGET_OBJECT_PATH,
    static_friction=BLOCK_STATIC_FRICTION,
    dynamic_friction=BLOCK_DYNAMIC_FRICTION,
    label="block",
)

sys.stderr.write("[OK] 1 block spawned in source bin (high-friction)\n")
simulation_app.update()

# ── Two independent ROS2 OmniGraph bridges (no cameras) ────────────────────
# ROS2Context ignores ROS_DOMAIN_ID unless told to read it -- its
# useDomainIDEnvVar input defaults False and domain_id defaults 0, so several
# Isaac instances started with different ROS_DOMAIN_IDs would all publish on
# domain 0 and fight over the same topic names. The domain is what isolates
# parallel collection instances from each other, so every graph in this file
# has to opt in.
DOMAIN_ID_FROM_ENV = ("ROS2Ctx.inputs:useDomainIDEnvVar", True)

def _build_robot_ros2_graph(graph_name, ns, robot_path):
    articulation_root_path = robot_path + "/link0"
    og.Controller.edit(
        {"graph_path": f"/{graph_name}", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                ("SimTime",   "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2Ctx",   "omni.isaac.ros2_bridge.ROS2Context"),
                ("PubJS",     "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                ("SubArm",    "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("ArmCtrl",   "omni.isaac.core_nodes.IsaacArticulationController"),
                ("SubGrip",   "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("GripCtrl",  "omni.isaac.core_nodes.IsaacArticulationController"),
                ("PubTF",     "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulse.outputs:execOut", "PubJS.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "SubArm.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "ArmCtrl.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "SubGrip.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "GripCtrl.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "PubTF.inputs:execIn"),
                ("ROS2Ctx.outputs:context",   "PubJS.inputs:context"),
                ("ROS2Ctx.outputs:context",   "SubArm.inputs:context"),
                ("ROS2Ctx.outputs:context",   "PubTF.inputs:context"),
                ("ROS2Ctx.outputs:context",   "SubGrip.inputs:context"),
                ("SimTime.outputs:simulationTime", "PubJS.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
                ("SubArm.outputs:jointNames",       "ArmCtrl.inputs:jointNames"),
                ("SubArm.outputs:positionCommand",  "ArmCtrl.inputs:positionCommand"),
                ("SubArm.outputs:velocityCommand",  "ArmCtrl.inputs:velocityCommand"),
                ("SubArm.outputs:effortCommand",    "ArmCtrl.inputs:effortCommand"),
                ("SubGrip.outputs:jointNames",      "GripCtrl.inputs:jointNames"),
                ("SubGrip.outputs:positionCommand", "GripCtrl.inputs:positionCommand"),
                ("SubGrip.outputs:velocityCommand", "GripCtrl.inputs:velocityCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                DOMAIN_ID_FROM_ENV,
                ("ArmCtrl.inputs:robotPath",  articulation_root_path),
                ("PubJS.inputs:topicName",    f"{ns}/joint_states"),
                ("SubArm.inputs:topicName",   f"{ns}/isaac_joint_commands"),
                ("PubJS.inputs:targetPrim",   [usdrt.Sdf.Path(articulation_root_path)]),
                ("GripCtrl.inputs:robotPath", articulation_root_path),
                ("SubGrip.inputs:topicName",  f"{ns}/gripper_joint_commands"),
                ("PubTF.inputs:targetPrims",  [usdrt.Sdf.Path(robot_path)]),
            ],
        },
    )


def _build_clock_graph():
    # Single shared /clock publisher (not namespaced -- same as
    # binpicking_scene.py). Both move_group instances and the trajectory
    # bridges run with use_sim_time:=true; without this, their ROS clocks
    # never advance and any time-based logic (execution timing, grace
    # periods) hangs indefinitely instead of erroring -- found by hanging
    # exactly like that on the first real test of this scene.
    og.Controller.edit(
        {"graph_path": "/ClockGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                ("SimTime",   "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2Ctx",   "omni.isaac.ros2_bridge.ROS2Context"),
                ("PubClock",  "omni.isaac.ros2_bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulse.outputs:execOut", "PubClock.inputs:execIn"),
                ("ROS2Ctx.outputs:context",   "PubClock.inputs:context"),
                ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
            ],
            og.Controller.Keys.SET_VALUES: [
                DOMAIN_ID_FROM_ENV,
            ],
        },
    )


# ── Front overview camera ───────────────────────────────────────────────────
# A VLA is fine-tuned on what the robot *sees*, and this scene had no cameras at
# all -- the heuristic controllers run on ground-truth /binpicking/object_pose,
# which is exactly the input a learned policy will not have. One scene camera
# looking at the whole workspace from the front is the minimum: it has to show
# both arms and all three trays at once, because the task the policy has to
# learn (block moves source -> handoff -> dest) is only legible across all of
# them. Wrist cameras are the usual second view and can be added the same way.
SCENE_CAM_PATH = "/World/SceneCam"
SCENE_CAM_TOPIC = "scene_camera/rgb"
SCENE_CAM_RESOLUTION = (640, 480)
# Same viewpoint as the GUI viewport, i.e. the view the workspace is normally
# judged from, pulled back far enough to hold both robots in frame.
SCENE_CAM_TARGET = (
    (SOURCE_WORLD[0] + HANDOFF_WORLD[0] + DEST_WORLD[0]) / 3.0,
    (SOURCE_WORLD[1] + HANDOFF_WORLD[1] + DEST_WORLD[1]) / 3.0,
    0.15,
)
# Distance is set by the tallest thing that must stay in frame, which is an arm
# parked at HOME (straight up) with its gripper at the top -- a policy cannot be
# trained on frames where the end effector is cropped out. Measured: at
# (-1.15, -0.95, 0.95) the trays fill more of the image but both grippers leave
# it, so that is not a trade worth making. This keeps everything in frame.
SCENE_CAM_EYE = (SCENE_CAM_TARGET[0] - 1.55, SCENE_CAM_TARGET[1] - 1.25, 1.25)


# Wrist cameras. The URDF already carries a RealSense-style mount on the
# gripper (camera_joint -> camera_link -> ... -> camera_color_optical_frame), so
# the camera goes exactly where the real robot's would be. Smaller than the
# scene camera on purpose: every render product costs simulator real-time
# factor, and the wrist view is a close-up that does not need the pixels.
WRIST_CAM_RESOLUTION = (320, 240)
WRIST_CAM_FRAME_SKIP = 3
# ROS optical convention (+Z forward, +Y down) vs USD camera convention
# (-Z forward, +Y up): a 180 degree turn about X maps one onto the other.
WRIST_CAM_OPTICAL_FRAME = "camera_color_optical_frame"


def _build_wrist_camera(robot_path: str, ns: str):
    parent_path = f"{robot_path}/{WRIST_CAM_OPTICAL_FRAME}"
    if not _stage.GetPrimAtPath(parent_path).IsValid():
        # Fall back to the camera body itself rather than silently mounting the
        # camera at the robot's origin, which would look plausible in the topic
        # list and be useless in the data.
        parent_path = f"{robot_path}/camera_link"
        if not _stage.GetPrimAtPath(parent_path).IsValid():
            raise RuntimeError(f"no camera mount prim under {robot_path}")
        sys.stderr.write(f"[WARN] {ns}: {WRIST_CAM_OPTICAL_FRAME} missing; mounting wrist cam on camera_link\n")

    cam_path = f"{parent_path}/WristCam"
    camera = UsdGeom.Camera.Define(_stage, Sdf.Path(cam_path))
    camera.CreateFocalLengthAttr(12.0)          # close-up, wide
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.716)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddRotateXOp().Set(180.0)

    import omni.replicator.core as rep

    render_product = rep.create.render_product(cam_path, WRIST_CAM_RESOLUTION)
    render_product_path = render_product.path if hasattr(render_product, "path") else str(render_product)
    topic = f"{ns}/wrist_camera/rgb"
    og.Controller.edit(
        {"graph_path": f"/WristCamGraph_{ns}", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick",  "omni.graph.action.OnPlaybackTick"),
                ("ROS2Ctx", "omni.isaac.ros2_bridge.ROS2Context"),
                ("CamRGB",  "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick",     "CamRGB.inputs:execIn"),
                ("ROS2Ctx.outputs:context", "CamRGB.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                DOMAIN_ID_FROM_ENV,
                ("CamRGB.inputs:renderProductPath", render_product_path),
                ("CamRGB.inputs:type", "rgb"),
                ("CamRGB.inputs:frameSkipCount", WRIST_CAM_FRAME_SKIP),
                ("CamRGB.inputs:topicName", topic),
                ("CamRGB.inputs:frameId", f"{ns}_wrist_camera"),
            ],
        },
    )
    return topic, cam_path


def _build_scene_camera():
    camera = UsdGeom.Camera.Define(_stage, Sdf.Path(SCENE_CAM_PATH))
    camera.CreateFocalLengthAttr(16.0)          # wide enough for both robots
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.716)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    # A USD camera looks down its own -Z with +Y up, which is what SetLookAt
    # builds a *view* matrix for -- the camera's transform is that inverted.
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*SCENE_CAM_EYE), Gf.Vec3d(*SCENE_CAM_TARGET), Gf.Vec3d(0, 0, 1))
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(view.GetInverse())

    import omni.replicator.core as rep

    # Create the render product once, here, rather than from an
    # IsaacCreateRenderProduct node inside the graph: the graph node runs every
    # evaluation and this only ever needs to exist once.
    render_product = rep.create.render_product(SCENE_CAM_PATH, SCENE_CAM_RESOLUTION)
    render_product_path = render_product.path if hasattr(render_product, "path") else str(render_product)

    og.Controller.edit(
        {"graph_path": "/SceneCamGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick",   "omni.graph.action.OnPlaybackTick"),
                ("ROS2Ctx",  "omni.isaac.ros2_bridge.ROS2Context"),
                ("CamRGB",   "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick",     "CamRGB.inputs:execIn"),
                ("ROS2Ctx.outputs:context", "CamRGB.inputs:context"),
            ],
            og.Controller.Keys.SET_VALUES: [
                DOMAIN_ID_FROM_ENV,
                ("CamRGB.inputs:renderProductPath", render_product_path),
                ("CamRGB.inputs:type", "rgb"),
                # Publish every Nth frame. Unthrottled this ran at ~57Hz and
                # took the simulator's real-time factor down to 0.76, which
                # broke the gripper (its ramp is timed in wall-clock seconds --
                # see dual_binpicking.launch.py). ~14Hz is more than a VLA
                # dataset needs and gives most of that back.
                ("CamRGB.inputs:frameSkipCount", 3),
                ("CamRGB.inputs:topicName", SCENE_CAM_TOPIC),
                ("CamRGB.inputs:frameId", "scene_camera"),
            ],
        },
    )
    return render_product_path


try:
    _build_robot_ros2_graph("ROS2GraphA", "robot_a", robot_a_path)
    _build_robot_ros2_graph("ROS2GraphB", "robot_b", robot_b_path)
    _build_clock_graph()
    sys.stderr.write("[OK] ROS2 OmniGraphs built for robot_a, robot_b, /clock\n")
    try:
        _rp_path = _build_scene_camera()
        sys.stderr.write(
            f"[OK] front scene camera at {tuple(round(v, 2) for v in SCENE_CAM_EYE)} "
            f"-> /{SCENE_CAM_TOPIC} ({SCENE_CAM_RESOLUTION[0]}x{SCENE_CAM_RESOLUTION[1]}, rp={_rp_path})\n"
        )
    except Exception as _cam_exc:  # a camera failing must not take the scene down
        carb.log_error(f"scene camera setup failed: {_cam_exc}")
        sys.stderr.write(f"[ERROR] scene camera setup failed: {_cam_exc}\n")
    # Switchable so the wrist cameras' cost can be isolated: every render
    # product costs real-time factor (0.77 with the scene camera alone, 0.58
    # with all three), and RTF is the variable the gripper timing was chasing.
    for _ns, _robot_path in (
        (("robot_a", robot_a_path), ("robot_b", robot_b_path))
        if os.environ.get("WRIST_CAMERAS", "1") != "0" else ()
    ):
        try:
            _topic, _cam_path = _build_wrist_camera(_robot_path, _ns)
            sys.stderr.write(
                f"[OK] {_ns} wrist camera at {_cam_path} -> /{_topic} "
                f"({WRIST_CAM_RESOLUTION[0]}x{WRIST_CAM_RESOLUTION[1]})\n"
            )
        except Exception as _wc_exc:
            carb.log_error(f"{_ns} wrist camera setup failed: {_wc_exc}")
            sys.stderr.write(f"[ERROR] {_ns} wrist camera setup failed: {_wc_exc}\n")
except Exception as e:
    carb.log_error(f"OmniGraph failed: {e}")
    import traceback; traceback.print_exc()
    simulation_app.close(); sys.exit(1)

simulation_app.update()

simulation_context.initialize_physics()
simulation_context.play()
simulation_app.update()


# ── Target block pose publisher (ground truth, shared) ─────────────────────
def _init_object_pose_publisher():
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
    except Exception as exc:
        sys.stderr.write(f"[WARN] rclpy unavailable; object pose topic disabled: {exc}\n")
        return None
    from std_msgs.msg import Empty
    from geometry_msgs.msg import Point
    did_init = False
    if not rclpy.ok():
        rclpy.init(args=None)
        did_init = True
    node = rclpy.create_node("dual_binpicking_object_pose_publisher")
    pub = node.create_publisher(PoseStamped, OBJECT_POSE_TOPIC, 10)
    # Teleport the block back to its spawn point on request. A failed carry can
    # fling it out of both robots' reach, and without this the only way back to
    # a runnable state is a full Isaac Sim restart (~1min) between attempts.
    ctx = {"rclpy": rclpy, "PoseStamped": PoseStamped, "node": node, "pub": pub,
           "did_init": did_init, "place_at": None}
    node.create_subscription(Empty, OBJECT_RESET_TOPIC,
                             lambda _msg: ctx.__setitem__("place_at", BLOCK_SPAWN_POS), 10)
    # Put the block anywhere, so one stage of the relay can be exercised on its
    # own instead of replaying the whole ~4min sequence to reach it.
    node.create_subscription(
        Point, OBJECT_PLACE_TOPIC,
        lambda msg: ctx.__setitem__("place_at", np.array([msg.x, msg.y, msg.z if msg.z > 0.0 else BLOCK_REST_Z])),
        10,
    )
    # Randomize where the block starts and what colour it is. Position is drawn
    # here rather than by the caller so the tray's inner extent -- which only
    # this file knows in full -- is what bounds it.
    node.create_subscription(Empty, OBJECT_RANDOMIZE_TOPIC,
                             lambda _msg: ctx.__setitem__("randomize", True), 10)
    sys.stderr.write(f"[INFO] Publishing target block pose: {TARGET_OBJECT_PATH} -> {OBJECT_POSE_TOPIC}\n")
    sys.stderr.write(f"[INFO] Block reset on: {OBJECT_RESET_TOPIC} (std_msgs/Empty)\n")
    sys.stderr.write(f"[INFO] Block teleport on: {OBJECT_PLACE_TOPIC} (geometry_msgs/Point)\n")
    sys.stderr.write(f"[INFO] Block randomize (pose + colour) on: {OBJECT_RANDOMIZE_TOPIC} (std_msgs/Empty)\n")
    ctx["randomize"] = False
    return ctx


def _random_block_pose_in_source():
    """A uniform point inside the source bin, clear of its walls and in reach.

    The bin's full inner extent spans 0.34x0.29m, whose far corner sits 0.70m
    from robot A's base -- and 0.67m is where robot B was measured repeatedly
    failing the identical relative geometry, with the tool arriving a
    deterministic 3.2mm and 0.00deg off. That is the arm running out of useful
    envelope, not an IK or accuracy problem, so a randomized spawn out there
    would just manufacture unpickable episodes. Rejection-sample instead of
    shrinking the box, so the reachable part of the bin stays uniformly covered
    rather than being cropped to an axis-aligned rectangle inside it.
    """
    half_w = max(0.0, SRC_W / 2.0 - WALL_T - RANDOM_PLACE_MARGIN)
    half_d = max(0.0, SRC_D / 2.0 - WALL_T - RANDOM_PLACE_MARGIN)
    for _ in range(200):
        x = SRC_X + random.uniform(-half_w, half_w)
        y = SRC_Y + random.uniform(-half_d, half_d)
        if math.hypot(x, y) <= RANDOM_MAX_REACH_M:
            return np.array([x, y, BLOCK_REST_Z])
    return np.array([SRC_X, SRC_Y, BLOCK_REST_Z])


def _set_block_color(rgb):
    """Recolour the block in place. Visual only -- mass, friction and collision
    are untouched, which is the point: colour has to vary without changing the
    physics the demonstrations were recorded under.

    Goes through the applied visual material, not the displayColor primvar.
    DynamicCuboid(color=...) builds a PreviewSurface material and binds it, and
    a bound material beats displayColor -- writing the primvar looked like it
    worked (no error, attribute set) and rendered exactly the same colour as
    before. Same silent-no-op shape as the friction binding earlier in this
    file, so verify by reading the value back.
    """
    target = np.array([float(c) for c in rgb])
    material = None
    try:
        material = _block.get_applied_visual_material()
    except Exception:
        material = None
    if material is not None:
        material.set_color(target)
        actual = np.asarray(material.get_color(), dtype=float).reshape(-1)[:3]
        if np.allclose(actual, target, atol=1e-3):
            return True
        sys.stderr.write(f"[WARN] block colour readback {actual} != requested {target}\n")

    prim = _stage.GetPrimAtPath(TARGET_OBJECT_PATH)
    if not prim.IsValid():
        return False
    gprim = UsdGeom.Gprim(prim)
    attr = gprim.GetDisplayColorAttr() or gprim.CreateDisplayColorAttr()
    attr.Set([Gf.Vec3f(*[float(c) for c in rgb])])
    return True


def _target_object_pose_msg(pub_ctx):
    import math
    prim = _stage.GetPrimAtPath(TARGET_OBJECT_PATH)
    if not prim.IsValid():
        return None
    world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    trans = world_tf.ExtractTranslation()
    quat = world_tf.ExtractRotationQuat()
    quat_imag = quat.GetImaginary()
    qx, qy, qz, qw = float(quat_imag[0]), float(quat_imag[1]), float(quat_imag[2]), float(quat.GetReal())
    q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_norm > 1e-9:
        qx, qy, qz, qw = qx / q_norm, qy / q_norm, qz / q_norm, qw / q_norm
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    msg = pub_ctx["PoseStamped"]()
    msg.header.stamp = pub_ctx["node"].get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(trans[0]), float(trans[1]), float(trans[2])
    msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
    msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
    return msg


_object_pose_pub_ctx = _init_object_pose_publisher()
_last_object_pose_pub_time = 0.0

sys.stderr.write("[OK] Dual-robot relay scene running.\n")
sys.stderr.write(f"     Robot A @ (0.00, 0.00)      Robot B @ ({B_OFFSET[0]:.2f}, {B_OFFSET[1]:.2f})\n")
sys.stderr.write(f"     Source  @ ({SOURCE_WORLD[0]:.2f}, {SOURCE_WORLD[1]:.2f})  Handoff @ ({HANDOFF_WORLD[0]:.2f}, {HANDOFF_WORLD[1]:.2f})  Dest @ ({DEST_WORLD[0]:.2f}, {DEST_WORLD[1]:.2f})\n")
sys.stderr.write("     Topics: /robot_a/joint_states /robot_a/isaac_joint_commands /robot_a/gripper_joint_commands\n")
sys.stderr.write("             /robot_b/joint_states /robot_b/isaac_joint_commands /robot_b/gripper_joint_commands\n")
sys.stderr.write(f"             {OBJECT_POSE_TOPIC} ({TARGET_OBJECT_PATH})\n")

while simulation_app.is_running():
    # GUI mode for visual inspection of the robot-B root_joint issue --
    # render on (headless=False above too). Switch both back once diagnosed.
    simulation_context.step(render=True)
    og.Controller.set(og.Controller.attribute("/ROS2GraphA/OnImpulse.state:enableImpulse"), True)
    og.Controller.set(og.Controller.attribute("/ROS2GraphB/OnImpulse.state:enableImpulse"), True)
    og.Controller.set(og.Controller.attribute("/ClockGraph/OnImpulse.state:enableImpulse"), True)
    if _object_pose_pub_ctx is not None:
        now = time.monotonic()
        if now - _last_object_pose_pub_time >= 1.0 / OBJECT_POSE_HZ:
            msg = _target_object_pose_msg(_object_pose_pub_ctx)
            if msg is not None:
                _object_pose_pub_ctx["pub"].publish(msg)
            _object_pose_pub_ctx["rclpy"].spin_once(_object_pose_pub_ctx["node"], timeout_sec=0.0)
            _last_object_pose_pub_time = now
        if _object_pose_pub_ctx.get("randomize"):
            _object_pose_pub_ctx["randomize"] = False
            _rand_pos = _random_block_pose_in_source()
            _rand_rgb = random.choice(OBJECT_COLORS)
            _rand_yaw = random.uniform(-math.pi, math.pi)
            _half = _rand_yaw / 2.0
            _block.set_world_pose(
                position=_rand_pos,
                orientation=np.array([math.cos(_half), 0.0, 0.0, math.sin(_half)]),  # w,x,y,z about Z
            )
            _block.set_linear_velocity(np.zeros(3))
            _block.set_angular_velocity(np.zeros(3))
            _set_block_color(_rand_rgb)
            sys.stderr.write(
                f"[INFO] block randomized: pos=({_rand_pos[0]:.3f}, {_rand_pos[1]:.3f}) "
                f"yaw={math.degrees(_rand_yaw):+.0f}deg rgb={_rand_rgb}\n"
            )
        if _object_pose_pub_ctx["place_at"] is not None:
            _target = _object_pose_pub_ctx["place_at"]
            _object_pose_pub_ctx["place_at"] = None
            # Zero the velocities too -- a block that was flung is still moving,
            # and teleporting it without clearing momentum just launches it again.
            _block.set_world_pose(position=_target, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            _block.set_linear_velocity(np.zeros(3))
            _block.set_angular_velocity(np.zeros(3))
            sys.stderr.write(f"[INFO] block moved to {_target}\n")

simulation_context.stop()
if _object_pose_pub_ctx is not None:
    _object_pose_pub_ctx["node"].destroy_node()
    if _object_pose_pub_ctx["did_init"]:
        _object_pose_pub_ctx["rclpy"].shutdown()
simulation_app.close()
