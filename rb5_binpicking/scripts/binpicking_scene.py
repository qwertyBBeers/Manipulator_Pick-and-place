"""
Isaac Sim bin picking scene — Phase 1 (ROS2-integrated).

새 패키지: rb5_binpicking  (rb5_isaac 코드 유지, 독립 실행)
  - 기존 rb5_isaac/urdf/rb5_with_tools.urdf 그대로 참조
  - bin + objects 환경 추가
  - 동일한 ROS2 토픽 제공

오브젝트:
  - DynamicSphere / DynamicCuboid / DynamicCylinder  (절차적)
  - 003_cracker_box_physics.usd  (YCB, 로컬 에셋)

ROS2 토픽 (rb5_isaac 과 동일):
  /joint_states   /tf   /clock
  /camera/color/image_raw   /camera/color/camera_info
  /camera/depth/image_rect_raw   /camera/depth/camera_info
  /isaac_joint_commands     /gripper_joint_commands
  /binpicking/object_pose

실행:
  source /opt/ros/humble/setup.bash && source ~/asl_ws/Manipulator/install/setup.bash
  ~/isaacsim/python.sh ~/asl_ws/Manipulator/rb5_binpicking/scripts/binpicking_scene.py
"""

import os, sys, random, time
import numpy as np

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "[ERROR] PyYAML is not available in this Python interpreter "
        "(~/isaacsim/python.sh). bin_geometry.yaml cannot be read without it "
        "and there is no hardcoded fallback — install pyyaml into the Isaac "
        "Sim Python environment.\n"
    )
    sys.exit(1)

# ── ROS_PACKAGE_PATH (AMENT_PREFIX_PATH 에서 자동 설정) ──────────────────────
if not os.environ.get("ROS_PACKAGE_PATH"):
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    if ament:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(p + "/share" for p in ament.split(":") if p)
        sys.stderr.write("[INFO] ROS_PACKAGE_PATH auto-set\n")

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
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid, DynamicSphere, DynamicCylinder
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

# ── 경로 설정 ────────────────────────────────────────────────────────────────
# 로봇 URDF: rb5_isaac 패키지의 것을 그대로 사용
URDF_PATH = os.path.expanduser(
    "~/asl_ws/Manipulator/rb5_isaac/urdf/rb5_with_tools.urdf"
)

# 로컬 YCB 에셋 자동 탐지 (Isaac Sim 설치 경로)
_extscache = os.path.expanduser("~/isaacsim/extscache")
_repl_dir = next(
    (d for d in os.listdir(_extscache) if d.startswith("omni.replicator.core")),
    ""
)
CRACKER_BOX_USD = os.path.join(
    _extscache, _repl_dir,
    "omni/replicator/core/tests/data/objects/003_cracker_box_physics.usd"
) if _repl_dir else ""

# ── Bin 파라미터: config/bin_geometry.yaml이 유일한 출처 (MoveIt 쪽
#    rb5_binpicking/bin_geometry.py와 동일 파일을 읽음). 여기서 하드코딩된
#    사본을 두지 않는다 — 값을 바꾸려면 그 YAML만 고치면 된다.
#    (Isaac Sim은 별도 파이썬 인터프리터라 ROS 패키지 import 대신 plain YAML
#    reader를 씀 — rb5_binpicking/bin_geometry.py의 docstring 참고.)


def _find_bin_geometry_yaml() -> str:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
        if not prefix:
            continue
        candidate = os.path.join(prefix, "share", "rb5_binpicking", "config", "bin_geometry.yaml")
        if os.path.isfile(candidate):
            return candidate
    # source-tree fallback for an uninstalled development checkout
    candidate = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config", "bin_geometry.yaml")
    )
    return candidate if os.path.isfile(candidate) else ""

def _load_bin_geometry():
    path = _find_bin_geometry_yaml()
    if not path:
        sys.stderr.write(
            "[ERROR] bin_geometry.yaml not found via AMENT_PREFIX_PATH or the "
            "source-tree fallback. Source ~/asl_ws/Manipulator/install/setup.bash "
            "before running this script, or check rb5_binpicking/config/.\n"
        )
        simulation_app.close(); sys.exit(1)
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    try:
        src, dst = data["source_bin"], data["destination_bin"]
        src_center = [float(v) for v in src["center"]]
        src_size   = [float(v) for v in src["inner_size"]]
        src_wall   = float(src["wall_thickness"])
        dst_center = [float(v) for v in dst["center"]]
        dst_size   = [float(v) for v in dst["inner_size"]]
    except (KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"[ERROR] bin_geometry.yaml malformed: {exc}\n")
        simulation_app.close(); sys.exit(1)                     
    sys.stderr.write(f"[INFO] bin geometry loaded from {path}\n")
    return src_center, src_size, src_wall, dst_center, dst_size


(
    (BIN_X, BIN_Y, BIN_Z),
    (BIN_W, BIN_D, BIN_H),
    WALL_T,
    (DEST_X, DEST_Y, DEST_Z),
    (DEST_W, DEST_D, DEST_H),
) = _load_bin_geometry()

# ── Demonstration target object ─────────────────────────────────────────────
TARGET_OBJECT_PATH = "/World/Objects/Cube0"
OBJECT_POSE_TOPIC = "/binpicking/object_pose"
OBJECT_POSE_HZ = 10.0

# ── 초기화 ───────────────────────────────────────────────────────────────────
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

if not os.path.exists(URDF_PATH):
    carb.log_error(f"URDF not found: {URDF_PATH}")
    simulation_app.close(); sys.exit(1)

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# 카메라 뷰포트: bin 위에서 비스듬히 내려다봄
viewports.set_camera_view(
    eye=np.array([BIN_X - 0.5, -0.8, 0.9]),
    target=np.array([BIN_X, BIN_Y, BIN_H / 2]),
)

# ── URDF 임포트 ──────────────────────────────────────────────────────────────
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
    carb.log_error("URDF import failed"); simulation_app.close(); sys.exit(1)

robot_prim_path        = "/" + str(stage_path).strip("/").split("/")[0]
articulation_root_path = robot_prim_path + "/link0"
sys.stderr.write(f"[INFO] robot: {robot_prim_path}\n")

# ── ArticulationRootAPI: root_joint → link0 ──────────────────────────────────
_stage = omni.usd.get_context().get_stage()
for api_cls in [UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI]:
    rj = _stage.GetPrimAtPath(robot_prim_path + "/root_joint")
    if rj.IsValid() and rj.HasAPI(api_cls):
        rj.RemoveAPI(api_cls)

_link0 = _stage.GetPrimAtPath(articulation_root_path)
UsdPhysics.ArticulationRootAPI.Apply(_link0)
_physx_art = PhysxSchema.PhysxArticulationAPI.Apply(_link0)
_physx_art.CreateEnabledSelfCollisionsAttr().Set(False)

# ── 조인트 드라이브 설정 ─────────────────────────────────────────────────────
for link, jnt in [
    ("link0","base"), ("link1","shoulder"), ("link2","elbow"),
    ("link3","wrist1"), ("link4","wrist2"), ("link5","wrist3"),
]:
    p = _stage.GetPrimAtPath(f"{robot_prim_path}/{link}/{jnt}")
    if p.IsValid():
        p.GetAttribute("drive:angular:physics:stiffness").Set(10000.0)
        p.GetAttribute("drive:angular:physics:damping").Set(1000.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1e9)

GRIPPER_PRIMARY_JOINT = "robotiq_85_left_knuckle_joint"

# left_knuckle is the only truly actuated DOF; the other 5 are passive
# followers that should track the same schedule (see trajectory_bridge.py
# GRIPPER_MIMIC) but have no real mechanical linkage enforcing that in this
# open-chain URDF import. Previously all 6 were driven with equally strong
# gains (7000/350/1e6): in free space that is harmless (nothing resists, so
# all 6 reach their scheduled target together), but the instant the fingertip
# touches an object, the primary knuckle is *meant* to stall against contact
# (that stall is how grasp success is detected) while the 5 followers -- each
# meeting different local resistance from the object -- kept forcing their
# own way toward their own scheduled target at the same high maxForce. That
# divergence is consistent with the reported "closes while lifting slightly,
# ends up gripping only the object's top edge": the fingertip pad tilts/climbs
# instead of staying flat because it keeps pushing through contact rather
# than yielding to it.
#
# A PhysX PhysxMimicJointAPI gear constraint would fix this exactly (follower
# angle tied to the primary's *actual* angle every substep) but that was
# tried and reverted -- it couples back into the whole articulation (PhysX
# docs: mimic joints are a two-way interaction, applying a reaction impulse
# to the reference joint too) and visibly destabilized the arm itself.
# This is a much smaller, lower-risk mitigation confined to independent
# per-joint gains (no cross-joint constraint, so it cannot propagate to the
# arm): make the followers weaker than the primary, so under contact
# resistance they yield/stall instead of forcing through it.
#
# Only the two finger_tip joints are the ones that actually touch an object,
# so only they should be softened for that reason. right_knuckle_joint and
# both inner_knuckle joints are different: robotiq_85_left/right_inner_knuckle_link
# are dead-end links in this URDF -- they never appear as the parent of any
# other joint. On the real hardware they'd pin-connect to the finger_tip
# link, closing the 4-bar loop, but that connecting joint doesn't exist here,
# so nothing physically holds them "aligned" with the finger assembly except
# their own drive tracking its commanded angle precisely. Softening them
# (README2.md §7.27) let gravity/disturbance pull them off that commanded
# angle, visibly separating them from the finger assembly -- a "part fell
# off" look (README2.md §7.28), not a grasp problem. Keeping them strong
# costs nothing since they never contact anything to yield to.
GRIPPER_SOFT_FOLLOWER_JOINTS = {
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
}
GRIPPER_FOLLOWER_JOINTS = GRIPPER_SOFT_FOLLOWER_JOINTS | {
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
}

for link, jnt in [
    ("robotiq_85_base_link",         "robotiq_85_left_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_right_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_left_inner_knuckle_joint"),
    ("robotiq_85_base_link",         "robotiq_85_right_inner_knuckle_joint"),
    ("robotiq_85_left_finger_link",  "robotiq_85_left_finger_tip_joint"),
    ("robotiq_85_right_finger_link", "robotiq_85_right_finger_tip_joint"),
]:
    p = _stage.GetPrimAtPath(f"{robot_prim_path}/{link}/{jnt}")
    if not p.IsValid():
        sys.stderr.write(f"[WARN] gripper joint not found: {jnt}\n"); continue
    stiff = p.GetAttribute("drive:angular:physics:stiffness")
    if not stiff or not str(stiff.GetTypeName()): continue
    if jnt in GRIPPER_SOFT_FOLLOWER_JOINTS:
        # Was 1200/150/3e4 (README2.md §7.19) -- still stiff enough that
        # under contact momentum a follower could overshoot past where the
        # primary knuckle actually stalled, letting the fingertip pad mesh
        # tilt and an edge poke inward into the object instead of the pad
        # closing flat (README2.md §7.27). Softened further so they yield
        # to contact resistance sooner; still enough to move the light
        # finger-tip links briskly in free air. Only applied to the two
        # finger_tip joints now -- see GRIPPER_SOFT_FOLLOWER_JOINTS comment
        # above for why the knuckle-family joints must NOT be softened
        # (README2.md §7.28).
        stiff.Set(600.0)
        p.GetAttribute("drive:angular:physics:damping").Set(120.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1.5e4)
    elif jnt in GRIPPER_FOLLOWER_JOINTS:
        # right_knuckle / both inner_knuckle joints: never touch the object,
        # so no reason to yield to contact -- keep them strong (same as the
        # primary) so they track their commanded angle precisely and don't
        # visibly sag away from the finger assembly (README2.md §7.28).
        stiff.Set(7000.0)
        p.GetAttribute("drive:angular:physics:damping").Set(350.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1e6)
    else:
        # Bumped from 5000/200 -- objects were slipping out mid-transfer
        # (reported: drops while swinging to the destination bin). Grip
        # force at a given finger gap scales with stiffness, so this holds
        # objects with less positional slack instead of needing a deeper
        # mechanical bite.
        stiff.Set(7000.0)
        p.GetAttribute("drive:angular:physics:damping").Set(350.0)
        p.GetAttribute("drive:angular:physics:maxForce").Set(1e6)
    tgt = p.GetAttribute("drive:angular:physics:targetPosition")
    if tgt and str(tgt.GetTypeName()): tgt.Set(0.0)

sys.stderr.write("[INFO] joint drives OK\n")
simulation_app.update()

# ── 그리퍼 fingertip 마찰력 ──────────────────────────────────────────────────
# URDF의 robotiq_85_left/right_finger_tip_link collision에는 이미
# <surface><friction><ode><mu1>100000.0</mu1>...>가 박혀 있어서 마찰을 최대로
# 의도했던 것으로 보이지만, 이건 Gazebo/ODE 전용 <gazebo> 확장 태그라 Isaac Sim의
# URDF importer는 표준 URDF만 읽고 이 블록을 완전히 무시함 -- 즉 지금까지
# fingertip은 PhysX 기본 마찰 계수(대략 0.5)로만 시뮬레이션됐고, 원래 의도했던
# "최대 마찰"은 전혀 반영이 안 되고 있었음. transfer 중 물체가 미끄러진다는
# 피드백과 맞아떨어짐. PhysX PhysicsMaterial로 명시적으로 다시 설정.
def _apply_high_friction(stage, link_prim_path, static_friction=1.2, dynamic_friction=1.0):
    link_prim = stage.GetPrimAtPath(link_prim_path)
    if not link_prim.IsValid():
        sys.stderr.write(f"[WARN] friction target not found: {link_prim_path}\n")
        return
    material = UsdShade.Material.Define(stage, Sdf.Path(link_prim_path + "/HighFrictionMaterial"))
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(0.0)
    # "max" combine: the effective friction at a contact is the larger of the
    # two materials' values, so this fingertip material dominates regardless
    # of what (default, untouched) friction the picked object itself has.
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr().Set("max")
    UsdShade.MaterialBindingAPI.Apply(link_prim).Bind(material, materialPurpose="physics")
    sys.stderr.write(f"[OK] high-friction physics material bound: {link_prim_path}\n")

for _fname in ("robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"):
    _apply_high_friction(_stage, f"{robot_prim_path}/{_fname}")

# ── 물리 씬 + 바닥 + 조명 ───────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/physicsScene").IsValid():
    sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)

omni.kit.commands.execute(
    "AddGroundPlaneCommand", stage=stage, planePath="/GroundPlane",
    axis="Z", size=5.0, position=Gf.Vec3f(0, 0, 0), color=Gf.Vec3f(0.45),
)
UsdLux.DomeLight.Define(stage, "/DomeLight").CreateIntensityAttr(1000)

# bin 위 스팟 조명 (실사감 향상)
rl = UsdLux.RectLight.Define(stage, "/BinSpotLight")
rl.CreateIntensityAttr(8000)
rl.CreateWidthAttr(0.5); rl.CreateHeightAttr(0.5)
xf = UsdGeom.Xformable(rl)
xf.AddTranslateOp().Set(Gf.Vec3d(BIN_X, BIN_Y, 1.4))
xf.AddRotateXYZOp().Set(Gf.Vec3d(-90, 0, 0))

# ── Bin 컨테이너 ─────────────────────────────────────────────────────────────
_bc = np.array([0.14, 0.14, 0.16])  # 소스 bin: 어두운 플라스틱
_dc = np.array([0.55, 0.48, 0.28])  # 데스티네이션 bin: 베이지/탄 색

def _fixed(path, pos, w, d, h, color=None):
    FixedCuboid(prim_path=path,
                position=np.array(pos),
                scale=np.array([w, d, h]),
                color=color if color is not None else _bc)

# 소스 bin (물건이 담긴 bin, 어두운 색)
_fixed("/World/Bin/Floor",  [BIN_X, BIN_Y, BIN_Z + WALL_T/2],                          BIN_W,  BIN_D,  WALL_T)
_fixed("/World/Bin/WallXp", [BIN_X + BIN_W/2 - WALL_T/2, BIN_Y, BIN_Z + BIN_H/2],     WALL_T, BIN_D,  BIN_H)
_fixed("/World/Bin/WallXn", [BIN_X - BIN_W/2 + WALL_T/2, BIN_Y, BIN_Z + BIN_H/2],     WALL_T, BIN_D,  BIN_H)
_fixed("/World/Bin/WallYp", [BIN_X, BIN_Y + BIN_D/2 - WALL_T/2, BIN_Z + BIN_H/2],     BIN_W,  WALL_T, BIN_H)
_fixed("/World/Bin/WallYn", [BIN_X, BIN_Y - BIN_D/2 + WALL_T/2, BIN_Z + BIN_H/2],     BIN_W,  WALL_T, BIN_H)

# 데스티네이션 bin (옮겨 담을 bin, 베이지 색)  — 로봇 왼쪽 (y+ 방향)
_fixed("/World/DestBin/Floor",  [DEST_X, DEST_Y, DEST_Z + WALL_T/2],                           DEST_W, DEST_D, WALL_T, _dc)
_fixed("/World/DestBin/WallXp", [DEST_X + DEST_W/2 - WALL_T/2, DEST_Y, DEST_Z + DEST_H/2],    WALL_T, DEST_D, DEST_H, _dc)
_fixed("/World/DestBin/WallXn", [DEST_X - DEST_W/2 + WALL_T/2, DEST_Y, DEST_Z + DEST_H/2],    WALL_T, DEST_D, DEST_H, _dc)
_fixed("/World/DestBin/WallYp", [DEST_X, DEST_Y + DEST_D/2 - WALL_T/2, DEST_Z + DEST_H/2],    DEST_W, WALL_T, DEST_H, _dc)
_fixed("/World/DestBin/WallYn", [DEST_X, DEST_Y - DEST_D/2 + WALL_T/2, DEST_Z + DEST_H/2],    DEST_W, WALL_T, DEST_H, _dc)

sys.stderr.write("[OK] source bin (dark) + destination bin (beige) created\n")

# ── bin 속 오브젝트 ──────────────────────────────────────────────────────────
# z 위에서 떨어뜨려 자연스럽게 쌓이게 함
random.seed(7)

def _rpos(z_offset=0.0):
    """bin 내부 랜덤 XY, z는 bin 상단 위.

    Wall clearance buffer was 0.03m -- with the fixed random.seed(7) below,
    that put Cube0 (TARGET_OBJECT_PATH, the tracked pick object) only
    ~5cm from the near wall (wall_xn) every single run, deterministically.
    That's the real reason grasp attempts kept dying against that wall
    (README2.md §7.23) -- moving the whole bin farther from the robot
    (§7.22) didn't help because BIN_X and this offset shift together, so
    Cube0's distance *from the wall* stayed the same regardless. Since the
    offset is a fixed fraction of this margin (same seed -> same underlying
    draw), only widening the margin itself changes the resulting clearance.
    0.10 puts Cube0 ~11cm from the near wall instead of ~5cm (verified by
    replaying the same seeded sequence with the new margin).
    """
    ix = BIN_W / 2 - WALL_T - 0.10
    iy = BIN_D / 2 - WALL_T - 0.10
    return np.array([
        BIN_X + random.uniform(-ix, ix),
        BIN_Y + random.uniform(-iy, iy),
        BIN_Z + BIN_H + 0.05 + z_offset,
    ])

PALETTE = [
    np.array([0.85, 0.15, 0.15]),  # 빨강
    np.array([0.15, 0.72, 0.15]),  # 초록
    np.array([0.15, 0.35, 0.90]),  # 파랑
    np.array([0.90, 0.75, 0.10]),  # 노랑
    np.array([0.75, 0.15, 0.75]),  # 보라
    np.array([0.90, 0.50, 0.10]),  # 주황
    np.array([0.10, 0.72, 0.72]),  # 청록
    np.array([0.55, 0.55, 0.55]),  # 회색
]

# 구 (3개)
for i in range(3):
    DynamicSphere(
        prim_path=f"/World/Objects/Sphere{i}",
        radius=0.022,
        position=_rpos(i * 0.06),
        color=PALETTE[i],
        mass=0.08,
    )

# 정육면체 박스 (3개)
for i in range(3):
    DynamicCuboid(
        prim_path=f"/World/Objects/Cube{i}",
        position=_rpos(i * 0.07),
        scale=np.array([0.042, 0.042, 0.042]),
        color=PALETTE[i + 3],
        mass=0.10,
    )

# 원통 (캔, 2개)
for i in range(2):
    DynamicCylinder(
        prim_path=f"/World/Objects/Cyl{i}",
        radius=0.020,
        height=0.065,
        position=_rpos(i * 0.08),
        color=PALETTE[i + 5],
        mass=0.06,
    )

# YCB 003_cracker_box (로컬 USD 에셋)
if CRACKER_BOX_USD and os.path.exists(CRACKER_BOX_USD):
    from omni.isaac.core.utils.stage import add_reference_to_stage
    for i in range(2):
        pp = f"/World/Objects/CrackerBox{i}"
        add_reference_to_stage(usd_path=CRACKER_BOX_USD, prim_path=pp)
        pr = stage.GetPrimAtPath(pp)
        if pr.IsValid():
            pos = _rpos(0.10 + i * 0.08)
            UsdGeom.XformCommonAPI(pr).SetTranslate(
                Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
            )
    sys.stderr.write("[OK] YCB cracker box loaded\n")
else:
    sys.stderr.write("[WARN] cracker box USD not found — skipping YCB asset\n")

sys.stderr.write(f"[OK] {3+3+2} procedural + up to 2 YCB objects spawned\n")
simulation_app.update()

# ── 카메라 prim 정의 (play() 전에 미리 생성) ────────────────────────────────
# 예전 방식: camera_link 밑에 새 prim을 만들고 RotateY(-90)+RotateZ(90)를 눈대중
# 으로 맞춰서 "얼추 맞는" 시야 방향을 만들었음 (주석부터 "makes that view vector
# match..."로, 수식이 아니라 경험적 튜닝임을 명시하고 있었음). roll까지 정확히
# 안 맞으면 depth->pointcloud 역투영에 회전 오차가 생기고, 그걸 pcd_offset_*
# 같은 평행이동 patch로만 보정하면 특정 거리에서만 맞고 다른 거리에선 계속
# 어긋남 (README2.md 카메라 transform 이슈).
#
# 새 방식: URDF의 camera_depth_optical_frame / camera_color_optical_frame 링크를
# 그대로 씀. import_config.merge_fixed_joints=False라서 이 fixed-joint 링크들도
# 독립 prim으로 보존되어 있고(질량/geometry 없는 링크라도), camera_depth_optical_joint/
# camera_color_optical_joint의 rpy로 URDF가 이미 해석적으로 정의해둔 "진짜" ROS
# 광학 좌표계(X=right, Y=down, Z=forward)임 → TF가 말하는 자세와 렌더링에 실제
# 쓰이는 자세가 항상 정확히 일치. 여기에 딱 하나, ROS 광학 좌표계(+Z가 정면) →
# Isaac/USD 카메라 좌표계(-Z가 정면) 변환만 추가하면 되는데, 이건 축 하나짜리
# 180도 회전(Rx(180°): X=right 그대로, Y=down→up, Z=forward→backward)이라 순서
# 모호성이 없음. 이 회전은 camera_depth_optical_frame '자체'에 걸면 TF로 나가는
# 값이 오염되므로, 위치 오프셋 0인 자식 prim을 하나 만들어서 그 자식에만 건다.
def _find_prim(*candidates):
    for c in candidates:
        if _stage.GetPrimAtPath(c).IsValid():
            return c
    return None


def _print_camera_alignment_check(optical_frame_path, render_prim_path, label):
    optical_prim = _stage.GetPrimAtPath(optical_frame_path)
    render_prim = _stage.GetPrimAtPath(render_prim_path)
    if not (optical_prim.IsValid() and render_prim.IsValid()):
        return
    t_optical = UsdGeom.Xformable(optical_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t_render = UsdGeom.Xformable(render_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p_optical = Gf.Vec3d(t_optical.ExtractTranslation())
    p_render = Gf.Vec3d(t_render.ExtractTranslation())
    diff_mm = (p_optical - p_render).GetLength() * 1000.0
    sys.stderr.write(
        f"[CHECK] {label}: optical_frame pos={tuple(round(v, 4) for v in p_optical)} "
        f"render prim pos={tuple(round(v, 4) for v in p_render)} diff={diff_mm:.2f}mm "
        f"(should be ~0mm -- render prim differs from the TF frame only by a fixed "
        f"local Rx(180) rotation, zero translation)\n"
    )


_depth_optical_frame = _find_prim(f"{robot_prim_path}/camera_depth_optical_frame")
_color_optical_frame = _find_prim(f"{robot_prim_path}/camera_color_optical_frame")

if _depth_optical_frame and _color_optical_frame:
    CAMERA_DEPTH_PATH = _depth_optical_frame + "/isaac_view"
    CAMERA_COLOR_PATH = _color_optical_frame + "/isaac_view"

    _depth_prim = UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_DEPTH_PATH))
    _color_prim = UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_COLOR_PATH))
    UsdGeom.Xformable(_depth_prim).AddRotateXOp().Set(180.0)
    UsdGeom.Xformable(_color_prim).AddRotateXOp().Set(180.0)

    sys.stderr.write(
        "[OK] Camera sensors attached directly to URDF camera_depth_optical_frame / "
        "camera_color_optical_frame (exact TF match, no empirical rotation tuning).\n"
    )
    simulation_app.update()
    _print_camera_alignment_check(_depth_optical_frame, CAMERA_DEPTH_PATH, "depth")
    _print_camera_alignment_check(_color_optical_frame, CAMERA_COLOR_PATH, "color")
else:
    # Fallback: the optical-frame links weren't found under the imported
    # articulation (unexpected given merge_fixed_joints=False, but don't hard
    # -fail). View direction will be approximately right but roll may not
    # exactly match TF's camera_depth_optical_frame -- the old, empirically
    # tuned path.
    sys.stderr.write(
        "[WARN] camera_depth_optical_frame/camera_color_optical_frame not found in the "
        "imported stage (expected to exist since merge_fixed_joints=False) -- falling "
        "back to manual rotation under camera_link. Depth/pointcloud geometry may not "
        "exactly match TF; see README2.md camera transform notes.\n"
    )
    _cam_link = _find_prim(
        f"{robot_prim_path}/camera_link",
        f"{robot_prim_path}/camera_bottom_screw_frame/camera_link",
    ) or f"{robot_prim_path}/camera_link"
    CAMERA_DEPTH_PATH = _cam_link + "/depth_camera"
    CAMERA_COLOR_PATH = _cam_link + "/color_camera"

    _depth_prim = UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_DEPTH_PATH))
    _color_prim = UsdGeom.Camera.Define(_stage, Sdf.Path(CAMERA_COLOR_PATH))
    UsdGeom.Xformable(_depth_prim).AddRotateYOp().Set(-90.0)
    UsdGeom.Xformable(_depth_prim).AddRotateZOp().Set(90.0)
    UsdGeom.Xformable(_color_prim).AddRotateYOp().Set(-90.0)
    UsdGeom.Xformable(_color_prim).AddRotateZOp().Set(90.0)

sys.stderr.write(f"[INFO] depth cam prim: {CAMERA_DEPTH_PATH}\n")
sys.stderr.write(f"[INFO] color cam prim: {CAMERA_COLOR_PATH}\n")

# ── OmniGraph: ROS2 bridge — color/depth 각각 별도 RenderProduct ────────────
try:
    og.Controller.edit(
        {"graph_path": "/ROS2Graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulse",    "omni.graph.action.OnImpulseEvent"),
                ("SimTime",      "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2Ctx",      "omni.isaac.ros2_bridge.ROS2Context"),
                ("PubJS",        "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                ("SubArm",       "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("ArmCtrl",      "omni.isaac.core_nodes.IsaacArticulationController"),
                ("SubGrip",      "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("GripCtrl",     "omni.isaac.core_nodes.IsaacArticulationController"),
                ("PubClock",     "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ("PubTF",        "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
                ("RenderProdC",  "omni.isaac.core_nodes.IsaacCreateRenderProduct"),  # color
                ("RenderProdD",  "omni.isaac.core_nodes.IsaacCreateRenderProduct"),  # depth
                ("CamRGB",       "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("CamDepth",     "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("CamRGBInfo",   "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("CamDepthInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulse.outputs:execOut", "PubJS.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "SubArm.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "PubClock.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "ArmCtrl.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "SubGrip.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "GripCtrl.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "PubTF.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "RenderProdC.inputs:execIn"),
                ("OnImpulse.outputs:execOut", "RenderProdD.inputs:execIn"),
                ("ROS2Ctx.outputs:context",   "PubJS.inputs:context"),
                ("ROS2Ctx.outputs:context",   "SubArm.inputs:context"),
                ("ROS2Ctx.outputs:context",   "PubClock.inputs:context"),
                ("ROS2Ctx.outputs:context",   "PubTF.inputs:context"),
                ("ROS2Ctx.outputs:context",   "SubGrip.inputs:context"),
                ("ROS2Ctx.outputs:context",   "CamRGB.inputs:context"),
                ("ROS2Ctx.outputs:context",   "CamDepth.inputs:context"),
                ("ROS2Ctx.outputs:context",   "CamRGBInfo.inputs:context"),
                ("ROS2Ctx.outputs:context",   "CamDepthInfo.inputs:context"),
                ("SimTime.outputs:simulationTime", "PubJS.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
                ("SubArm.outputs:jointNames",       "ArmCtrl.inputs:jointNames"),
                ("SubArm.outputs:positionCommand",  "ArmCtrl.inputs:positionCommand"),
                ("SubArm.outputs:velocityCommand",  "ArmCtrl.inputs:velocityCommand"),
                ("SubArm.outputs:effortCommand",    "ArmCtrl.inputs:effortCommand"),
                ("SubGrip.outputs:jointNames",      "GripCtrl.inputs:jointNames"),
                ("SubGrip.outputs:positionCommand", "GripCtrl.inputs:positionCommand"),
                ("SubGrip.outputs:velocityCommand", "GripCtrl.inputs:velocityCommand"),
                # color camera chain
                ("RenderProdC.outputs:execOut",           "CamRGB.inputs:execIn"),
                ("RenderProdC.outputs:renderProductPath", "CamRGB.inputs:renderProductPath"),
                ("RenderProdC.outputs:execOut",           "CamRGBInfo.inputs:execIn"),
                ("RenderProdC.outputs:renderProductPath", "CamRGBInfo.inputs:renderProductPath"),
                # depth camera chain
                ("RenderProdD.outputs:execOut",           "CamDepth.inputs:execIn"),
                ("RenderProdD.outputs:renderProductPath", "CamDepth.inputs:renderProductPath"),
                ("RenderProdD.outputs:execOut",           "CamDepthInfo.inputs:execIn"),
                ("RenderProdD.outputs:renderProductPath", "CamDepthInfo.inputs:renderProductPath"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArmCtrl.inputs:robotPath",   articulation_root_path),
                ("PubJS.inputs:topicName",     "joint_states"),
                ("SubArm.inputs:topicName",    "isaac_joint_commands"),
                ("PubJS.inputs:targetPrim",    [usdrt.Sdf.Path(articulation_root_path)]),
                ("GripCtrl.inputs:robotPath",  articulation_root_path),
                ("SubGrip.inputs:topicName",   "gripper_joint_commands"),
                ("PubTF.inputs:targetPrims",   [usdrt.Sdf.Path(robot_prim_path)]),
                # color render product → color camera prim
                ("RenderProdC.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_COLOR_PATH)]),
                ("RenderProdC.inputs:width",  640),
                ("RenderProdC.inputs:height", 480),
                # depth render product → depth camera prim
                ("RenderProdD.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_DEPTH_PATH)]),
                ("RenderProdD.inputs:width",  640),
                ("RenderProdD.inputs:height", 480),
                ("CamRGB.inputs:topicName",   "camera/color/image_raw"),
                ("CamRGB.inputs:frameId",     "camera_color_optical_frame"),
                ("CamRGB.inputs:type",        "rgb"),
                ("CamRGBInfo.inputs:topicName", "camera/color/camera_info"),
                ("CamRGBInfo.inputs:frameId",   "camera_color_optical_frame"),
                ("CamRGBInfo.inputs:type",      "camera_info"),
                ("CamDepth.inputs:topicName", "camera/depth/image_rect_raw"),
                ("CamDepth.inputs:frameId",   "camera_depth_optical_frame"),
                ("CamDepth.inputs:type",      "depth"),
                ("CamDepthInfo.inputs:topicName", "camera/depth/camera_info"),
                ("CamDepthInfo.inputs:frameId",   "camera_depth_optical_frame"),
                ("CamDepthInfo.inputs:type",      "camera_info"),
            ],
        },
    )
    sys.stderr.write("[OK] ROS2 OmniGraph built\n")
except Exception as e:
    carb.log_error(f"OmniGraph failed: {e}")
    import traceback; traceback.print_exc()
    simulation_app.close(); sys.exit(1)

simulation_app.update()

# ── 물리 시작 (OmniGraph 빌드 후) ───────────────────────────────────────────
simulation_context.initialize_physics()
simulation_context.play()
simulation_app.update()

# ── D435i 카메라 intrinsics 설정 (play() 이후 렌더러 활성화 후) ──────────────
import math
from omni.isaac.sensor import Camera as IsaacCamera

# D435i color: 640×480, K matrix from camera_info topic
_W, _H   = 640, 480
_PX_UM   = 1.4   # pixel size μm (D435i)
_K_color = [612.418, 0.0, 309.723, 0.0, 612.362, 245.359, 0.0, 0.0, 1.0]
_K_depth = [390.0,   0.0, 320.0,   0.0, 390.0,   240.0,   0.0, 0.0, 1.0]

def _apply_d435i_intrinsics(cam, K, add_depth=False):
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    focal_mm = (fx + fy) / 2 * _PX_UM * 1e-3
    h_ap_mm  = _PX_UM * 1e-3 * _W
    v_ap_mm  = _PX_UM * 1e-3 * _H
    diag     = 2 * math.sqrt(max(cx, _W - cx)**2 + max(cy, _H - cy)**2)
    diag_fov = 2 * math.atan2(diag, fx + fy) * 180 / math.pi
    cam.set_focal_length(focal_mm / 10.0)          # mm → cm (Isaac Sim unit)
    cam.set_focus_distance(0.5)
    cam.set_lens_aperture(1.8 * 100.0)             # f/1.8
    cam.set_horizontal_aperture(h_ap_mm / 10.0)
    cam.set_vertical_aperture(v_ap_mm / 10.0)
    cam.set_clipping_range(0.1, 10.0)
    cam.set_projection_type("fisheyePolynomial")
    cam.set_rational_polynomial_properties(_W, _H, cx, cy, diag_fov, [0.0] * 8)
    if add_depth:
        cam.add_distance_to_image_plane_to_frame()

_cam_color = IsaacCamera(prim_path=CAMERA_COLOR_PATH, frequency=30, resolution=(_W, _H))
_cam_color.initialize()
_apply_d435i_intrinsics(_cam_color, _K_color)

_cam_depth = IsaacCamera(prim_path=CAMERA_DEPTH_PATH, frequency=30, resolution=(_W, _H))
_cam_depth.initialize()
_apply_d435i_intrinsics(_cam_depth, _K_depth, add_depth=True)

sys.stderr.write("[INFO] D435i color + depth cameras initialized\n")
simulation_app.update()


# ── Target object pose publisher ────────────────────────────────────────────
def _init_object_pose_publisher():
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
    except Exception as exc:
        sys.stderr.write(f"[WARN] rclpy unavailable; object pose topic disabled: {exc}\n")
        return None

    try:
        did_init = False
        if not rclpy.ok():
            rclpy.init(args=None)
            did_init = True
        node = rclpy.create_node("binpicking_object_pose_publisher")
        pub = node.create_publisher(PoseStamped, OBJECT_POSE_TOPIC, 10)
        sys.stderr.write(
            f"[INFO] Publishing target object pose: {TARGET_OBJECT_PATH} -> {OBJECT_POSE_TOPIC}\n"
        )
        return {
            "rclpy": rclpy,
            "PoseStamped": PoseStamped,
            "node": node,
            "pub": pub,
            "did_init": did_init,
        }
    except Exception as exc:
        sys.stderr.write(f"[WARN] object pose publisher disabled: {exc}\n")
        return None


def _target_object_pose_msg(pub_ctx):
    prim = _stage.GetPrimAtPath(TARGET_OBJECT_PATH)
    if not prim.IsValid():
        return None

    world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    trans = world_tf.ExtractTranslation()
    quat = world_tf.ExtractRotationQuat()
    quat_imag = quat.GetImaginary()

    # ExtractRotationQuat() on a matrix that also carries this prim's scale
    # (DynamicCuboid(scale=...)) does not reliably return a unit quaternion --
    # observed norm ~0.53 instead of 1.0 in practice. A non-normalized
    # quaternion is not a valid rotation and corrupts any downstream code
    # that consumes this orientation (e.g. grasp-orientation alignment).
    # Normalize explicitly before publishing.
    qx, qy, qz, qw = float(quat_imag[0]), float(quat_imag[1]), float(quat_imag[2]), float(quat.GetReal())
    q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_norm > 1e-9:
        qx, qy, qz, qw = qx / q_norm, qy / q_norm, qz / q_norm, qw / q_norm
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

    msg = pub_ctx["PoseStamped"]()
    msg.header.stamp = pub_ctx["node"].get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = float(trans[0])
    msg.pose.position.y = float(trans[1])
    msg.pose.position.z = float(trans[2])
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


_object_pose_pub_ctx = _init_object_pose_publisher()
_last_object_pose_pub_time = 0.0

sys.stderr.write("[OK] Bin picking scene running.\n")
sys.stderr.write(f"     Source bin  @ ({BIN_X:.2f}, {BIN_Y:.2f})  {int(BIN_W*100)}x{int(BIN_D*100)}x{int(BIN_H*100)} cm  [dark]\n")
sys.stderr.write(f"     Dest   bin  @ ({DEST_X:.2f}, {DEST_Y:.2f})  {int(DEST_W*100)}x{int(DEST_D*100)}x{int(DEST_H*100)} cm  [beige]\n")
sys.stderr.write("     Topics: /joint_states /tf /clock\n")
sys.stderr.write("             /camera/color/image_raw /camera/color/camera_info\n")
sys.stderr.write("             /camera/depth/image_rect_raw /camera/depth/camera_info\n")
sys.stderr.write(f"             {OBJECT_POSE_TOPIC} ({TARGET_OBJECT_PATH})\n")

while simulation_app.is_running():
    simulation_context.step(render=True)
    og.Controller.set(
        og.Controller.attribute("/ROS2Graph/OnImpulse.state:enableImpulse"), True
    )
    if _object_pose_pub_ctx is not None:
        now = time.monotonic()
        if now - _last_object_pose_pub_time >= 1.0 / OBJECT_POSE_HZ:
            msg = _target_object_pose_msg(_object_pose_pub_ctx)
            if msg is not None:
                _object_pose_pub_ctx["pub"].publish(msg)
            _object_pose_pub_ctx["rclpy"].spin_once(
                _object_pose_pub_ctx["node"], timeout_sec=0.0
            )
            _last_object_pose_pub_time = now

simulation_context.stop()
if _object_pose_pub_ctx is not None:
    _object_pose_pub_ctx["node"].destroy_node()
    if _object_pose_pub_ctx["did_init"]:
        _object_pose_pub_ctx["rclpy"].shutdown()
simulation_app.close()
