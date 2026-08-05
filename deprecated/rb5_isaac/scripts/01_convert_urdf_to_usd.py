"""
RB5-850E URDF to USD conversion script.
Run with:  ~/isaacsim/python.sh scripts/01_convert_urdf_to_usd.py
Output:    ~/rb5_isaac_assets/rb5_850e.usd

최초 1회 실행. URDF를 USD로 저장함.
"""

import os, sys
from isaacsim import SimulationApp

kit = SimulationApp({"headless": True})

import omni.kit.commands
from pxr import Usd, Sdf

URDF_PATH = os.path.expanduser(
    "~/asl_ws/install/rbpodo_description/share/rbpodo_description/robots/rb5_850e.urdf"
)

OUTPUT_DIR  = os.path.expanduser("~/rb5_isaac_assets")
OUTPUT_USD  = os.path.join(OUTPUT_DIR, "rb5_850e.usd")
LOG_FILE    = "/tmp/urdf2usd.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

open(LOG_FILE, "w").close()   # reset log
log(f"[INFO] URDF: {URDF_PATH}")
log(f"[INFO] USD:  {OUTPUT_USD}")

# ── 1. Import ────────────────────────────────────────────────────────────────
_, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
import_config.merge_fixed_joints = False
import_config.convex_decomp      = False
import_config.import_inertia_tensor = True
import_config.fix_base            = True
import_config.distance_scale      = 1.0
import_config.make_instanceable   = True

status, stage_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    get_articulation_root=True,
    dest_path=OUTPUT_USD,
)

log(f"[INFO] import status={status}  stage_path={stage_path}")

# ── 2. Set defaultPrim on the saved USD (before kit.close) ───────────────────
if status and os.path.exists(OUTPUT_USD):
    saved = Usd.Stage.Open(OUTPUT_USD)
    if saved:
        # list all prims
        for p in saved.Traverse():
            log(f"  prim: {p.GetPath()} [{p.GetTypeName()}]")
        # use the top-level path from stage_path
        top = "/" + str(stage_path).strip("/").split("/")[0] if stage_path else None
        candidate = saved.GetPrimAtPath(top) if top else None
        if candidate and candidate.IsValid():
            saved.SetDefaultPrim(candidate)
            saved.GetRootLayer().Save()
            log(f"[OK] defaultPrim set to {top} and saved")
        else:
            log(f"[WARN] prim {top} not found in saved USD")
    else:
        log("[ERROR] Could not open saved USD")
else:
    log("[ERROR] Import failed or USD not found")

kit.close()
