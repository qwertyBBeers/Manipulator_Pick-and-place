"""Shared grasp-state machinery for the curriculum stages (Reach / GraspLift /
Transport / Curriculum). Centralized here so `observations.py`, `rewards.py`,
`terminations.py`, and each stage's env cfg all read the same definitions
instead of re-deriving them (avoids the "grasp defined differently in three
places" bug class).

Two things live here that don't exist anywhere else in this package yet:

1. **The reference grasp orientation.** Not invented -- ported directly from
   the working ROS/MoveIt pipeline's `moveit_pick_place.py`
   (`SAVED_POSES["gripping"]["orientation"]`, ROS xyzw convention, converted
   to IsaacLab's wxyz convention below) combined with
   `generate_grasp_candidates()`'s own `yaw_quaternion(object_yaw) *
   gripping_orientation` composition -- i.e. this is the exact top-down
   grasp orientation formula already live-validated on this robot, not a
   guess.
2. **Per-env scratch counters** (consecutive-step conditions, "ever grasped"
   sticky flags). `ManagerBasedRLEnv` has no built-in generic mutable state
   slot for custom MDP terms, so these are lazily attached as an attribute
   on the `env` object itself (`env._rb5_pp_state`), reset by
   `events.reset_rb5_pp_state` (mode="reset") and updated idempotently once
   per step (keyed off `env.common_step_counter`, since both a reward term
   and a termination term may read/advance the same counter in the same
   step -- see `_step_once`).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from rb5_isaaclab.robots.rb5_850e import FINGERTIP_LINK_NAMES, GRIPPER_PRIMARY_JOINT

# ---------------------------------------------------------------------------
# Grasp geometry constants
# ---------------------------------------------------------------------------

# ROS xyzw [0.532, 0.528, 0.458, 0.478] -> IsaacLab wxyz.
GRIPPING_QUAT_WXYZ = (0.478, 0.532, 0.528, 0.458)

# Height above the object's own center the pre-grasp target sits at. Object
# is a 0.042m cube (half-height 0.021m); the Robotiq 2F-85 fingertip pad
# starts ~0.02-0.03m below the tcp frame at full open, so ~0.10m clears the
# bin wall (bin wall_thickness 0.007m + object half-height) comfortably
# without the arm having to dive in past the wrist. Matches the
# `pre_grasp_offset_z = 0.08 to 0.12 m` range requested; picked the midpoint
# since (unlike the ROS pipeline's `grasp_offset_z`, which was tuned live
# against a real/simulated gripper over many trials) this hasn't been
# empirically validated for the RL task yet.
PRE_GRASP_OFFSET_Z = 0.10

# Passed to place_and_release/object_placed-style checks throughout.
DEFAULT_POSITION_THRESHOLD = 0.03
DEFAULT_VELOCITY_THRESHOLD = 0.05
DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.4  # rad, primary knuckle angle
DEFAULT_ORIENTATION_THRESHOLD = 0.20  # rad, matches terminations.reach_success's default

# Contact sensors (see rb5_850e.py's FINGERTIP_LINK_NAMES, confirmed against
# the converted USD by inspection -- not guessed) are registered under these
# scene attribute / sensor names by `add_grasp_contact_sensors` below.
LEFT_CONTACT_SENSOR_NAME = "left_fingertip_contact"
RIGHT_CONTACT_SENSOR_NAME = "right_fingertip_contact"

# Fingertip-vs-floor (not object) -- see `add_floor_contact_sensors` /
# `rewards.floor_contact_penalty`. Separate sensors, not an extra filter
# path added to the object sensors above, because `fingertip_contact_forces`'s
# `.view(num_envs, 3)` assumes exactly one filtered body (M=1) -- adding a
# second filter target would change `force_matrix_w`'s shape to (N, 1, 2, 3)
# and silently break that reshape.
LEFT_FLOOR_CONTACT_SENSOR_NAME = "left_fingertip_floor_contact"
RIGHT_FLOOR_CONTACT_SENSOR_NAME = "right_fingertip_floor_contact"

# Contact force (N) above which a fingertip is considered "touching the
# object" -- the object is light (0.10kg, ~1N weight), so this just needs to
# be well above simulation noise floor, not a realistic force-controlled
# grasp threshold.
CONTACT_FORCE_THRESHOLD = 0.5

# Number of consecutive control steps a condition must hold before counting
# as "stable" (grasp) or "success" (reach/place) -- avoids single-frame
# physics-noise flukes counting as success.
STABLE_GRASP_STEPS = 5


def add_grasp_contact_sensors(scene_cfg) -> None:
    """Attach fingertip-vs-object contact sensors to a scene cfg instance.

    Call from a stage's `__post_init__` (mirrors IsaacLab's own
    `dexsuite_kuka_allegro_env_cfg.py` pattern of `setattr`-ing
    `ContactSensorCfg` entries onto an already-constructed scene cfg,
    verified against that file rather than invented). Only stages that need
    grasp detection (GraspLift, Transport, Curriculum) call this -- Reach
    doesn't need it.
    """
    left_link, right_link = FINGERTIP_LINK_NAMES
    setattr(
        scene_cfg,
        LEFT_CONTACT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + left_link,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ),
    )
    setattr(
        scene_cfg,
        RIGHT_CONTACT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + right_link,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ),
    )


def add_floor_contact_sensors(scene_cfg, floor_prim_path: str = "{ENV_REGEX_NS}/SourceBinFloor") -> None:
    """Attach fingertip-vs-floor contact sensors to a scene cfg instance --
    used by `rewards.floor_contact_penalty` to penalize genuine floor
    contact directly, replacing an earlier height-based proxy that risked
    penalizing legitimate grasp descent (the "right" height threshold turned
    out to be genuinely unclear -- see `reach_grasp_env_cfg.py`'s git
    history / 2026-08-04 discussion). Requires the scene to have a floor
    asset at `floor_prim_path` (true for every stage's `SourceBinFloor`)."""
    left_link, right_link = FINGERTIP_LINK_NAMES
    setattr(
        scene_cfg,
        LEFT_FLOOR_CONTACT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + left_link,
            filter_prim_paths_expr=[floor_prim_path],
        ),
    )
    setattr(
        scene_cfg,
        RIGHT_FLOOR_CONTACT_SENSOR_NAME,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + right_link,
            filter_prim_paths_expr=[floor_prim_path],
        ),
    )


def fingertip_floor_contact_forces(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(left_force_mag, right_force_mag) against the floor, each (num_envs,)
    -- same shape/reshape reasoning as `fingertip_contact_forces`."""
    left: ContactSensor = env.scene.sensors[LEFT_FLOOR_CONTACT_SENSOR_NAME]
    right: ContactSensor = env.scene.sensors[RIGHT_FLOOR_CONTACT_SENSOR_NAME]
    left_force = left.data.force_matrix_w.view(env.num_envs, 3)
    right_force = right.data.force_matrix_w.view(env.num_envs, 3)
    return torch.norm(left_force, dim=-1), torch.norm(right_force, dim=-1)


def any_floor_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bool (num_envs,): either fingertip touching the floor above
    `CONTACT_FORCE_THRESHOLD` -- unlike `bilateral_contact`, this is an OR
    (one finger clipping the floor is already the failure mode being
    guarded against), not an AND."""
    left_force, right_force = fingertip_floor_contact_forces(env)
    return (left_force > CONTACT_FORCE_THRESHOLD) | (right_force > CONTACT_FORCE_THRESHOLD)


def fingertip_contact_forces(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(left_force_mag, right_force_mag), each shape (num_envs,).

    `force_matrix_w` is (N, B, M, 3) with B=1 (one sensor body: the
    fingertip link) and M=1 (one filtered body: the object) here, per
    `ContactSensorCfg`'s own docstring -- `.view(num_envs, 3)` flattens that,
    matching the verified pattern in
    `isaaclab_tasks/.../dexsuite/mdp/rewards.py::contacts`.
    """
    left: ContactSensor = env.scene.sensors[LEFT_CONTACT_SENSOR_NAME]
    right: ContactSensor = env.scene.sensors[RIGHT_CONTACT_SENSOR_NAME]
    left_force = left.data.force_matrix_w.view(env.num_envs, 3)
    right_force = right.data.force_matrix_w.view(env.num_envs, 3)
    return torch.norm(left_force, dim=-1), torch.norm(right_force, dim=-1)


def bilateral_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bool (num_envs,): both fingertips simultaneously in contact with the
    object above `CONTACT_FORCE_THRESHOLD`. Contact-based, NOT gripper-angle
    based -- see module docstring / this package's known limitation that
    `gripper_opening` alone was previously (mis)used as a grasp proxy."""
    left_force, right_force = fingertip_contact_forces(env)
    return (left_force > CONTACT_FORCE_THRESHOLD) & (right_force > CONTACT_FORCE_THRESHOLD)


def desired_grasp_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Target gripper orientation (robot-root frame, wxyz), yaw-aligned to
    the object -- `yaw_quat(object_orientation) * GRIPPING_QUAT_WXYZ`, the
    same composition `moveit_pick_place.py::generate_grasp_candidates` uses
    (ROS quaternion-multiply order there: `_quaternion_multiply(yaw_quat,
    gripping_quat)`, i.e. yaw-quat applied first / on the left -- matched
    here with `quat_mul(yaw, gripping)`)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3], object.data.root_quat_w
    )
    gripping_quat = torch.tensor(GRIPPING_QUAT_WXYZ, device=env.device, dtype=object_quat_b.dtype).expand_as(
        object_quat_b
    )
    return math_utils.quat_mul(math_utils.yaw_quat(object_quat_b), gripping_quat)


def pre_grasp_target_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    offset_z: float = PRE_GRASP_OFFSET_Z,
) -> torch.Tensor:
    """Pre-grasp target position (robot-root frame): object position + a
    fixed Z offset, NOT the object center itself (see module docstring)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object.data.root_pos_w[:, :3]
    )
    offset = torch.zeros_like(object_pos_b)
    offset[:, 2] = offset_z
    return object_pos_b + offset

def ee_pos_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """(ee_pos_b, ee_quat_b): end-effector (tcp) pose in the robot root
    frame, read from the `ee_frame` FrameTransformer already used by the
    reward functions in `rewards.py`."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    return math_utils.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)

# ---------------------------------------------------------------------------
# Per-env scratch state (consecutive-step counters, sticky flags)
# ---------------------------------------------------------------------------

def _state(env: ManagerBasedRLEnv) -> dict:
    if not hasattr(env, "_rb5_pp_state"):
        env._rb5_pp_state = {}
    return env._rb5_pp_state

def get_counter(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Get (creating if needed) a persistent (num_envs,) float32 buffer."""
    state = _state(env)
    if name not in state:
        state[name] = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    return state[name]


def get_tensor_buffer(env: ManagerBasedRLEnv, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Get (creating if needed) a persistent (num_envs, *shape) float32 buffer.

    Same lazy per-env scratch-state mechanism as `get_counter`, generalized
    to non-scalar buffers (e.g. `rewards.action_jerk_penalty`'s per-joint
    action history). Zeroed automatically on reset by
    `events.reset_rb5_pp_state` alongside every other buffer in this dict --
    `buf[env_ids] = 0.0` broadcasts correctly regardless of trailing shape."""
    state = _state(env)
    if name not in state:
        state[name] = torch.zeros(env.num_envs, *shape, device=env.device, dtype=torch.float32)
    return state[name]

def step_once(env: ManagerBasedRLEnv, key: str, fn) -> None:
    """Run `fn()` at most once per env step, regardless of how many reward/
    termination terms call `step_once` with the same `key` this step
    (manager-based envs evaluate reward terms and termination terms
    separately in the same step, both of which may need to advance the same
    counter -- without this guard the counter would double-advance)."""
    state = _state(env)
    step_key = f"__step__{key}"
    if state.get(step_key) != env.common_step_counter:
        fn()
        state[step_key] = env.common_step_counter

def advance_consecutive_counter(env: ManagerBasedRLEnv, name: str, condition: torch.Tensor) -> torch.Tensor:
    """Increment `name`'s counter where `condition` is True, reset to 0
    elsewhere; returns the (post-update) counter. Idempotent per step (see
    `step_once`)."""
    counter = get_counter(env, name)

    def _update():
        counter[:] = torch.where(condition, counter + 1.0, torch.zeros_like(counter))

    step_once(env, name, _update)
    return counter

def set_sticky_flag(env: ManagerBasedRLEnv, name: str, condition: torch.Tensor) -> torch.Tensor:
    """Once `condition` is True for an env, `name`'s flag stays 1.0 until
    the next episode reset (used for e.g. "has this env ever achieved a
    valid grasp/lift", so a drop penalty only fires after a real grasp, not
    just because the object starts on the bin floor)."""
    flag = get_counter(env, name)

    def _update():
        flag[:] = torch.maximum(flag, condition.float())

    step_once(env, name, _update)
    return flag


def stable_grasp(env: ManagerBasedRLEnv, min_steps: int = STABLE_GRASP_STEPS) -> torch.Tensor:
    """Bool (num_envs,): bilateral contact has held for >= `min_steps`
    consecutive control steps."""
    counter = advance_consecutive_counter(env, "bilateral_contact_streak", bilateral_contact(env))
    return counter >= min_steps


def grasped_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_center_distance: float = 0.06,
) -> torch.Tensor:
    """Conservative grasp definition (per spec): bilateral fingertip contact
    AND the object is actually near the grasp center (tcp), not just
    touching a fingertip in passing while sliding past."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    near_center = torch.norm(object.data.root_pos_w[:, :3] - ee_pos_w, dim=1) < max_center_distance
    return bilateral_contact(env) & near_center


def object_inside_destination_xy(
    env: ManagerBasedRLEnv,
    margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object's world XY is inside the destination bin's
    actual inner footprint (from `bin_geometry.yaml`, not just "close to the
    bin center" by Euclidean distance)."""
    from ..bin_geometry import load_bin_geometry

    _, dest_bin = load_bin_geometry()
    object: RigidObject = env.scene[object_cfg.name]
    x, y = object.data.root_pos_w[:, 0], object.data.root_pos_w[:, 1]
    half_x = dest_bin.inner_size[0] / 2.0 - margin
    half_y = dest_bin.inner_size[1] / 2.0 - margin
    inside_x = (x - dest_bin.center[0]).abs() < half_x
    inside_y = (y - dest_bin.center[1]).abs() < half_y
    return inside_x & inside_y


# ---------------------------------------------------------------------------
# Evaluation-oriented physical-state predicates. Thin, named wrappers around
# the machinery above -- exist so `evaluate_policy.py` (and any future
# diagnostic script) reads off ONE shared definition of "is this stage
# physically complete" instead of re-deriving thresholds inline. Where an
# equivalent already existed above (bilateral contact, stable grasp, grasped,
# inside-destination) callers should keep using that function directly rather
# than an extra pass-through -- these fill the gaps that didn't already have
# a named predicate.
# ---------------------------------------------------------------------------


def is_near_pregrasp(
    env: ManagerBasedRLEnv,
    position_threshold: float = DEFAULT_POSITION_THRESHOLD,
    orientation_threshold: float = DEFAULT_ORIENTATION_THRESHOLD,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Bool (num_envs,): end-effector within `position_threshold`/
    `orientation_threshold` of the pre-grasp target -- the same condition
    `terminations.reach_success` gates on (that function delegates here)."""
    ee_pos_b, ee_quat_b = ee_pos_quat_b(env, robot_cfg, ee_frame_cfg)
    target_pos_b = pre_grasp_target_pos_b(env, robot_cfg, object_cfg)
    target_quat_b = desired_grasp_quat_b(env, robot_cfg, object_cfg)
    pos_ok = torch.norm(target_pos_b - ee_pos_b, dim=-1) < position_threshold
    ori_ok = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b) < orientation_threshold
    return pos_ok & ori_ok


def is_object_lifted(
    env: ManagerBasedRLEnv,
    min_height: float,
    source_floor_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object is more than `min_height` above the source
    bin floor (relative, unlike `rewards.object_is_lifted`'s absolute-world-Z
    check -- matches the height calc `rewards.continuous_lift_reward` uses)."""
    object: RigidObject = env.scene[object_cfg.name]
    return (object.data.root_pos_w[:, 2] - source_floor_height) > min_height


def is_object_above_safe_height(
    env: ManagerBasedRLEnv,
    safe_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object world Z above `safe_height` -- same
    threshold `rewards.object_to_goal_position_reward`/`object_height_safety_reward`
    gate transport-tracking reward on."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] > safe_height


def is_object_near_destination(
    env: ManagerBasedRLEnv,
    radius: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object's world XY within `radius` (Euclidean, NOT
    the exact footprint check `object_inside_destination_xy` does) of the
    destination bin center -- a looser "in the vicinity" signal, useful as a
    funnel stage between "transporting" and "inside the exact footprint"."""
    from ..bin_geometry import load_bin_geometry

    _, dest_bin = load_bin_geometry()
    object: RigidObject = env.scene[object_cfg.name]
    dx = object.data.root_pos_w[:, 0] - dest_bin.center[0]
    dy = object.data.root_pos_w[:, 1] - dest_bin.center[1]
    return torch.sqrt(dx * dx + dy * dy) < radius


def is_gripper_open(
    env: ManagerBasedRLEnv,
    gripper_open_threshold: float = DEFAULT_GRIPPER_OPEN_THRESHOLD,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
) -> torch.Tensor:
    """Bool (num_envs,): primary knuckle angle below `gripper_open_threshold`
    -- same threshold used inline throughout `rewards.py`/`terminations.py`."""
    robot: RigidObject = env.scene[robot_cfg.name]
    knuckle_angle = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    return knuckle_angle < gripper_open_threshold


def is_object_stable(
    env: ManagerBasedRLEnv,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object linear speed below `velocity_threshold` --
    the "settled, not just passing through" check used throughout
    `rewards.py`'s place/transport terms."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.norm(object.data.root_lin_vel_w, dim=1) < velocity_threshold


def is_object_released(
    env: ManagerBasedRLEnv,
    gripper_open_threshold: float = DEFAULT_GRIPPER_OPEN_THRESHOLD,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[GRIPPER_PRIMARY_JOINT]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): gripper open AND no valid grasp contact -- a
    genuine release, not just a gripper-angle reading (an open commanded
    target with the object still wedged in the mechanism would fail this)."""
    return is_gripper_open(env, gripper_open_threshold, robot_cfg) & ~grasped_object(env, object_cfg)


def is_at_place_target(
    env: ManagerBasedRLEnv,
    margin: float,
    max_height_above_floor: float,
    velocity_threshold: float,
    dest_floor_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bool (num_envs,): object is inside the destination footprint, near
    the floor, and settled -- everything `rewards.full_place_success_condition`
    checks EXCEPT gripper state. Used as the scripted release TRIGGER for
    `actions.AutoReleaseAction` (the Place stage's mirror of `AutoGraspAction`)
    -- checking gripper-open/not-grasped here would be circular, since this
    condition is what decides whether to open the gripper in the first
    place."""
    object: RigidObject = env.scene[object_cfg.name]
    inside_footprint = object_inside_destination_xy(env, margin, object_cfg)
    near_floor = (object.data.root_pos_w[:, 2] - dest_floor_height) < max_height_above_floor
    settled = is_object_stable(env, velocity_threshold, object_cfg)
    return inside_footprint & near_floor & settled
