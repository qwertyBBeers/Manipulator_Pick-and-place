#!/usr/bin/env bash
# One self-contained collection instance: Isaac scene + MoveIt stack + episode
# logger + a loop of relay batches. Several of these run side by side to
# collect faster.
#
# Isolation is by ROS_DOMAIN_ID, not by renaming topics. Every instance uses
# the same topic names inside its own domain, so nothing in the scene, the
# launch file or the controller needs to know it is one of several. Note that
# Isaac's ROS2Context node ignores ROS_DOMAIN_ID unless useDomainIDEnvVar is
# set, which dual_binpicking_scene.py now does -- without it every instance
# would publish on domain 0 and they would drive each other's robots.
#
# Usage: collect_instance.sh <instance-id> <batches> <cycles-per-batch> [gui]
#   instance-id  also becomes ROS_DOMAIN_ID and the dataset subdirectory
#   gui          pass "gui" to show this instance's Isaac window (default headless)
#
# Env: TARGET_SUCCESS  stop once this many successful episodes exist across all
#                      instances (0 = ignore, just run the batches)
#
# Logs: $LOG_ROOT/inst<id>_{scene,launch,collect,logger}.log
set -u

INSTANCE="${1:?usage: collect_instance.sh <instance-id> <batches> <cycles> [gui]}"
BATCHES="${2:-40}"
CYCLES="${3:-3}"
DISPLAY_MODE="${4:-headless}"
TARGET_SUCCESS="${TARGET_SUCCESS:-0}"

DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/hdd/relay_datasets}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs}"
OUT_DIR="$DATA_ROOT/relay_inst${INSTANCE}"
mkdir -p "$LOG_ROOT" "$OUT_DIR"

export ROS_DOMAIN_ID="$INSTANCE"
export ROS_LOCALHOST_ONLY=1
if [ "$DISPLAY_MODE" = "gui" ]; then export ISAAC_HEADLESS=0; else export ISAAC_HEADLESS=1; fi

SCENE_PID=""
STACK_PID=""
LOGGER_PID=""

log() { echo "[inst$INSTANCE $(date +%H:%M:%S)] $*"; }

kill_tree() {  # kill a pid and anything it spawned that still names our domain
  local pid="$1"
  [ -n "$pid" ] || return 0
  pkill -9 -P "$pid" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
}

kill_in_domain() {
  local pat="$1"
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    # /proc/<pid>/environ is the environment the process was EXECed with, so it
    # only carries ROS_DOMAIN_ID for children we spawned after exporting it --
    # which is exactly the set we want to kill, and never this script itself.
    if grep -qz "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" "/proc/$p/environ" 2>/dev/null; then
      kill -9 "$p" 2>/dev/null
    fi
  done
}

teardown_stack() {
  for pat in "relay_pick_place.py" "dual_binpicking.launch.py" \
             "moveit_ros_move_group/move_group" "namespaced_trajectory_bridge.py" \
             "dual_scene_setup.py" "dual_binpicking_scene.py"; do
    kill_in_domain "$pat"
  done
  sleep 5
}

cleanup() {
  log "stopping"
  kill_in_domain "dual_episode_logger.py"
  teardown_stack
}
trap cleanup EXIT INT TERM

cd "$DIR" || exit 1

start_scene() {
  local logfile="$LOG_ROOT/inst${INSTANCE}_scene.log"
  # Only look at output produced by THIS start. The log is appended across
  # restarts (a crash backtrace is worth keeping), so a plain grep would match
  # the previous run's "relay scene running" and report the new scene up
  # instantly -- after which MoveIt starts against a simulator that does not
  # exist yet.
  local offset=1
  [ -f "$logfile" ] && offset=$(( $(wc -c < "$logfile") + 1 ))

  log "starting Isaac scene (domain $ROS_DOMAIN_ID, $DISPLAY_MODE)"
  setsid nohup ~/isaacsim/python.sh dual_binpicking_scene.py >> "$logfile" 2>&1 < /dev/null &
  SCENE_PID=$!
  for _ in $(seq 1 120); do
    tail -c "+$offset" "$logfile" 2>/dev/null | grep -q "relay scene running" && break
    sleep 5
  done
  if ! tail -c "+$offset" "$logfile" 2>/dev/null | grep -q "relay scene running"; then
    log "scene did not come up; see $logfile"
    return 1
  fi
  log "scene up"
}

start_stack() {
  log "starting MoveIt stack"
  setsid nohup ros2 launch "$DIR/dual_binpicking.launch.py" \
    >> "$LOG_ROOT/inst${INSTANCE}_launch.log" 2>&1 < /dev/null &
  STACK_PID=$!
  sleep 35
}

start_logger() {
  log "starting episode logger -> $OUT_DIR"
  setsid nohup python3 dual_episode_logger.py --out-dir "$OUT_DIR" \
    >> "$LOG_ROOT/inst${INSTANCE}_logger.log" 2>&1 < /dev/null &
  LOGGER_PID=$!
  sleep 8
}

scene_process_alive() {
  for p in $(pgrep -f "dual_binpicking_scene.py" 2>/dev/null); do
    grep -qz "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" "/proc/$p/environ" 2>/dev/null && return 0
  done
  return 1
}

sim_healthy() {
  # "Is the process there" is NOT a health check. Overnight both collectors
  # kept a live Isaac process whose simulation had stopped, sailed through a
  # process-existence check, and burned 400 batches at ~10s each producing
  # empty failure episodes -- 2806 of them. What actually distinguishes a
  # working simulator is that its clock advances, so ask that instead. This
  # also covers the crash case, since a dead process publishes no clock.
  scene_process_alive || return 1
  python3 - <<'PY'
import sys, time
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

rclpy.init()
node = Node("sim_health_probe")
seen = []
node.create_subscription(Clock, "/clock", lambda m: seen.append(m.clock.sec + m.clock.nanosec * 1e-9), 10)
deadline = time.monotonic() + 12.0
while time.monotonic() < deadline and (len(seen) < 2 or seen[-1] - seen[0] < 0.5):
    rclpy.spin_once(node, timeout_sec=0.2)
advanced = len(seen) >= 2 and (seen[-1] - seen[0]) >= 0.5
node.destroy_node()
rclpy.shutdown()
sys.exit(0 if advanced else 1)
PY
}

successes_so_far() { ls -d "$DATA_ROOT"/relay_inst*/success/episode_* 2>/dev/null | wc -l; }

rebuild() {
  kill_in_domain "dual_episode_logger.py"
  teardown_stack
  start_scene || { log "could not restart the scene; giving up"; return 1; }
  start_stack
  start_logger
}

# A healthy 3-cycle batch takes minutes; anything under a minute means it is
# failing upstream rather than working. A hung one has to be cut off eventually.
MIN_BATCH_SECONDS="${MIN_BATCH_SECONDS:-60}"
MAX_SHORT_STREAK="${MAX_SHORT_STREAK:-3}"
BATCH_TIMEOUT="${BATCH_TIMEOUT:-1800}"
SHORT_STREAK=0

set +u
source /opt/ros/humble/setup.bash
source /home/hh/asl_ws/Manipulator/install/setup.bash
set -u

start_scene || exit 1
start_stack
start_logger

for i in $(seq 1 "$BATCHES"); do
  if [ "$TARGET_SUCCESS" -gt 0 ] && [ "$(successes_so_far)" -ge "$TARGET_SUCCESS" ]; then
    log "target of $TARGET_SUCCESS successful episodes reached; stopping"
    break
  fi

  # Two independent signals, because overnight each of these happened and the
  # other check would not have caught it:
  #   - the simulation stops while its process lives -> sim_healthy()
  #   - a batch returns immediately, over and over -> SHORT_BATCH streak
  if ! sim_healthy; then
    log "simulation is not advancing -- rebuilding this instance"
    rebuild || exit 1
    SHORT_STREAK=0
  fi

  log "batch $i/$BATCHES"
  batch_start=$(date +%s)
  # A relay can hang outright: instance 1 sat inside one batch for 24h after a
  # DDS goal response went missing, so the loop never reached any check at all.
  timeout --signal=KILL "$BATCH_TIMEOUT" \
    python3 relay_pick_place.py --cycles "$CYCLES" --randomize \
    >> "$LOG_ROOT/inst${INSTANCE}_collect.log" 2>&1
  batch_seconds=$(( $(date +%s) - batch_start ))

  if [ "$batch_seconds" -lt "$MIN_BATCH_SECONDS" ]; then
    SHORT_STREAK=$(( SHORT_STREAK + 1 ))
    log "batch $i finished in ${batch_seconds}s (< ${MIN_BATCH_SECONDS}s) -- streak $SHORT_STREAK"
    if [ "$SHORT_STREAK" -ge "$MAX_SHORT_STREAK" ]; then
      log "$SHORT_STREAK consecutive short batches -- rebuilding this instance"
      rebuild || exit 1
      SHORT_STREAK=0
    fi
  else
    SHORT_STREAK=0
  fi
  sleep 3
done
log "collection finished"
