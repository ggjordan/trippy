#!/bin/bash
# requeue_with_masks.sh — submit masked SIBLING training runs for kk-coherent configs.
# Usage: scripts/requeue_with_masks.sh [--dry-run] [--masks-dir DIR] <config.yaml> [config2.yaml ...]
#
# Context: kk-coherent trained without person-exclusion masks, so Jordan's kids show up as
# ghosts in TRIPS outputs (research/trips-metal.md, experiments/MASKS.md). Jordan wants BOTH
# the existing unmasked results AND masked results (2026-09-06 correction) -- NOT a
# replacement. So this script never touches, edits, or dequeues the original config or its
# GPU-queue entry. For each given config it instead:
#   1. Writes a SIBLING config (same directory, "_masked" suffix on the filename) that is a
#      copy of the original with two changes:
#        - `masks_dir: <DIR>` inserted right after the `scene_root:` line.
#        - `run_dir:` given a `-masked` suffix on its final path component, so the masked
#          run gets its own output dir and its own GPU-queue job name (queue_training.sh
#          names the job after run_dir's basename) instead of colliding with the original.
#   2. Submits the sibling via scripts/queue_training.sh (prio 70, `trippy train --report`,
#      same as every other trippy training job -- see that script's header).
#
# --dry-run: the sibling config is written to a SCRATCH file under $TRIPPY_OUTPUT/tmp/ (never
#   under experiments/), printed, submitted through queue_training.sh --dry-run (which forwards
#   to gpu_submit.sh --dry-run: job file written/printed only -- no queue/runner/memory checks,
#   no call to Splats' submit.sh, no research/trips-metal.md write), then deleted. Nothing
#   under experiments/ is created or modified in --dry-run mode.
# Without --dry-run: the sibling config is written PERMANENTLY next to the original
#   (<name>_masked.yaml) and actually submitted to the real GPU queue.
#
# Invariants:
#   - Refuses (exit 2) if a config is missing, has no top-level `scene_root:` containing
#     "kk-coherent" (masks were only generated for kk-coherent -- see experiments/MASKS.md),
#     or has no top-level `run_dir:` key.
#   - Never edits, deletes, dequeues, or resubmits the ORIGINAL config or its queue file.
#   - Idempotent in real mode: if the permanent sibling config already exists, it is reused
#     unchanged (a partial prior run doesn't get clobbered) and still (re)submitted.
#   - This script never touches the GPU queue directly; it only ever hands configs to
#     scripts/queue_training.sh, which hands the job to scripts/gpu_submit.sh --train.
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-$PWD/output}}

usage() {
  echo "usage: requeue_with_masks.sh [--dry-run] [--masks-dir DIR] <config.yaml> [config2.yaml ...]" >&2
}

DRYRUN=0
# Default: where make_masks3.py wrote the 238 kk-coherent person-exclusion masks
# (experiments/MASKS.md). Override with --masks-dir for a different scene/location.
MASKS_DIR="/Users/nzbirdranch/trippy/output/masks/kk-coherent"
CONFIGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRYRUN=1; shift ;;
    --masks-dir) MASKS_DIR=${2:-}; shift 2 ;;
    -*) usage; exit 2 ;;
    *) CONFIGS+=("$1"); shift ;;
  esac
done

[ ${#CONFIGS[@]} -gt 0 ] || { usage; exit 2; }

for CONFIG in "${CONFIGS[@]}"; do
  [ -f "$CONFIG" ] || { echo "✗ config not found: $CONFIG" >&2; exit 2; }

  SCENE_ROOT=$(grep -m1 "^scene_root:" "$CONFIG" | sed 's/^scene_root:[[:space:]]*//')
  case "$SCENE_ROOT" in
    *kk-coherent*) ;;
    *)
      echo "✗ $CONFIG: scene_root ($SCENE_ROOT) does not reference kk-coherent -- refusing" >&2
      echo "  (masks were only generated for kk-coherent; see experiments/MASKS.md)" >&2
      exit 2
      ;;
  esac

  RUN_DIR=$(grep -m1 "^run_dir:" "$CONFIG" | sed 's/^run_dir:[[:space:]]*//')
  [ -n "$RUN_DIR" ] || { echo "✗ $CONFIG has no top-level 'run_dir:' key" >&2; exit 2; }

  case "$CONFIG" in
    *.yaml) BASE=${CONFIG%.yaml} ;;
    *.yml) BASE=${CONFIG%.yml} ;;
    *) echo "✗ $CONFIG: not a .yaml/.yml file" >&2; exit 2 ;;
  esac
  PERM_SIBLING="${BASE}_masked.yaml"

  if [ "$DRYRUN" -eq 1 ]; then
    mkdir -p "$TRIPPY_OUTPUT/tmp"
    # BSD mktemp (macOS) only replaces a trailing run of Xs -- a suffix after them (e.g.
    # ".yaml") is left as literal "XXXXXX" and every call returns the SAME non-unique name.
    # Keep Xs at the true end, then rename to add the extension.
    _TMP=$(mktemp "$TRIPPY_OUTPUT/tmp/requeue_masked.XXXXXX")
    SIBLING="${_TMP}.yaml"
    mv "$_TMP" "$SIBLING"
  else
    SIBLING="$PERM_SIBLING"
  fi

  if [ "$DRYRUN" -eq 0 ] && [ -f "$SIBLING" ]; then
    echo "ℹ $SIBLING already exists; reusing it unchanged" >&2
  else
    awk -v masks_dir="$MASKS_DIR" '
      /^scene_root:/ { print; print "masks_dir: " masks_dir; next }
      /^run_dir:/ { sub(/^run_dir:[ \t]*/, ""); print "run_dir: " $0 "-masked"; next }
      { print }
    ' "$CONFIG" > "$SIBLING"
  fi

  echo "-- masked sibling for $CONFIG --"
  echo "   masks_dir: $MASKS_DIR"
  echo "   run_dir:   $RUN_DIR-masked"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "   written:   $SIBLING (scratch, --dry-run; deleted after submit preview)"
  else
    echo "   written:   $SIBLING (permanent; original config untouched)"
  fi

  QARGS=()
  [ "$DRYRUN" -eq 1 ] && QARGS+=(--dry-run)
  bash "$REPO_ROOT/scripts/queue_training.sh" "$SIBLING" "${QARGS[@]+"${QARGS[@]}"}"

  if [ "$DRYRUN" -eq 1 ]; then rm -f "$SIBLING"; fi
done
