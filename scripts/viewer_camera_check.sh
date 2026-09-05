#!/bin/bash
# viewer_camera_check.sh — prove, without a human, that moving the camera changes
# what trips-viewer renders.
#
# Renders the SAME bundle twice through the viewer's own `--screenshot` path,
# once at the chosen training view and once yawed `--yaw` degrees off it, then
# reports the mean absolute difference between the two PNGs. A viewer whose
# input is wired up produces two different images; the one Jordan tested on
# 2026-09-06 would have produced two identical ones, because the drag never
# reached the controller (see docs/decisions/ADR-0006 and research/trips-metal.md).
#
# Usage: scripts/viewer_camera_check.sh [--binary PATH] [--scale F] [--yaw DEG]
#                                       [--threshold T] <bundle-dir> <out-dir>
# Invariants:
#   - PUBLIC SCENES ONLY. This script writes PNG renders to disk, and agents are
#     forbidden from opening renders of Jordan's own scenes (AGENTS.md §6). It
#     therefore refuses any bundle whose manifest name is not a known-public one
#     (the Zenodo/Tanks-and-Temples scenes), rather than trusting the caller.
#   - Runs the viewer headlessly: no window, two frames of GPU work per run.
#     Submit it through scripts/gpu_submit.sh like any other GPU work.
#   - Exits non-zero when the two frames are the same, which is the failure the
#     check exists to catch.
# Related docs: docs/USER_GUIDE.md; docs/decisions/ADR-0006-viewer-integration.md.
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
# The release binary is built into the MAIN checkout's target dir (that is where
# scripts/open_mac_viewer.sh points the delivered launcher, so it outlives any
# worktree). `--git-common-dir` is the shared `.git` for a checkout and all its
# worktrees, so this resolves to the main checkout from either.
MAIN_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

# Scenes that may be rendered to a file by an automated check: the public TRIPS
# / Tanks-and-Temples ones. Anything else is Jordan's own capture.
PUBLIC_SCENES="horse train truck lighthouse playground m60"

BINARY="$MAIN_ROOT/rust/target/release/trips-viewer"
[ -x "$BINARY" ] || BINARY="$REPO_ROOT/rust/target/release/trips-viewer"
SCALE=0.35
YAW=12
THRESHOLD=1.0

usage() {
  echo 'usage: viewer_camera_check.sh [--binary PATH] [--scale F] [--yaw DEG] [--threshold T] <bundle-dir> <out-dir>' >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --binary)    BINARY=${2:?--binary needs a value}; shift 2 ;;
    --scale)     SCALE=${2:?--scale needs a value}; shift 2 ;;
    --yaw)       YAW=${2:?--yaw needs a value}; shift 2 ;;
    --threshold) THRESHOLD=${2:?--threshold needs a value}; shift 2 ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) break ;;
  esac
done

BUNDLE=${1:?$(usage)}
OUTDIR=${2:?$(usage)}

[ -f "$BUNDLE/bundle.json" ] || { echo "✗ no bundle.json in $BUNDLE" >&2; exit 2; }
[ -x "$BINARY" ] || { echo "✗ viewer binary not found at $BINARY" >&2; exit 3; }

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv/bin/python"; fi

SCENE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$BUNDLE/bundle.json")
ALLOWED=0
for public in $PUBLIC_SCENES; do
  [ "$SCENE" = "$public" ] && ALLOWED=1
done
if [ "$ALLOWED" -ne 1 ]; then
  echo "✗ refusing to render \"$SCENE\" to a file: this check writes PNGs and only" >&2
  echo "  public scenes ($PUBLIC_SCENES) may be written by an automated run." >&2
  echo "  Jordan's own scenes are checked by Jordan, in the window." >&2
  exit 4
fi

mkdir -p "$OUTDIR"
A="$OUTDIR/camera-yaw-0.png"
B="$OUTDIR/camera-yaw-$YAW.png"

echo "▶ $BINARY $BUNDLE --screenshot ... --camera-yaw-deg 0 / $YAW  (scale $SCALE)"
"$BINARY" "$BUNDLE" --screenshot "$A" --camera-yaw-deg 0   --scale "$SCALE" --frames 1
"$BINARY" "$BUNDLE" --screenshot "$B" --camera-yaw-deg "$YAW" --scale "$SCALE" --frames 1

"$PY" - "$A" "$B" "$THRESHOLD" <<'PY'
import sys
import numpy as np
from PIL import Image

a_path, b_path, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float64)
b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float64)
if a.shape != b.shape:
    raise SystemExit(f"FAIL shapes differ: {a.shape} vs {b.shape}")

diff = np.abs(a - b)
mad = float(diff.mean())
changed = float((diff.max(axis=2) > 2.0).mean())
mse = float(((a - b) ** 2).mean())
print(f"CAMERA-DIFF mean|a-b| = {mad:.3f}/255   changed pixels = {100 * changed:.1f}%"
      f"   rms = {np.sqrt(mse):.3f}   size = {a.shape[1]}x{a.shape[0]}")
print(f"  a = {a_path}")
print(f"  b = {b_path}")
if mad <= threshold:
    raise SystemExit(
        f"FAIL: the two frames are effectively identical (mean|a-b| {mad:.3f} <= {threshold}); "
        "a camera change is not reaching the renderer"
    )
print(f"PASS: a scripted camera change reaches the renderer (threshold {threshold})")
PY
