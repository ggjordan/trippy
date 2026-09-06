#!/bin/bash
# viewer_splat_check.sh — prove, without a human, that the LIVE splat reaches the
# frame at a pose no camera ever stood at.
#
# Renders the same bundle three times through trips-viewer's own `--screenshot`
# path, all of them yawed `--yaw` degrees off a capture view (so the camera is
# NOT pinned and no precomputed render is usable):
#
#   a  --blend-mode mix --mix 0   pure splat, live from bundle.json's splat_ply
#   b  --blend-mode mix --mix 1   pure TRIPS
#   c  --blend-mode mix --mix 0 --no-live-splat   the CONTROL
#
# a vs b must DIFFER: that is the live path working. c vs b must be IDENTICAL:
# with the ply closed there is no splat off a capture view, so the renderer
# falls back to TRIPS and says so -- which is exactly the behaviour this change
# replaces, and pinning it here is what stops "a differs from b" being an
# artefact of something else in the frame.
#
# Usage: scripts/viewer_splat_check.sh [--binary PATH] [--scale F] [--yaw DEG]
#                                      [--threshold T] <bundle-dir> <out-dir>
# Invariants:
#   - PUBLIC / SYNTHETIC SCENES ONLY. This writes PNG renders to disk and agents
#     are forbidden from opening renders of Jordan's scenes (AGENTS.md section
#     6), so it refuses any bundle whose manifest name is not on the allow-list
#     below rather than trusting the caller. Same guard, same reason, as
#     scripts/viewer_camera_check.sh.
#   - Runs the viewer headlessly: no window, a couple of frames of GPU work per
#     run. Submit it through scripts/gpu_submit.sh like any other GPU work.
#   - Exits non-zero when the live frame matches the TRIPS frame, which is the
#     failure the check exists to catch.
# Related docs: docs/USER_GUIDE.md "Blend panel";
#   docs/decisions/ADR-0006-viewer-integration.md; rust/crates/trips-viewer/src/splat.rs.
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
MAIN_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

# Scenes an automated check may render to a file: the public TRIPS /
# Tanks-and-Temples ones, plus the generated fixtures tools/ writes.
PUBLIC_SCENES="horse train truck lighthouse playground m60 synthetic synthetic-splat"

# Prefer THIS checkout's binary (a worktree building the feature has its own
# target dir); fall back to the main checkout's, which is where the delivered
# launchers point.
BINARY="$REPO_ROOT/rust/target/release/trips-viewer"
[ -x "$BINARY" ] || BINARY="$MAIN_ROOT/rust/target/release/trips-viewer"
SCALE=1.0
YAW=20
THRESHOLD=1.0

usage() {
  echo 'usage: viewer_splat_check.sh [--binary PATH] [--scale F] [--yaw DEG] [--threshold T] <bundle-dir> <out-dir>' >&2
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
PLY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("blend",{}).get("splat_ply",""))' "$BUNDLE/bundle.json")
ALLOWED=0
for public in $PUBLIC_SCENES; do
  [ "$SCENE" = "$public" ] && ALLOWED=1
done
if [ "$ALLOWED" -ne 1 ]; then
  echo "✗ refusing to render \"$SCENE\" to a file: this check writes PNGs and only" >&2
  echo "  public or synthetic scenes ($PUBLIC_SCENES) may be written by an automated run." >&2
  echo "  Jordan's own scenes are checked by Jordan, in the window." >&2
  exit 4
fi
if [ -z "$PLY" ]; then
  echo "✗ $BUNDLE/bundle.json has no blend.splat_ply: there is no live splat to check." >&2
  echo "  Re-export the bundle from a checkpoint whose point source is a Gaussian PLY." >&2
  exit 5
fi
echo "▶ scene \"$SCENE\", splat_ply $PLY"

mkdir -p "$OUTDIR"
A="$OUTDIR/splat-live-mix0.png"
B="$OUTDIR/trips-mix1.png"
C="$OUTDIR/control-no-live-splat-mix0.png"

echo "▶ $BINARY $BUNDLE --camera-yaw-deg $YAW (scale $SCALE), mix 0 / mix 1 / control"
"$BINARY" "$BUNDLE" --screenshot "$A" --camera-yaw-deg "$YAW" --scale "$SCALE" --frames 1 \
  --blend-mode mix --mix 0
"$BINARY" "$BUNDLE" --screenshot "$B" --camera-yaw-deg "$YAW" --scale "$SCALE" --frames 1 \
  --blend-mode mix --mix 1
"$BINARY" "$BUNDLE" --screenshot "$C" --camera-yaw-deg "$YAW" --scale "$SCALE" --frames 1 \
  --blend-mode mix --mix 0 --no-live-splat

"$PY" - "$A" "$B" "$C" "$THRESHOLD" <<'PY'
import sys

import numpy as np
from PIL import Image

a_path, b_path, c_path, threshold = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


a, b, c = load(a_path), load(b_path), load(c_path)
if not (a.shape == b.shape == c.shape):
    raise SystemExit(f"FAIL shapes differ: {a.shape} vs {b.shape} vs {c.shape}")


def report(label, x, y):
    diff = np.abs(x - y)
    mad = float(diff.mean())
    changed = float((diff.max(axis=2) > 2.0).mean())
    print(f"SPLAT-DIFF {label}: mean|a-b| = {mad:.3f}/255   changed pixels = {100 * changed:.1f}%")
    return mad


live = report("live splat (mix 0) vs TRIPS (mix 1)", a, b)
control = report("control, ply closed (mix 0) vs TRIPS (mix 1)", c, b)
print(f"  a = {a_path}\n  b = {b_path}\n  c = {c_path}   size = {a.shape[1]}x{a.shape[0]}")

if live <= threshold:
    raise SystemExit(
        f"FAIL: mix 0 and mix 1 are effectively identical (mean|a-b| {live:.3f} <= {threshold}); "
        "the live splat is not reaching the composited frame"
    )
if control > 1e-9:
    raise SystemExit(
        f"FAIL: with --no-live-splat the mix-0 frame is NOT the TRIPS frame "
        f"(mean|c-b| {control:.6f}); off a capture view there is no splat to show, so the "
        "fallback must be TRIPS exactly -- something else in the frame is moving"
    )
print(f"PASS: the live splat reaches the frame at an off-capture pose (threshold {threshold}), "
      "and closing the ply restores the old TRIPS-only fallback exactly")
PY
