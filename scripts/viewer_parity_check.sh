#!/bin/bash
# viewer_parity_check.sh — prove, as NUMBERS, that trips-viewer renders what the
# bundle says, for any bundle including Jordan's own scenes.
#
# For each requested view it renders the same frame twice:
#   1. through `trips-viewer --screenshot` (the Rust rasteriser + U-Net + tone mapper), and
#   2. through `trippy bundle-parity` (trippy's Python reference, reading the SAME
#      three bundle files and nothing else),
# then prints PSNR between the two 8-bit PNGs plus, for each of them, per-channel
# mean/p01/p50/p99 brightness and the saturated / crushed-black pixel fractions.
#
# Usage: scripts/viewer_parity_check.sh [options] <bundle-dir> <out-dir>
#   --binary PATH   trips-viewer to run (default: the MAIN checkout's release build)
#   --views "A B"   dataset view indices (default: the bundle's own default_view)
#   --scale F       render scale, 1.0 = the view's own size (default 1.0)
#   --half-net      run the viewer's U-Net in f16 (what the shipped launcher does)
#   --label NAME    tag for the JSON/PNG filenames (default: the bundle dir's name)
#
# Invariants:
#   - NUMBERS ONLY. Unlike scripts/viewer_camera_check.sh this never asks anyone to
#     look at the PNGs it writes, so it is safe on private scenes and does not
#     restrict itself to the public ones. The PNGs stay under <out-dir> (which must
#     be outside the repo — this repo is public) and are read only by numpy.
#   - Both halves render the SAME camera: `--scale 1.0` with a pinned view is the
#     view's own fx/fy/cx/cy/R/t on both sides (see
#     trippy/render/bundle_render.py `view_camera`).
#   - Needs the GPU (wgpu for the viewer, MPS for the Python half), so it runs
#     ONLY inside a scripts/gpu_submit.sh job, never directly.
#   - Bash 3.2 safe: no arrays.
# Related docs: docs/decisions/ADR-0006-viewer-integration.md ("verification without
#   a window"), docs/LIMITATIONS.md, trippy/render/bundle_render.py.
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
# The release binary lives in the MAIN checkout's target dir; that is the path the
# delivered launchers bake, so it outlives any worktree.
MAIN_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

BINARY="$MAIN_ROOT/rust/target/release/trips-viewer"
[ -x "$BINARY" ] || BINARY="$REPO_ROOT/rust/target/release/trips-viewer"
VIEWS=""
SCALE=1.0
HALF_NET=""
LABEL=""

usage() {
  echo 'usage: viewer_parity_check.sh [--binary PATH] [--views "A B"] [--scale F] [--half-net] [--label NAME] <bundle-dir> <out-dir>' >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --binary)   BINARY=${2:?--binary needs a value}; shift 2 ;;
    --views)    VIEWS=${2:?--views needs a value}; shift 2 ;;
    --scale)    SCALE=${2:?--scale needs a value}; shift 2 ;;
    --label)    LABEL=${2:?--label needs a value}; shift 2 ;;
    --half-net) HALF_NET="--half-net"; shift ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) break ;;
  esac
done

BUNDLE=${1:?$(usage)}
OUTDIR=${2:?$(usage)}

[ -f "$BUNDLE/bundle.json" ] || { echo "✗ no bundle.json in $BUNDLE" >&2; exit 2; }
[ -x "$BINARY" ] || { echo "✗ viewer binary not found at $BINARY" >&2; exit 3; }
case "$OUTDIR" in
  "$REPO_ROOT"/*) echo "✗ refusing to write renders inside the public repo ($OUTDIR)" >&2; exit 5 ;;
esac

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv/bin/python"; fi

[ -n "$LABEL" ] || LABEL=$(basename "$BUNDLE")
PRECISION=f32
[ -z "$HALF_NET" ] || PRECISION=f16
TAG="$LABEL-$PRECISION-s$SCALE"

if [ -z "$VIEWS" ]; then
  VIEWS=$("$PY" -c 'import json,sys
doc = json.load(open(sys.argv[1]))
print(doc["views"][doc.get("default_view", 0)]["index"])' "$BUNDLE/bundle.json")
fi

mkdir -p "$OUTDIR"
SHOTS=""
for view in $VIEWS; do
  # Five-digit index in the name is what `trippy bundle-parity --compare` matches on.
  shot=$(printf '%s/%s_view_%05d_viewer.png' "$OUTDIR" "$TAG" "$view")
  echo "▶ $BINARY $BUNDLE --view $view --scale $SCALE $HALF_NET --screenshot $shot"
  "$BINARY" "$BUNDLE" --view "$view" --scale "$SCALE" $HALF_NET --screenshot "$shot" --frames 2
  SHOTS="$SHOTS $shot"
done

echo "▶ trippy bundle-parity --bundle $BUNDLE --view $VIEWS --scale $SCALE (mps)"
PYTHONPATH="$REPO_ROOT" "$PY" -m trippy.cli bundle-parity \
  --bundle "$BUNDLE" \
  --view $VIEWS \
  --scale "$SCALE" \
  --device mps \
  --out "$OUTDIR/$TAG-python" \
  --compare $SHOTS \
  | tee "$OUTDIR/$TAG.json"

echo "✓ $OUTDIR/$TAG.json"
