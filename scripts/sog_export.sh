#!/bin/bash
# sog_export.sh — compress a Gaussian-splat PLY to SOG (+ a compressed .ply second
# artefact) with @playcanvas/splat-transform, and prove the SOG round-trips.
# Usage: scripts/sog_export.sh [--check] [--gpu n|cpu] [--out-dir DIR] [--name STEM] <input.ply>
# Invariants (ADR-0008-supersplat.md Stage 3, item 8):
#   - Installs @playcanvas/splat-transform at a PINNED version into a LOCAL
#     node_modules under $TRIPPY_OUTPUT/tools/splat-transform -- never a global
#     install (AGENTS.md Sec 6's "no installing outside ./.venv" has the same spirit
#     for node). Re-running with the pin already installed is a no-op.
#   - `--gpu n|cpu` (default: `cpu`) selects splat-transform's own `-g` device for
#     SOG compression. CPU is the safe default for a direct/interactive run (never
#     touches the shared Metal device outside AGENTS.md Sec 6's GPU-queue path).
#     `--gpu 0` (or another adapter index from `splat-transform --list-gpus`) uses
#     the native WebGPU (Dawn) backend -- confirmed working WITHOUT approving the
#     `webgpu` package's blocked npm postinstall script (`--list-gpus` enumerates
#     the adapter either way) -- and is only ever passed from INSIDE a
#     scripts/gpu_submit.sh queue slot, never run against the GPU directly by this
#     script itself (2026-09-10: the first CPU-only run of this script on the real
#     8.5M-point shade-keep PLY ran 14h with no progress and was killed; the
#     GPU-queue path replaces the CPU default here for both real conversions).
#   - Never writes to <input.ply>; it is only ever read. Never copies it into the
#     repo (AGENTS.md Sec 6): all outputs land under $TRIPPY_OUTPUT/sog/<stem>/.
#   - Verifies the SOG round-trips: converts it back to a scratch PLY and compares
#     `numGaussians` (a count, not pixels) across input / .sog / .compressed.ply /
#     round-trip -- the honesty check this script exists to make, not just "the
#     tool exited 0". The scratch round-trip file is deleted after comparing.
#   - `--check`: prints toolchain readiness (node version, npm, free memory, pin
#     already installed or not) and exits 0 WITHOUT installing or converting
#     anything -- the guard-clause tests in tests/test_sog_export_script.py use
#     this path so the CPU suite never needs network or minutes of runtime
#     (same convention as scripts/web_build.sh's `--check`).
#   - Bash 3.2 safe, `set -u` clean: no array unpack, quoted expansions.
# Related docs: docs/decisions/ADR-0008-supersplat.md Stage 3; docs/QUEST.md;
#   scripts/quest_viewer_bootstrap.sh (consumes the .sog this script produces);
#   scripts/supersplat_bootstrap.sh (the idempotent-install style this follows).
set -eu
cd "$(dirname "$0")/.."

# The pinned splat-transform release. 3.3.3 is the exact version SuperSplat 3.0.0
# itself devDependency-pins (verified: `node_modules/@playcanvas/splat-transform
# /package.json` inside the Stage-1 checkout), so a SOG built here round-trips
# through the self-hosted SuperSplat editor with no version skew. Bump deliberately
# and re-check that pin before moving this one.
SPLAT_TRANSFORM_PIN="3.3.3"

# The same "conversion of a Karekare-scale PLY is lighter than a training job but
# still real CPU/RAM work" caution line `trippy export-splat-layers` used directly
# (STATE.md 2026-09-09: ran at ~12 GB free, "well under the task's 10 GB caution
# line"). Not scripts/cpu_heavy.sh's 28 GB -- that guard is sized for a training-
# scale job; this one loads at most one ~2 GB PLY plus the tool's own working set.
# The machine OOM'd once already (2026-09-05); this stays a named, commented number.
MIN_FREE_GB=10

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

TOOL_DIR="$TRIPPY_OUTPUT/tools/splat-transform"
BIN="$TOOL_DIR/node_modules/.bin/splat-transform"

usage() {
  echo "usage: sog_export.sh [--check] [--gpu n|cpu] [--out-dir DIR] [--name STEM] <input.ply>" >&2
}

_node_ok() {
  command -v node >/dev/null 2>&1 || { echo "✗ node not found on PATH" >&2; return 1; }
  ver=$(node --version | sed 's/^v//')
  major=$(printf '%s' "$ver" | cut -d. -f1)
  minor=$(printf '%s' "$ver" | cut -d. -f2)
  case "$major" in ''|*[!0-9]*) echo "✗ could not parse node --version: $ver" >&2; return 1 ;; esac
  case "$minor" in ''|*[!0-9]*) minor=0 ;; esac
  ok=0
  if [ "$major" -gt 20 ]; then ok=1; fi
  if [ "$major" -eq 20 ] && [ "$minor" -ge 19 ]; then ok=1; fi
  if [ "$ok" -ne 1 ]; then echo "✗ node $ver is older than the required 20.19" >&2; return 1; fi
  echo "node $ver OK (>= 20.19)"
  return 0
}

_freegb() {
  vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%d", (f+i+s)*16384/1073741824}'
}

_pin_installed() {
  [ -x "$BIN" ] || return 1
  # The version banner is printed on stderr (verified empirically: `-v`'s stdout
  # is empty), so both streams are merged before grepping for it.
  cur=$("$BIN" -v 2>&1 | sed -n 's/^splat-transform v\([0-9.]*\).*/\1/p')
  [ "$cur" = "$SPLAT_TRANSFORM_PIN" ]
}

# Installs @playcanvas/splat-transform locally, pinned, idempotent. Deliberately
# does NOT approve its `webgpu` dependency's postinstall script (npm 11 blocks it
# by default and prints a warning) -- that script fetches a native Dawn/WebGPU
# binary this repo never uses, because every call below passes `-g cpu`.
_ensure_splat_transform() {
  if _pin_installed; then
    echo "  splat-transform $SPLAT_TRANSFORM_PIN already installed at $TOOL_DIR"
    return 0
  fi
  mkdir -p "$TOOL_DIR"
  cat > "$TOOL_DIR/package.json" <<EOF
{
  "name": "trippy-sog-export-tool",
  "private": true,
  "description": "Local, pinned, gitignored install of @playcanvas/splat-transform for trippy's scripts/sog_export.sh and scripts/quest_viewer_bootstrap.sh. Never vendored, never global.",
  "dependencies": {
    "@playcanvas/splat-transform": "$SPLAT_TRANSFORM_PIN"
  }
}
EOF
  echo "▶ npm install (in $TOOL_DIR — a LOCAL install, not global; AGENTS.md Sec 6)"
  ( cd "$TOOL_DIR" && npm install --no-audit --no-fund )
  _pin_installed || { echo "✗ install finished but $BIN is not at $SPLAT_TRANSFORM_PIN" >&2; exit 1; }
}

# --- arg parsing ---------------------------------------------------------------
CHECK=0
OUT_DIR=""
STEM=""
INPUT=""
GPU_ARG="cpu"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1; shift ;;
    --gpu) GPU_ARG=${2:?--gpu needs a value (an adapter index, or "cpu")}; shift 2 ;;
    --out-dir) OUT_DIR=${2:?--out-dir needs a value}; shift 2 ;;
    --name) STEM=${2:?--name needs a value}; shift 2 ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) INPUT=$1; shift ;;
  esac
done

if [ "$CHECK" -eq 1 ]; then
  _node_ok || exit 1
  command -v npm >/dev/null 2>&1 || { echo "✗ npm not found on PATH" >&2; exit 1; }
  echo "npm $(npm --version) OK"
  echo "free memory: $(_freegb) GB (need >= $MIN_FREE_GB GB to convert)"
  if _pin_installed; then
    echo "splat-transform $SPLAT_TRANSFORM_PIN: already installed at $TOOL_DIR"
  else
    echo "splat-transform $SPLAT_TRANSFORM_PIN: would install into $TOOL_DIR (local, pinned, not global)"
  fi
  echo "would convert (gpu=$GPU_ARG): <input.ply> -> <out-dir>/<stem>.sog, <out-dir>/<stem>.compressed.ply"
  echo "would round-trip the .sog back to a scratch PLY and compare Gaussian counts"
  exit 0
fi

[ -n "$INPUT" ] || { usage; exit 2; }
[ -f "$INPUT" ] || { echo "✗ input not found: $INPUT" >&2; exit 2; }
INPUT_ABS=$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")

if [ -z "$STEM" ]; then
  STEM=$(basename "$INPUT_ABS")
  STEM=${STEM%.ply}
  STEM=${STEM%.compressed}
fi

[ -n "$OUT_DIR" ] || OUT_DIR="$TRIPPY_OUTPUT/sog/$STEM"
mkdir -p "$OUT_DIR"

_node_ok
command -v npm >/dev/null 2>&1 || { echo "✗ npm not found on PATH" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "✗ python3 not found on PATH (needed to parse --info json)" >&2; exit 1; }

FREEGB=$(_freegb)
[ "$FREEGB" -ge "$MIN_FREE_GB" ] || { echo "✗ only ${FREEGB} GB free; need >= $MIN_FREE_GB GB" >&2; exit 5; }
echo "free memory: ${FREEGB} GB (>= $MIN_FREE_GB GB OK)"

_ensure_splat_transform

echo "GPU device for SOG compression: $GPU_ARG ($([ "$GPU_ARG" = cpu ] && echo "CPU, safe for a direct run" || echo "native WebGPU adapter -- only from inside a GPU-queue slot"))"

_count() {
  # $1 = path to a splat file (or `null` is never passed here — always a real path).
  # Always -g cpu: reading structural metadata does no SH compression, so this never
  # needs the GPU regardless of what $GPU_ARG is for the real conversion below.
  "$BIN" "$1" null --info json -g cpu -q | python3 -c 'import json,sys; print(json.load(sys.stdin)["numGaussians"])'
}

SOG_PATH="$OUT_DIR/$STEM.sog"
COMPRESSED_PATH="$OUT_DIR/$STEM.compressed.ply"
mkdir -p "$TRIPPY_OUTPUT/tmp"
ROUNDTRIP_PATH="$TRIPPY_OUTPUT/tmp/sog_export_roundtrip_$STEM.ply"

echo "▶ reading input: $INPUT_ABS"
INPUT_COUNT=$(_count "$INPUT_ABS")
echo "  input Gaussians: $INPUT_COUNT"

echo "▶ writing $SOG_PATH"
"$BIN" "$INPUT_ABS" "$SOG_PATH" -g "$GPU_ARG" -w -q
SOG_COUNT=$(_count "$SOG_PATH")

echo "▶ writing $COMPRESSED_PATH"
"$BIN" "$INPUT_ABS" "$COMPRESSED_PATH" -g "$GPU_ARG" -w -q
COMPRESSED_COUNT=$(_count "$COMPRESSED_PATH")

echo "▶ round-trip check: $SOG_PATH -> scratch PLY -> count"
"$BIN" "$SOG_PATH" "$ROUNDTRIP_PATH" -g cpu -w -q
ROUNDTRIP_COUNT=$(_count "$ROUNDTRIP_PATH")
rm -f "$ROUNDTRIP_PATH"

echo
echo "Gaussian counts: input=$INPUT_COUNT sog=$SOG_COUNT compressed.ply=$COMPRESSED_COUNT roundtrip=$ROUNDTRIP_COUNT"
if [ "$INPUT_COUNT" != "$SOG_COUNT" ] || [ "$INPUT_COUNT" != "$COMPRESSED_COUNT" ] || [ "$INPUT_COUNT" != "$ROUNDTRIP_COUNT" ]; then
  echo "✗ Gaussian count mismatch -- the SOG/compressed export is not a faithful 1:1 conversion" >&2
  exit 1
fi
echo "✓ counts match: $INPUT_COUNT Gaussians preserved through .sog and .compressed.ply, and round-trip verified"

echo
echo "sizes:"
ls -la "$INPUT_ABS" "$SOG_PATH" "$COMPRESSED_PATH"
