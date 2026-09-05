#!/bin/bash
# bootstrap.sh — idempotent developer-machine setup for trippy. Safe to run every session.
# Usage: scripts/bootstrap.sh
# Invariants:
#   - Idempotent: safe to re-run; only creates/updates what's missing.
#   - Never touches the GPU (no MPS work, no gpu_submit.sh call here).
#   - Python is always ./.venv/bin/python, created below by `uv sync` (own venv, Python 3.13).
#   - Reads $SPLATS_ROOT read-only; only checks that its GPU-queue/review tools exist.
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then . ./.env; fi
SPLATS_ROOT=${SPLATS_ROOT:-/Users/nzbirdranch/Splats}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

git config core.hooksPath .githooks

if [ -f .env.example ]; then
  cp -n .env.example .env
fi

UV=""
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
else
  echo "✗ uv not found on PATH or at ~/.local/bin/uv" >&2
  echo "  Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! "$UV" python find 3.13 >/dev/null 2>&1; then
  echo "▶ installing Python 3.13 via uv"
  "$UV" python install 3.13
fi

echo "▶ uv sync"
"$UV" sync --quiet

SUBMIT_OK="present"
[ -x "$SPLATS_ROOT/tools/gpu_queue/submit.sh" ] || { SUBMIT_OK="MISSING"; echo "⚠ warning: $SPLATS_ROOT/tools/gpu_queue/submit.sh not found" >&2; }
REVIEW_OK="present"
[ -x "$SPLATS_ROOT/tools/review_add.sh" ] || { REVIEW_OK="MISSING"; echo "⚠ warning: $SPLATS_ROOT/tools/review_add.sh not found" >&2; }

RUNNER_ALIVE="no"
if [ -f "$SPLATS_ROOT/tools/gpu_queue/runner.pid" ]; then
  rpid=$(cat "$SPLATS_ROOT/tools/gpu_queue/runner.pid" 2>/dev/null || true)
  if [ -n "$rpid" ] && kill -0 "$rpid" 2>/dev/null; then
    RUNNER_ALIVE="yes (pid $rpid)"
  fi
fi

FREEGB=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%d", (f+i+s)*16384/1073741824}')

mkdir -p output/jobs output/logs output/runs output/deliver

echo "trippy bootstrap: hooksPath=.githooks venv=./.venv SPLATS_ROOT=$SPLATS_ROOT"
echo "  gpu_queue submit.sh: $SUBMIT_OK | review_add.sh: $REVIEW_OK"
echo "  GPU-queue runner alive: $RUNNER_ALIVE"
echo "  free memory: ${FREEGB} GB (need >=28 GB for GPU jobs)"
