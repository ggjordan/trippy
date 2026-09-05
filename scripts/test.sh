#!/bin/bash
# test.sh — run trippy's CPU test suite (build.sh first, then pytest).
# Usage: scripts/test.sh
#        RUN_GPU_TESTS=1 scripts/test.sh   -> refuses; GPU tests only run inside a queue job
# Invariants:
#   - CPU-only here: TRIPS_DEVICE=cpu, PYTORCH_ENABLE_MPS_FALLBACK=0. No MPS/GPU work in this script.
#   - The `gpu` pytest marker is excluded; those tests run only via scripts/gpu_submit.sh.
#   - Rust tests only run when rust/Cargo.toml exists.
set -eu
cd "$(dirname "$0")/.."
# Worktrees have no .venv of their own: fall back to the main checkout's venv and make the
# worktree's package shadow the editable install (PYTHONPATH=.).
PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv/bin/python"; fi
export PYTHONPATH=.

if [ "${RUN_GPU_TESTS:-0}" = "1" ]; then
  echo "GPU tests only run inside a queue job: scripts/gpu_submit.sh gpu-tests -- python -m pytest -m gpu tests"
  exit 2
fi

scripts/build.sh

echo "▶ pytest (CPU, not gpu)"
TRIPS_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=0 "$PY" -m pytest -q -m "not gpu" tests

if [ -f rust/Cargo.toml ]; then
  # Scoped to trippy's own crates (brush-pyramid, brush-unet), not the full
  # Brush fork vendored at rust/brush-trips (a separate workspace/submodule;
  # see docs/decisions/ADR-0005-brush-fork-layout.md). That full suite is
  # exercised separately via scripts/cpu_heavy.sh, never on every push.
  echo "▶ cargo test (rust/: brush-pyramid, brush-unet)"
  ( cd rust && cargo test -p brush-pyramid -p brush-unet -q )
fi

echo "✓ tests OK"
