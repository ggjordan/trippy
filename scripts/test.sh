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

if [ "${RUN_GPU_TESTS:-0}" = "1" ]; then
  echo "GPU tests only run inside a queue job: scripts/gpu_submit.sh gpu-tests -- python -m pytest -m gpu tests"
  exit 2
fi

scripts/build.sh

PY=./.venv/bin/python
echo "▶ pytest (CPU, not gpu)"
TRIPS_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=0 "$PY" -m pytest -q -m "not gpu" tests

if [ -f rust/Cargo.toml ]; then
  echo "▶ cargo test (rust/)"
  ( cd rust && cargo test --workspace -q )
fi

echo "✓ tests OK"
