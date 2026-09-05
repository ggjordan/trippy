#!/bin/bash
# build.sh — compile check for trippy. Used by the pre-push hook; also runnable standalone.
# Usage: scripts/build.sh
# Invariants:
#   - Python is always ./.venv/bin/python (created by scripts/bootstrap.sh via `uv sync`).
#   - Rust steps only run when rust/Cargo.toml exists.
#   - <30 s; exits non-zero on any failure.
set -eu
cd "$(dirname "$0")/.."
# Worktrees have no .venv of their own: fall back to the main checkout's venv and make the
# worktree's package shadow the editable install (PYTHONPATH=.).
PY=./.venv/bin/python
if [ ! -x "$PY" ]; then PY="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv/bin/python"; fi
export PYTHONPATH=.

[ -x "$PY" ] || { echo "✗ $PY not found; run scripts/bootstrap.sh first" >&2; exit 1; }

echo "▶ compileall"
"$PY" -m compileall -q trippy tests tools

RUFF="$(dirname "$PY")/ruff"
if [ -x "$RUFF" ]; then
  echo "▶ ruff check"
  "$RUFF" check .
fi

echo "▶ import trippy"
"$PY" -c "import trippy"

if [ -f rust/Cargo.toml ]; then
  # rust/Cargo.toml is trippy's OWN thin workspace (brush-pyramid, brush-unet
  # only) -- NOT the rust/brush-trips submodule's workspace, which vendors the
  # whole Brush fork and would blow the <30s budget. See
  # docs/decisions/ADR-0005-brush-fork-layout.md.
  echo "▶ cargo check (rust/: brush-pyramid, brush-unet)"
  ( cd rust && cargo check -p brush-pyramid -p brush-unet )
  echo "▶ cargo check --features gpu --all-targets (type-check only; GPU tests run in queue jobs)"
  ( cd rust && cargo check -p brush-pyramid -p brush-unet --features gpu --all-targets --offline )
  # trips-viewer only exists with the gpu feature graph, so it is checked here
  # rather than in scripts/test.sh: `cargo check` reuses the metadata the line
  # above just produced (seconds), while `cargo test` would have to codegen
  # Burn/CubeCL/wgpu in debug (minutes). Its 37 unit tests are CPU-only and are
  # run in release next to the viewer build:
  #   cargo test --release -p trips-viewer
  echo "▶ cargo check -p trips-viewer --all-targets (the viewer, incl. its tests)"
  ( cd rust && cargo check -p trips-viewer --all-targets --offline )
fi

echo "✓ build OK"
