#!/bin/bash
# build.sh — compile check for trippy. Used by the pre-push hook; also runnable standalone.
# Usage: scripts/build.sh
# Invariants:
#   - Python is always ./.venv/bin/python (created by scripts/bootstrap.sh via `uv sync`).
#   - Rust steps only run when rust/Cargo.toml exists.
#   - <30 s; exits non-zero on any failure.
set -eu
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
[ -x "$PY" ] || { echo "✗ $PY not found; run scripts/bootstrap.sh first" >&2; exit 1; }

echo "▶ compileall"
"$PY" -m compileall -q trippy tests tools

if [ -x ./.venv/bin/ruff ]; then
  echo "▶ ruff check"
  ./.venv/bin/ruff check .
fi

echo "▶ import trippy"
"$PY" -c "import trippy"

if [ -f rust/Cargo.toml ]; then
  echo "▶ cargo check (rust/)"
  ( cd rust && cargo check --workspace )
fi

echo "✓ build OK"
