#!/bin/bash
# queue_training.sh — submit a self-reporting training run to the GPU queue.
# Usage: scripts/queue_training.sh <config.yaml> [--max-minutes M] [--dry-run]
# Invariants:
#   - <config.yaml> must exist and parse as YAML with a top-level `run_dir:` key
#     (exit 2 otherwise) -- the run never even reaches gpu_submit.sh without one.
#   - The queue job name is the config's own `run_dir:` basename, so
#     output/jobs/trippy-<name>.sh and the GPU-queue log line up with
#     output/runs/.../<name> without Jordan having to cross-reference anything.
#   - Always submits `trippy train --config <config.yaml> --report` (this task's
#     brief: no orchestrator step should be needed after training finishes) at
#     `gpu_submit.sh --train` priority (70, behind Splats' own jobs at 60).
#   - --dry-run forwards to gpu_submit.sh (writes/prints the job file only; no
#     queue/runner/memory checks, no submit.sh call, no research/trips-metal.md write).
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

usage() {
  echo "usage: queue_training.sh <config.yaml> [--max-minutes M] [--dry-run]" >&2
}

CONFIG=""
MAX_MINUTES=""
DRYRUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --max-minutes) MAX_MINUTES=${2:-}; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    -*) usage; exit 2 ;;
    *)
      if [ -n "$CONFIG" ]; then usage; exit 2; fi
      CONFIG="$1"
      shift
      ;;
  esac
done

[ -n "$CONFIG" ] || { usage; exit 2; }
[ -f "$CONFIG" ] || { echo "✗ config not found: $CONFIG" >&2; exit 2; }

# Same venv-python resolution as gpu_submit.sh: git worktrees have no .venv of
# their own, so fall back to the main checkout's interpreter.
VENV_PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  MAIN_ROOT=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
  if [ -n "$MAIN_ROOT" ]; then VENV_PY="$(dirname "$MAIN_ROOT")/.venv/bin/python"; fi
fi
[ -x "$VENV_PY" ] || VENV_PY="python3"

RUN_DIR=$("$VENV_PY" - "$CONFIG" <<'PYEOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f) or {}
run_dir = data.get("run_dir")
if not run_dir:
    sys.exit(1)
print(run_dir)
PYEOF
) || { echo "✗ $CONFIG has no top-level 'run_dir:' key (or is not valid YAML)" >&2; exit 2; }

NAME=$(basename "$RUN_DIR")
case "$NAME" in
  *[!A-Za-z0-9_.-]*|'')
    echo "✗ run_dir basename ($NAME, from $CONFIG's run_dir: $RUN_DIR) must match [A-Za-z0-9_.-]+" >&2
    echo "  (gpu_submit.sh's job-name rule -- edit run_dir in the config)" >&2
    exit 2
    ;;
esac

CMD=(trippy train --config "$CONFIG" --report)
if [ -n "$MAX_MINUTES" ]; then CMD+=(--max-minutes "$MAX_MINUTES"); fi

SUBMIT_ARGS=(--train)
[ "$DRYRUN" -eq 1 ] && SUBMIT_ARGS+=(--dry-run)
SUBMIT_ARGS+=("$NAME" --)

exec bash "$REPO_ROOT/scripts/gpu_submit.sh" "${SUBMIT_ARGS[@]}" "${CMD[@]}"
