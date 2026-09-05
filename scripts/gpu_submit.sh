#!/bin/bash
# gpu_submit.sh — submit a self-contained job to Splats' GPU queue.
# Usage: scripts/gpu_submit.sh [--prio N | --train] [--wait] [--dry-run] <name> -- <command...>
#   Default prio 15 (trippy short jobs live at 10-19). --train sets prio 70 (behind Splats' 60).
#   --wait  block until done and print the log (execs scripts/gpu_wait.sh).
#   --dry-run  write and print the job file only; skip all queue/runner/memory checks and never
#              calls Splats' submit.sh or writes to research/trips-metal.md.
# Invariants:
#   - This script NEVER touches the GPU itself; it only ever hands a script to Splats' queue.
#   - Refuses (exit 3) if the command text mentions gpu_lock.sh — only the queue runner holds that lock.
#   - Unless --dry-run: refuses if the GPU-queue runner is not alive (exit 4) or free memory < 28 GB (exit 5).
#   - The generated job file uses only absolute paths and trippy's own ./.venv, never the caller's PATH.
set -eu
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}; _pre_env_splats=${SPLATS_ROOT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}; SPLATS_ROOT=${_pre_env_splats:-${SPLATS_ROOT:-}}
SPLATS_ROOT=${SPLATS_ROOT:-/Users/nzbirdranch/Splats}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

usage() {
  echo "usage: gpu_submit.sh [--prio N | --train] [--wait] [--dry-run] <name> -- <command...>" >&2
}

PRIO=15
WAIT=0
DRYRUN=0
NAME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --prio) PRIO=${2:-}; shift 2 ;;
    --train) PRIO=70; shift ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) NAME="$1"; shift ;;
  esac
done

CMD=("$@")

[ -n "$NAME" ] || { usage; exit 2; }
[ ${#CMD[@]} -gt 0 ] || { echo "✗ missing -- <command...>" >&2; usage; exit 2; }

case "$PRIO" in
  ''|*[!0-9]*) echo "✗ --prio must be numeric" >&2; exit 2 ;;
esac
if [ "$PRIO" -lt 10 ] || { [ "$PRIO" -gt 19 ] && [ "$PRIO" -ne 70 ]; }; then
  echo "✗ prio must be 10-19 (trippy short jobs) or 70 (trainings, behind Splats' 60)." >&2
  echo "  Jordan can override this by editing the scripts/gpu_submit.sh call." >&2
  exit 2
fi

case "$NAME" in
  *[!A-Za-z0-9_.-]*|'') echo "✗ name must match [A-Za-z0-9_.-]+" >&2; exit 2 ;;
esac
JOBNAME="trippy-$NAME"

CMDSTR="${CMD[*]}"
case "$CMDSTR" in
  *gpu_lock.sh*)
    echo "✗ refused: command must not reference gpu_lock.sh (the queue runner holds the lock)" >&2
    exit 3
    ;;
esac

if [ "$DRYRUN" -eq 0 ]; then
  RUNNER_OK=0
  if [ -f "$SPLATS_ROOT/tools/gpu_queue/runner.pid" ]; then
    rpid=$(cat "$SPLATS_ROOT/tools/gpu_queue/runner.pid" 2>/dev/null || true)
    if [ -n "$rpid" ] && kill -0 "$rpid" 2>/dev/null; then RUNNER_OK=1; fi
  fi
  [ "$RUNNER_OK" -eq 1 ] || { echo "✗ GPU-queue runner not alive ($SPLATS_ROOT/tools/gpu_queue/runner.pid)" >&2; exit 4; }

  FREEGB=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%d", (f+i+s)*16384/1073741824}')
  [ "$FREEGB" -ge 28 ] || { echo "✗ only ${FREEGB} GB free; need >=28 GB before submitting a GPU job" >&2; exit 5; }
fi

# Rewrite a leading `python` -> the absolute venv python; a leading `trippy` -> the venv CLI module.
FIRST="${CMD[0]}"
REST=()
if [ ${#CMD[@]} -gt 1 ]; then REST=("${CMD[@]:1}"); fi
VENV_PY="$REPO_ROOT/.venv/bin/python"
# Git worktrees have no .venv: fall back to the main checkout's interpreter (PYTHONPATH=. below
# keeps the worktree's package first on sys.path).
if [ ! -x "$VENV_PY" ]; then VENV_PY="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv/bin/python"; fi
EXTRA=()
case "$FIRST" in
  python) NEWHEAD="$VENV_PY" ;;
  trippy) NEWHEAD="$VENV_PY"; EXTRA=(-m trippy.cli) ;;
  *) NEWHEAD="$FIRST" ;;
esac

FINAL=("$NEWHEAD")
if [ ${#EXTRA[@]} -gt 0 ]; then FINAL+=("${EXTRA[@]+"${EXTRA[@]}"}"); fi
if [ ${#REST[@]} -gt 0 ]; then FINAL+=("${REST[@]+"${REST[@]}"}"); fi

EXEC_LINE="exec"
for tok in "${FINAL[@]+"${FINAL[@]}"}"; do
  EXEC_LINE="$EXEC_LINE $(printf '%q' "$tok")"
done

mkdir -p "$TRIPPY_OUTPUT/jobs"
JOBFILE="$TRIPPY_OUTPUT/jobs/$JOBNAME.sh"

cat > "$JOBFILE" <<EOF
#!/bin/bash
set -eu
export TRIPPY_ROOT="$REPO_ROOT"
export SPLATS_ROOT="$SPLATS_ROOT"
export TRIPPY_OUTPUT="$TRIPPY_OUTPUT"
export RUST_LOG=info
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTHONPATH="$REPO_ROOT"
export PATH="$(dirname "$VENV_PY"):\$PATH"
cd "$REPO_ROOT"
$EXEC_LINE
EOF
chmod +x "$JOBFILE"

if [ "$DRYRUN" -eq 1 ]; then
  echo "-- dry run: would submit prio=$PRIO name=$JOBNAME --"
  echo "-- job file: $JOBFILE --"
  cat "$JOBFILE"
  exit 0
fi

set +e
bash "$SPLATS_ROOT/tools/gpu_queue/submit.sh" "$PRIO" "$JOBNAME" "$JOBFILE"
SUBMIT_RC=$?
set -e
echo "submit.sh rc=$SUBMIT_RC"
[ "$SUBMIT_RC" -eq 0 ] || { echo "✗ submit.sh refused the job (see message above)" >&2; exit "$SUBMIT_RC"; }
echo "  done: $SPLATS_ROOT/tools/gpu_queue/done/$JOBNAME.rc"
echo "  log:  $SPLATS_ROOT/tools/gpu_queue/logs/$JOBNAME.log"

mkdir -p research
[ -f research/trips-metal.md ] || printf '# trips-metal — GPU/CPU research log (running: date, question, job name, numbers, verdict, artifact)\n\n' > research/trips-metal.md
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf -- '- %s submitted job %s prio %s: %s\n' "$TS" "$JOBNAME" "$PRIO" "$CMDSTR" >> research/trips-metal.md

if [ "$WAIT" -eq 1 ]; then
  exec "$(dirname "$0")/gpu_wait.sh" "$NAME"
fi
