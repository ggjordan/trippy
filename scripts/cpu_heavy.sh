#!/bin/bash
# cpu_heavy.sh — run one CPU-heavy background job at a time (single global lock + memory guard).
# Usage: scripts/cpu_heavy.sh <name> -- <command...>
#        scripts/cpu_heavy.sh --status
#        scripts/cpu_heavy.sh --release
# Invariants:
#   - Only one heavy CPU job at a time; the lock file holds a pid and stale locks (dead pid) are
#     cleared automatically.
#   - Refuses to start if already held (exit 4) or if free memory < 28 GB (exit 5).
#   - Never touches the GPU queue.
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then . ./.env; fi
SPLATS_ROOT=${SPLATS_ROOT:-/Users/nzbirdranch/Splats}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}
LOCK="$TRIPPY_OUTPUT/.cpu_heavy.lock"

mkdir -p "$TRIPPY_OUTPUT/logs"

lock_pid() { [ -f "$LOCK" ] && cat "$LOCK" 2>/dev/null || true; }
lock_alive() {
  p=$(lock_pid)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

case "${1:-}" in
  --status)
    if lock_alive; then echo "held by pid $(lock_pid)"; else echo "free"; fi
    exit 0
    ;;
  --release)
    rm -f "$LOCK"
    echo "released"
    exit 0
    ;;
esac

NAME=${1:?usage: cpu_heavy.sh <name> -- <command...>}
shift
[ "${1:-}" = "--" ] || { echo "usage: cpu_heavy.sh <name> -- <command...>" >&2; exit 2; }
shift
CMD=("$@")
[ ${#CMD[@]} -gt 0 ] || { echo "✗ missing -- <command...>" >&2; exit 2; }

if lock_alive; then
  echo "✗ cpu_heavy already held by pid $(lock_pid) (scripts/cpu_heavy.sh --status / --release)" >&2
  exit 4
elif [ -f "$LOCK" ]; then
  echo "ℹ clearing stale lock (pid $(lock_pid) not running)" >&2
  rm -f "$LOCK"
fi

FREEGB=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%d", (f+i+s)*16384/1073741824}')
[ "$FREEGB" -ge 28 ] || { echo "✗ only ${FREEGB} GB free; need >=28 GB" >&2; exit 5; }

LOG="$TRIPPY_OUTPUT/logs/$NAME.log"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
disown
echo "$PID" > "$LOCK"
echo "started pid $PID"
echo "log: $LOG"
