#!/bin/bash
# gpu_wait.sh — poll Splats' GPU queue for a trippy job's completion.
# Usage: scripts/gpu_wait.sh <name> [timeout-min]
#   <name> may be given with or without the "trippy-" prefix. Default timeout 720 minutes.
# Invariants:
#   - Read-only against the queue (polls done/<name>.rc every 30s); on timeout it touches
#     nothing in the queue and exits 124.
#   - On completion, copies the job's log into $TRIPPY_OUTPUT/logs/ and exits with the job's rc.
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then . ./.env; fi
SPLATS_ROOT=${SPLATS_ROOT:-/Users/nzbirdranch/Splats}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

RAW_NAME=${1:?usage: gpu_wait.sh <name> [timeout-min]}
TIMEOUT_MIN=${2:-720}

case "$RAW_NAME" in
  trippy-*) NAME="$RAW_NAME" ;;
  *) NAME="trippy-$RAW_NAME" ;;
esac

Q="$SPLATS_ROOT/tools/gpu_queue"
RCFILE="$Q/done/$NAME.rc"
LOGFILE="$Q/logs/$NAME.log"

NOW=$(date +%s)
DEADLINE=$((NOW + TIMEOUT_MIN * 60))

echo "waiting for $RCFILE (timeout ${TIMEOUT_MIN} min)"
while [ ! -f "$RCFILE" ]; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then
    echo "✗ timeout after ${TIMEOUT_MIN} min waiting for $NAME" >&2
    exit 124
  fi
  sleep 30
done

RC=$(cat "$RCFILE")
mkdir -p "$TRIPPY_OUTPUT/logs"
if [ -f "$LOGFILE" ]; then
  cp "$LOGFILE" "$TRIPPY_OUTPUT/logs/$NAME.log"
  echo "---- last 30 lines: $TRIPPY_OUTPUT/logs/$NAME.log ----"
  tail -n 30 "$LOGFILE"
fi
echo "rc=$RC"
exit "$RC"
