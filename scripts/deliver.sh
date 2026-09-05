#!/bin/bash
# deliver.sh — the ONLY sanctioned way for trippy to hand Jordan a deliverable.
# Usage: scripts/deliver.sh [--dry-run] <artifact> <name> "<why>"
# Invariants:
#   - <artifact> must resolve to a path under $TRIPPY_OUTPUT or $SPLATS_ROOT/output (exit 2 otherwise);
#     scenes/plys/checkpoints are never copied elsewhere.
#   - Never writes into $SPLATS_ROOT/output/Jordan-Review directly — that's Splats' review_add.sh's job.
#   - A directory containing index.html gets an OPEN_<NAME>.command launcher (python3 http.server
#     bound to 127.0.0.1 only) generated alongside it and delivered too.
#   - --dry-run prints what would be generated/run and touches neither review_add.sh nor
#     research/trips-metal.md.
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then . ./.env; fi
SPLATS_ROOT=${SPLATS_ROOT:-/Users/nzbirdranch/Splats}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

DRYRUN=0
if [ "${1:-}" = "--dry-run" ]; then DRYRUN=1; shift; fi

ARTIFACT=${1:?usage: deliver.sh [--dry-run] <artifact> <name> "<why>"}
NAME=${2:?usage: deliver.sh [--dry-run] <artifact> <name> "<why>"}
WHY=${3:?usage: deliver.sh [--dry-run] <artifact> <name> "<why>"}

[ -e "$ARTIFACT" ] || { echo "✗ artifact not found: $ARTIFACT" >&2; exit 2; }
if [ -d "$ARTIFACT" ]; then
  ABS=$(cd "$ARTIFACT" && pwd)
else
  ABS="$(cd "$(dirname "$ARTIFACT")" && pwd)/$(basename "$ARTIFACT")"
fi

case "$ABS" in
  "$TRIPPY_OUTPUT"|"$TRIPPY_OUTPUT"/*) ;;
  "$SPLATS_ROOT/output"|"$SPLATS_ROOT/output"/*) ;;
  *)
    echo "✗ artifact must be under \$TRIPPY_OUTPUT ($TRIPPY_OUTPUT) or \$SPLATS_ROOT/output ($SPLATS_ROOT/output)" >&2
    exit 2
    ;;
esac

CMDFILE=""
if [ -d "$ABS" ] && [ -f "$ABS/index.html" ]; then
  NAME_UPPER=$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]')
  PORT_OFFSET=$(printf '%s' "$NAME" | cksum | awk '{print $1 % 150}')
  PORT=$((8800 + PORT_OFFSET))
  DELIVER_DIR="$TRIPPY_OUTPUT/deliver/$NAME"
  CMDFILE="$DELIVER_DIR/OPEN_${NAME_UPPER}.command"
  CMDCONTENT="#!/bin/bash
# Double-click me. Starts a tiny local web server (needed: browsers block asset loads when a
# viewer is opened straight from disk) and opens the $NAME viewer in your browser.
# Nothing leaves this machine -- the server listens on 127.0.0.1 only.
cd \"$ABS\" || exit 1
PORT=$PORT
if ! curl -s -o /dev/null \"http://127.0.0.1:\$PORT/index.html\"; then
  nohup python3 -m http.server \"\$PORT\" --bind 127.0.0.1 >/dev/null 2>&1 &
  sleep 1
fi
open \"http://127.0.0.1:\$PORT/index.html\"
echo \"$NAME viewer is at http://127.0.0.1:\$PORT/  (you can close this window)\"
"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "-- dry run: would write $CMDFILE --"
    printf '%s\n' "$CMDCONTENT"
  else
    mkdir -p "$DELIVER_DIR"
    printf '%s\n' "$CMDCONTENT" > "$CMDFILE"
    chmod +x "$CMDFILE"
    echo "wrote $CMDFILE"
  fi
fi

REVIEW_ADD="$SPLATS_ROOT/tools/review_add.sh"
if [ "$DRYRUN" -eq 1 ]; then
  echo "-- dry run: would run --"
  echo "bash \"$REVIEW_ADD\" \"$ABS\" \"$NAME\" \"$WHY\""
  if [ -n "$CMDFILE" ]; then
    echo "bash \"$REVIEW_ADD\" \"$CMDFILE\" \"$NAME-launcher\" \"Double-click to view $NAME in a browser (127.0.0.1 only)\""
  fi
  exit 0
fi

bash "$REVIEW_ADD" "$ABS" "$NAME" "$WHY"
if [ -n "$CMDFILE" ]; then
  bash "$REVIEW_ADD" "$CMDFILE" "$NAME-launcher" "Double-click to view $NAME in a browser (127.0.0.1 only)"
fi

mkdir -p research
[ -f research/trips-metal.md ] || printf '# trips-metal — GPU/CPU research log (running: date, question, job name, numbers, verdict, artifact)\n\n' > research/trips-metal.md
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf -- '- %s delivered %s: %s (%s)\n' "$TS" "$NAME" "$WHY" "$ABS" >> research/trips-metal.md
