#!/bin/bash
# open_supersplat.sh — generate the double-click launcher for the self-hosted SuperSplat editor.
# Usage: scripts/open_supersplat.sh [--dist PATH]
# Invariants (ADR-0008-supersplat.md Stage 1, item 2):
#   - Serves scripts/supersplat_bootstrap.sh's dist/ with `python3 -m http.server` bound to
#     127.0.0.1 ONLY, on a fixed port (never 0.0.0.0) -- never the hosted superspl.at/editor.
#   - Writes ONLY into $TRIPPY_OUTPUT/deliver/supersplat/, same shape as
#     scripts/open_mac_viewer.sh: the generated .command hardcodes absolute paths and depends
#     on nothing from the caller's shell (Finder gives it none of those).
#   - The .command's header carries the red-box warning from ADR-0008 Sec 3/Sec 5 task 5:
#     File -> Publish uploads scene.ply to PlayCanvas; self-hosted it 404s, but never click it.
#   - Deliver the generated .command with scripts/deliver.sh, same as every other launcher.
# Related docs: docs/decisions/ADR-0008-supersplat.md; docs/USER_GUIDE.md
#   "Editing the cleaned splat in SuperSplat"; scripts/supersplat_bootstrap.sh.
set -eu
cd "$(dirname "$0")/.."
# The MAIN checkout, not a worktree: the generated .command outlives the worktree it was
# made in (same reasoning as open_mac_viewer.sh).
REPO_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

# Fixed, not derived from deliver.sh's name-hash formula -- SuperSplat is a standing tool
# (one instance, reopened many sessions), not a one-off per-run viewer, so a stable port is
# easier to remember and to rule out as "already running" (the curl check below).
SUPERSPLAT_PORT=8877
DIST_DIR="$TRIPPY_OUTPUT/tools/supersplat/dist"

usage() { echo "usage: open_supersplat.sh [--dist PATH]" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dist) DIST_DIR=${2:?--dist needs a value}; shift 2 ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) usage; exit 2 ;;
  esac
done

[ -f "$DIST_DIR/index.html" ] || {
  echo "✗ no built SuperSplat at $DIST_DIR — run scripts/supersplat_bootstrap.sh first" >&2
  exit 3
}
DIST_ABS=$(cd "$DIST_DIR" && pwd)

# Where the two-layer PLY pairs from `trippy export-splat-layers` land (task 3). This is a
# printed hint only -- Jordan drags files from Finder, this script never touches them.
CLEAN_REVIEW_DIR="\$HOME/Splats/output/Jordan-Review/2-open-in-brush"

DELIVER_DIR="$TRIPPY_OUTPUT/deliver/supersplat"
CMDFILE="$DELIVER_DIR/OPEN_SUPERSPLAT.command"
mkdir -p "$DELIVER_DIR"

cat > "$CMDFILE" <<LAUNCHER
#!/bin/bash
# Double-click me. Opens the self-hosted SuperSplat 3.0 splat editor in your browser.
#
# ================================================================================
#  DO NOT click File -> Publish. It is one click away from Export in the same menu.
#  Publish tries to upload scene.ply to PlayCanvas's servers. Self-hosted like this
#  it 404s and goes nowhere -- but never click it on a real scene. Use Export instead
#  (PLY / compressed PLY / SOG / SPZ / .splat all stay on this machine).
# ================================================================================
#
# Everything here runs on THIS machine: a static build served on 127.0.0.1 only.
# Nothing is uploaded anywhere. See docs/decisions/ADR-0008-supersplat.md Sec 3 for
# the full privacy audit and research/trips-metal.md for the measured request list.
#
# HOW TO USE (see docs/USER_GUIDE.md "Editing the cleaned splat in SuperSplat" for more):
#   1. Drag BOTH files from a cleaned pair into the browser window:
#        $CLEAN_REVIEW_DIR/<name>-keep.ply
#        $CLEAN_REVIEW_DIR/<name>-fog.ply
#      They open as two layers in the Scene panel (show/hide/solo each one).
#   2. Solo the "-fog" layer to see exactly what TRIPS flagged; hide it once you've
#      checked it, and finish cleaning up the "-keep" layer with the selection tools.
#   3. Export when done (File -> Export -> PLY, or SOG for a compressed file).
#
DIST="$DIST_ABS"
PORT=$SUPERSPLAT_PORT

if [ ! -f "\$DIST/index.html" ]; then
  echo "SuperSplat is not built at:"
  echo "  \$DIST"
  echo
  echo "Rebuild it with:"
  echo "  $REPO_ROOT/scripts/supersplat_bootstrap.sh"
  echo
  echo "(press return to close)"
  read -r _
  exit 1
fi

cd "\$DIST" || exit 1
if ! curl -s -o /dev/null "http://127.0.0.1:\$PORT/index.html"; then
  nohup python3 -m http.server "\$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
  sleep 1
fi
open "http://127.0.0.1:\$PORT/index.html"
echo "SuperSplat is at http://127.0.0.1:\$PORT/  (you can close this window)"
echo
echo "Cleaned splat layer pairs to drag in are usually in:"
echo "  $CLEAN_REVIEW_DIR"
LAUNCHER

chmod +x "$CMDFILE"
echo "wrote $CMDFILE"
echo "  dist: $DIST_ABS"
echo "  port: $SUPERSPLAT_PORT (127.0.0.1 only)"
echo
echo "Deliver it with:"
echo "  scripts/deliver.sh \"$CMDFILE\" supersplat \"Self-hosted SuperSplat 3.0 splat editor (127.0.0.1 only) -- drag in a -keep/-fog layer pair, never click Publish\""
