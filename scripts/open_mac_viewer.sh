#!/bin/bash
# open_mac_viewer.sh — generate a double-click launcher for the native TRIPS viewer.
# Usage: scripts/open_mac_viewer.sh [--scale F] [--flags "..."] [--binary PATH] <bundle-dir> <name>
# Invariants:
#   - Writes ONLY into $TRIPPY_OUTPUT/deliver/<name>/, which is where deliver.sh expects
#     to find an artifact; it never writes into $SPLATS_ROOT/output/Jordan-Review/.
#   - <bundle-dir> must contain bundle.json (exit 2 otherwise) — a launcher that opens a
#     folder with no scene in it is worse than no launcher.
#   - The generated .command hardcodes ABSOLUTE paths and never depends on the caller's
#     PATH, cwd or shell, because Finder gives it none of those.
#   - Nothing leaves the machine: the viewer is a local binary reading local files.
# Related docs: docs/USER_GUIDE.md; docs/decisions/ADR-0006-viewer-integration.md.
set -eu
cd "$(dirname "$0")/.."
# The MAIN checkout, not this worktree: the generated .command outlives the
# worktree it was made in, so every absolute path baked into it must point at
# somewhere that still exists after the branch is merged and the worktree is
# removed. `--git-common-dir` is the shared `.git` for both.
REPO_ROOT=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd)

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}; _pre_env_splats=${SPLATS_ROOT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}; SPLATS_ROOT=${_pre_env_splats:-${SPLATS_ROOT:-}}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

# The viewer's default speed setting, chosen from the measured table in
# research/trips-metal.md (v0.4.0, horse bundle, M3 Ultra):
#   --half-net     runs the U-Net in f16. 2.58x on the whole frame and 59.8 dB
#                  against the exact pipeline -- i.e. free. This is the lever that
#                  matters: the network is ~89% of the frame, the rasteriser ~11%.
#   --scale 0.75   45.3 ms, 22.1 fps in a 1080p window. `-` and `=` change it live.
# Deliberately NOT included: --packed-sort (1.7 ms for 0.75 dB) and --cap-fragments
# (nothing for 42 dB). What Jordan sees should be what the pipeline produces.
DEFAULT_FLAGS="--half-net --scale 0.75"
BINARY="$REPO_ROOT/rust/target/release/trips-viewer"
SCALE=""

usage() {
  echo 'usage: open_mac_viewer.sh [--scale F] [--flags "..."] [--binary PATH] <bundle-dir> <name>' >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --scale)  SCALE=${2:?--scale needs a value}; shift 2 ;;
    --flags)  DEFAULT_FLAGS=${2:?--flags needs a value}; shift 2 ;;
    --binary) BINARY=${2:?--binary needs a value}; shift 2 ;;
    --) shift; break ;;
    -*) usage; exit 2 ;;
    *) break ;;
  esac
done

BUNDLE=${1:?$(usage)}
NAME=${2:?$(usage)}

[ -d "$BUNDLE" ] || { echo "✗ bundle directory not found: $BUNDLE" >&2; exit 2; }
BUNDLE_ABS=$(cd "$BUNDLE" && pwd)
[ -f "$BUNDLE_ABS/bundle.json" ] || {
  echo "✗ $BUNDLE_ABS has no bundle.json — make one with:" >&2
  echo "    trippy export-bundle --checkpoint <ckpt> --out $BUNDLE_ABS" >&2
  exit 2
}

if [ ! -x "$BINARY" ]; then
  echo "✗ viewer binary not found at $BINARY" >&2
  echo "  build it with:" >&2
  echo "    bash scripts/cpu_heavy.sh trips-viewer-build -- bash -c 'cd $REPO_ROOT/rust && cargo build --release -p trips-viewer'" >&2
  exit 3
fi

# A caller-supplied --scale replaces the default rather than appending a second
# one (last-wins would work, but a duplicated flag in a delivered launcher looks
# like a bug to whoever reads it next).
if [ -n "$SCALE" ]; then
  DEFAULT_FLAGS=$(printf '%s' "$DEFAULT_FLAGS" | sed 's/--scale [0-9.]*//')
  DEFAULT_FLAGS="$DEFAULT_FLAGS --scale $SCALE"
fi

NAME_UPPER=$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]')
DELIVER_DIR="$TRIPPY_OUTPUT/deliver/$NAME"
CMDFILE="$DELIVER_DIR/OPEN_TRIPS_MAC_${NAME}.command"
mkdir -p "$DELIVER_DIR"

cat > "$CMDFILE" <<LAUNCHER
#!/bin/bash
# Double-click me. Opens the "$NAME" TRIPS scene in the native Mac viewer.
#
# Everything runs on this machine: a local binary reading local files. Nothing
# is uploaded anywhere.
#
#   left-drag         turn around the scene (orbit) — or look around, in free mode
#   right/middle-drag pan sideways
#   scroll            orbit: closer / further      free: change fly speed
#   W A S D           move           Q / E   down / up
#   R                 back to the view it opened at
#   N / P             next / previous real camera of the capture
#   F                 orbit <-> free fly
#   V                 cycle view: network -> raw level-0 -> coverage
#   - / =             render scale   TAB     hide the panel
#
# It opens on a real camera of the capture and orbits inside it, so you cannot
# get lost. Speed is set from how far apart the real cameras are, so one tap of
# W is a step, not a teleport.
#
# The panel's top line is the frame time and fps. "coverage" is the honesty
# view: dark means the rasteriser drew nothing there and the network invented
# every pixel of it.
BIN="$BINARY"
BUNDLE="$BUNDLE_ABS"

if [ ! -x "\$BIN" ]; then
  echo "The viewer binary is missing:"
  echo "  \$BIN"
  echo
  echo "Rebuild it with:"
  echo "  cd $REPO_ROOT/rust && cargo build --release -p trips-viewer"
  echo
  echo "(press return to close)"
  read -r _
  exit 1
fi

echo "Opening $NAME ..."
"\$BIN" "\$BUNDLE" $DEFAULT_FLAGS
status=\$?
if [ \$status -ne 0 ]; then
  echo
  echo "The viewer exited with status \$status. The message above says why."
  echo "(press return to close)"
  read -r _
fi
LAUNCHER

chmod +x "$CMDFILE"
echo "wrote $CMDFILE"
echo "  binary: $BINARY"
echo "  bundle: $BUNDLE_ABS"
echo "  flags:  $DEFAULT_FLAGS"
echo
echo "Deliver it with:"
echo "  scripts/deliver.sh \"$CMDFILE\" <deliver-name> \"<why>\""
