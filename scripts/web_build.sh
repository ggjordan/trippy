#!/bin/bash
# web_build.sh -- build the Brush fork's web viewer (wasm-pack + vite) into a static dist dir.
# Purpose: v0.5.0 groundwork. Proves the desktop web-viewer toolchain (Rust -> wasm -> vite bundle
# -> static files servable on 127.0.0.1) end to end. This builds the STOCK Brush web demo only;
# dropping in the TRIPS render pipeline (brush-pyramid/brush-unet output) is a later task -- see
# docs/WEB_VIEWER.md "Next: wiring TRIPS in" for where that hook goes.
# Usage: scripts/web_build.sh [--dev] [--check]
#   --dev     wasm-pack --dev (fast, unoptimized, includes dwarf debug info) instead of --release
#   --check   dry run: verify npm/wasm-pack/wasm32 target are present and print the plan, do not
#             build anything. For CI/tests that want to confirm the toolchain without paying for
#             a full wasm-pack + vite build.
# Invariants:
#   - Operates on rust/brush-trips/apps/brush-app/web (a submodule); never edits submodule files.
#   - Output is copied to $TRIPPY_OUTPUT/web/brush-dist/ (gitignored under output/), NOT left in
#     the submodule's own dist/, so scripts/deliver.sh's artifact-location check
#     ($TRIPPY_OUTPUT or $SPLATS_ROOT/output) passes.
#   - Builds with BRUSH_BASE_PATH=/ (default), not the package.json `npm run build`'s
#     BRUSH_BASE_PATH=/brush-demo (that base path is for GitHub Pages; a `/brush-demo`-rooted
#     bundle 404s every asset when served from a plain http.server root).
#   - wasm-pack build compiles Burn/CubeCL/wgpu/egui to wasm32 -- heavy. Route through
#     scripts/cpu_heavy.sh, never run this script itself inside another cpu_heavy.sh job.
#   - Does not install wasm-pack or the wasm32-unknown-unknown target; it checks for both and
#     exits with the exact command to fix it, per AGENTS.md's "don't fake unsupported APIs" rule.
#   - Uses `npm ci`, not `npm install`, in the submodule: the submodule's package-lock.json lives
#     at the npm-workspace root (rust/brush-trips/package-lock.json, workspaces = apps/brush-app/web
#     + apps/brush-js/web), and `npm install` run from a workspace member directory rewrites that
#     root lockfile (observed: it drops an "extraneous" brush_nextjs workspace entry the lockfile
#     carries but this checkout doesn't have on disk), which is an edit to a submodule file --
#     forbidden by rust/README.md's "submodule, not a subtree" model. `npm ci` installs from the
#     existing lockfile without writing to it.
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}
WEB_DIR="rust/brush-trips/apps/brush-app/web"
DIST_OUT="$TRIPPY_OUTPUT/web/brush-dist"

[ -d "$WEB_DIR" ] || {
  echo "✗ $WEB_DIR not found -- is the brush-trips submodule initialised? (git submodule update --init)" >&2
  exit 2
}

command -v npm >/dev/null 2>&1 || { echo "✗ npm not found" >&2; exit 2; }
command -v wasm-pack >/dev/null 2>&1 || {
  echo "✗ wasm-pack not found -- install with: npm i -g wasm-pack  (or: cargo install wasm-pack)" >&2
  exit 2
}
rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$' || {
  echo "✗ wasm32-unknown-unknown target missing -- run: rustup target add wasm32-unknown-unknown" >&2
  exit 2
}

MODE="release"
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --dev) MODE="dev" ;;
    --check) CHECK=1 ;;
    *) echo "✗ unknown argument: $arg (expected --dev and/or --check)" >&2; exit 2 ;;
  esac
done

if [ "$CHECK" -eq 1 ]; then
  echo "✓ toolchain OK: npm=$(command -v npm), wasm-pack=$(command -v wasm-pack), wasm32-unknown-unknown installed"
  echo "  would run: npm ci ($WEB_DIR)"
  echo "  would run: wasm-pack build ($MODE) via npm run build:wasm-$MODE"
  echo "  would run: BRUSH_BASE_PATH=/ npx vite build ($WEB_DIR)"
  echo "  would copy $WEB_DIR/dist -> $DIST_OUT"
  exit 0
fi

echo "▶ npm ci ($WEB_DIR) -- not npm install, see the invariant note at the top of this file"
( cd "$WEB_DIR" && npm ci )

echo "▶ wasm-pack build ($MODE)"
if [ "$MODE" = "dev" ]; then
  ( cd "$WEB_DIR" && npm run build:wasm-dev )
else
  ( cd "$WEB_DIR" && npm run build:wasm-release )
fi

echo "▶ vite build (base path '/' for local serving; package.json's own \`npm run build\` target is /brush-demo for GH Pages, not what we want here)"
( cd "$WEB_DIR" && BRUSH_BASE_PATH=/ npx vite build )

mkdir -p "$(dirname "$DIST_OUT")"
rm -rf "$DIST_OUT"
cp -R "$WEB_DIR/dist" "$DIST_OUT"

echo "✓ web build OK -> $DIST_OUT"
echo "  Serve with: scripts/deliver.sh $DIST_OUT <name> \"<why>\"  (generates OPEN_<NAME>.command bound to 127.0.0.1)"
