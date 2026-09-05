#!/bin/bash
# web_build.sh -- build a web viewer to a static dist dir servable on 127.0.0.1.
#
# TWO targets:
#   (default)  the STOCK Brush web demo, from rust/brush-trips/apps/brush-app/web via wasm-pack +
#              vite. This is the v0.5.0 groundwork build (docs/WEB_VIEWER.md "Build").
#   --trips    trippy's OWN web viewer: rust/crates/trips-web (wasm-pack --target web, no vite, no
#              npm install) plus web/index.html + web/trips.js plus a copy of a TRIPS bundle. This
#              is the real v0.5.0 deliverable -- the same brush-pyramid rasteriser and brush-unet
#              decoder the native trips-viewer runs, compiled to wasm32 and driving WebGPU.
#
# Usage: scripts/web_build.sh [--trips] [--dev] [--check] [--bundle DIR] [--out DIR]
#   --trips      build trippy's viewer instead of the stock Brush demo
#   --dev        wasm-pack --dev (fast, unoptimized, dwarf debug info) instead of --release
#   --check      dry run: verify the toolchain and print the plan; build nothing. For tests that
#                want to confirm the toolchain without paying for a full wasm-pack build.
#   --bundle DIR --trips only: the TRIPS bundle directory to copy in.
#                Default $TRIPPY_OUTPUT/brush/horse_bundle (the PUBLIC horse scene).
#   --out DIR    override the output directory.
# Invariants:
#   - --trips uses `wasm-pack --target web`, which emits a plain ES module. There is deliberately
#     no bundler: index.html loads ./pkg/trips_web.js with <script type="module">, so the build has
#     no npm install step of its own and the dist dir is exactly what a static file server needs.
#   - --trips copies the bundle's THREE files (bundle.json, points.npz, weights.safetensors) into
#     dist/bundle/. ~80 MB, which the browser fetches over loopback. The default bundle is the
#     PUBLIC horse scene; pointing --bundle at anything derived from Jordan's private scenes would
#     put a render of them in a delivered artifact, so the default is the one to keep.
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

MODE="release"
CHECK=0
TRIPS=0
BUNDLE=""
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) MODE="dev"; shift ;;
    --check) CHECK=1; shift ;;
    --trips) TRIPS=1; shift ;;
    --bundle) BUNDLE=${2:?--bundle needs a directory}; shift 2 ;;
    --out) OUT=${2:?--out needs a directory}; shift 2 ;;
    *) echo "✗ unknown argument: $1 (expected --trips, --dev, --check, --bundle DIR, --out DIR)" >&2; exit 2 ;;
  esac
done

if [ "$TRIPS" -eq 1 ]; then
  DIST_OUT=${OUT:-$TRIPPY_OUTPUT/web/trips-dist}
  BUNDLE=${BUNDLE:-$TRIPPY_OUTPUT/brush/horse_bundle}
else
  DIST_OUT=${OUT:-$TRIPPY_OUTPUT/web/brush-dist}
fi

# The submodule is needed for BOTH targets: the stock build lives in it, and trips-web reaches into
# it by path for brush-cube / brush-sort / brush-prefix-sum (ADR-0005).
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

if [ "$TRIPS" -eq 1 ]; then
  [ -d "$BUNDLE" ] || {
    echo "✗ bundle directory not found: $BUNDLE" >&2
    echo "  Export one with: .venv/bin/python -m trippy export-bundle ... (docs/USER_GUIDE.md)" >&2
    exit 2
  }
  [ -f "$BUNDLE/bundle.json" ] || {
    echo "✗ $BUNDLE has no bundle.json -- that is not a TRIPS bundle directory" >&2
    exit 2
  }

  if [ "$CHECK" -eq 1 ]; then
    echo "✓ toolchain OK: npm=$(command -v npm), wasm-pack=$(command -v wasm-pack), wasm32-unknown-unknown installed"
    echo "  target: trips-web (trippy's own viewer; no vite, no npm install)"
    echo "  would run: wasm-pack build rust/crates/trips-web --$MODE --target web --out-dir $DIST_OUT/pkg"
    echo "  would copy web/index.html + web/trips.js -> $DIST_OUT"
    echo "  would copy $BUNDLE -> $DIST_OUT/bundle"
    exit 0
  fi

  echo "▶ wasm-pack build ($MODE) rust/crates/trips-web --target web"
  rm -rf "$DIST_OUT"
  mkdir -p "$DIST_OUT"
  # --target web emits an ES module usable straight from <script type="module">; no bundler.
  # The manifest lives in trippy's own thin workspace, so nothing in the submodule is touched.
  ( cd rust && wasm-pack build crates/trips-web "--$MODE" --target web --out-dir "$DIST_OUT/pkg" )

  echo "▶ copy the page"
  cp web/index.html web/trips.js "$DIST_OUT/"

  echo "▶ copy the bundle ($BUNDLE -> $DIST_OUT/bundle)"
  mkdir -p "$DIST_OUT/bundle"
  # Only the three files the manifest schema defines; never a whole directory of unknowns.
  cp "$BUNDLE/bundle.json" "$DIST_OUT/bundle/"
  MANIFEST_POINTS=$(sed -n 's/.*"points"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$BUNDLE/bundle.json" | head -1)
  MANIFEST_WEIGHTS=$(sed -n 's/.*"weights"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$BUNDLE/bundle.json" | head -1)
  [ -n "$MANIFEST_POINTS" ] && [ -n "$MANIFEST_WEIGHTS" ] || {
    echo "✗ could not read \"points\"/\"weights\" out of $BUNDLE/bundle.json" >&2
    exit 2
  }
  cp "$BUNDLE/$MANIFEST_POINTS" "$DIST_OUT/bundle/"
  cp "$BUNDLE/$MANIFEST_WEIGHTS" "$DIST_OUT/bundle/"

  echo "✓ trips web build OK -> $DIST_OUT"
  du -sh "$DIST_OUT" 2>/dev/null || true
  echo "  Serve with: scripts/deliver.sh $DIST_OUT <name> \"<why>\"  (generates OPEN_<NAME>.command bound to 127.0.0.1)"
  exit 0
fi

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
