#!/bin/bash
# supersplat_bootstrap.sh — clone + build SuperSplat 3.0 for local, self-hosted use.
# Usage: scripts/supersplat_bootstrap.sh
# Invariants (ADR-0008-supersplat.md Stage 1, item 1):
#   - Idempotent: safe to re-run; a matching clone at the pinned tag with dist/ already
#     built is a no-op except reprinting the dist path.
#   - Bash 3.2 safe, `set -u` clean: no array unpack, quoted expansions throughout.
#   - Clones OUTSIDE the repo, into $TRIPPY_OUTPUT/tools/supersplat -- output/ is
#     gitignored, so this never vendors SuperSplat's MIT code into trippy's own tree
#     (ADR-0008 Sec 3 licence rule: pin a tag, never vendor).
#   - Pins a TAG, not `main`, so a future upstream change cannot silently add a beacon
#     to a scene-loaded editor (ADR-0008 Sec 3 "operational guards").
#   - Public code download only: this script fetches PlayCanvas' own public repo over
#     git/npm. No local file (scene, ply, checkpoint) is read or sent by this script.
#   - Refuses on Node < 20.19 (SuperSplat's own engine requirement).
# Related docs: docs/decisions/ADR-0008-supersplat.md; scripts/open_supersplat.sh
#   (serves the dist/ this script builds); scripts/bootstrap.sh (the idempotent-script
#   style this follows).
set -eu
cd "$(dirname "$0")/.."

# The pinned SuperSplat release. v3.0.0 is the "rebuilt on WebGPU" release ADR-0008
# read and audited (Sec "Sources read", Sec 3 privacy verdict). Bump this deliberately,
# re-read the new tag's src/publish.ts and src/index.html for anything that changed the
# privacy verdict, and update the ADR before bumping.
SUPERSPLAT_PIN="v3.0.0"
SUPERSPLAT_REPO="https://github.com/playcanvas/supersplat"

# Exported env wins over .env (tests and worktrees set TRIPPY_OUTPUT explicitly).
_pre_env_out=${TRIPPY_OUTPUT:-}
if [ -f .env ]; then . ./.env; fi
TRIPPY_OUTPUT=${_pre_env_out:-${TRIPPY_OUTPUT:-}}
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}

CHECKOUT_DIR="$TRIPPY_OUTPUT/tools/supersplat"

# --- Node version gate -------------------------------------------------------------
command -v node >/dev/null 2>&1 || { echo "✗ node not found on PATH" >&2; exit 1; }
NODE_VERSION=$(node --version | sed 's/^v//')
NODE_MAJOR=$(printf '%s' "$NODE_VERSION" | cut -d. -f1)
NODE_MINOR=$(printf '%s' "$NODE_VERSION" | cut -d. -f2)
case "$NODE_MAJOR" in ''|*[!0-9]*) echo "✗ could not parse node --version output: $NODE_VERSION" >&2; exit 1 ;; esac
case "$NODE_MINOR" in ''|*[!0-9]*) NODE_MINOR=0 ;; esac
NODE_OK=0
if [ "$NODE_MAJOR" -gt 20 ]; then NODE_OK=1; fi
if [ "$NODE_MAJOR" -eq 20 ] && [ "$NODE_MINOR" -ge 19 ]; then NODE_OK=1; fi
if [ "$NODE_OK" -ne 1 ]; then
  echo "✗ node $NODE_VERSION is older than the required 20.19 (SuperSplat's own engine requirement)" >&2
  exit 1
fi
echo "node $NODE_VERSION OK (>= 20.19)"

command -v npm >/dev/null 2>&1 || { echo "✗ npm not found on PATH" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "✗ git not found on PATH" >&2; exit 1; }

mkdir -p "$(dirname "$CHECKOUT_DIR")"

# --- Clone or update, pinned to the tag ---------------------------------------------
if [ -d "$CHECKOUT_DIR/.git" ]; then
  echo "▶ existing checkout at $CHECKOUT_DIR — verifying pin"
  CURRENT_TAG=$(git -C "$CHECKOUT_DIR" describe --tags --exact-match 2>/dev/null || true)
  if [ "$CURRENT_TAG" != "$SUPERSPLAT_PIN" ]; then
    echo "▶ fetching tags and checking out $SUPERSPLAT_PIN (was: ${CURRENT_TAG:-unknown})"
    git -C "$CHECKOUT_DIR" fetch --tags origin
    git -C "$CHECKOUT_DIR" checkout "$SUPERSPLAT_PIN"
  else
    echo "  already at $SUPERSPLAT_PIN"
  fi
else
  if [ -e "$CHECKOUT_DIR" ]; then
    echo "✗ $CHECKOUT_DIR exists but is not a git checkout — remove it and re-run" >&2
    exit 1
  fi
  echo "▶ cloning $SUPERSPLAT_REPO @ $SUPERSPLAT_PIN into $CHECKOUT_DIR"
  git clone --branch "$SUPERSPLAT_PIN" --depth 1 "$SUPERSPLAT_REPO" "$CHECKOUT_DIR"
fi

ACTUAL_TAG=$(git -C "$CHECKOUT_DIR" describe --tags --exact-match 2>/dev/null || true)
if [ "$ACTUAL_TAG" != "$SUPERSPLAT_PIN" ]; then
  echo "✗ checkout at $CHECKOUT_DIR is not pinned to $SUPERSPLAT_PIN (describe: ${ACTUAL_TAG:-none})" >&2
  exit 1
fi

# --- Install + build (npm ci pulls packages; this is the only network step) --------
DIST_DIR="$CHECKOUT_DIR/dist"
NEED_BUILD=1
if [ -f "$DIST_DIR/index.html" ] && [ -d "$CHECKOUT_DIR/node_modules" ]; then
  echo "  dist/ and node_modules/ already present — skipping npm ci/build (idempotent re-run)"
  NEED_BUILD=0
fi

if [ "$NEED_BUILD" -eq 1 ]; then
  echo "▶ npm ci (in $CHECKOUT_DIR — a LOCAL install, not global; AGENTS.md Sec 6)"
  ( cd "$CHECKOUT_DIR" && npm ci )
  echo "▶ npm run build"
  ( cd "$CHECKOUT_DIR" && npm run build )
fi

[ -f "$DIST_DIR/index.html" ] || { echo "✗ build finished but $DIST_DIR/index.html is missing" >&2; exit 1; }

echo "SuperSplat $SUPERSPLAT_PIN built."
echo "dist: $DIST_DIR"
echo
echo "Serve it with:"
echo "  scripts/open_supersplat.sh"
