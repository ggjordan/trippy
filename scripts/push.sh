#!/bin/sh
# push.sh — the ONLY way to push. Computes and pushes the next sequential build-NNNN tag on HEAD.
# Usage: scripts/push.sh [remote] [branch]   (defaults: origin, current branch)
# Invariants: the pre-push hook runs scripts/test.sh (and the file guard) before any push lands;
#             tags are monotonic build-0001, build-0002, ... and are never reused or force-moved.
set -e
cd "$(dirname "$0")/.."
REMOTE=${1:-origin}; BRANCH=${2:-$(git rev-parse --abbrev-ref HEAD)}
git fetch -q "$REMOTE" --tags || true
LAST=$(git tag -l 'build-*' | sed 's/build-//' | sort -n | tail -1); LAST=${LAST:-0}
NEXT=$(printf 'build-%04d' $((10#$LAST + 1)))
git push "$REMOTE" "$BRANCH"            # pre-push hook runs here (trailer + scripts/test.sh)
git tag -a "$NEXT" -m "Build $NEXT on $BRANCH ($(git rev-parse --short HEAD))"
git push -q "$REMOTE" "$NEXT"
echo "✓ pushed $BRANCH and tagged $NEXT"
