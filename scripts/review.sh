#!/bin/sh
# review.sh — Reviewer helper. Shows the working-tree diff (or a range) and the review checklist
# from AGENTS.md. With --trailer "<role>/<model>/<effort>" it commits staged changes with the trailer.
# Usage: scripts/review.sh [range]              show diff + checklist
#        scripts/review.sh --trailer X -m "msg" commit staged with Reviewed-by trailer
# Invariants: the checklist extraction is numbering-independent (starts at "### Review checklist",
#             stops at the next "## " heading), so AGENTS.md section renumbering never breaks it.
set -e
cd "$(dirname "$0")/.."
if [ "$1" = "--trailer" ]; then
  T=$2; shift 2; git commit "$@" --trailer "Reviewed-by: $T"; exit $?
fi
RANGE=${1:-}
echo "================ DIFF ================"
if [ -n "$RANGE" ]; then git diff --stat "$RANGE"; git diff "$RANGE"; else git status --short; git diff --stat HEAD; git diff HEAD; fi
echo "============= CHECKLIST =============="
awk '/^### Review checklist/{f=1;next} /^## /{f=0} f' AGENTS.md
