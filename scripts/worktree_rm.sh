#!/bin/bash
# worktree_rm.sh — remove a subagent worktree WITHOUT losing run artefacts.
# Usage: scripts/worktree_rm.sh <name>     (removes .worktrees/<name> and branch refs are left alone)
# Invariants: if the worktree contains a non-empty output/ (an agent ran without TRIPPY_OUTPUT), it
#   is moved to $TRIPPY_OUTPUT/_rescued/<name>-<timestamp>/ before removal. Learned 2026-09-06 when
#   EXP-0005's 53 minutes of Gaussian renders vanished with `git worktree remove --force`.
set -eu
cd "$(dirname "$0")/.."
name=${1:?usage: worktree_rm.sh <name>}
wt=".worktrees/$name"
[ -d "$wt" ] || { echo "no such worktree: $wt"; exit 2; }
TRIPPY_OUTPUT=${TRIPPY_OUTPUT:-$PWD/output}
if [ -d "$wt/output" ] && [ -n "$(find "$wt/output" -type f -print -quit 2>/dev/null)" ]; then
  dest="$TRIPPY_OUTPUT/_rescued/$name-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$(dirname "$dest")"
  mv "$wt/output" "$dest"
  echo "rescued $wt/output -> $dest"
fi
git worktree remove --force "$wt"
echo "removed $wt"
