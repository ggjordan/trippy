#!/bin/sh
# release.sh vX.Y.Z — write VERSION, move CHANGELOG Unreleased -> version, mirror the Rust
# workspace version if rust/Cargo.toml exists, tag, and create a GitHub Release.
# Usage: scripts/release.sh vX.Y.Z   (run on main after merging; requires gh auth)
# Invariants: VERSION file is the single source of truth for the version; rust/Cargo.toml's
#             [workspace.package] version is kept in sync when the Rust workspace exists;
#             only files that exist are staged (VERSION CHANGELOG.md rust/Cargo.toml rust/Cargo.lock).
set -e
cd "$(dirname "$0")/.."
V=${1:?usage: scripts/release.sh vX.Y.Z}; NUM=${V#v}
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "✗ release from main only"; exit 1; }
printf '%s\n' "$NUM" > VERSION
if [ -f rust/Cargo.toml ]; then
  sed -i '' "/\\[workspace.package\\]/,/^\\[/ s/^version = \".*\"/version = \"$NUM\"/" rust/Cargo.toml
  ( cd rust && cargo update --workspace --offline ) || true
fi
DATE=$(date +%Y-%m-%d)
python3 - "$V" "$DATE" <<'PY'
import sys,re;v,d=sys.argv[1:];p='CHANGELOG.md';s=open(p).read()
s=s.replace('## [Unreleased]', f'## [Unreleased]\n\n## [{v}] - {d}',1);open(p,'w').write(s)
PY
NOTES=$(awk "/^## \\[$V\\]/{f=1;next} /^## \\[/{f=0} f" CHANGELOG.md)
FILES=""
for f in VERSION CHANGELOG.md rust/Cargo.toml rust/Cargo.lock; do
  [ -f "$f" ] && FILES="$FILES $f"
done
git add $FILES
git commit -q -m "Release $V" -m "Reviewed-by: Orchestrator/release-script/n-a"
git tag -a "$V" -m "trippy $V"
scripts/push.sh origin main
git push -q origin "$V"
gh release create "$V" --title "trippy $V" --notes "$NOTES"
echo "✓ released $V"
