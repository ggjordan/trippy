# ADR-0005: Brush fork layout — submodule to a public fork, not a subtree

Date: 2026-09-06 · Status: Accepted

## Context

v0.4.0 ports the locked TRIPS design (v0.3.0) from Python/PyTorch to Rust/Burn/CubeCL
for the production viewers (`apps/brush-app` native + web). That means bringing
ArthurBrussee's Brush (Apache-2.0, https://github.com/ArthurBrussee/brush) into
trippy at `rust/brush-trips`, at the same upstream commit Splats' working fork
(`~/Splats/tools/brush-final`, not itself a git repo — it lives inside the
family-splats repo) started from: `8b7f5c6c0638892204b540d9aced219f62fc2192`
(2026-08-17), plus Splats' three patches (`~/Splats/tools/patches/*.patch`:
`brush-robust`, `brush-appearance`, `brush-surface`).

Two ways to vendor an upstream Rust project inside a public repo:

1. **`git subtree`**: squash (or full-history) merge of upstream into
   `rust/brush-trips/`, committed directly into trippy's own history. Self-contained
   — a plain `git clone` of trippy gets everything, no second remote to manage. But
   trippy's history balloons by Brush's full commit graph (or loses it, if squashed),
   and there's no natural place to push fixes back upstream or track patches as
   reviewable commits.
2. **`git submodule`** pointing at a new fork (`ggjordan/brush` on GitHub, public).
   Needs `gh repo fork` (or equivalent) to exist, and a fresh `git clone --recursive`
   (or `git submodule update --init`) to populate it. Splats' three patches become
   ordinary commits on a branch in the fork, which is easier to read, diff, and
   eventually rebase onto a newer upstream commit than replaying `.patch` files by
   hand each time.

## Decision

**Submodule to a public fork.** `gh` was already authenticated as `ggjordan` with
`repo` scope, and `gh repo fork ArthurBrussee/brush --clone=false` is a single,
already-available command — the condition for preferring the submodule route over
the subtree fallback.

- Fork: **https://github.com/ggjordan/brush** (public, Apache-2.0, forked from
  ArthurBrussee/brush).
- `rust/brush-trips` is a git submodule (`.gitmodules`) pointing at that fork,
  `branch = trippy-fork`.
- The fork's `trippy-fork` branch is `8b7f5c6c06...` (upstream, unmodified) plus
  three merge commits, one per Splats patch, each starting from its own
  `patch-<name>` branch off the same base commit and merged in with `git merge`
  (not `git apply`/`git am` in sequence) so overlapping hunks in shared files
  (`brush-train/config.rs`, `train.rs`, `lib.rs`; `brush-dataset/scene.rs`,
  `scene_loader.rs`) go through a real three-way merge instead of fuzzy patch
  offsets. All branches (`upstream-base`, `patch-robust`, `patch-appearance`,
  `patch-surface`, `trippy-fork`) are pushed to the fork for traceability. See
  `rust/README.md` for which patch commit did what and any conflicts resolved by
  hand.

## Reachability

A fresh `git clone --recursive` (or `git clone` followed by
`git submodule update --init`) of trippy resolves `rust/brush-trips` to the pinned
commit on `https://github.com/ggjordan/brush.git`, which is public and requires no
credentials to read. `git submodule status` is clean (no local modifications inside
the submodule; all patch work is committed and pushed to the fork).

## Consequences

### Kept separate from trippy's own Rust workspace

`rust/brush-trips` is its own Cargo workspace (upstream Brush's, version `1.0.0`,
many crates). trippy's own placeholder crates (`brush-pyramid`, `brush-unet`) live
in a **separate, thin virtual workspace** at `rust/Cargo.toml` (members
`crates/brush-pyramid`, `crates/brush-unet` — physically outside
`rust/brush-trips/`, not nested under it). Two independent Cargo workspaces cannot
share a member crate, and nesting them under the submodule's `crates/` (as
`rust/README.md`'s forward-looking v0.4.0 structure diagram shows for when the real
integration lands) would force that conflict today for no benefit. `scripts/build.sh`
/ `scripts/test.sh` only ever run `cargo check`/`cargo test -p brush-pyramid -p
brush-unet` inside `rust/`, so every push stays fast; the full Brush build only runs
manually via `scripts/cpu_heavy.sh` (see rust/README.md).

### Updating the fork

Rebasing onto a newer upstream commit, or adding a fourth patch, means: fetch
`ArthurBrussee/brush` into a clone of `ggjordan/brush`, rebase or re-merge
`trippy-fork` on top, push, then bump the submodule's pinned commit in trippy with
an ordinary `git -C rust/brush-trips checkout <sha>` + `git add rust/brush-trips`.
No `.patch` file replay needed going forward — the patches are now first-class
commits in the fork's history.

### Licensing

The fork (and therefore `rust/brush-trips`) is Apache-2.0, inherited from upstream
Brush; trippy's own thin `rust/` workspace crates are MIT per ADR-0004. This mirrors
`docs/UPSTREAM.md`'s existing statement of the split and needs no change: a `NOTICE`
file attributing ArthurBrussee's work still ships when the Brush fork is distributed
as part of trippy.

## Related

- `docs/UPSTREAM.md` — commit pins and patch provenance.
- `rust/README.md` — build instructions, patch-by-patch changelog, conflict notes.
- `.gitmodules`.
- ADR-0004 (public repo + licensing split).
