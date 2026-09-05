# ADR-0003: VERSION file as single source of truth for Python and Rust

Date: 2026-09-05 · Status: Accepted

## Context

The trippy project spans Python (research) and Rust (production viewers). Both need version numbers for:
- Release tags and GitHub Releases (vX.Y.Z).
- Build artifacts and diagnostics (build-NNNN tags).
- Dependency specifications and reproducibility.

Two approaches exist:

1. **Scattered versions**: Python reads `pyproject.toml`, Rust reads `Cargo.toml`. Each file holds its version independently.
2. **Single source**: one `VERSION` file at the repo root; both Python and Rust read it. Fewer sync points, one source of truth.

## Decision

Adopt **Option 2**: the root-level `VERSION` file is the single source of truth. Both `pyproject.toml` (Python) and `Cargo.toml` (Rust) dynamically read this file at build time.

- **FORMAT**: `VERSION` contains a single line: `vX.Y.Z` (e.g., `v0.1.0`). No additional metadata.
- **Python**: `pyproject.toml` uses dynamic versioning (setuptools backend) and reads `VERSION` at install time.
- **Rust**: Cargo build script (`build.rs`) reads `VERSION` and writes it to an env var or embedded constant.
- **Tagging**: `build-NNNN` tags mark every push (auto-incremented by `scripts/push.sh`). `vX.Y.Z` tags mark releases (explicit via `scripts/release.sh vX.Y.Z`).
- **Validation**: `tests/test_version_sync.py` asserts that `VERSION` content matches parsed version strings in both ecosystems at test time.

## Consequences

### Single point of maintenance

- Bump `VERSION`, tag release, both Python and Rust pick it up automatically.
- No risk of Python and Rust drifting to different versions.
- Release process is simpler: one edit, one commit, one tag.

### Build-time determination

- Python: `pip install .` reads the current `VERSION` and bakes it into the installed package.
- Rust: `cargo build` reads `VERSION` and embeds it (allows `trippy --version` to print the exact build tag).
- Test validation: `test_version_sync.py` runs on CI to catch any drift.

### Release process

1. Edit `VERSION` file (e.g., `v0.1.0`).
2. `scripts/release.sh vX.Y.Z` (same as step 1).
3. Tag: `git tag vX.Y.Z`.
4. Push: `scripts/push.sh` (also creates `build-NNNN` tag).
5. `gh release create vX.Y.Z --notes "..."`.

Both Python and Rust versions are now in sync.

## Alternatives considered

### Why not PEP 440 / Cargo semver directly?

- **No agreed format**: Python uses PEP 440; Rust uses semantic versioning. While both support `vX.Y.Z`, build metadata and pre-release tags differ.
- **Duplication**: two files to edit, two places to get out of sync.
- **CI complexity**: release CI must parse and validate both formats, increasing failure modes.

### Why not derive from git tags?

- **Late binding**: version is not known until after the build; `trippy --version` requires a git repo (fails in distributions).
- **Offline builds**: some Rust toolchains (cross-compilation, CI) may not have `.git` available.

## Maintenance

- `scripts/release.sh vX.Y.Z` edits `VERSION` and commits it in the same step.
- `tests/test_version_sync.py` is run by `test.sh`; it fails if `VERSION` and parsed versions diverge.
- `scripts/push.sh` refuses to push if tests don't pass (includes version sync).

## Related files

- `VERSION`: root-level file, one line, tracked in git.
- `pyproject.toml`: `dynamic = ["version"]` + read from VERSION at install time.
- `Cargo.toml` (when present): version is derived at build time via `build.rs`.
- `tests/test_version_sync.py`: validates agreement.
