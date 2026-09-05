"""VERSION file, trippy.__version__, and (if present) Cargo workspace version must agree.

Module: tests.test_version_sync
Invariants under test: there is exactly one place a human/agent edits the
    version (VERSION); everything else (setuptools dynamic version,
    __version__, a future Rust workspace) derives from or matches it.
Related docs: docs/SPEC.md "Scripts and hooks" item "Version truth";
    scripts/release.sh (writes VERSION, mirrors Cargo workspace version).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import trippy

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_file_matches_package_version() -> None:
    version_file = REPO_ROOT / "VERSION"
    if not version_file.exists():
        pytest.skip("VERSION file not present yet")
    assert version_file.read_text().strip() == trippy.__version__


def test_cargo_workspace_version_matches_if_present() -> None:
    cargo_toml = REPO_ROOT / "rust" / "Cargo.toml"
    if not cargo_toml.exists():
        pytest.skip("rust/Cargo.toml not present yet")

    text = cargo_toml.read_text()
    match = re.search(r"\[workspace\.package\][^\[]*?version\s*=\s*\"([^\"]+)\"", text, re.DOTALL)
    assert match, "rust/Cargo.toml has no [workspace.package] version"
    assert match.group(1) == trippy.__version__
