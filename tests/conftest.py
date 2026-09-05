"""Shared pytest fixtures.

Module: tests.conftest
Invariants: never reads/copies real scene data into the repo; only points
    at ~/Splats read-only, and skips cleanly when absent (this repo must
    run green on a machine without ~/Splats too).
Related docs: /tmp/trippy-plan.md decision D12 (public repo, no photos/
    plys/checkpoints from Jordan's scenes ever committed; test fixtures
    synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SPLATS_SCENE = Path("/Users/nzbirdranch/Splats/scenes/karekare/kk-coherent/sparse_txt")


@pytest.fixture
def splats_scene() -> Path:
    """Path to a real COLMAP sparse-txt model outside the repo, if present.

    Skips the test cleanly when ~/Splats (or this scene) isn't on the
    machine, so CI/other machines without ~/Splats stay green.
    """
    if not _SPLATS_SCENE.exists():
        pytest.skip(f"real scene not available at {_SPLATS_SCENE}")
    return _SPLATS_SCENE
