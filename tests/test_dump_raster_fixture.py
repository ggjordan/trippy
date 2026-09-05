"""The synthetic rasteriser fixtures the Rust parity tests read.

Module: tests.test_dump_raster_fixture
Invariants: these tests never *rewrite* the fixtures. They re-render from the
    same seed and compare, so a semantic change anywhere in `trippy.raster`
    fails here instead of silently invalidating
    `rust/crates/brush-pyramid`'s parity tests (which have no way to notice
    that the .npy files they trust have drifted). If a change is intended,
    re-run `tools/dump_raster_fixture.py` and review the diff.
    The compression check matters for the same reason: `brush_pyramid::npz`
    is a minimal reader that supports stored and deflate only, and the two
    files are deliberately written one of each so both branches are covered.
Related docs: tools/dump_raster_fixture.py (the writer);
    rust/crates/brush-pyramid/tests/parity_cpu.rs (the reader).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools.dump_raster_fixture import (
    FIXTURE_MODES,
    FIXTURE_NUM_CHANNELS,
    FIXTURE_NUM_LAYERS,
    FIXTURE_NUM_POINTS,
    FIXTURE_PIXEL_CENTERS,
    FIXTURE_SEED,
    build_scene,
    fixture_name,
    render_reference,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "synthetic"

# The whole tree must stay comfortably small: it is committed to a public repo
# and read by a Rust test that should run in milliseconds.
MAX_TOTAL_BYTES = 1_000_000

# Re-rendering from the same seed is deterministic, so this is a float32
# round-trip tolerance, not a numerical-agreement one.
TOL = 1e-6

ALL_FIXTURES = [(mode, pc) for mode in FIXTURE_MODES for pc in FIXTURE_PIXEL_CENTERS]


@pytest.fixture(scope="module")
def scene() -> dict:
    """The synthetic scene, rebuilt from the pinned seed."""
    return build_scene(np.random.default_rng(FIXTURE_SEED))


def _fixture_dir(mode: str, pixel_center: str) -> Path:
    path = FIXTURE_ROOT / fixture_name(mode, pixel_center)
    if not path.is_dir():
        pytest.fail(f"{path} missing; run tools/dump_raster_fixture.py")
    return path


def test_every_mode_and_pixel_centre_has_a_fixture() -> None:
    """All six (mode, pixel_center) directories exist and nothing else does."""
    found = sorted(p.name for p in FIXTURE_ROOT.glob("raster_fixture_*"))
    expected = sorted(fixture_name(mode, pc) for mode, pc in ALL_FIXTURES)
    assert found == expected


def test_the_whole_fixture_tree_stays_small() -> None:
    """Public repo: synthetic fixtures only, and under 1 MB in total."""
    total = sum(p.stat().st_size for p in FIXTURE_ROOT.rglob("*") if p.is_file())
    assert total < MAX_TOTAL_BYTES, f"fixtures grew to {total} bytes"


@pytest.mark.parametrize(("mode", "pixel_center"), ALL_FIXTURES)
def test_points_are_stored_and_expected_is_deflated(mode: str, pixel_center: str) -> None:
    """Both branches of the Rust npz reader stay covered by the fixtures."""
    directory = _fixture_dir(mode, pixel_center)
    with zipfile.ZipFile(directory / "points.npz") as archive:
        methods = {info.compress_type for info in archive.infolist()}
    assert methods == {zipfile.ZIP_STORED}, "points.npz must be written with np.savez"

    with zipfile.ZipFile(directory / "expected.npz") as archive:
        methods = {info.compress_type for info in archive.infolist()}
    assert methods == {zipfile.ZIP_DEFLATED}, "expected.npz must use np.savez_compressed"


@pytest.mark.parametrize(("mode", "pixel_center"), ALL_FIXTURES)
def test_stored_points_match_the_seeded_scene(scene: dict, mode: str, pixel_center: str) -> None:
    """The committed point set is exactly what the seed regenerates."""
    directory = _fixture_dir(mode, pixel_center)
    with np.load(directory / "points.npz") as data:
        for key, source in (("xyz", "xyz"), ("size", "size"), ("feat", "feat"), ("conf", "conf")):
            np.testing.assert_allclose(data[key], scene[source], rtol=0, atol=0)
        assert data["xyz"].shape == (FIXTURE_NUM_POINTS, 3)
        assert data["feat"].shape == (FIXTURE_NUM_POINTS, FIXTURE_NUM_CHANNELS)


@pytest.mark.parametrize(("mode", "pixel_center"), ALL_FIXTURES)
def test_expected_images_still_match_a_fresh_render(
    scene: dict, mode: str, pixel_center: str
) -> None:
    """Guard against silent drift in trippy.raster invalidating the Rust tests."""
    directory = _fixture_dir(mode, pixel_center)
    layers, aux = render_reference(scene, mode, pixel_center)
    meta = json.loads((directory / "meta.json").read_text())

    assert meta["num_fragments"] == int(aux["num_fragments"])
    assert meta["fragments_per_layer"] == [int(v) for v in aux["fragments_per_layer"]]

    with np.load(directory / "expected.npz") as data:
        for layer in range(FIXTURE_NUM_LAYERS):
            np.testing.assert_allclose(
                data[f"layer_{layer}"], layers[layer].numpy(), rtol=0, atol=TOL
            )
            np.testing.assert_allclose(
                data[f"t_final_{layer}"], aux["t_final"][layer].numpy(), rtol=0, atol=TOL
            )
            np.testing.assert_array_equal(
                data[f"n_used_{layer}"], aux["n_used"][layer].numpy().astype(np.int32)
            )


@pytest.mark.parametrize(("mode", "pixel_center"), ALL_FIXTURES)
def test_params_json_matches_the_raster_constants(mode: str, pixel_center: str) -> None:
    """The Rust side reads its PyramidParams from here; keep it honest."""
    from trippy.constants import (
        RASTER_ALPHA_MIN,
        RASTER_MAX_FRAGS,
        RASTER_T_CUTOFF,
        RASTER_ZNEAR,
    )

    params = json.loads((_fixture_dir(mode, pixel_center) / "params.json").read_text())
    assert params["mode"] == mode
    assert params["pixel_center"] == pixel_center
    assert params["pyramid_halving"] == "ceil"
    assert params["num_layers"] == FIXTURE_NUM_LAYERS
    assert params["num_channels"] == FIXTURE_NUM_CHANNELS
    assert params["max_frags"] == RASTER_MAX_FRAGS
    assert params["t_cutoff"] == pytest.approx(RASTER_T_CUTOFF)
    assert params["alpha_min"] == pytest.approx(RASTER_ALPHA_MIN)
    assert params["znear"] == pytest.approx(RASTER_ZNEAR)
    assert len(params["background"]) == FIXTURE_NUM_CHANNELS


@pytest.mark.parametrize(("mode", "pixel_center"), ALL_FIXTURES)
def test_both_compositing_stop_rules_are_exercised(mode: str, pixel_center: str) -> None:
    """A fixture that never hits max_frags or t_cutoff would pass with both
    `break`s in blend_fwd removed. The two deliberate clusters in
    `build_scene` exist to stop that."""
    from trippy.constants import RASTER_MAX_FRAGS, RASTER_T_CUTOFF

    directory = _fixture_dir(mode, pixel_center)
    capped = 0
    cut_off = 0
    with np.load(directory / "expected.npz") as data:
        for layer in range(FIXTURE_NUM_LAYERS):
            n_used = data[f"n_used_{layer}"]
            t_final = data[f"t_final_{layer}"]
            capped += int((n_used == RASTER_MAX_FRAGS).sum())
            cut_off += int(((t_final < RASTER_T_CUTOFF) & (n_used < RASTER_MAX_FRAGS)).sum())
    assert capped > 0, "no pixel reached the max_frags cap"
    assert cut_off > 0, "no pixel reached the t_cutoff early-out"
