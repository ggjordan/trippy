"""Crop equivalence: the Gaussian render is cropped/zoomed exactly like the photo.

Module: tests.test_hybrid_a_crop
Invariants under test: `GaussianInputs.crop_frame` and the photo's own
    `trippy.scene.dataset.crop` call, given the same `(size, zoom, center)`,
    produce (a) a bit-identical adjusted `K`, (b) the same validity/overshoot
    footprint, and (c) windows that agree pixel-for-pixel with a hand-written
    gather. This is design A's single most dangerous failure mode -- a render
    misaligned by even a pixel would train the network on a lie -- so it is
    tested against the *photo path itself*, not against a re-derivation.
    Also covers the render-resolution mismatch case (a w1008 render set reused
    by a narrower training run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_hybrid_a_helpers import write_fake_render

from trippy.hybrid.config_a import HybridConfig
from trippy.hybrid.gaussian_input import GaussianInputs
from trippy.scene.dataset import crop as dataset_crop

H, W = 24, 32
K_BASE = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
NAME = "IMG_0.jpg"

# (size, zoom, center) triples: no zoom, zoomed in, zoomed out, and a window that
# overshoots the frame on both axes (the padding/mask case).
CROP_CASES = [
    (8, 1.0, (16.0, 12.0)),
    (8, 2.0, (10.0, 7.0)),
    (8, 0.5, (16.0, 12.0)),
    (8, 1.0, (1.0, 1.0)),
    (16, 1.5, (20.0, 18.0)),
]


def _item(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rgb = torch.from_numpy(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    return {"rgb": rgb, "K": K_BASE.clone()}


def _inputs(tmp_path: Path, height: int = H, width: int = W, **overrides) -> GaussianInputs:
    renders = tmp_path / "renders"
    write_fake_render(renders, Path(NAME).stem, height=height, width=width, seed=3)
    kwargs = {
        "enabled": True,
        "renders_dir": str(renders),
        "depth_scale": 5.0,
        "mask_by_alpha": False,
        "dropout_gaussian_p": 0.0,
    }
    kwargs.update(overrides)
    return GaussianInputs.build(HybridConfig(**kwargs), [NAME])


@pytest.mark.parametrize(("size", "zoom", "center"), CROP_CASES)
def test_render_crop_uses_a_bit_identical_k_adjust(
    tmp_path: Path, size: int, zoom: float, center: tuple[float, float]
) -> None:
    inputs = _inputs(tmp_path)
    item = _item()
    photo = dataset_crop(item, size=size, zoom=zoom, center=center)

    block = inputs.frame(NAME, (H, W))
    assert block is not None
    render = dataset_crop(
        {"rgb": block.permute(1, 2, 0).contiguous(), "K": item["K"]},
        size=size,
        zoom=zoom,
        center=center,
    )
    assert torch.equal(render["K"], photo["K"]), "render K-adjust diverged from the photo's"
    assert torch.equal(render["mask"], photo["mask"]), "render/photo validity masks differ"

    got = inputs.crop_frame(NAME, item, size=size, zoom=zoom, center=center)
    assert got is not None
    assert torch.equal(got, render["rgb"].permute(2, 0, 1))


@pytest.mark.parametrize(("size", "zoom", "center"), CROP_CASES)
def test_render_crop_matches_a_hand_written_gather(
    tmp_path: Path, size: int, zoom: float, center: tuple[float, float]
) -> None:
    """Independent re-derivation of the window (docs/GEOMETRY.md pixel-centre convention)."""
    inputs = _inputs(tmp_path)
    item = _item()
    block = inputs.frame(NAME, (H, W))
    assert block is not None

    window = size / zoom
    x0 = center[0] - window / 2.0
    y0 = center[1] - window / 2.0
    out_idx = np.arange(size) + 0.5
    src_col = np.floor(x0 + out_idx / size * window).astype(np.int64)
    src_row = np.floor(y0 + out_idx / size * window).astype(np.int64)
    valid = ((src_row >= 0) & (src_row < H))[:, None] & ((src_col >= 0) & (src_col < W))[None, :]
    gathered = block[:, np.clip(src_row, 0, H - 1), :][:, :, np.clip(src_col, 0, W - 1)]
    expected = gathered * torch.from_numpy(valid).to(gathered.dtype)

    got = inputs.crop_frame(NAME, item, size=size, zoom=zoom, center=center)
    assert got is not None
    assert torch.allclose(got, expected)


def test_overshoot_zeroes_the_render_exactly_where_the_photo_is_padded(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    item = _item()
    size, zoom, center = 16, 1.0, (2.0, 2.0)
    photo = dataset_crop(item, size=size, zoom=zoom, center=center)
    got = inputs.crop_frame(NAME, item, size=size, zoom=zoom, center=center)

    padded = photo["mask"] == 0.0
    assert bool(padded.any()), "this case is meant to overshoot the frame"
    assert got is not None
    assert torch.count_nonzero(got[:, padded]) == 0


def test_crop_of_a_larger_render_set_still_matches_the_photo_grid(tmp_path: Path) -> None:
    """A w1008 render set reused by a narrower run: resampled first, then cropped identically."""
    inputs = _inputs(tmp_path, height=H * 2, width=W * 2)
    item = _item()
    size, zoom, center = 8, 1.0, (16.0, 12.0)
    got = inputs.crop_frame(NAME, item, size=size, zoom=zoom, center=center)
    assert got is not None
    assert got.shape == (5, size, size)

    block = inputs.frame(NAME, (H, W))
    assert block is not None and block.shape == (5, H, W)
    expected = dataset_crop(
        {"rgb": block.permute(1, 2, 0).contiguous(), "K": item["K"]},
        size=size,
        zoom=zoom,
        center=center,
    )
    assert torch.equal(got, expected["rgb"].permute(2, 0, 1))
    assert torch.equal(expected["K"], dataset_crop(item, size=size, zoom=zoom, center=center)["K"])


def test_crop_of_a_missing_render_is_none(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    assert inputs.crop_frame("NOPE.jpg", _item(), size=8, zoom=1.0, center=(16.0, 12.0)) is None
