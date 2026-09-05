"""Tests for `trippy.hybrid.gaussian_input`: block building, pooling, attach, dropout.

Module: tests.test_hybrid_a_inputs
Invariants under test: the Gaussian block's channel order and contents
    (`mask_by_alpha`, depth normalisation); the measured depth scale;
    `attach`'s channel bookkeeping in both modes and its zero-fill for a
    missing/dropped frame; and that dropping zeroes **exactly** the Gaussian
    channels, leaving every TRIPS channel bit-identical.
All fixtures are synthetic (tests/test_hybrid_a_helpers.py); no MPS, no
`~/Splats`, no photo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_hybrid_a_helpers import FAKE_DEPTH_MAX, FAKE_DEPTH_MIN, write_fake_render

from trippy.hybrid.config_a import HybridConfig
from trippy.hybrid.gaussian_input import (
    GaussianInputs,
    block_from_arrays,
    measure_depth_scale,
    resample_to,
    stem_of,
)

H, W = 12, 16


def _arrays(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "rgb": rng.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32),
        "alpha": rng.uniform(0.0, 1.0, (H, W)).astype(np.float32),
        "depth": rng.uniform(1.0, 9.0, (H, W)).astype(np.float32),
    }


def _inputs(tmp_path: Path, **overrides) -> tuple[GaussianInputs, list[str]]:
    names = ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg"]
    renders = tmp_path / "renders"
    for i, name in enumerate(names):
        write_fake_render(renders, stem_of(name), height=H, width=W, seed=i)
    kwargs = {"enabled": True, "renders_dir": str(renders), "depth_scale": 6.0}
    kwargs.update(overrides)
    return GaussianInputs.build(HybridConfig(**kwargs), names), names


# --- block building ---


def test_block_channel_order_and_contents() -> None:
    arrays = _arrays()
    block = block_from_arrays(arrays, ["rgb", "alpha", "depth"], depth_scale=2.0, mask_by_alpha=False)
    assert block.shape == (5, H, W)
    assert torch.allclose(block[0:3], torch.from_numpy(arrays["rgb"]).permute(2, 0, 1))
    assert torch.allclose(block[3], torch.from_numpy(arrays["alpha"]))
    assert torch.allclose(block[4], torch.from_numpy(arrays["depth"]) / 2.0)


def test_mask_by_alpha_multiplies_rgb_only() -> None:
    arrays = _arrays()
    plain = block_from_arrays(arrays, ["rgb", "alpha"], depth_scale=1.0, mask_by_alpha=False)
    masked = block_from_arrays(arrays, ["rgb", "alpha"], depth_scale=1.0, mask_by_alpha=True)
    alpha = torch.from_numpy(arrays["alpha"])
    assert torch.allclose(masked[0:3], plain[0:3] * alpha)
    assert torch.allclose(masked[3], plain[3])


def test_block_subset_drops_the_unrequested_groups() -> None:
    block = block_from_arrays(_arrays(), ["rgb"], depth_scale=1.0, mask_by_alpha=False)
    assert block.shape == (3, H, W)


def test_block_rejects_a_non_positive_depth_scale() -> None:
    with pytest.raises(ValueError, match="depth_scale must be positive"):
        block_from_arrays(_arrays(), ["depth"], depth_scale=0.0, mask_by_alpha=False)


# --- depth scale ---


def test_measure_depth_scale_lands_inside_the_rendered_depth_range(tmp_path: Path) -> None:
    renders = tmp_path / "renders"
    names = [f"IMG_{i}.jpg" for i in range(4)]
    for i, name in enumerate(names):
        write_fake_render(renders, stem_of(name), height=H, width=W, seed=i)
    scale = measure_depth_scale(renders, names)
    assert FAKE_DEPTH_MIN < scale < FAKE_DEPTH_MAX


def test_measure_depth_scale_falls_back_when_no_render_exists(tmp_path: Path) -> None:
    assert measure_depth_scale(tmp_path, ["IMG_0.jpg"], fallback=17.0) == 17.0


def test_build_resolves_depth_scale_in_place(tmp_path: Path) -> None:
    renders = tmp_path / "renders"
    write_fake_render(renders, "IMG_0", height=H, width=W)
    cfg = HybridConfig(enabled=True, renders_dir=str(renders))
    assert cfg.depth_scale is None
    GaussianInputs.build(cfg, ["IMG_0.jpg"])
    assert cfg.depth_scale is not None and cfg.depth_scale > 0.0


def test_build_skips_the_measurement_when_depth_is_not_requested(tmp_path: Path) -> None:
    cfg = HybridConfig(enabled=True, renders_dir=str(tmp_path), channels=["rgb", "alpha"])
    GaussianInputs.build(cfg, [])
    assert cfg.depth_scale is None


# --- frame access ---


def test_frame_is_resampled_to_the_requested_size(tmp_path: Path) -> None:
    inputs, names = _inputs(tmp_path)
    block = inputs.frame(names[0], (H // 2, W // 2))
    assert block is not None
    assert block.shape == (5, H // 2, W // 2)


def test_missing_render_is_none_by_default_and_raises_when_asked(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    assert inputs.has("IMG_0.jpg") is True
    assert inputs.has("NOPE.jpg") is False
    assert inputs.frame("NOPE.jpg", (H, W)) is None

    strict, _names = _inputs(tmp_path, missing="error")
    with pytest.raises(FileNotFoundError, match="no render triple"):
        strict.frame("NOPE.jpg", (H, W))


def test_available_names_reports_render_coverage(tmp_path: Path) -> None:
    inputs, names = _inputs(tmp_path)
    assert inputs.available_names([*names, "NOPE.jpg"]) == names


# --- resampling ---


def test_resample_to_shrinks_by_area_average() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    out = resample_to(x, (2, 2))
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4.0)


def test_resample_to_is_identity_at_the_same_size() -> None:
    x = torch.rand(2, 5, 7)
    assert resample_to(x, (5, 7)) is x


# --- attach: channel bookkeeping ---


def _pyramid(channels: int = 4, levels: int = 3, base: int = 16) -> list[torch.Tensor]:
    return [torch.rand(1, channels, base >> i, base >> i) for i in range(levels)]


def test_attach_all_levels_widens_every_level_with_pooled_content(tmp_path: Path) -> None:
    inputs, names = _inputs(tmp_path, mode="all_levels")
    levels = _pyramid()
    block = torch.ones(5, 16, 16) * 0.25
    out = inputs.attach(levels, block)

    assert len(out) == len(levels)
    for level, (before, after) in enumerate(zip(levels, out, strict=True)):
        assert after.shape[1] == before.shape[1] + 5
        assert after.shape[-2:] == before.shape[-2:]
        assert torch.allclose(after[:, :4], before), f"TRIPS channels changed at level {level}"
        assert torch.allclose(after[:, 4:], torch.full_like(after[:, 4:], 0.25))
    assert inputs.has(names[0])


def test_attach_concat_level0_zeroes_the_coarse_levels(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path, mode="concat_level0")
    levels = _pyramid()
    block = torch.ones(5, 16, 16)
    out = inputs.attach(levels, block)

    assert torch.allclose(out[0][:, 4:], torch.ones_like(out[0][:, 4:]))
    for coarse in out[1:]:
        assert torch.count_nonzero(coarse[:, 4:]) == 0


def test_attach_with_no_block_zero_fills_every_level(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path)
    levels = _pyramid()
    out = inputs.attach(levels, None)
    for before, after in zip(levels, out, strict=True):
        assert after.shape[1] == before.shape[1] + 5
        assert torch.allclose(after[:, :4], before)
        assert torch.count_nonzero(after[:, 4:]) == 0


def test_dropping_zeroes_exactly_the_gaussian_channels(tmp_path: Path) -> None:
    """The ablation's contract: TRIPS channels untouched, Gaussian channels all zero."""
    inputs, _ = _inputs(tmp_path)
    levels = _pyramid()
    kept = inputs.attach(levels, torch.rand(5, 16, 16) + 0.5)
    dropped = inputs.attach(levels, None)
    for keep, drop in zip(kept, dropped, strict=True):
        assert torch.equal(keep[:, :4], drop[:, :4])
        assert torch.count_nonzero(keep[:, 4:]) > 0
        assert torch.count_nonzero(drop[:, 4:]) == 0


# --- dropout ---


def test_should_drop_honours_the_probability_bounds(tmp_path: Path) -> None:
    never, _ = _inputs(tmp_path, dropout_gaussian_p=0.0)
    always, _ = _inputs(tmp_path, dropout_gaussian_p=1.0)
    generator = torch.Generator().manual_seed(0)
    assert not any(never.should_drop(generator) for _ in range(20))
    assert all(always.should_drop(generator) for _ in range(20))


def test_should_drop_is_roughly_the_configured_fraction(tmp_path: Path) -> None:
    inputs, _ = _inputs(tmp_path, dropout_gaussian_p=0.25)
    generator = torch.Generator().manual_seed(0)
    drops = sum(inputs.should_drop(generator) for _ in range(2000))
    assert 0.20 < drops / 2000 < 0.30
