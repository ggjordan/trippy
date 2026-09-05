"""Tests for trippy.hybrid.dataset_c: pyramid construction and render<->photo pairing.

Module: tests.test_hybrid_c_dataset
Invariants under test: `build_pyramid` produces `num_layers` tensors, finest first, each half
    the previous level's spatial size (floor division for odd sizes); `paired_names` pairs by
    filename stem and silently excludes photos with no matching render triple; `crop_pair` /
    `sample_crop_origin` never produce an out-of-bounds window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trippy.hybrid import dataset_c


def test_build_pyramid_shapes_halve_each_level() -> None:
    x = torch.rand(4, 32, 32)
    levels = dataset_c.build_pyramid(x, num_layers=4)
    assert len(levels) == 4
    expected_sizes = [32, 16, 8, 4]
    for level, size in zip(levels, expected_sizes, strict=True):
        assert level.shape == (4, size, size)
    assert torch.equal(levels[0], x)


def test_build_pyramid_floor_halves_odd_sizes() -> None:
    x = torch.rand(4, 17, 17)
    levels = dataset_c.build_pyramid(x, num_layers=3)
    assert [tuple(lvl.shape[-2:]) for lvl in levels] == [(17, 17), (8, 8), (4, 4)]


def test_build_pyramid_rejects_non_positive_num_layers() -> None:
    with pytest.raises(ValueError):
        dataset_c.build_pyramid(torch.rand(4, 8, 8), num_layers=0)


def test_paired_names_excludes_photos_without_a_render_triple(tmp_path: Path) -> None:
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir()
    for stem in ("IMG_0", "IMG_2"):
        (renders_dir / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        np.save(renders_dir / f"{stem}.alpha.npy", np.zeros((2, 2), dtype=np.float16))
        np.save(renders_dir / f"{stem}.depth.npy", np.zeros((2, 2), dtype=np.float16))
    # IMG_1 has only a png (render_splat_views crashed mid-write, say) -- must not count as paired.
    (renders_dir / "IMG_1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    photo_names = ["IMG_0.jpg", "IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]
    paired = dataset_c.paired_names(renders_dir, photo_names)
    assert paired == ["IMG_0.jpg", "IMG_2.jpg"]


def test_render_to_tensor_channel_order_and_range() -> None:
    h, w = 5, 6
    arrays = {
        "rgb": np.full((h, w, 3), 0.5, dtype=np.float32),
        "alpha": np.full((h, w), 0.75, dtype=np.float32),
        "depth": np.full((h, w), 10.0, dtype=np.float32),
    }
    t4 = dataset_c.render_to_tensor(arrays, channels=4)
    assert t4.shape == (4, h, w)
    assert torch.allclose(t4[:3], torch.full((3, h, w), 0.5))
    assert torch.allclose(t4[3], torch.full((h, w), 0.75))

    t5 = dataset_c.render_to_tensor(arrays, channels=5)
    assert t5.shape == (5, h, w)
    assert torch.allclose(t5[4], torch.full((h, w), 0.5))  # 10.0 / HYBRID_C_DEPTH_NORM_SCALE (20.0)


def test_render_to_tensor_rejects_unsupported_channels() -> None:
    arrays = {
        "rgb": np.zeros((2, 2, 3), dtype=np.float32),
        "alpha": np.zeros((2, 2), dtype=np.float32),
        "depth": np.zeros((2, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError):
        dataset_c.render_to_tensor(arrays, channels=3)


def test_crop_pair_and_sample_crop_origin_stay_in_bounds() -> None:
    render = torch.rand(4, 20, 30)
    photo = torch.rand(3, 20, 30)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        y0, x0 = dataset_c.sample_crop_origin(20, 30, size=8, generator=gen)
        assert 0 <= y0 <= 12
        assert 0 <= x0 <= 22
        render_crop, photo_crop = dataset_c.crop_pair(render, photo, size=8, y0=y0, x0=x0)
        assert render_crop.shape == (4, 8, 8)
        assert photo_crop.shape == (3, 8, 8)
        assert torch.equal(render_crop, render[:, y0 : y0 + 8, x0 : x0 + 8])


def test_crop_pair_raises_when_window_overshoots() -> None:
    render = torch.rand(4, 10, 10)
    photo = torch.rand(3, 10, 10)
    with pytest.raises(ValueError):
        dataset_c.crop_pair(render, photo, size=8, y0=5, x0=5)
