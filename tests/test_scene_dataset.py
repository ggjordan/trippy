"""Tests for trippy.scene.dataset: undistort cache, crop masking, splits wiring.

Module: tests.test_scene_dataset
Invariants under test:
    1. `camera.undistort_maps` + `grid_sample` reproduce, to sub-pixel
       accuracy, an independently-computed analytic forward-distortion
       mapping (the "undistort round trip").
    2. `dataset.crop` never fakes padding as content: pixels where a crop
       window overshoots the source image get rgb == 0 and mask == 0,
       exactly (docs/GEOMETRY.md bug class 3).
    3. `SceneDataset`'s second construction over the same cache_root/width
       hits the on-disk cache (no recompute) and returns identical items.
All synthetic fixtures are generated in-test into tmp_path; no photos are
committed. The real-scene case (`splats_scene` fixture) builds a dataset
for the FIRST 3 images only, into a tmp cache -- never the full scene, and
never under ~/Splats.
Related docs: docs/GEOMETRY.md "Undistortion and image cache";
    docs/ARCHITECTURE.md "Module overview" (trippy/scene).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from trippy.geom import camera as camera_geom
from trippy.scene import dataset as scene_dataset


def test_undistort_maps_matches_analytic_distortion() -> None:
    """grid_sample of a coordinate-encoding image via undistort_maps recovers
    the analytically-computed source pixel-centre coordinate to < 0.5 px."""
    width_src, height_src = 640, 480
    fx_src = fy_src = 500.0
    cx_src, cy_src = 320.0, 240.0
    dist = camera_geom.OpenCVDistortion(k1=-0.12, k2=0.02, p1=0.0005, p2=-0.0003)

    width_dst = 320
    scale = width_dst / width_src
    height_dst = round(height_src * scale)
    fx_dst, fy_dst = fx_src * scale, fy_src * scale
    cx_dst, cy_dst = cx_src * scale, cy_src * scale

    grid = camera_geom.undistort_maps(
        fx_src=fx_src,
        fy_src=fy_src,
        cx_src=cx_src,
        cy_src=cy_src,
        width_src=width_src,
        height_src=height_src,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        width_dst=width_dst,
        height_dst=height_dst,
        distortion=dist,
    )
    assert grid.shape == (height_dst, width_dst, 2)

    # A coordinate-encoding "image": channel 0/1 hold each source pixel's own
    # continuous pixel-centre (u, v). Bilinear interpolation of an affine
    # field is exact, so this isolates undistort_maps' coordinate-convention
    # correctness from 8-bit/JPEG quantization noise.
    rows = np.arange(height_src, dtype=np.float32)
    cols = np.arange(width_src, dtype=np.float32)
    v_centre, u_centre = np.meshgrid(rows + 0.5, cols + 0.5, indexing="ij")
    coord_img = np.stack([u_centre, v_centre], axis=0)  # (2, H_src, W_src)

    src_t = torch.from_numpy(coord_img).unsqueeze(0)  # (1, 2, H_src, W_src)
    grid_t = torch.from_numpy(grid).unsqueeze(0)  # (1, H_dst, W_dst, 2)
    decoded = F.grid_sample(src_t, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False)
    decoded = decoded.squeeze(0).numpy()  # (2, H_dst, W_dst)
    decoded_u, decoded_v = decoded[0], decoded[1]

    # Independently-computed analytic expectation (same distortion formula,
    # applied directly to a destination pixel grid, not routed through
    # undistort_maps' internal grid array).
    rows_d = np.arange(height_dst, dtype=np.float64) + 0.5
    cols_d = np.arange(width_dst, dtype=np.float64) + 0.5
    row_d, col_d = np.meshgrid(rows_d, cols_d, indexing="ij")
    x = (col_d - cx_dst) / fx_dst
    y = (row_d - cy_dst) / fy_dst
    uv_d = dist.distort(np.stack([x.ravel(), y.ravel()], axis=1))
    x_d, y_d = uv_d[:, 0].reshape(row_d.shape), uv_d[:, 1].reshape(row_d.shape)
    expected_u = fx_src * x_d + cx_src
    expected_v = fy_src * y_d + cy_src

    # Restrict comparison to destination pixels whose source ray lands well
    # inside the source frame (grid_sample's zero-padding beyond +/-1, and
    # bilinear blending with it near +/-1, are expected -- not under test
    # here).
    interior = (np.abs(grid[..., 0]) < 0.9) & (np.abs(grid[..., 1]) < 0.9)
    assert interior.sum() > 1000, "expected most of the destination frame to be interior"

    err_u = np.abs(decoded_u[interior] - expected_u[interior])
    err_v = np.abs(decoded_v[interior] - expected_v[interior])
    assert np.max(err_u) < 0.5, f"max u error {np.max(err_u):.4f}px"
    assert np.max(err_v) < 0.5, f"max v error {np.max(err_v):.4f}px"


def test_undistort_maps_pinhole_source_is_plain_resize() -> None:
    """Zero distortion degenerates undistort_maps to an affine resize grid."""
    width_src, height_src = 200, 100
    fx = fy = 100.0
    cx, cy = 100.0, 50.0
    width_dst, height_dst = 100, 50
    scale = width_dst / width_src

    grid = camera_geom.undistort_maps(
        fx_src=fx,
        fy_src=fy,
        cx_src=cx,
        cy_src=cy,
        width_src=width_src,
        height_src=height_src,
        fx_dst=fx * scale,
        fy_dst=fy * scale,
        cx_dst=cx * scale,
        cy_dst=cy * scale,
        width_dst=width_dst,
        height_dst=height_dst,
        distortion=None,
    )
    # Corner destination pixel (0, 0) has continuous centre (0.5, 0.5); at
    # `scale`, that maps to source continuous coordinate 0.5/scale (a plain
    # resize has no distortion, so the mapping is just 1/scale).
    expected_gx = 2.0 * (0.5 / scale) / width_src - 1.0
    expected_gy = 2.0 * (0.5 / scale) / height_src - 1.0
    assert grid[0, 0, 0] == pytest.approx(expected_gx, abs=1e-6)
    assert grid[0, 0, 1] == pytest.approx(expected_gy, abs=1e-6)


def _make_item(rgb: torch.Tensor) -> dict:
    return {"rgb": rgb, "K": torch.eye(3, dtype=torch.float32)}


def test_crop_corner_overshoot_masks_and_zeros_padding() -> None:
    height, width = 20, 20
    rgb = torch.full((height, width, 3), 255, dtype=torch.uint8)
    item = _make_item(rgb)

    size = 10
    # Centre the crop window exactly on the image's top-left corner: half
    # the crop overshoots past both edges.
    out = scene_dataset.crop(item, size=size, zoom=1.0, center=(0.0, 0.0))

    assert out["rgb"].shape == (size, size, 3)
    assert out["mask"].shape == (size, size)

    half = size // 2
    # Interior source-side quadrant (rows/cols >= half) is valid content.
    assert torch.all(out["mask"][half:, half:] == 1.0)
    assert torch.all(out["rgb"][half:, half:] == 255)
    # The overshoot quadrant (rows/cols < half, mapping to negative source
    # coordinates) must be exactly zero -- never treated as image content.
    assert torch.all(out["mask"][:half, :] == 0.0)
    assert torch.all(out["mask"][:, :half] == 0.0)
    assert torch.all(out["rgb"][:half, :] == 0)
    assert torch.all(out["rgb"][:, :half] == 0)


def test_crop_fully_interior_has_no_padding() -> None:
    height, width = 40, 40
    rgb = torch.arange(height * width * 3, dtype=torch.uint8).reshape(height, width, 3) % 255
    item = _make_item(rgb)
    out = scene_dataset.crop(item, size=8, zoom=1.0, center=(20.0, 20.0))
    assert torch.all(out["mask"] == 1.0)
    expected = rgb[16:24, 16:24, :]
    assert torch.equal(out["rgb"], expected)


def test_crop_updates_intrinsics() -> None:
    rgb = torch.zeros((100, 100, 3), dtype=torch.uint8)
    K = torch.tensor([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    item = {"rgb": rgb, "K": K}
    out = scene_dataset.crop(item, size=20, zoom=2.0, center=(50.0, 50.0))
    assert out["K"][0, 0] == pytest.approx(100.0)
    assert out["K"][1, 1] == pytest.approx(100.0)


# --- SceneDataset cache tests (synthetic scene) ---


def _write_txt_scene(scene_root: Path, images: list[tuple[str, np.ndarray]]) -> None:
    """A minimal 1-camera PINHOLE scene with no lens distortion.

    `images`: list of (name, rgb uint8 array (H, W, 3)) -- all must share
    one size (one shared camera, kept simple for this cache test).
    """
    images_dir = scene_root / "images"
    images_dir.mkdir(parents=True)
    height, width = images[0][1].shape[:2]
    for name, arr in images:
        assert arr.shape[:2] == (height, width), "test fixture: all images must share one camera size"
        PILImage.fromarray(arr, mode="RGB").save(images_dir / name)

    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    lines = []
    for i, (name, _arr) in enumerate(images, start=1):
        lines.append(f"{i} 1.0 0.0 0.0 0.0 {float(i)} 0.0 0.0 1 {name}")
        lines.append("")  # zero observations: genuine blank POINTS2D line.
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse_dir / "points3D.txt").write_text("")


def _synthetic_gradient(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def test_scene_dataset_cache_hit_on_second_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_root = tmp_path / "scene"
    cache_root = tmp_path / "cache"
    images = [
        ("a.jpg", _synthetic_gradient(30, 40, seed=0)),
        ("b.jpg", _synthetic_gradient(30, 40, seed=1)),
    ]
    _write_txt_scene(scene_root, images)

    ds1 = scene_dataset.SceneDataset(scene_root, width=20, cache_root=cache_root, device="cpu")
    assert len(ds1) == 2
    item1 = ds1[0]
    assert item1["rgb"].shape == (15, 20, 3)  # 40->20 halves height 30->15
    assert item1["rgb"].dtype == torch.uint8
    assert item1["K"].shape == (3, 3)
    assert item1["K"].dtype == torch.float32

    calls = {"n": 0}
    original = scene_dataset.SceneDataset._undistort_image

    def counting_undistort(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(scene_dataset.SceneDataset, "_undistort_image", counting_undistort)

    ds2 = scene_dataset.SceneDataset(scene_root, width=20, cache_root=cache_root, device="cpu")
    assert calls["n"] == 0, "second construction must not recompute undistortion (cache hit expected)"
    assert len(ds2) == 2

    item2 = ds2[0]
    assert torch.equal(item1["rgb"], item2["rgb"])
    assert torch.equal(item1["K"], item2["K"])
    assert torch.equal(item1["qvec"], item2["qvec"])
    assert torch.equal(item1["tvec"], item2["tvec"])
    assert item1["name"] == item2["name"] == "a.jpg"


def test_scene_dataset_stale_cache_raises(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    cache_root = tmp_path / "cache"
    images = [("a.jpg", _synthetic_gradient(20, 20, seed=0))]
    _write_txt_scene(scene_root, images)

    scene_dataset.SceneDataset(scene_root, width=10, cache_root=cache_root, device="cpu")

    meta_path = cache_root / "scene" / "w10" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["images"]["a.jpg"]["K"][0][0] = 999999.0
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(AssertionError):
        scene_dataset.SceneDataset(scene_root, width=10, cache_root=cache_root, device="cpu")


def test_scene_dataset_limit_restricts_to_first_n(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    cache_root = tmp_path / "cache"
    images = [(f"IMG_{i}.jpg", _synthetic_gradient(20, 20, seed=i)) for i in range(5)]
    _write_txt_scene(scene_root, images)

    ds = scene_dataset.SceneDataset(scene_root, width=10, cache_root=cache_root, device="cpu", limit=3)
    assert len(ds) == 3
    assert ds.names == sorted(name for name, _ in images)[:3]


def test_resolve_sparse_dir_prefers_bin(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    (scene_root / "sparse" / "0").mkdir(parents=True)
    (scene_root / "sparse_txt").mkdir(parents=True)
    assert scene_dataset.resolve_sparse_dir(scene_root) == scene_root / "sparse" / "0"

    scene_root2 = tmp_path / "scene2"
    (scene_root2 / "sparse_txt").mkdir(parents=True)
    assert scene_dataset.resolve_sparse_dir(scene_root2) == scene_root2 / "sparse_txt"

    with pytest.raises(FileNotFoundError):
        scene_dataset.resolve_sparse_dir(tmp_path / "scene3")


# --- real-scene integration test (skips cleanly without ~/Splats) ---


def test_scene_dataset_real_scene_first_three_images(splats_scene: Path, tmp_path: Path) -> None:
    """Build a dataset for the FIRST 3 kk-coherent images only, at width 504.

    Never processes the full (219-registered-image) scene, and writes only
    into `tmp_path` -- never under ~/Splats.
    """
    scene_root = splats_scene.parent
    t0 = time.time()
    ds = scene_dataset.SceneDataset(scene_root, width=504, cache_root=tmp_path, device="cpu", limit=3)
    elapsed = time.time() - t0

    assert len(ds) == 3
    for i in range(len(ds)):
        item = ds[i]
        assert item["rgb"].shape[1] == 504
        assert item["rgb"].shape[2] == 3
        assert item["rgb"].dtype == torch.uint8
        assert item["K"].shape == (3, 3)
        assert item["index"] == i

    # Cache hit path exercised too: rebuilding must not reprocess anything.
    ds2 = scene_dataset.SceneDataset(scene_root, width=504, cache_root=tmp_path, device="cpu", limit=3)
    assert torch.equal(ds[0]["rgb"], ds2[0]["rgb"])

    assert elapsed < 60.0, f"building 3 images took {elapsed:.1f}s"
