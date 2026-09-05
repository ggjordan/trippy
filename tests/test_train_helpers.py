"""Shared synthetic-scene builders for tests/test_train_*.py (not a test module itself).

Module: tests.test_train_helpers
Invariants: every fixture built here is synthetic (generated numpy/torch
    data), never a photo or a real Splats scene (AGENTS.md: test fixtures
    must be synthetic only). Images are rendered from a known colour point
    cloud via the differentiable CPU reference path
    (`trippy.raster.pyramid.render_pyramid`, device cpu), so a training run
    against a (near-identical) point source has an achievable target --
    this is the "synthetic COLMAP scene ... rendered from 300 coloured
    points via ref path so the target is achievable" fixture the task brief
    asks for. Not named `test_*` at module scope (only helper functions),
    so pytest collects it without expecting any tests inside; other
    `tests/test_train_*.py` files import it directly (pytest's default
    "prepend" import mode puts `tests/` on `sys.path`, see pyproject.toml
    `testpaths = ["tests"]` and the absence of `tests/__init__.py`).
    The synthetic photos carry real EXIF (ExposureTime + ISO, see
    `_exif_for`) so the tone mapper's exposure initialisation is exercised
    by CPU tests -- without it every EV would be 0 and
    `Trainer._initial_exposure` would be untestable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from trippy.constants import EXIF_TAG_EXIF_IFD_POINTER, EXIF_TAG_EXPOSURE_TIME, EXIF_TAG_ISO
from trippy.points.source import PointSet
from trippy.raster.pyramid import render_pyramid
from trippy.train.config import PointSourceConfig, TrainConfig
from trippy.train.export import write_gaussian_ply

IMG_WIDTH = 48
IMG_HEIGHT = 36
FX = FY = 48.0
CX, CY = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0
N_POINTS = 300
N_IMAGES = 4

# EXIF written into every synthetic photo: a realistic *absolute* exposure (EV ~ 8.2,
# i.e. a gain of 2**-8.2 = 1/294 if a trainer forgets to centre it on the scene mean --
# see Trainer._initial_exposure) with only a small spread between images, so the
# scene stays fittable by a single texture once the mean is removed.
EXIF_EXPOSURE_TIMES = (1 / 320.0, 1 / 300.0, 1 / 280.0, 1 / 260.0)
EXIF_ISO = 100


def synthetic_point_set(n_points: int = N_POINTS, seed: int = 0) -> PointSet:
    """A random coloured point cloud in front of the synthetic cameras below."""
    rng = np.random.default_rng(seed)
    xyz = np.stack(
        [
            rng.uniform(-2.0, 2.0, n_points),
            rng.uniform(-1.5, 1.5, n_points),
            rng.uniform(4.0, 8.0, n_points),
        ],
        axis=1,
    ).astype(np.float32)
    rgb0 = rng.uniform(0.05, 0.95, (n_points, 3)).astype(np.float32)
    size0 = np.full(n_points, 0.05, dtype=np.float32)
    conf0 = np.full(n_points, 0.95, dtype=np.float32)
    provenance = np.ones(n_points, dtype=np.uint8)
    return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)


def camera_pose(image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """World->camera (R, t) for the i-th synthetic image: identity rotation, translated along x."""
    R = torch.eye(3, dtype=torch.float64)
    t = torch.tensor([-image_index * 0.3, 0.0, 0.0], dtype=torch.float64)
    return R, t


def render_reference_image(point_set: PointSet, image_index: int, num_layers: int = 5) -> np.ndarray:
    """Render one synthetic photo (uint8 HxWx3) from `point_set` via the CPU reference path."""
    xyz = torch.from_numpy(point_set.xyz.astype(np.float64))
    feat = torch.from_numpy(point_set.rgb0.astype(np.float64))
    size = torch.from_numpy(point_set.size0.astype(np.float64))
    conf = torch.from_numpy(point_set.conf0.astype(np.float64))
    K = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=torch.float64)
    R, t = camera_pose(image_index)
    layers, _aux = render_pyramid(
        xyz, size, feat, conf, K, R, t, (IMG_HEIGHT, IMG_WIDTH),
        num_layers=num_layers, mode="broadcast", compute_dtype=torch.float64,
    )  # fmt: skip
    img = layers[0].clamp(0.0, 1.0).permute(1, 2, 0)[:, :, :3].numpy()
    return (img * 255.0).round().astype(np.uint8)


def _exif_for(image_index: int) -> PILImage.Exif:
    """EXIF block (ExposureTime + ISO) for the `image_index`-th synthetic photo.

    Real EXIF is what makes the tone mapper's exposure initialisation
    non-trivial: `Trainer._initial_exposure` must subtract the scene-mean
    EV (TRIPS `NeuralScene.cpp:38`) or every prediction is divided by
    `2 ** mean(EV)`. Without EXIF here every EV would be 0 and that bug
    would be invisible to the CPU suite -- see
    tests/test_train_regression.py.
    """
    exif = PILImage.Exif()
    ifd = exif.get_ifd(EXIF_TAG_EXIF_IFD_POINTER)
    ifd[EXIF_TAG_EXPOSURE_TIME] = EXIF_EXPOSURE_TIMES[image_index % len(EXIF_EXPOSURE_TIMES)]
    ifd[EXIF_TAG_ISO] = EXIF_ISO
    return exif


def build_synthetic_scene(tmp_path: Path, n_images: int = N_IMAGES, seed: int = 0) -> tuple[Path, PointSet]:
    """Write a minimal 1-camera PINHOLE COLMAP scene with `n_images` rendered photos.

    Returns:
        (scene_root, point_set): `scene_root` contains `images/` and
        `sparse_txt/`; `point_set` is the ground-truth cloud the photos
        were rendered from (world frame matches `camera_pose`'s convention).
    """
    point_set = synthetic_point_set(seed=seed)
    scene_root = tmp_path / "scene"
    images_dir = scene_root / "images"
    images_dir.mkdir(parents=True)
    names = [f"IMG_{i}.jpg" for i in range(n_images)]
    for i, name in enumerate(names):
        img = render_reference_image(point_set, i)
        PILImage.fromarray(img, mode="RGB").save(images_dir / name, exif=_exif_for(i))

    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {IMG_WIDTH} {IMG_HEIGHT} {FX} {FY} {CX} {CY}\n")

    lines = []
    for i, name in enumerate(names, start=1):
        _R, t = camera_pose(i - 1)
        tx = float(t[0])
        lines.append(f"{i} 1.0 0.0 0.0 0.0 {tx} 0.0 0.0 1 {name}")
        lines.append("")  # zero observations
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse_dir / "points3D.txt").write_text("")

    return scene_root, point_set


def build_synthetic_ply(tmp_path: Path, point_set: PointSet, color_noise_std: float = 0.05, seed: int = 1) -> Path:
    """Write `point_set` (optionally colour-perturbed) as a GaussianPlySource-readable PLY.

    A small amount of colour noise gives training loss room to decrease
    (an exact-match point source would already score ~0 loss at step 0).
    """
    rng = np.random.default_rng(seed)
    rgb = np.clip(point_set.rgb0 + rng.normal(0.0, color_noise_std, point_set.rgb0.shape), 0.0, 1.0)
    path = tmp_path / "source.ply"
    write_gaussian_ply(path, point_set.xyz, rgb, point_set.conf0, point_set.size0, provenance=point_set.provenance)
    return path


def tiny_train_config(scene_root: Path, ply_path: Path, run_dir: Path, cache_root: Path, **overrides) -> TrainConfig:
    """A TrainConfig sized for the tiny synthetic scene above (CPU, seconds not minutes)."""
    defaults: dict = {
        "scene_root": str(scene_root),
        "cache_root": str(cache_root),
        "run_dir": str(run_dir),
        "width": IMG_WIDTH,
        "crop": 24,
        "zoom_min": 1.0,
        "zoom_max": 1.0,
        "epochs": 2,
        "train_factor": 0.5,
        "layers": 3,
        "feature_channels": 4,
        "heldout_k": 8,
        "eval_every": 1,
        "checkpoint_every": 1,
        "eval_lpips": False,
        "loss_vgg": 0.0,
        "loss_lpips": 0.0,
        "seed": 0,
        "device": "cpu",
        "point_source": PointSourceConfig(type="gaussian", path=str(ply_path), min_opacity=0.0),
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)
