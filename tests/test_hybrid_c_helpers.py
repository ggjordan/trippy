"""Shared synthetic scene + render fixtures for tests/test_hybrid_c_*.py (not a test module).

Module: tests.test_hybrid_c_helpers
Invariants: every fixture built here is synthetic (generated numpy data), never a photo or a
    real Splats scene (AGENTS.md: test fixtures must be synthetic only). The synthetic
    "render" is the photo plus Gaussian noise (clipped to [0, 1]) so a U-Net has a learnable
    signal to remove -- an exact-match render would already score ~0 loss at step 0, mirroring
    `tests/test_train_helpers.py::build_synthetic_ply`'s colour-noise rationale. Not named
    `test_*` at module scope (only helper functions), so pytest collects it without expecting
    any tests inside (see that module's own docstring for the same convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from trippy.hybrid.config_c import HybridCConfig

IMG_WIDTH = 64
IMG_HEIGHT = 48
FX = FY = 64.0
CX, CY = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0
N_IMAGES = 8


def _synthetic_photo(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)


def build_synthetic_scene(tmp_path: Path, n_images: int = N_IMAGES) -> tuple[Path, list[str]]:
    """A minimal 1-camera PINHOLE COLMAP scene with `n_images` random photos.

    Returns:
        (scene_root, names): `scene_root` contains `images/` and `sparse_txt/`; `names` is
        the sorted list of registered image filenames.
    """
    scene_root = tmp_path / "scene"
    images_dir = scene_root / "images"
    images_dir.mkdir(parents=True)
    names = [f"IMG_{i}.jpg" for i in range(n_images)]
    for i, name in enumerate(names):
        PILImage.fromarray(_synthetic_photo(seed=i), mode="RGB").save(images_dir / name)

    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {IMG_WIDTH} {IMG_HEIGHT} {FX} {FY} {CX} {CY}\n")

    lines = []
    for i, name in enumerate(names, start=1):
        lines.append(f"{i} 1.0 0.0 0.0 0.0 {float(i) * 0.1} 0.0 0.0 1 {name}")
        lines.append("")  # zero observations: genuine blank POINTS2D line.
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse_dir / "points3D.txt").write_text("")

    return scene_root, names


def build_synthetic_renders(
    scene_root: Path, out_dir: Path, names: list[str], noise_std: float = 0.15, seed: int = 0
) -> Path:
    """Write a render triple (rgb png, alpha npy, depth npy) per name, near the photo + noise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        stem = Path(name).stem
        with PILImage.open(scene_root / "images" / name) as img:
            photo = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        noisy = np.clip(photo + rng.normal(0.0, noise_std, photo.shape), 0.0, 1.0)
        rgb_u8 = np.round(noisy * 255.0).astype(np.uint8)
        PILImage.fromarray(rgb_u8, mode="RGB").save(out_dir / f"{stem}.png")

        alpha = rng.uniform(0.3, 1.0, size=photo.shape[:2]).astype(np.float32)
        depth = rng.uniform(1.0, 10.0, size=photo.shape[:2]).astype(np.float32)
        np.save(out_dir / f"{stem}.alpha.npy", alpha.astype(np.float16))
        np.save(out_dir / f"{stem}.depth.npy", depth.astype(np.float16))
    return out_dir


def tiny_hybrid_c_config(
    scene_root: Path, renders_dir: Path, run_dir: Path, cache_root: Path, **overrides
) -> HybridCConfig:
    """A HybridCConfig sized for the tiny synthetic scene above (CPU, seconds not minutes)."""
    defaults: dict = {
        "scene_root": str(scene_root),
        "renders_dir": str(renders_dir),
        "run_dir": str(run_dir),
        "cache_root": str(cache_root),
        "width": IMG_WIDTH,
        "crop": 16,
        "layers": 3,
        "channels": 4,
        "epochs": 2,
        "train_factor": 1.0,
        "heldout_k": 4,
        "forced_heldout": ["IMG_1.jpg"],
        "eval_every": 1,
        "checkpoint_every": 1,
        "eval_lpips": False,
        "loss_lpips": 0.0,
        "seed": 0,
        "device": "cpu",
    }
    defaults.update(overrides)
    return HybridCConfig(**defaults)
