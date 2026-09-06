"""Shared synthetic fixtures for tests/test_hybrid_a_*.py (not a test module itself).

Module: tests.test_hybrid_a_helpers
Invariants: everything built here is synthetic (generated arrays), never a
    photo, a Splats render, or the real 1.7 GB Gaussian PLY -- AGENTS.md
    requires synthetic-only fixtures, and no test may touch MPS or
    `~/Splats`. `write_fake_renders` produces exactly the on-disk layout
    `trippy.hybrid.render_splat_views` writes (`<stem>.png` uint8 rgb,
    `<stem>.depth.npy` / `<stem>.alpha.npy` float16), so the hybrid loader is
    exercised against the real file contract rather than a mock.
    `FakeGsrender` stands in for Splats' `gsrender.render`/`load_ply` pair so
    `trippy.hybrid.gsrender_live` can be tested end to end on the CPU.
Related docs: docs/EXPERIMENTS.md "Hybrid design A"; tests/test_train_helpers.py
    (the synthetic COLMAP scene these fixtures pair with).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from test_train_helpers import (
    IMG_HEIGHT,
    IMG_WIDTH,
    build_synthetic_ply,
    build_synthetic_scene,
    tiny_train_config,
)

from trippy.hybrid.config_a import HybridConfig
from trippy.train.config import TrainConfig

#: Camera-space depth range the fake renders fill (world units), so a measured
#: median depth scale is a known, finite, positive number.
FAKE_DEPTH_MIN = 4.0
FAKE_DEPTH_MAX = 8.0


def write_fake_render(
    renders_dir: Path,
    stem: str,
    height: int = IMG_HEIGHT,
    width: int = IMG_WIDTH,
    rgb: np.ndarray | None = None,
    seed: int = 0,
    alpha: np.ndarray | float | None = None,
) -> dict[str, Path]:
    """Write one synthetic render triple in `render_splat_views`' exact layout.

    Args:
        renders_dir: destination directory (created if missing).
        stem: `Path(image_name).stem` -- the pairing key.
        height, width: render size.
        rgb: (H, W, 3) float in [0, 1] to use as the render's colour; None
            generates a deterministic gradient/noise mix from `seed`.
        seed: RNG seed for the generated channels.
        alpha: (H, W) float in [0, 1], or a scalar broadcast over the frame;
            None generates a random coverage map. A test that needs "the splat
            covers this pixel completely" (the blend gate's saturated extreme,
            tests/test_hybrid_gate_trainer.py) passes 1.0 here.

    Returns:
        `{"rgb": png path, "depth": npy path, "alpha": npy path}`.
    """
    renders_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if rgb is None:
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        rgb = np.clip(ramp + rng.normal(0.0, 0.1, (height, width, 3)).astype(np.float32), 0.0, 1.0)
    if alpha is None:
        alpha = rng.uniform(0.2, 1.0, (height, width))
    alpha = np.clip(np.broadcast_to(np.asarray(alpha, dtype=np.float32), (height, width)), 0.0, 1.0)
    depth = rng.uniform(FAKE_DEPTH_MIN, FAKE_DEPTH_MAX, (height, width)).astype(np.float32)

    paths = {
        "rgb": renders_dir / f"{stem}.png",
        "depth": renders_dir / f"{stem}.depth.npy",
        "alpha": renders_dir / f"{stem}.alpha.npy",
    }
    PILImage.fromarray(np.round(np.asarray(rgb) * 255.0).astype(np.uint8), mode="RGB").save(paths["rgb"])
    np.save(paths["depth"], depth.astype(np.float16))
    np.save(paths["alpha"], alpha.astype(np.float16))
    return paths


def write_fake_renders(
    renders_dir: Path,
    names: list[str],
    scene_root: Path | None = None,
    height: int = IMG_HEIGHT,
    width: int = IMG_WIDTH,
    alpha: np.ndarray | float | None = None,
) -> Path:
    """Write a render triple for every name in `names`.

    When `scene_root` is given, each render's rgb is that image's own photo
    (so the Gaussian channels carry real signal and a hybrid training step has
    something to exploit); otherwise a generated pattern is used.
    """
    for i, name in enumerate(names):
        rgb = None
        if scene_root is not None:
            photo_path = Path(scene_root) / "images" / name
            with PILImage.open(photo_path) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        write_fake_render(
            renders_dir, Path(name).stem, height=height, width=width, rgb=rgb, seed=i, alpha=alpha
        )
    return renders_dir


def hybrid_train_config(
    tmp_path: Path, render_alpha: np.ndarray | float | None = None, **hybrid_overrides: Any
) -> tuple[TrainConfig, list[str]]:
    """A tiny CPU `TrainConfig` with hybrid design A enabled and fake renders on disk.

    Args:
        tmp_path: pytest tmp dir the scene, renders and run dir are built under.
        render_alpha: forwarded to `write_fake_renders` -- pass 1.0 for
            fully-covered renders (see `write_fake_render`).
        hybrid_overrides: merged into the `hybrid:` block.

    Returns:
        `(cfg, names)` -- `names` is the synthetic scene's image list, all of
        which have a render triple unless a caller deletes some.
    """
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    names = sorted(p.name for p in (scene_root / "images").iterdir())
    renders_dir = tmp_path / "renders"
    write_fake_renders(renders_dir, names, scene_root=scene_root, alpha=render_alpha)

    hybrid = {"enabled": True, "renders_dir": str(renders_dir), "dropout_gaussian_p": 0.0}
    hybrid.update(hybrid_overrides)
    cfg = tiny_train_config(
        scene_root,
        ply_path,
        tmp_path / "run",
        tmp_path / "cache",
        hybrid=HybridConfig(**hybrid),
    )
    return cfg, names


class FakeGsrender:
    """Stand-in for Splats' `gsrender` module: a `load_ply`/`render` pair, CPU only.

    Records every call so a test can assert the PLY is loaded exactly once per
    process (the real one is 1.7 GB) and that `render` received the pose it
    was meant to.
    """

    def __init__(self, height: int = IMG_HEIGHT, width: int = IMG_WIDTH) -> None:
        self.height = height
        self.width = width
        self.load_calls: list[str] = []
        self.render_calls: list[dict[str, Any]] = []

    def load_ply(self, path: str) -> tuple[str, None, None]:
        self.load_calls.append(path)
        return (f"fake-gaussians:{path}", None, None)

    def render(
        self,
        gaussians: Any,
        viewmat: np.ndarray,
        k: np.ndarray,
        width: int,
        height: int,
        dev: str = "cpu",
        max_hw: int = 400,
        min_opacity: float = 0.02,
        return_depth: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.render_calls.append(
            {
                "gaussians": gaussians,
                "viewmat": np.asarray(viewmat).copy(),
                "K": np.asarray(k).copy(),
                "width": width,
                "height": height,
                "dev": dev,
                "max_hw": max_hw,
                "min_opacity": min_opacity,
            }
        )
        rgb = np.full((height, width, 3), 0.5, dtype=np.float32)
        depth = np.full((height, width), 6.0, dtype=np.float32)
        alpha = np.full((height, width), 0.75, dtype=np.float32)
        return rgb, depth, alpha
