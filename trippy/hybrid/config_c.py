"""HybridCConfig: YAML-loadable configuration for the Design C render->photo trainer.

Module: trippy.hybrid.config_c
Invariants: mirrors `trippy.train.config.TrainConfig`'s "state only what differs from the
    default" YAML convention (`HybridCConfig(**yaml.safe_load(...))`), but is its own
    dataclass -- Design C's trainer does not build a `TrainConfig` (no point source, no
    rasteriser). Every default that has a TRIPS/trippy precedent reuses that exact constant
    (lr_network, loss weights, heldout_k, ...) rather than inventing a new number, so the two
    trainers agree wherever the underlying component (U-Net, NeuralCamera, losses, split) is
    literally the same code.
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement";
    trippy.hybrid.train_c.HybridCTrainer (consumes this config).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from trippy.constants import (
    HYBRID_C_DEFAULT_CHANNELS,
    LOSS_DEFAULT_WEIGHT_L1,
    LOSS_DEFAULT_WEIGHT_SSIM,
    SHADE_FRAMES_KK,
    TRAIN_DEFAULT_CHECKPOINT_EVERY,
    TRAIN_DEFAULT_CROP,
    TRAIN_DEFAULT_EPOCHS,
    TRAIN_DEFAULT_EVAL_EVERY,
    TRAIN_DEFAULT_HELDOUT_K,
    TRAIN_DEFAULT_LAYERS,
    TRAIN_DEFAULT_LR_EXPOSURE,
    TRAIN_DEFAULT_LR_NETWORK,
    TRAIN_DEFAULT_LR_RESPONSE,
    TRAIN_DEFAULT_SEED,
    TRAIN_DEFAULT_TRAIN_FACTOR,
    TRAIN_DEFAULT_WIDTH,
)

# Design C's loss is "L1 + SSIM + LPIPS" (task brief) -- unlike TrainConfig's TRIPS-parity
# default (loss_lpips=0, vgg carries perceptual loss), this experiment has no vgg term at all
# and turns lpips on by default.
HYBRID_C_DEFAULT_LOSS_LPIPS = 1.0


@dataclass
class HybridCConfig:
    """Full configuration for `trippy.hybrid.train_c.HybridCTrainer`.

    Attributes:
        scene_root: COLMAP scene root (images/ + sparse/0 or sparse_txt) -- must be the same
            scene `renders_dir` was rendered from.
        renders_dir: directory of `trippy.hybrid.render_splat_views` output (rgb/depth/alpha
            triples), e.g. `output/hybrid-c/renders/w1008`.
        run_dir: output run directory (checkpoints/, eval_ep*/, log.txt, metrics.jsonl).
        width: `SceneDataset` width; must match the width `renders_dir` was rendered at, or
            every photo/render pair mismatches in (H, W) and `crop_pair` raises.
        cache_root: `SceneDataset` cache root override; None uses
            `trippy.config.load_settings()`'s default.
        crop: training crop side length (must be divisible by `2 ** (layers - 1)` so
            `build_pyramid` never needs to interpolate a level's crop window into existence --
            it always halves a `crop`-sized window exactly).
        channels: `HYBRID_C_DEFAULT_CHANNELS` (4, rgb+alpha) or 5 (+ depth) -- see
            `trippy.hybrid.dataset_c.render_to_tensor`.
        layers: pyramid levels fed to the U-Net (`NetworkConfig.num_layers`).
        epochs, train_factor: epoch schedule, same meaning as `TrainConfig` (one epoch is
            `ceil(train_factor * n_train)` steps).
        heldout_k, forced_heldout: `trippy.scene.splits.split_with_forced_heldout` inputs;
            `forced_heldout` defaults to `SHADE_FRAMES_KK` per the task brief.
        lr_network, lr_exposure, lr_response: optimiser param-group learning rates, reusing
            `TrainConfig`'s TRIPS-derived values for the same components.
        loss_l1, loss_ssim, loss_lpips: `trippy.net.losses.LossWeights` fields (no `vgg`
            weight in this design -- see `HYBRID_C_DEFAULT_LOSS_LPIPS`).
        eval_lpips: gates constructing the (possibly network-fetched) LPIPS metric backbone
            during `evaluate()`, same escape hatch as `TrainConfig.eval_lpips`.
        eval_every, checkpoint_every: epoch cadence for `evaluate()`/`save_checkpoint()`.
        seed, device, max_minutes: misc / wall-clock budget (see `TrainConfig`).
        limit_images: test/debug hook forwarded to `SceneDataset`'s own `limit`.
    """

    scene_root: str = ""
    renders_dir: str = ""
    run_dir: str = "output/runs/hybrid-c/default"
    width: int = TRAIN_DEFAULT_WIDTH
    cache_root: str | None = None

    crop: int = TRAIN_DEFAULT_CROP
    channels: int = HYBRID_C_DEFAULT_CHANNELS
    layers: int = TRAIN_DEFAULT_LAYERS

    epochs: int = TRAIN_DEFAULT_EPOCHS
    train_factor: float = TRAIN_DEFAULT_TRAIN_FACTOR

    heldout_k: int = TRAIN_DEFAULT_HELDOUT_K
    forced_heldout: list[str] = field(default_factory=lambda: list(SHADE_FRAMES_KK))

    lr_network: float = TRAIN_DEFAULT_LR_NETWORK
    lr_exposure: float = TRAIN_DEFAULT_LR_EXPOSURE
    lr_response: float = TRAIN_DEFAULT_LR_RESPONSE

    loss_l1: float = LOSS_DEFAULT_WEIGHT_L1
    loss_ssim: float = LOSS_DEFAULT_WEIGHT_SSIM
    loss_lpips: float = HYBRID_C_DEFAULT_LOSS_LPIPS
    eval_lpips: bool = True

    eval_every: int = TRAIN_DEFAULT_EVAL_EVERY
    checkpoint_every: int = TRAIN_DEFAULT_CHECKPOINT_EVERY

    seed: int = TRAIN_DEFAULT_SEED
    device: str = "cpu"
    max_minutes: float | None = None
    limit_images: int | None = None

    def __post_init__(self) -> None:
        if self.crop <= 0:
            raise ValueError(f"crop must be positive, got {self.crop}")
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1, got {self.layers}")
        min_multiple = 2 ** (self.layers - 1)
        if self.crop % min_multiple != 0:
            raise ValueError(
                f"crop ({self.crop}) must be divisible by 2**(layers-1) ({min_multiple}) so "
                "every pyramid level halves an exact crop window"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

    def to_dict(self) -> dict:
        """Plain-dict representation, YAML-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> HybridCConfig:
        """Build a HybridCConfig from a plain dict; unknown keys raise (typo safety)."""
        return cls(**(data or {}))

    @classmethod
    def load_yaml(cls, path: str | Path) -> HybridCConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path
