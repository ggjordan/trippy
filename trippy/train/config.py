"""TrainConfig: the trainer's YAML-loadable configuration dataclass.

Module: trippy.train.config
Invariants: every default cites its `configs/train_normalnet.ini` source (or
    says explicitly that it is a trippy addition/scaling) in
    trippy.constants -- see that module's "train/" section header. Loading
    from YAML is `TrainConfig(**yaml.safe_load(...))`: any key present in
    the file overrides that one default, so a config file only needs to
    state what differs from train_normalnet.ini's scaled-for-trippy
    defaults. Fractions (`lock_cameras_frac`, `lock_structure_frac`,
    `vgg_start_frac`) are stored, not absolute epoch counts, so the
    proportion of a run spent locked/pre-VGG matches TRIPS regardless of
    how many epochs a given trippy run actually does (docs/TRIPS_REFERENCE.md
    Sec. 7's `lock_camera_params_epochs=100` etc. are all fractions of that
    config's own `num_epochs=600`).
Related docs: docs/TRIPS_REFERENCE.md Sec. 7 (losses/schedule, all
    defaults); docs/ARCHITECTURE.md "train/" section; docs/EXPERIMENTS.md
    "Training runs" (config file format, output layout).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trippy.constants import (
    DEFAULT_MIN_OPACITY,
    EVAL_CALIBRATE_DEFAULT_LR,
    EVAL_CALIBRATE_DEFAULT_STEPS,
    FORCED_HELDOUT_MODE_ALL,
    FORCED_HELDOUT_MODES,
    LOSS_DEFAULT_WEIGHT_L1,
    LOSS_DEFAULT_WEIGHT_LPIPS,
    LOSS_DEFAULT_WEIGHT_SSIM,
    LOSS_DEFAULT_WEIGHT_VGG,
    RASTER_MODES,
    RASTER_PIXEL_CENTERS,
    RASTER_PYRAMID_HALVINGS,
    TRAIN_DEFAULT_BACKGROUND,
    TRAIN_DEFAULT_CHECKPOINT_EVERY,
    TRAIN_DEFAULT_CHECKPOINT_KEEP_EVERY,
    TRAIN_DEFAULT_CHECKPOINT_KEEP_LAST,
    TRAIN_DEFAULT_CROP,
    TRAIN_DEFAULT_CROPS_PER_STEP,
    TRAIN_DEFAULT_EPOCHS,
    TRAIN_DEFAULT_EVAL_EVERY,
    TRAIN_DEFAULT_FEATURE_CHANNELS,
    TRAIN_DEFAULT_HELDOUT_K,
    TRAIN_DEFAULT_LAYERS,
    TRAIN_DEFAULT_LR_BACKGROUND,
    TRAIN_DEFAULT_LR_CONFIDENCE,
    TRAIN_DEFAULT_LR_EXPOSURE,
    TRAIN_DEFAULT_LR_NETWORK,
    TRAIN_DEFAULT_LR_POINTS,
    TRAIN_DEFAULT_LR_POSES,
    TRAIN_DEFAULT_LR_RESPONSE,
    TRAIN_DEFAULT_LR_SIZE,
    TRAIN_DEFAULT_LR_TEXTURE,
    TRAIN_DEFAULT_MODE,
    TRAIN_DEFAULT_PIXEL_CENTER,
    TRAIN_DEFAULT_PYRAMID_HALVING,
    TRAIN_DEFAULT_SEED,
    TRAIN_DEFAULT_TRAIN_FACTOR,
    TRAIN_DEFAULT_WIDTH,
    TRAIN_DEFAULT_ZOOM_MAX,
    TRAIN_DEFAULT_ZOOM_MIN,
    TRAIN_EVAL_MAX_SHEET_IMAGES,
    TRAIN_EXTENT_MARGIN_FRAC,
    TRAIN_EXTENT_PENALTY_WEIGHT_DEFAULT,
    TRAIN_LOCK_CAMERAS_FRAC,
    TRAIN_LOCK_STRUCTURE_FRAC,
    TRAIN_LR_DECAY_FACTOR,
    TRAIN_LR_DECAY_PATIENCE,
    TRAIN_VGG_START_FRAC,
)
from trippy.hybrid.config_a import HybridConfig


def steps_per_epoch(train_factor: float, n_train: int) -> int:
    """`ceil(train_factor * n_train)`, floored at 1 (see TRAIN_DEFAULT_TRAIN_FACTOR)."""
    return max(1, math.ceil(train_factor * n_train))


@dataclass
class PointSourceConfig:
    """Config-file description of a `trippy.points.PointSource` (docs/SPEC.md D4).

    Attributes:
        type: "gaussian" (GaussianPlySource), "colmap" (ColmapSparseSource),
            "union" (UnionSource of `sources`), or "npz" (a PointSet dumped
            via `PointSet.save_npz`, loaded back verbatim -- useful for
            synthetic test fixtures and for point sets produced by another
            tool offline).
        path: PLY/sparse-dir/npz path (unused when type == "union").
        min_opacity, size_mode, max_points, seed: forwarded to
            GaussianPlySource (ignored by other types).
        sources: child PointSourceConfigs, only read when type == "union".
        voxel: forwarded to UnionSource's voxel-dedupe (only read when
            type == "union"); None disables dedupe.
    """

    type: str = "gaussian"
    path: str = ""
    min_opacity: float = DEFAULT_MIN_OPACITY
    size_mode: str = "scale"
    max_points: int | None = None
    seed: int = TRAIN_DEFAULT_SEED
    sources: list[PointSourceConfig] = field(default_factory=list)
    voxel: float | None = None

    def __post_init__(self) -> None:
        self.sources = [s if isinstance(s, PointSourceConfig) else PointSourceConfig(**s) for s in self.sources]

    def to_source(self) -> Any:
        """Build the `trippy.points.PointSource` this config describes (not built yet)."""
        # Deferred imports: trippy.points is a large, torch-free import surface trainer
        # code doesn't otherwise need at module load time (e.g. `trippy eval --help`).
        from trippy.points.colmap_sparse import ColmapSparseSource
        from trippy.points.gaussian_ply import GaussianPlySource
        from trippy.points.source import PointSet, PointSource
        from trippy.points.union import UnionSource

        if self.type == "gaussian":
            return GaussianPlySource(
                self.path,
                min_opacity=self.min_opacity,
                size_mode=self.size_mode,
                max_points=self.max_points,
                seed=self.seed,
            )
        if self.type == "colmap":
            return ColmapSparseSource(self.path)
        if self.type == "union":
            return UnionSource([s.to_source() for s in self.sources], voxel=self.voxel)
        if self.type == "npz":

            class _NpzPointSource(PointSource):
                """Loads a PointSet previously written by PointSet.save_npz verbatim."""

                def __init__(self, path: str) -> None:
                    self.path = path

                def describe(self) -> dict:
                    return {"type": "npz", "path": str(self.path)}

                def build(self) -> PointSet:
                    return PointSet.load_npz(self.path)

            return _NpzPointSource(self.path)
        raise ValueError(f"unknown point source type {self.type!r}")


@dataclass
class TrainConfig:
    """Full trainer configuration; every field has a scaled-for-trippy default.

    See trippy.constants "train/" section for the exact TRIPS ini citation
    (or trippy-addition rationale) behind each default value.
    """

    # --- scene / io ---
    scene_root: str = ""
    cache_root: str | None = None  # None => `<TRIPPY_OUTPUT>/cache` (trippy.config.load_settings)
    run_dir: str = "output/runs/default"
    width: int = TRAIN_DEFAULT_WIDTH
    limit_images: int | None = None  # test/debug hook, forwarded to SceneDataset's own `limit`

    # --- crop / zoom sampling (docs/TRIPS_REFERENCE.md Sec. 7 "Crop augmentation") ---
    crop: int = TRAIN_DEFAULT_CROP
    zoom_min: float = TRAIN_DEFAULT_ZOOM_MIN
    zoom_max: float = TRAIN_DEFAULT_ZOOM_MAX
    crops_per_step: int = TRAIN_DEFAULT_CROPS_PER_STEP

    # --- epoch schedule ---
    epochs: int = TRAIN_DEFAULT_EPOCHS
    train_factor: float = TRAIN_DEFAULT_TRAIN_FACTOR
    lock_cameras_frac: float = TRAIN_LOCK_CAMERAS_FRAC
    lock_structure_frac: float = TRAIN_LOCK_STRUCTURE_FRAC
    vgg_start_frac: float = TRAIN_VGG_START_FRAC

    # --- optimiser (trippy.constants "TRAIN_DEFAULT_LR_*") ---
    lr_network: float = TRAIN_DEFAULT_LR_NETWORK
    lr_texture: float = TRAIN_DEFAULT_LR_TEXTURE
    lr_background: float = TRAIN_DEFAULT_LR_BACKGROUND
    lr_points: float = TRAIN_DEFAULT_LR_POINTS
    lr_size: float = TRAIN_DEFAULT_LR_SIZE
    lr_poses: float = TRAIN_DEFAULT_LR_POSES
    lr_confidence: float = TRAIN_DEFAULT_LR_CONFIDENCE
    lr_exposure: float = TRAIN_DEFAULT_LR_EXPOSURE
    lr_response: float = TRAIN_DEFAULT_LR_RESPONSE
    lr_decay_factor: float = TRAIN_LR_DECAY_FACTOR
    lr_decay_patience: int = TRAIN_LR_DECAY_PATIENCE

    # --- losses (trippy.net.losses.LossWeights defaults, reused directly) ---
    loss_vgg: float = LOSS_DEFAULT_WEIGHT_VGG
    loss_l1: float = LOSS_DEFAULT_WEIGHT_L1
    loss_ssim: float = LOSS_DEFAULT_WEIGHT_SSIM
    loss_lpips: float = LOSS_DEFAULT_WEIGHT_LPIPS
    extent_penalty_weight: float = TRAIN_EXTENT_PENALTY_WEIGHT_DEFAULT
    extent_margin: float = TRAIN_EXTENT_MARGIN_FRAC

    # --- render (trippy.raster.pyramid.render_pyramid / trippy.net.unet.NetworkConfig) ---
    # `mode` defaults to "trips", the layer rule the published TRIPS checkpoints render
    # with. `pixel_center` stays on trippy's own "half" convention: trippy undistorts its
    # scenes with pixel centres at i + 0.5 (docs/GEOMETRY.md), so "integer" -- which exists
    # to reproduce a TRIPS checkpoint bit-for-bit -- would put the render half a pixel off
    # its own ground truth. See docs/GEOMETRY.md "Pixel-centre convention".
    mode: str = TRAIN_DEFAULT_MODE
    pixel_center: str = TRAIN_DEFAULT_PIXEL_CENTER
    pyramid_halving: str = TRAIN_DEFAULT_PYRAMID_HALVING
    layers: int = TRAIN_DEFAULT_LAYERS
    feature_channels: int = TRAIN_DEFAULT_FEATURE_CHANNELS
    background: float = TRAIN_DEFAULT_BACKGROUND

    # --- point source (docs/SPEC.md D4) ---
    point_source: PointSourceConfig = field(default_factory=PointSourceConfig)

    # --- split / eval (trippy.scene.splits) ---
    heldout_k: int = TRAIN_DEFAULT_HELDOUT_K
    forced_heldout: list[str] = field(default_factory=list)
    # "all" (default, unchanged behaviour) holds every forced frame out; "alternate" holds out
    # every other one and forces the rest into training. The two protocols answer different
    # questions -- see trippy.scene.splits.partition_forced and docs/EXPERIMENTS.md
    # "Forced hold-out protocols".
    forced_heldout_mode: str = FORCED_HELDOUT_MODE_ALL
    # Test-time photometric calibration at eval (trippy.constants "test-time photometric
    # calibration"): fit ONLY this image's exposure (+ white balance when
    # eval_calibrate_white_balance) against its own photo before scoring it. Default False:
    # every training-time eval stays on the strict protocol; `trippy eval --calibrate` turns
    # it on for a diagnostic re-run, which reports both numbers side by side.
    eval_calibrate_camera: bool = False
    eval_calibrate_white_balance: bool = False
    eval_calibrate_steps: int = EVAL_CALIBRATE_DEFAULT_STEPS
    eval_calibrate_lr: float = EVAL_CALIBRATE_DEFAULT_LR
    eval_every: int = TRAIN_DEFAULT_EVAL_EVERY
    checkpoint_every: int = TRAIN_DEFAULT_CHECKPOINT_EVERY
    eval_lpips: bool = True  # see trainer.py docstring: gated so CPU tests don't require a
    # network-reachable LPIPS/VGG backbone unless explicitly asked for.
    # Rows in the per-epoch eval contact sheet (photo | render | raw L0 | coverage), forced
    # held-out frames first -- was a hardcoded constant (TRAIN_EVAL_MAX_SHEET_IMAGES), now a
    # config field so a run can widen/narrow it without a code change; default is unchanged
    # behaviour (trippy.train.trainer.Trainer.evaluate).
    eval_max_images: int = TRAIN_EVAL_MAX_SHEET_IMAGES

    # --- checkpoint retention (trippy.train.retention; task brief 2026-09-06, disk at 94%) ---
    # `Trainer.save_checkpoint` keeps checkpoint_latest.pt, checkpoint_best.pt (the best
    # held-out PSNR so far), every epoch that is a multiple of `checkpoint_keep_every`, and
    # the `checkpoint_keep_last` most recent epoch files, deleting the rest.
    checkpoint_keep_every: int = TRAIN_DEFAULT_CHECKPOINT_KEEP_EVERY
    checkpoint_keep_last: int = TRAIN_DEFAULT_CHECKPOINT_KEEP_LAST

    # --- hybrid: the Gaussian splat render as extra U-Net input channels (design A) ---
    # `enabled: false` (the default) is a hard no-op -- see trippy.hybrid.config_a.
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # --- misc ---
    seed: int = TRAIN_DEFAULT_SEED
    device: str = "cpu"
    max_minutes: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.point_source, dict):
            self.point_source = PointSourceConfig(**self.point_source)
        if isinstance(self.hybrid, dict):
            self.hybrid = HybridConfig(**self.hybrid)
        if self.crop <= 0:
            raise ValueError(f"crop must be positive, got {self.crop}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.feature_channels < 3:
            raise ValueError(f"feature_channels must be >= 3 (rgb0 seeds channels 0:3), got {self.feature_channels}")
        if self.mode not in RASTER_MODES:
            raise ValueError(f"mode must be one of {RASTER_MODES}, got {self.mode!r}")
        if self.pixel_center not in RASTER_PIXEL_CENTERS:
            raise ValueError(
                f"pixel_center must be one of {RASTER_PIXEL_CENTERS}, got {self.pixel_center!r}"
            )
        if self.pyramid_halving not in RASTER_PYRAMID_HALVINGS:
            raise ValueError(
                f"pyramid_halving must be one of {RASTER_PYRAMID_HALVINGS}, got {self.pyramid_halving!r}"
            )
        if self.forced_heldout_mode not in FORCED_HELDOUT_MODES:
            raise ValueError(
                f"forced_heldout_mode must be one of {FORCED_HELDOUT_MODES}, got {self.forced_heldout_mode!r}"
            )
        if self.eval_calibrate_steps < 0:
            raise ValueError(f"eval_calibrate_steps must be >= 0, got {self.eval_calibrate_steps}")
        if self.eval_calibrate_lr <= 0:
            raise ValueError(f"eval_calibrate_lr must be positive, got {self.eval_calibrate_lr}")
        if self.eval_max_images <= 0:
            raise ValueError(f"eval_max_images must be positive, got {self.eval_max_images}")
        if self.checkpoint_keep_every <= 0:
            raise ValueError(f"checkpoint_keep_every must be positive, got {self.checkpoint_keep_every}")
        if self.checkpoint_keep_last < 0:
            raise ValueError(f"checkpoint_keep_last must be >= 0, got {self.checkpoint_keep_last}")

    @property
    def net_input_channels(self) -> int:
        """Channels per U-Net input level: TRIPS features, plus the Gaussian block if hybrid.

        `trippy.net.unet.MultiScaleUnet2dDecOnlySmallFixed` requires every
        pyramid level to carry exactly `NetworkConfig.num_input_channels`
        channels (its `forward` validates this), so the Gaussian block widens
        *all* levels even in `concat_level0` mode -- the coarse levels simply
        carry zeros there. With `hybrid.enabled` false this is exactly
        `feature_channels`, i.e. the pre-design-A value.
        """
        return self.feature_channels + (self.hybrid.num_channels if self.hybrid.enabled else 0)

    @property
    def lock_cameras_epochs(self) -> int:
        """Epochs, from the start of a run, that pose deltas are held fixed."""
        return round(self.lock_cameras_frac * self.epochs)

    @property
    def lock_structure_epochs(self) -> int:
        """Epochs, from the start of a run, that xyz/size are held fixed."""
        return round(self.lock_structure_frac * self.epochs)

    @property
    def vgg_start_epoch(self) -> int:
        """First epoch the perceptual (vgg) loss term is added in."""
        return round(self.vgg_start_frac * self.epochs)

    def steps_per_epoch(self, n_train: int) -> int:
        """See module-level `steps_per_epoch`."""
        return steps_per_epoch(self.train_factor, n_train)

    def to_dict(self) -> dict:
        """Plain-dict (nested dataclasses included) representation, YAML-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        """Build a TrainConfig from a plain dict (e.g. `yaml.safe_load`'s output).

        Only keys present in `data` override the scaled-for-trippy defaults;
        unknown keys raise (a config typo should fail loudly, not silently
        no-op).
        """
        return cls(**(data or {}))

    @classmethod
    def load_yaml(cls, path: str | Path) -> TrainConfig:
        """Load a TrainConfig from a YAML file (see class docstring)."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> Path:
        """Write this config's resolved values to `path` as YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path
