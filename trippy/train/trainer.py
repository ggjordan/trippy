"""Trainer: the TRIPS-style crop/render/tone-map/loss training loop.

Module: trippy.train.trainer
Invariants: every render call goes through `trippy.raster.pyramid.
    render_pyramid` unmodified -- on `device="cpu"` this dispatches to the
    fully-differentiable `ref_torch` path (so every method here is
    CPU-testable today), and on `device="mps"` to the Metal kernel, which
    is forward-only until `blend_bwd` lands; nothing in this module assumes
    which backward path is active, so it "just works" once Metal gradients
    plug in (no API change needed here, per this task's brief).
    Crop rendering never renders the full frame and slices it: the crop's
    adjusted intrinsics (`trippy.scene.dataset.crop`'s K update) are handed
    straight to `render_pyramid` with `image_hw = (crop, crop)`, so only the
    crop's fragments are ever rasterised (the "K-adjust" strategy -- see
    `tests/test_train_crop_equivalence.py` for the proof this equals
    cropping a full render of the same points/pose).
    Hybrid design A (`cfg.hybrid.enabled`) concatenates a Gaussian-splat
    render onto every pyramid level before the U-Net (trippy.hybrid.
    gaussian_input). It is strictly additive: with `hybrid.enabled` false
    `self.hybrid is None` and every code path below is the one that existed
    before design A, down to the `NetworkConfig` channel count.
    Point removal (`cfg.point_removal`, TRIPS's own confidence rule, or
    trippy's relative analogue -- `trippy.train.prune_config
    .PointRemovalConfig.mode`; and `cfg.shade_prune`, trippy's audit-aligned
    heuristic, same mode choice) is the one thing
    here that changes the *shape* of the trained state mid-run: see
    `_apply_keep_mask`, which index-selects every per-point parameter and
    its Adam moments together so moment row i keeps belonging to point i,
    and `load_state`, which resizes before loading so such a checkpoint can
    be resumed and re-evaluated. Both default to disabled, in which case
    the point count is fixed for the whole run exactly as before.
    Structure/camera "locking" (docs/TRIPS_REFERENCE.md Sec. 7) is
    implemented by toggling `requires_grad_` on the frozen parameters for
    the locked epoch range, not by zeroing an optimizer-group learning
    rate -- this composes cleanly with the global `ReduceLROnPlateau`
    schedule (a zeroed-then-restored lr would otherwise fight the
    scheduler's own decay of that same group).
Related docs: docs/TRIPS_REFERENCE.md Sec. 2 (point parametrisation), Sec. 6
    (neural camera / tone mapping), Sec. 7 (losses/schedule, crop
    augmentation, locking); docs/ARCHITECTURE.md "train/" section;
    docs/EXPERIMENTS.md "Training runs" (output layout, what gets logged).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from trippy.config import load_settings, pick_device
from trippy.constants import (
    EVAL_EXPOSURE_MODES,
    SCENE_CACHE_META_FILENAME,
    SHADE_FRAMES_KK,
    TRAIN_CHECKPOINT_BEST_FILENAME,
    TRAIN_CHECKPOINT_BEST_JSON_FILENAME,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_CHECKPOINT_FILENAME_FMT,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
    TRAIN_EVAL_DIRNAME_FMT,
    TRAIN_EVAL_METRICS_FILENAME,
    TRAIN_EVAL_SHEET_JPEG_FILENAME,
    TRAIN_EVAL_SHEET_JPEG_QUALITY,
    TRAIN_EXPORT_FILENAME,
    TRAIN_LOG_FILENAME,
    TRAIN_LPIPS_METRIC_NET,
    TRAIN_METRICS_FILENAME,
    TRAIN_PSNR_EPS,
)
from trippy.geom import xform_b
from trippy.hybrid.gaussian_input import GaussianInputs
from trippy.net.camera_model import NeuralCamera, interpolate_from_train_neighbours
from trippy.net.losses import LossWeights, TripsLoss, _LazyLPIPS, l1_loss, mse_loss, ssim
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.raster.pyramid import render_pyramid
from trippy.render.sheets import colorize, contact_sheet
from trippy.scene import splits
from trippy.scene.dataset import SceneDataset, resolve_sparse_dir
from trippy.scene.dataset import crop as dataset_crop
from trippy.train import checkpoint_io, export, prune, retention
from trippy.train.config import TrainConfig
from trippy.train.params import PointParams, PoseParams


def _center_crop_like(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Centre-crop the last two dims of `x` down to (target_h, target_w).

    The U-Net's output can be a few pixels smaller than its input when the
    input isn't divisible by `2 ** (num_layers - 1)` (trippy.net.unet
    module docstring, "CombineBridge / odd-size handling"); the loss target
    and mask must shrink to match. A no-op when `x` is already that size.
    """
    h, w = x.shape[-2], x.shape[-1]
    dh = max(0, (h - target_h) // 2)
    dw = max(0, (w - target_w) // 2)
    return x[..., dh : dh + target_h, dw : dw + target_w]


def _save_eval_sheet_jpeg(path: Path, arr: np.ndarray, quality: int) -> Path:
    """Save a per-epoch eval contact sheet as JPEG rather than PNG.

    `trippy.render.sheets.contact_sheet` already returns a uint8 (H, W, 3)
    array, so this is a thin wrapper -- kept local to this module (rather
    than added to `trippy.render.sheets`) because it is specific to the
    eval-sheet disk-usage tradeoff (task brief 2026-09-06): a quick
    per-epoch progress check tolerates lossy compression, unlike
    `candidate-report`'s honesty/coverage PNGs (`trippy.render.candidate`),
    which Jordan inspects pixel-for-pixel and which this function is never
    used for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=quality)
    return path


def best_global_gain(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """The scalar `g` minimising `||g * pred - target||^2` -- a *diagnostic*, not a metric.

    Closed form (least squares in one unknown):
    `g = sum(pred * target) / sum(pred * pred)`. Applied to an already
    tone-mapped prediction it answers exactly one question: how much of
    this frame's error is a single global brightness factor (a mis-set
    per-image exposure gain) rather than structure? A frame whose PSNR
    jumps by many dB under `g` is being scored on its exposure, not on
    its geometry. It never touches training and is never reported as the
    run's PSNR (see `Trainer.evaluate`'s "psnr_gain" per-image key and
    docs/EXPERIMENTS.md "Per-image exposure diagnostics").

    Args:
        pred, target: (B, C, H, W) tensors in the same space.
        mask: (B, 1, H, W) validity mask (broadcast over channels), or None.

    Returns:
        `g` as a python float; 1.0 when `pred` is identically zero (no
        gain can fix an all-black prediction, so report the no-op).
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask
    denom = float((pred * pred).sum().item())
    if denom <= 0.0:
        return 1.0
    return float((pred * target).sum().item()) / denom


def _psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """-10 log10(masked MSE), the single definition every number in this module uses."""
    mse = mse_loss(pred, target, mask)
    return float((-10.0 * torch.log10(mse + TRAIN_PSNR_EPS)).item())


def _masked_mean_value(x: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean pixel value of `x` over `mask` (all channels), as a python float."""
    if mask is None:
        return float(x.mean().item())
    total = float(mask.sum().item()) * x.shape[1]
    if total <= 0.0:
        return 0.0
    return float((x * mask).sum().item()) / total


def _shade_and_other(names: list[str], forced_heldout: list[str]) -> tuple[list[str], list[str]]:
    """Partition an already-held-out `names` list into shade frames vs other held-out frames.

    Shade frames are `forced_heldout` (a config's own explicit "always hold these frames out"
    list, `cfg.forced_heldout`) when non-empty, else `SHADE_FRAMES_KK` (the kk-coherent scene's
    known shade region under the trees) -- intersected with `names` either way, so a scene with
    neither yields an empty shade set rather than raising. This is *not*
    `trippy.scene.splits.split_with_forced_heldout` -- that function partitions the *whole*
    dataset into train/heldout; this one further partitions an already-held-out set of names
    purely for reporting (docs/EXPERIMENTS.md "Leaderboard").

    Returns:
        `(shade_names, other_names)`, both in `names`'s own order, partitioning it exactly.
    """
    shade_set = (set(forced_heldout) if forced_heldout else set(SHADE_FRAMES_KK)) & set(names)
    shade = [n for n in names if n in shade_set]
    other = [n for n in names if n not in shade_set]
    return shade, other


def _aggregate_group(names: list[str], per_image: dict[str, dict], suffix: str = "") -> dict:
    """`{"n", "psnr", "ssim", "lpips"}` -- mean over `names` from a per-image metrics dict.

    `psnr`/`ssim`/`lpips` are all `None` when `names` is empty (no images in this group, e.g. a
    scene with no `SHADE_FRAMES_KK` and no `forced_heldout`); `lpips` alone is also `None` when
    `cfg.eval_lpips` was False for the run that produced `per_image` (no per-image lpips values to
    average).

    "names" is the group's own member list (so a consumer -- e.g. the `trippy eval`
    diagnostics table -- can tell shade from other without re-deriving the split).

    `suffix` selects which family of per-image keys to average: `""` reads
    `psnr`/`ssim`/`lpips` (the strict, uncalibrated numbers), `"_calibrated"` reads
    `psnr_calibrated`/... (present only when `Trainer.evaluate` ran with
    `calibrate=True`). Missing keys average to `None` rather than raising, so an
    older `per_image` dict aggregates fine.
    """
    psnr_vals = [per_image[n].get(f"psnr{suffix}") for n in names]
    psnr_vals = [v for v in psnr_vals if v is not None]
    ssim_vals = [v for v in (per_image[n].get(f"ssim{suffix}") for n in names) if v is not None]
    lpips_vals = [v for v in (per_image[n].get(f"lpips{suffix}") for n in names) if v is not None]
    return {
        "n": len(names),
        "names": list(names),
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
    }


class Trainer:
    """Owns a scene, a trained point cloud, the U-Net, the tone mapper, and the optimiser.

    Construction builds (or loads from cache) the dataset, the point
    source, and every trainable module; call `fit()` to run the full
    schedule, or `train_step()`/`evaluate()` directly for finer control
    (e.g. from tests).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        # Seed the *global* torch RNG, not just `self._rng`: the U-Net's and the
        # NeuralCamera's weight initialisation go through torch's default generator,
        # so without this two runs of the same config start from different networks
        # and their held-out numbers are not comparable (observed: 6.7 dB vs 8.4 dB
        # at init on the same EXP-0003 smoke config).
        torch.manual_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        settings = load_settings()
        cache_root = Path(cfg.cache_root) if cfg.cache_root else settings.trippy_output / "cache"

        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / TRAIN_CHECKPOINT_DIRNAME
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / TRAIN_LOG_FILENAME
        self.metrics_path = self.run_dir / TRAIN_METRICS_FILENAME

        self.dataset = SceneDataset(
            cfg.scene_root, cfg.width, cache_root, device=self.device, limit=cfg.limit_images
        )
        self._name_to_index = {name: i for i, name in enumerate(self.dataset.names)}
        self.train_names, self.heldout_names = splits.split_with_forced_heldout(
            self.dataset.names, cfg.forced_heldout, k=cfg.heldout_k, mode=cfg.forced_heldout_mode
        )
        if not self.train_names:
            raise ValueError("split produced an empty train set -- check heldout_k/forced_heldout vs dataset size")
        # For `exposure_mode="neighbours"` (trippy.net.camera_model.interpolate_from_train_neighbours):
        # per-frame-index flag, True for a TRAINING frame, aligned with `self.dataset.names`'s own
        # order (== `NeuralCamera.exposures_values`'s row order, since both are built from
        # `len(self.dataset.names)`). Computed once -- the split never changes after __init__.
        self._train_name_set = set(self.train_names)
        self._is_train_frame = [name in self._train_name_set for name in self.dataset.names]

        point_source = cfg.point_source.to_source()
        point_set = point_source.build()
        self.point_source_describe = point_source.describe()
        self.point_params = PointParams(point_set, feature_channels=cfg.feature_channels, seed=cfg.seed).to(
            self.device
        )
        self.pose_params = PoseParams(len(self.dataset.names)).to(self.device)
        self.background = nn.Parameter(
            torch.full((cfg.feature_channels,), float(cfg.background), device=self.device)
        )

        # Design A: the Gaussian render's channels are appended to every U-Net input
        # level, so the network -- and only the network -- gets wider. Point features,
        # background and the rasteriser all stay at `cfg.feature_channels`.
        # `GaussianInputs.build` also resolves `cfg.hybrid.depth_scale` in place, so
        # `save_checkpoint`'s `cfg.to_dict()` records the exact normaliser used.
        self.hybrid: GaussianInputs | None = None
        if cfg.hybrid.enabled:
            self.hybrid = GaussianInputs.build(cfg.hybrid, list(self.dataset.names))
            n_with_render = len(self.hybrid.available_names(list(self.dataset.names)))
            self._log(
                f"hybrid design A: mode={cfg.hybrid.mode} channels={cfg.hybrid.channels} "
                f"(+{self.hybrid.num_channels} net input channels), depth_scale="
                f"{cfg.hybrid.depth_scale}, renders for {n_with_render}/{len(self.dataset.names)} images"
            )
        # Set by callers that render poses no photo exists for (see `gaussian_for_pose`);
        # `trippy.train.eval.build_trainer_from_checkpoint` installs a lazy live-gsrender
        # provider here, and `trippy.render.candidate.render_candidate` can override it.
        self.gaussian_provider: Callable[..., torch.Tensor | None] | None = None

        network_cfg = NetworkConfig(num_input_channels=cfg.net_input_channels, num_layers=cfg.layers)
        self.net = MultiScaleUnet2dDecOnlySmallFixed(network_cfg).to(self.device)

        first_item = self.dataset[0]
        full_h, full_w = int(first_item["rgb"].shape[0]), int(first_item["rgb"].shape[1])
        self.camera = NeuralCamera(
            image_height=full_h,
            image_width=full_w,
            num_frames=len(self.dataset.names),
            initial_exposure=self._initial_exposure(),
        ).to(self.device)
        # TRIPS freezes vignette + white-balance by default (fix_vignette=true, fix_wb=true;
        # docs/TRIPS_REFERENCE.md Sec. 7) even though both are applied in the forward pass;
        # exposure and response are NOT in that fix_* list, so they stay trainable below.
        if self.camera.vignette_net is not None:
            for p in self.camera.vignette_net.parameters():
                p.requires_grad_(False)
        if self.camera.white_balance_values is not None:
            self.camera.white_balance_values.requires_grad_(False)

        self.loss_fn = TripsLoss(
            LossWeights(vgg=0.0, l1=cfg.loss_l1, ssim=cfg.loss_ssim, lpips=cfg.loss_lpips)
        ).to(self.device)
        # Gated on cfg.eval_lpips (default True) so a caller that wants CPU tests to never
        # touch a (possibly network-fetched) VGG backbone can opt out cleanly.
        self._eval_lpips = _LazyLPIPS(net=TRAIN_LPIPS_METRIC_NET) if cfg.eval_lpips else None

        group_specs: list[tuple[str, list[nn.Parameter], float]] = [
            ("points", [self.point_params.xyz], cfg.lr_points),
            ("size", [self.point_params.raw_size], cfg.lr_size),
            ("conf", [self.point_params.raw_conf], cfg.lr_confidence),
            ("texture", [self.point_params.feat], cfg.lr_texture),
            ("background", [self.background], cfg.lr_background),
            ("poses", [self.pose_params.delta], cfg.lr_poses),
            ("network", list(self.net.parameters()), cfg.lr_network),
        ]
        if self.camera.exposures_values is not None:
            group_specs.append(("exposure", [self.camera.exposures_values], cfg.lr_exposure))
        if self.camera.camera_response is not None:
            group_specs.append(("response", list(self.camera.camera_response.parameters()), cfg.lr_response))
        self._group_index = {name: i for i, (name, _, _) in enumerate(group_specs)}
        self.optimizer = torch.optim.Adam([{"params": params, "lr": lr} for _, params, lr in group_specs])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=cfg.lr_decay_factor, patience=cfg.lr_decay_patience
        )

        self._rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.epoch = 0
        self.global_step = 0

        # Point removal bookkeeping (trippy.train.prune). `_shade_views` is the audit
        # region, built lazily from the COLMAP model the first time a prune pass or an
        # eval needs it and memoised for the rest of the run (a full points3D parse);
        # `_shade_region_error` records why it could not be built, so a scene whose
        # shade frames are not registered logs a reason instead of raising.
        self.points_removed_total = 0
        self._shade_views: list[prune.ShadeView] | None = None
        self._shade_region_error: str | None = None

        # Checkpoint retention bookkeeping (trippy.train.retention): `_best_epoch`/
        # `_best_psnr` track the best held-out PSNR seen by ANY `evaluate()` call, updated
        # there; `save_checkpoint` only promotes a checkpoint to checkpoint_best.pt when it
        # is saving the exact epoch that currently holds that record (see save_checkpoint's
        # docstring for what happens when eval_every != checkpoint_every).
        self._best_epoch: int | None = None
        self._best_psnr: float = float("-inf")

    # --- construction helpers ---

    def _initial_exposure(self) -> torch.Tensor:
        """Per-image EV_log2 from cached EXIF, **relative to the scene mean**.

        Simplified from TRIPS's full formula (docs/TRIPS_REFERENCE.md Sec.
        6: `log2(FNumber^2 / ExposureTime) + log2(ISO/100) -
        ExposureBiasValue`) -- `trippy.scene.dataset`'s EXIF reader only
        extracts ExposureTime and ISO (no FNumber/ExposureBiasValue), so
        the f-number and bias terms are omitted here.

        The scene mean is subtracted because that is what TRIPS does and
        because `NeuralCamera` applies the value as a *gain*,
        `x = x * 2 ** -EV`: `colmap2adop.cpp:105` stores
        `dataset.ini`'s `scene_exposure_value = mean(EV over all images)`
        and `NeuralScene.cpp:38` initialises the per-frame exposure as
        `f.exposure_value - scene_exposure_value`. So the initialisation
        only ever encodes *relative* brightness differences between
        images, and the average image starts at gain 1. Absolute EVs
        (kk-coherent: mean 6.14, i.e. gain 2**-6.14 = 1/70) would instead
        divide every prediction by ~70 before the response LUT, which no
        amount of training at `lr_exposure=5e-4` can undo -- the bug that
        produced the 1.6 dB EXP-0003 smoke run (research/trips-metal.md,
        2026-09-06).

        **Images with no usable EXIF get the scene mean (relative EV 0),
        not absolute 0.** TRIPS's `colmap2adop` writes a literal `0` into
        `exposure.txt` for a missing-EXIF image (`colmap2adop.cpp:32-36`),
        which is harmless there because its scenes' `scene_exposure_value`
        is itself 0. Here the scene mean is ~5.87 EV, so the old
        `ev = 0.0` fallback produced a *relative* EV of -5.87, i.e. a gain
        of `2 ** 5.87 = 58.5x` -- and a held-out frame's exposure is never
        trained, so that 58x stuck for the whole run. On kk-coherent 10 of
        219 images have no EXIF, 6 of them land in the held-out split, and
        all 6 scored 6.2-6.9 dB in EXP-0003 full2-broadcast (vs 16.9 dB
        mean over the 27 held-out frames that do have EXIF) -- see
        `experiments/EXP-0003-kk-trips-train/README.md` "The exposure
        artefact". Falling back to the mean makes a missing-EXIF image
        start at gain 1, the neutral place `NeuralCamera`'s own zero-init
        would put it.
        """
        meta = json.loads((self.dataset.cache_dir / SCENE_CACHE_META_FILENAME).read_text())["images"]
        values: list[float | None] = []
        for name in self.dataset.names:
            info = meta.get(name, {})
            exposure_time = info.get("exposure_time")
            iso = info.get("iso")
            if exposure_time and exposure_time > 0 and iso and iso > 0:
                values.append(math.log2(1.0 / exposure_time) + math.log2(iso / 100.0))
            else:
                values.append(None)  # no usable EXIF -> scene mean, i.e. relative 0 (see docstring)
        if not values:
            return torch.zeros(0, dtype=torch.float32)
        known = [v for v in values if v is not None]
        mean = sum(known) / len(known) if known else 0.0
        n_missing = len(values) - len(known)
        if n_missing:
            self._log(
                f"exposure init: {n_missing}/{len(values)} images have no usable EXIF "
                f"(ExposureTime/ISO); they start at the scene mean EV {mean:.3f}, i.e. gain 1.0"
            )
        return torch.tensor([mean if v is None else v for v in values], dtype=torch.float32) - mean

    def _apply_locks(self, epoch: int) -> None:
        """Freeze pose deltas / xyz+size during their respective lock periods (see module docstring)."""
        locked_cameras = epoch < self.cfg.lock_cameras_epochs
        locked_structure = epoch < self.cfg.lock_structure_epochs
        self.pose_params.delta.requires_grad_(not locked_cameras)
        self.point_params.xyz.requires_grad_(not locked_structure)
        self.point_params.raw_size.requires_grad_(not locked_structure)

    def _extent_penalty(self) -> torch.Tensor:
        """Soft AABB penalty keeping xyz near the initial point cloud's bbox (docs/SPEC.md "extent inflation")."""
        bbox_min, bbox_max = self.point_params.bbox_min, self.point_params.bbox_max
        center = (bbox_min + bbox_max) / 2.0
        half_extent = (bbox_max - bbox_min) / 2.0 * (1.0 + self.cfg.extent_margin)
        excess = torch.relu((self.point_params.xyz - center).abs() - half_extent)
        return (excess * excess).mean()

    def _pose_for(self, item: dict, frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        R0 = xform_b.qvec2R(item["qvec"])
        t0 = item["tvec"]
        return self.pose_params.compose_pose(frame_index, R0, t0)

    def _render(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        image_hw: tuple[int, int],
        gaussian: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """Render + decode one image: raster pyramid -> U-Net. Returns (net_out, layers, aux).

        `gaussian` is design A's `(G, H, W)` Gaussian block at level-0
        resolution (None = zeros, or no hybrid at all); `layers` is always the
        *TRIPS-only* pyramid, so every honesty artifact ("raw L0", coverage)
        keeps meaning exactly what it meant before design A.
        """
        layers, aux = render_pyramid(
            self.point_params.xyz,
            self.point_params.size(),
            self.point_params.feat,
            self.point_params.conf(),
            K,
            R,
            t,
            image_hw,
            num_layers=self.cfg.layers,
            mode=self.cfg.mode,
            bg=self.background,
            pixel_center=self.cfg.pixel_center,
            pyramid_halving=self.cfg.pyramid_halving,
        )
        inputs = [layer.unsqueeze(0) for layer in layers]
        if self.hybrid is not None:
            inputs = self.hybrid.attach(inputs, gaussian)
        net_out = self.net(inputs)
        return net_out, layers, aux

    def _tone_map(self, net_out: torch.Tensor, frame_index: int) -> torch.Tensor:
        frame_index_t = torch.tensor([frame_index], device=self.device, dtype=torch.long)
        return self.camera(net_out, frame_index_t)

    def gaussian_for_pose(
        self,
        name: str | None,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        """Design A's Gaussian block for an **arbitrary** pose, or None (-> zeros).

        Deliberately does NOT fall back to `name`'s precomputed render. A
        `CameraPose.image_name` means "anchored to that image" -- every dolly
        and off-path pose is *displaced* from the photographed one
        (`trippy.render.offpath.offpath_poses`, `trippy.render.dolly`), so
        reusing that image's render here would feed the network a Gaussian
        image taken from the wrong camera. The precomputed renders are used
        only where the pose genuinely is the image's own: `evaluate()`, which
        calls `self.hybrid.frame` directly.

        `name` is still passed through to the provider (it may want it for
        logging or for a caller-supplied exact-pose cache); resolution is
        otherwise: `self.gaussian_provider` if a caller installed one
        (`trippy.hybrid.gsrender_live.gaussian_provider_for` renders the PLY
        live at this exact pose), else None -- an honest all-zero block, the
        TRIPS-only state `hybrid.dropout_gaussian_p` trained the network to
        survive. Always None when the run is not hybrid, so non-hybrid callers
        can invoke this unconditionally.
        """
        if self.hybrid is None or self.gaussian_provider is None:
            return None
        return self.gaussian_provider(name, K, R, t, image_hw)

    def render_at_pose(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        image_hw: tuple[int, int],
        frame_index: int = 0,
        image_name: str | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
        """Public render+tone-map at an arbitrary pose (no pose-delta refinement applied).

        Unlike `train_step`/`evaluate`, `R`/`t` are used exactly as given
        (no `PoseParams.compose_pose` lookup) -- this is for poses that
        don't correspond to any training image index, e.g. off-path
        honesty renders (`trippy.train.eval.render_offpath`) or a future
        dolly camera path.

        Args:
            K, R, t: pinhole intrinsics / world->camera pose.
            image_hw: (H, W) to render at.
            frame_index: which trained per-image exposure/response
                parameters to apply (default 0, an arbitrary but
                deterministic stand-in when there is no natural frame).
            image_name: the registered image this pose is *anchored to*, if
                any -- passed straight through to design A's
                `gaussian_for_pose` (which renders the Gaussian block live at
                this pose; it never substitutes that image's own precomputed
                render, see there). Ignored when not hybrid.

        Returns:
            (pred, layers, aux): `pred` is the toned-mapped (1, 3, H', W')
            image (see `_render` for the odd-size caveat); `layers`/`aux`
            are `render_pyramid`'s raw pyramid outputs.
        """
        gaussian = self.gaussian_for_pose(image_name, K, R, t, image_hw)
        net_out, layers, aux = self._render(K, R, t, image_hw, gaussian=gaussian)
        pred = self._tone_map(net_out, frame_index)
        return pred, layers, aux

    def _sample_zoom(self) -> float:
        lo, hi = self.cfg.zoom_min, self.cfg.zoom_max
        return lo + torch.rand((), generator=self._rng).item() * (hi - lo)

    def _sample_crop_center(self, height: int, width: int, zoom: float) -> tuple[float, float]:
        """Uniform random centre such that the `crop/zoom` window stays inside the frame.

        TRIPS's `RandomImageCrop` (`Dataset.cpp:264`) samples a crop that
        lies within the source image; `crop_prefere_border=true`
        (docs/TRIPS_REFERENCE.md Sec. 7) only biases *where* inside it
        lands. trippy samples uniformly inside instead of biasing, and
        falls back to the frame centre on whichever axis the window is
        larger than the image (`trippy.scene.dataset.crop`'s validity mask
        then covers the unavoidable overshoot).

        Sampling the centre over the *whole* frame -- what this method used
        to do -- makes roughly half of every crop's pixels padding
        (mask == 0): they cost a full rasterisation and contribute nothing
        to the loss, halving the useful signal per step.
        """
        window = self.cfg.crop / zoom

        def _axis(extent: int) -> float:
            low, high = window / 2.0, extent - window / 2.0
            if high <= low:  # window wider than the image on this axis: centre it
                return extent / 2.0
            return low + torch.rand((), generator=self._rng).item() * (high - low)

        return _axis(width), _axis(height)

    # --- training ---

    def train_step(
        self,
        name: str | None = None,
        zoom: float | None = None,
        center: tuple[float, float] | None = None,
    ) -> dict:
        """Sample one crop, render, tone-map, backprop, step.

        Args:
            name, zoom, center: override the random sampling (used by tests
                that need a deterministic, repeatable crop); None samples
                per the trainer's own schedule.

        Returns:
            A dict of scalar metrics for this step (also appended to
            `metrics.jsonl`).
        """
        self.net.train()
        self.camera.train()

        if name is None:
            pick = torch.randint(0, len(self.train_names), (), generator=self._rng).item()
            name = self.train_names[pick]
        frame_index = self._name_to_index[name]
        item = self.dataset[frame_index]
        height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])

        zoom = self._sample_zoom() if zoom is None else zoom
        center = self._sample_crop_center(height, width, zoom) if center is None else center
        cropped = dataset_crop(item, size=self.cfg.crop, zoom=zoom, center=center)

        target = cropped["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        mask = cropped["mask"].unsqueeze(0).unsqueeze(0)

        # Design A: the Gaussian render is cropped through the *same* function with the
        # *same* (size, zoom, center), so its K-adjust is identical to the photo's by
        # construction (tests/test_hybrid_a_crop.py). `dropped` is ablation 1: a fraction
        # of crops see zeros instead, so the net cannot rely on the Gaussians everywhere.
        gaussian = None
        dropped = False
        if self.hybrid is not None:
            dropped = self.hybrid.should_drop(self._rng)
            if not dropped:
                gaussian = self.hybrid.crop_frame(name, item, self.cfg.crop, zoom, center)

        R, t = self._pose_for(item, frame_index)
        net_out, _layers, _aux = self._render(
            cropped["K"], R, t, (self.cfg.crop, self.cfg.crop), gaussian=gaussian
        )
        pred = self._tone_map(net_out, frame_index)

        target = _center_crop_like(target, pred.shape[-2], pred.shape[-1])
        mask = _center_crop_like(mask, pred.shape[-2], pred.shape[-1])

        image_loss = self.loss_fn(pred, target, mask)
        extent_penalty = self._extent_penalty()
        camera_reg = self.camera.regularizer()
        total = image_loss + self.cfg.extent_penalty_weight * extent_penalty + camera_reg

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nonfinite_grads = self._sanitise_gradients()
        self.optimizer.step()
        self.camera.apply_constraints()

        self.global_step += 1
        record = {
            "step": self.global_step,
            "epoch": self.epoch,
            "image": name,
            "zoom": float(zoom),
            "loss": float(total.detach().item()),
            "image_loss": float(image_loss.detach().item()),
            "extent_penalty": float(extent_penalty.detach().item()),
            "camera_reg": float(camera_reg.detach().item()),
            "nonfinite_grads": int(nonfinite_grads.item()),
        }
        if self.hybrid is not None:
            record["gaussian_dropped"] = bool(dropped)
            record["gaussian_present"] = bool(gaussian is not None)
        self._append_metrics(record)
        return record

    def _sanitise_gradients(self) -> torch.Tensor:
        """Zero every non-finite gradient entry before the optimizer step; return how many.

        Containment, not a cure: a single degenerate fragment can make the
        rasteriser's backward emit a NaN gradient for one point (see
        docs/LIMITATIONS.md, "NaN gradient out of the rasteriser backward" --
        the root cause lives in `trippy.raster`, outside this module).
        Adam turns one NaN gradient into a permanently NaN parameter, and
        because `_extent_penalty` reduces over *all* points, one NaN `xyz`
        row then makes the reported total loss NaN for the rest of the run
        while the images still look fine -- exactly the kind of silent
        corruption that must not reach a checkpoint. Zeroing the offending
        entries keeps the run alive and puts the count in `metrics.jsonl`
        so it is visible rather than silent.
        """
        count: torch.Tensor | None = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                bad = ~torch.isfinite(param.grad)
                count = bad.sum() if count is None else count + bad.sum()
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return count if count is not None else torch.zeros((), dtype=torch.long, device=self.device)

    # --- evaluation ---

    def _interpolated_camera_override(
        self, frame_index: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Exposure/WB override for `frame_index`, interpolated from its nearest TRAIN neighbours.

        `exposure_mode="neighbours"`'s own building block (`trippy.net.camera_model.
        interpolate_from_train_neighbours`, porting `NeuralCameraImpl::InterpolateFromNeighbors`,
        `NeuralCamera.cpp:481-520`): never reads `frame_index`'s own photo, only the current
        (possibly still-initial, for a frame never sampled by `train_step`) rows of every
        TRAINING frame. Read-only -- `.detach()`d inputs, nothing written back to the module,
        same "never touches the checkpoint" contract as `calibrate_frame`.

        Returns:
            `(exposure_override, white_balance_override)`, each shaped like a single row of
            the corresponding `NeuralCamera` parameter with a leading batch dim of 1 (ready for
            `NeuralCamera.forward_with`'s `exposure`/`white_balance` kwargs), or `None` when
            that parameter is disabled (`enable_exposure=False` / `enable_white_balance=False`).
        """
        exposure_override = None
        if self.camera.exposures_values is not None:
            exposure_override = interpolate_from_train_neighbours(
                self.camera.exposures_values.detach(), self._is_train_frame, frame_index
            ).unsqueeze(0)
        white_balance_override = None
        if self.camera.white_balance_values is not None:
            white_balance_override = interpolate_from_train_neighbours(
                self.camera.white_balance_values.detach(), self._is_train_frame, frame_index
            ).unsqueeze(0)
        return exposure_override, white_balance_override

    def calibrate_frame(
        self,
        net_out: torch.Tensor,
        target: torch.Tensor,
        frame_index: int,
        mask: torch.Tensor | None = None,
        steps: int | None = None,
        lr: float | None = None,
        calibrate_white_balance: bool | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Fit ONLY this frame's exposure (+ optionally white balance) to its own photo.

        Why this is legitimate, and what it is not:

        - A held-out image's row of `NeuralCamera.exposures_values` never
          receives a gradient during training (only sampled train frames
          do), so at eval it still carries its EXIF initialisation. That
          scalar is a property of the *camera that took the photo*, not of
          the scene reconstruction, and getting it wrong costs PSNR that
          has nothing to do with geometry or with the network.
        - TRIPS ships exactly this knob: `optimize_eval_camera` runs a
          per-epoch "EvalRefine" pass over the **test** crops that steps
          the camera and pose optimisers with texture/network frozen
          (`third_party/TRIPS/src/apps/train.cpp:591-596, 693-697`;
          `NeuralScene.cpp:1473-1503`), and `interpolate_eval_settings`
          instead copies a test frame's exposure/WB from its neighbouring
          train frames (`NeuralCamera.cpp:481-520`). Both default to false
          in `configs/train_normalnet.ini:48-49` and in the released
          checkpoint, which is why trippy's own default is off too.
        - trippy's version is strictly weaker than TRIPS's: only the
          photometric scalars move. The point cloud, the poses, the U-Net,
          the vignette and the response LUT are all untouched, nothing is
          written back into this module (the fitted tensors are local, see
          `NeuralCamera.forward_with`), and the number is always reported
          *next to* the uncalibrated one, never instead of it.
        - It does use the held-out photo, so a calibrated PSNR is not a
          pure novel-view number. It answers one question only: how much
          of the gap is photometric.

        Args:
            net_out: (1, 3, H, W) pre-tone-map network output for this frame.
            target: (1, 3, H, W) ground-truth photo, same size.
            frame_index: dataset index of the frame (selects the starting
                exposure/WB row).
            mask: (1, 1, H, W) validity mask, or None (all valid).
            steps: Adam steps (default `cfg.eval_calibrate_steps`).
            lr: Adam learning rate (default `cfg.eval_calibrate_lr`).
            calibrate_white_balance: also fit the red/blue white-balance
                gains (green is pinned to its reference, exactly as
                `NeuralCamera.apply_constraints` does during training).
                Default `cfg.eval_calibrate_white_balance`.

        Returns:
            `(pred, info)`: the tone-mapped prediction with the fitted
            values applied, and `{"exposure_before", "exposure_after",
            "exposure_delta_ev", "gain_before", "gain_after",
            "white_balance", "exposure_warm_start", "steps", "l1_before",
            "l1_after"}` -- `l1_before` is measured at the warm start, not
            at the stored exposure.
        """
        steps = self.cfg.eval_calibrate_steps if steps is None else steps
        lr = self.cfg.eval_calibrate_lr if lr is None else lr
        if calibrate_white_balance is None:
            calibrate_white_balance = self.cfg.eval_calibrate_white_balance

        index_t = torch.tensor([frame_index], device=net_out.device, dtype=torch.long)
        net_out = net_out.detach()
        target = target.detach()

        if self.camera.exposures_values is not None:
            start = self.camera.exposures_values[index_t].detach().clone()
        else:
            start = torch.zeros(1, 1, 1, 1, device=net_out.device, dtype=net_out.dtype)

        # Warm start: Adam moves an exposure by at most ~lr per step, so a frame that starts
        # 5.87 EV (58x) off -- kk-coherent's missing-EXIF frames -- would need more steps than
        # the budget. `best_global_gain` on the *uncalibrated* prediction says how much
        # brighter/darker the output has to get; -log2(g) converts that into EV. It is exact
        # only if the response LUT were linear, so it is a starting point, not the answer:
        # Adam still runs from there.
        with torch.no_grad():
            pred0 = self.camera.forward_with(net_out, index_t)
            gain0 = best_global_gain(pred0, target, mask)
        if gain0 > 0.0:
            start = start - math.log2(gain0)
        exposure = start.clone().requires_grad_(True)
        params = [exposure]

        white_balance = None
        wb_reference = None
        if calibrate_white_balance:
            if self.camera.white_balance_values is not None:
                wb_start = self.camera.white_balance_values[index_t].detach().clone()
            else:
                wb_start = torch.ones(1, 3, 1, 1, device=net_out.device, dtype=net_out.dtype)
            wb_reference = wb_start.clone()
            white_balance = wb_start.clone().requires_grad_(True)
            params.append(white_balance)

        def _predict() -> torch.Tensor:
            return self.camera.forward_with(
                net_out, index_t, exposure=exposure, white_balance=white_balance
            )

        with torch.no_grad():
            l1_before = float(l1_loss(_predict(), target, mask).item())

        if steps > 0:
            optimizer = torch.optim.Adam(params, lr=lr)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = l1_loss(_predict(), target, mask)
                loss.backward()
                optimizer.step()
                if white_balance is not None and wb_reference is not None:
                    # NeuralCamera.apply_constraints: green is pinned to its reference.
                    with torch.no_grad():
                        white_balance[:, 1:2] = wb_reference[:, 1:2]

        with torch.no_grad():
            pred = _predict()
            l1_after = float(l1_loss(pred, target, mask).item())

        before_ev = (
            float(self.camera.exposures_values[index_t].detach().reshape(-1)[0].item())
            if self.camera.exposures_values is not None
            else 0.0
        )
        after_ev = float(exposure.detach().reshape(-1)[0].item())
        info = {
            "exposure_before": before_ev,
            "exposure_after": after_ev,
            "exposure_delta_ev": after_ev - before_ev,
            "gain_before": float(2.0**-before_ev),
            "gain_after": float(2.0**-after_ev),
            "white_balance": (
                [float(v) for v in white_balance.detach().reshape(-1)] if white_balance is not None else None
            ),
            "exposure_warm_start": float(start.reshape(-1)[0].item()),
            "steps": int(steps),
            "l1_before": l1_before,
            "l1_after": l1_after,
        }
        return pred.detach(), info


    def evaluate(
        self,
        names: list[str] | None = None,
        epoch: int | None = None,
        eval_dirname: str | None = None,
        calibrate: bool | None = None,
        exposure_mode: str | None = None,
    ) -> dict:
        """Full-frame held-out evaluation: PSNR/SSIM(/LPIPS) + an honesty contact sheet.

        Args:
            names: image names to evaluate; defaults to `self.heldout_names`.
            epoch: label used for the output directory / metrics record;
                defaults to `self.epoch`.
            eval_dirname: output directory name under `run_dir`; defaults to
                `TRAIN_EVAL_DIRNAME_FMT.format(epoch=epoch)` (what every
                mid-training/`--report` eval has always used).
                `trippy.train.eval.evaluate_checkpoint` passes a
                `eval_manual_<timestamp>` name instead, so a standalone
                `trippy eval --checkpoint` re-run never collides with the
                checkpoint's own epoch directory.
            calibrate: also report *test-time photometrically calibrated*
                metrics -- per image, this frame's exposure (and, with
                `cfg.eval_calibrate_white_balance`, its red/blue white
                balance) is fitted to its own photo by
                `calibrate_frame` while everything else stays frozen.
                Defaults to `cfg.eval_calibrate_camera` (False), so no
                training-time eval changes. The strict numbers are always
                computed and reported as well; the calibrated ones land in
                separate keys and never overwrite them. Independent of
                `exposure_mode` below (see that arg for how the two relate).
            exposure_mode: which exposure/white-balance a HELD-OUT image's
                "_eval"-suffixed numbers (below) are computed with --
                `trippy.constants.EVAL_EXPOSURE_MODES`
                ("own"/"neighbours"/"calibrate"). Defaults to
                `cfg.eval_exposure_mode` ("neighbours"). A name that is one
                of `self.train_names` always uses "own" regardless of this
                setting (TRIPS's own `InterpolateFromNeighbors` is likewise
                only ever called on `not_training_indices` --
                `NeuralCamera.cpp:481-520`, `train.cpp:1604-1611`). This
                setting never changes the plain `psnr`/`ssim`/`lpips`/
                `psnr_mean`/`shade`/`other` fields below -- those always use
                each frame's own raw, never-trained exposure, exactly as
                before this feature existed.

        Returns:
            {"epoch", "n_images", "psnr_mean", "ssim_mean", "lpips_mean"
            (None if `cfg.eval_lpips` is False), "names", "per_image"
            (`{name: {"psnr", "ssim", "lpips"}}`, one entry per evaluated
            image), "shade" and "other" (`{"n", "psnr", "ssim", "lpips"}`,
            the `_shade_and_other` split of `names` -- "shade" is
            `cfg.forced_heldout` when non-empty, else `SHADE_FRAMES_KK`,
            intersected with `names`)}. Every `per_image` entry also
            carries the exposure diagnostics (no training, no extra
            render, always computed from each frame's own raw exposure
            regardless of `exposure_mode`): "pred_mean"/"target_mean"/
            "brightness_ratio" (mean photo brightness over mean predicted
            brightness -- a mis-set per-image gain shows up here directly),
            "gain_best" (the closed-form global gain of `best_global_gain`)
            and "psnr_gain" (PSNR after applying it), plus "exposure_ev"/
            "exposure_gain", the frame's own exposure (whether or not it
            was used for this row's headline number). With `calibrate` on,
            "psnr_calibrated"/"ssim_calibrated"/"lpips_calibrated"/
            "calibration" are added per image and "psnr_mean_calibrated",
            "shade_calibrated" and "other_calibrated" at the top level
            (`calibrated: false` and no such keys otherwise).

            "exposure_mode" (top level): the resolved mode this call used.
            Per image, `exposure_mode`, `psnr_eval`, `ssim_eval`,
            `lpips_eval` -- the headline number under this feature: for a
            TRAINING-set name this is always identical to `psnr`/`ssim`/
            `lpips` ("own"); for a held-out name it is computed under the
            resolved `exposure_mode` ("own": identical to the plain fields;
            "neighbours": `_interpolated_camera_override`'s exposure/WB;
            "calibrate": `calibrate_frame`'s fit, reusing the same fit
            `calibrate=True` above would have produced, computed at most
            once per image either way). "psnr_mean_eval", "shade_eval" and
            "other_eval" aggregate these at the top level, always (this is
            the number a caller should treat as "the" held-out PSNR under
            this feature -- see `trippy.constants` "eval_exposure_mode").

            Also writes `<run_dir>/<eval_dirname>/metrics.json` (the full
            dict above) and, for up to `cfg.eval_max_images` images
            (forced-held-out shade frames first), `sheet.jpg` (JPEG quality
            `TRAIN_EVAL_SHEET_JPEG_QUALITY`, not PNG -- see module-level
            `_save_eval_sheet_jpeg`): photo | render | raw level-0 |
            coverage, one row per image (docs/EXPERIMENTS.md "Mandatory
            honesty sheet"). Also updates `self._best_epoch`/
            `self._best_psnr` when this eval's `psnr_mean` beats every
            prior eval -- `save_checkpoint` reads those to decide whether
            to write `checkpoint_best.pt`. Always appends an `{"eval":
            True, ...}` row (same shape, minus "names") to `metrics.jsonl`,
            so `trippy.render.leaderboard` picks up the shade split from
            the most recent call to this method, whether that call came
            from mid-training, `--report`, or a standalone `trippy eval
            --checkpoint`.
        """
        self.net.eval()
        self.camera.eval()
        names = list(self.heldout_names) if names is None else list(names)
        epoch = self.epoch if epoch is None else epoch
        calibrate = self.cfg.eval_calibrate_camera if calibrate is None else calibrate
        exposure_mode = self.cfg.eval_exposure_mode if exposure_mode is None else exposure_mode
        if exposure_mode not in EVAL_EXPOSURE_MODES:
            raise ValueError(f"exposure_mode must be one of {EVAL_EXPOSURE_MODES}, got {exposure_mode!r}")

        forced = set(self.cfg.forced_heldout)
        sheet_names = sorted(names, key=lambda n: (n not in forced, n))[: self.cfg.eval_max_images]

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        lpips_vals: list[float] = []
        psnr_cal_vals: list[float] = []
        psnr_eval_vals: list[float] = []
        ssim_eval_vals: list[float] = []
        lpips_eval_vals: list[float] = []
        per_image: dict[str, dict] = {}
        sheet_images: list[np.ndarray] = []
        sheet_labels: list[str] = []

        with torch.no_grad():
            for name in names:
                frame_index = self._name_to_index[name]
                item = self.dataset[frame_index]
                height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
                target = item["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                mask = torch.ones((1, 1, height, width), device=self.device)

                R, t = self._pose_for(item, frame_index)
                # Renders exist on disk for every registered image (held-out included),
                # so held-out eval never needs a live gsrender pass.
                gaussian = self.hybrid.frame(name, (height, width)) if self.hybrid is not None else None
                net_out, layers, aux = self._render(
                    item["K"], R, t, (height, width), gaussian=gaussian
                )
                pred = self._tone_map(net_out, frame_index)

                target_c = _center_crop_like(target, pred.shape[-2], pred.shape[-1])
                mask_c = _center_crop_like(mask, pred.shape[-2], pred.shape[-1])

                # `mse_loss` averages over every (channel, pixel) element the mask keeps.
                # Dividing a 3-channel error sum by a 1-channel mask sum (what this used
                # to do) inflates the MSE 3x and costs exactly 4.77 dB of reported PSNR.
                mse = mse_loss(pred, target_c, mask_c)
                psnr_val = float((-10.0 * torch.log10(mse + TRAIN_PSNR_EPS)).item())
                ssim_val = float(ssim(pred, target_c, mask_c).item())
                lpips_val = (
                    float(self._eval_lpips(pred, target_c, mask_c).item())
                    if self._eval_lpips is not None
                    else None
                )
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)
                if lpips_val is not None:
                    lpips_vals.append(lpips_val)

                # Exposure diagnostics: free (no render, no training), always recorded --
                # see `best_global_gain` and docs/EXPERIMENTS.md "Per-image exposure
                # diagnostics".
                pred_mean = _masked_mean_value(pred, mask_c)
                target_mean = _masked_mean_value(target_c, mask_c)
                gain_best = best_global_gain(pred, target_c, mask_c)
                exposure_ev = (
                    float(self.camera.exposures_values[frame_index].detach().reshape(-1)[0].item())
                    if self.camera.exposures_values is not None
                    else 0.0
                )
                per_image[name] = {
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    "pred_mean": pred_mean,
                    "target_mean": target_mean,
                    "brightness_ratio": (target_mean / pred_mean) if pred_mean > 0 else None,
                    "gain_best": gain_best,
                    "psnr_gain": _psnr(pred * gain_best, target_c, mask_c),
                    "exposure_ev": exposure_ev,
                    "exposure_gain": float(2.0**-exposure_ev),
                }

                # Headline "_eval" numbers (docs/EXPERIMENTS.md "Test-time camera
                # calibration"): a training-set name always uses its own (never-overridden)
                # exposure, exactly like TRIPS's InterpolateFromNeighbors only ever touching
                # `not_training_indices`; a held-out name uses the resolved `exposure_mode`.
                row_mode = "own" if name in self._train_name_set else exposure_mode

                # `calibrate_frame` is expensive (an Adam loop) -- run it at most once per
                # image even when both the legacy `calibrate` side-column and
                # `exposure_mode="calibrate"`'s headline number need it.
                pred_cal: torch.Tensor | None = None
                cal_info: dict | None = None
                if calibrate or row_mode == "calibrate":
                    with torch.enable_grad():
                        pred_cal, cal_info = self.calibrate_frame(net_out, target_c, frame_index, mask=mask_c)
                    pred_cal = _center_crop_like(pred_cal, pred.shape[-2], pred.shape[-1])

                if calibrate:
                    psnr_cal = _psnr(pred_cal, target_c, mask_c)
                    per_image[name].update(
                        {
                            "psnr_calibrated": psnr_cal,
                            "ssim_calibrated": float(ssim(pred_cal, target_c, mask_c).item()),
                            "lpips_calibrated": (
                                float(self._eval_lpips(pred_cal, target_c, mask_c).item())
                                if self._eval_lpips is not None
                                else None
                            ),
                            "calibration": cal_info,
                        }
                    )
                    psnr_cal_vals.append(psnr_cal)

                if row_mode == "own":
                    eval_pred = pred
                elif row_mode == "neighbours":
                    exposure_override, wb_override = self._interpolated_camera_override(frame_index)
                    frame_index_t = torch.tensor([frame_index], device=self.device, dtype=torch.long)
                    eval_pred = self.camera.forward_with(
                        net_out, frame_index_t, exposure=exposure_override, white_balance=wb_override
                    )
                    eval_pred = _center_crop_like(eval_pred, pred.shape[-2], pred.shape[-1])
                else:  # "calibrate"
                    eval_pred = pred_cal

                psnr_eval = _psnr(eval_pred, target_c, mask_c)
                ssim_eval = float(ssim(eval_pred, target_c, mask_c).item())
                lpips_eval = (
                    float(self._eval_lpips(eval_pred, target_c, mask_c).item())
                    if self._eval_lpips is not None
                    else None
                )
                per_image[name].update(
                    {
                        "exposure_mode": row_mode,
                        "psnr_eval": psnr_eval,
                        "ssim_eval": ssim_eval,
                        "lpips_eval": lpips_eval,
                    }
                )
                psnr_eval_vals.append(psnr_eval)
                ssim_eval_vals.append(ssim_eval)
                if lpips_eval is not None:
                    lpips_eval_vals.append(lpips_eval)

                if name in sheet_names:
                    raw = layers[0][:3].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    coverage = (1.0 - aux["t_final"][0]).clamp(0.0, 1.0).cpu().numpy()
                    coverage_rgb = colorize(coverage, 0.0, 1.0)
                    photo_np = target_c[0].permute(1, 2, 0).cpu().numpy()
                    pred_np = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    sheet_images += [photo_np, pred_np, raw, coverage_rgb]
                    sheet_labels += [f"{name} photo", "render", "raw L0", "coverage"]

        shade_names, other_names = _shade_and_other(names, self.cfg.forced_heldout)

        metrics = {
            "epoch": epoch,
            "n_images": len(names),
            "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
            "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
            "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
            "names": names,
            "per_image": per_image,
            "shade": _aggregate_group(shade_names, per_image),
            "other": _aggregate_group(other_names, per_image),
            "calibrated": bool(calibrate),
            # Headline numbers under `exposure_mode` (trippy.constants "eval_exposure_mode"):
            # always populated (unlike the "_calibrated" block below, which is opt-in) since
            # "neighbours" is the resolved default -- see `evaluate`'s own docstring for how
            # this relates to the plain psnr_mean/shade/other above.
            "exposure_mode": exposure_mode,
            "psnr_mean_eval": float(np.mean(psnr_eval_vals)) if psnr_eval_vals else 0.0,
            "ssim_mean_eval": float(np.mean(ssim_eval_vals)) if ssim_eval_vals else 0.0,
            "lpips_mean_eval": float(np.mean(lpips_eval_vals)) if lpips_eval_vals else None,
            "shade_eval": _aggregate_group(shade_names, per_image, suffix="_eval"),
            "other_eval": _aggregate_group(other_names, per_image, suffix="_eval"),
            # Point count + the shade audit's dark-mass fraction, computed in-process
            # from the current parameters (trippy.train.prune.dark_mass_stats), so a
            # removal run's effect on the metric is visible per eval rather than only
            # at export time. See `point_stats`.
            "points": self.point_stats(),
        }
        if calibrate:
            metrics["psnr_mean_calibrated"] = float(np.mean(psnr_cal_vals)) if psnr_cal_vals else 0.0
            metrics["shade_calibrated"] = _aggregate_group(shade_names, per_image, suffix="_calibrated")
            metrics["other_calibrated"] = _aggregate_group(other_names, per_image, suffix="_calibrated")

        eval_dirname = eval_dirname if eval_dirname is not None else TRAIN_EVAL_DIRNAME_FMT.format(epoch=epoch)
        eval_dir = self.run_dir / eval_dirname
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / TRAIN_EVAL_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))
        if sheet_images:
            sheet = contact_sheet(sheet_images, sheet_labels, cols=4)
            _save_eval_sheet_jpeg(eval_dir / TRAIN_EVAL_SHEET_JPEG_FILENAME, sheet, TRAIN_EVAL_SHEET_JPEG_QUALITY)

        if metrics["psnr_mean"] > self._best_psnr:
            self._best_psnr = metrics["psnr_mean"]
            self._best_epoch = epoch

        self._append_metrics({"eval": True, **{k: v for k, v in metrics.items() if k != "names"}})
        self._log(f"epoch {epoch}: eval psnr={metrics['psnr_mean']:.3f} ssim={metrics['ssim_mean']:.4f}")
        return metrics

    # --- point removal (trippy.train.prune) ---

    #: PointParams attribute name -> the optimiser group (`self._group_index`) it lives in.
    #: Every one of these is a per-point parameter whose first dimension is the point
    #: count, so removing points means index-selecting all four *and* their Adam moments.
    _POINT_PARAM_GROUPS = (("xyz", "points"), ("raw_size", "size"), ("raw_conf", "conf"), ("feat", "texture"))

    def shade_views(self) -> list[prune.ShadeView] | None:
        """The shade audit's region for this scene, built once and memoised.

        Built from `cfg.shade_prune.scene_txt` when set, otherwise from the
        run's own scene (`trippy.scene.dataset.resolve_sparse_dir`). Returns
        None -- and records why in `self._shade_region_error` -- when the
        model cannot be read or a configured shade frame is not registered
        in it, which is the normal case for every scene except kk-coherent.
        This never raises: measuring the audit statistic must not be able to
        kill a training run.
        """
        if self._shade_views is not None or self._shade_region_error is not None:
            return self._shade_views
        cfg = self.cfg.shade_prune
        try:
            sparse_dir = Path(cfg.scene_txt) if cfg.scene_txt else resolve_sparse_dir(self.cfg.scene_root)
            views = prune.build_shade_region(sparse_dir, cfg.frames, cfg.znear_frac, cfg.zfar_frac)
        except (OSError, ValueError, KeyError) as exc:
            self._shade_region_error = f"{type(exc).__name__}: {exc}"
            self._log(f"shade region unavailable ({self._shade_region_error}); dark-mass logging disabled")
            return None
        self._shade_views = views
        self._log(
            "shade region built from "
            + ", ".join(f"{v.name}(d={v.d:.2f}, z in [{v.znear:.2f},{v.zfar:.2f}], {v.nobs} obs)" for v in views)
        )
        return views

    def _point_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xyz, rgb, conf) as float64 numpy, in exactly the form `export_ply` writes them.

        `rgb` is `clip(feat[:, :3], 0, 1)` and `conf` is `sigmoid(10*raw_conf)`;
        `trippy.train.export.write_gaussian_ply` writes `(rgb - 0.5)/SH_C0`
        and `logit(conf)`, and `depthprior_shade_audit.py` inverts both --
        so the statistic computed from these is the statistic that script
        reports on the exported PLY, not an approximation of it.
        """
        with torch.no_grad():
            xyz = self.point_params.xyz.detach().cpu().numpy().astype(np.float64)
            rgb = np.clip(self.point_params.feat[:, :3].detach().cpu().numpy().astype(np.float64), 0.0, 1.0)
            conf = self.point_params.conf().detach().cpu().numpy().astype(np.float64)
        return xyz, rgb, conf

    def point_stats(self) -> dict:
        """Point count, points removed so far, and the in-region dark-mass fraction.

        Logged into every `evaluate()` metrics record (and so into
        `metrics.jsonl`) so the number Jordan's complaint is measured by is
        visible *during* training rather than only after an export plus a
        Splats-tool run. The `shade_region` sub-dict is
        `trippy.train.prune.dark_mass_stats`'s output, or `{"error": ...}`
        when the region could not be built, or absent entirely when
        `cfg.shade_prune.log_dark_mass` is off.
        """
        stats: dict = {
            "n_points": len(self.point_params),
            "n_removed_total": self.points_removed_total,
        }
        if not self.cfg.shade_prune.log_dark_mass:
            return stats
        views = self.shade_views()
        if views is None:
            stats["shade_region"] = {"error": self._shade_region_error}
            return stats
        xyz, rgb, conf = self._point_arrays()
        stats["shade_region"] = prune.dark_mass_stats(views, xyz, rgb, conf, self.cfg.shade_prune.lum_threshold)
        return stats

    def _apply_keep_mask(self, keep: np.ndarray, reason: str) -> int:
        """Drop every point `keep` is False for, rebuilding params AND optimiser state.

        This is trippy's equivalent of TRIPS's
        `NeuralScene::RemovePoints` + `ShrinkTextureOptimizer`
        (`src/lib/data/NeuralScene.cpp:1375-1470`,
        `src/lib/models/MyAdam.cu:346-374`): every per-point parameter is
        index-selected onto the survivors, and so are that parameter's Adam
        first/second moments, so moment row `i` still belongs to point `i`
        afterwards. TRIPS zeroes the gradients after the surgery
        (`NeuralScene.cpp:1436-1447`); here the freshly built
        `nn.Parameter`s simply have `grad is None`, which is the same thing
        for the next `optimizer.step()`. The Adam `step` counter is a scalar
        and is carried across unchanged (TRIPS keeps `param_steps` too, just
        index-selected, since its counter is per element).

        Args:
            keep: (N,) bool array over the *current* point count.
            reason: short tag for the run log (e.g. "point_removal").

        Returns:
            How many points were removed (0 when `keep` is all True, in
            which case nothing is rebuilt at all).
        """
        n_before = len(self.point_params)
        if keep.shape != (n_before,):
            raise ValueError(f"keep mask must be ({n_before},), got {keep.shape}")
        n_removed = int(n_before - keep.sum())
        if n_removed <= 0:
            return 0

        index = torch.from_numpy(np.flatnonzero(keep).astype(np.int64)).to(self.device)
        for attr, group_name in self._POINT_PARAM_GROUPS:
            old_param = getattr(self.point_params, attr)
            new_param = nn.Parameter(
                old_param.detach().index_select(0, index).clone(), requires_grad=old_param.requires_grad
            )
            state = self.optimizer.state.pop(old_param, None)
            if state is not None:
                self.optimizer.state[new_param] = {
                    key: (
                        value.index_select(0, index.to(value.device)).clone()
                        if torch.is_tensor(value) and value.shape == old_param.shape
                        else value
                    )
                    for key, value in state.items()
                }
            group = self.optimizer.param_groups[self._group_index[group_name]]
            group["params"] = [new_param if p is old_param else p for p in group["params"]]
            setattr(self.point_params, attr, new_param)

        self.point_params.provenance = self.point_params.provenance.index_select(0, index)
        # init_conf travels with its point exactly like provenance: it is metadata (each
        # point's OWN confidence at construction, for the relative removal test), never
        # optimizer state, so it is index-selected here and left alone everywhere else.
        self.point_params.init_conf = self.point_params.init_conf.index_select(0, index)
        # bbox_min/bbox_max are deliberately NOT recomputed: the extent penalty is
        # anchored to the *initial* cloud's bounding box (see `_extent_penalty`), and
        # shrinking it after a prune would start penalising points that were always
        # inside the scene.
        self.points_removed_total += n_removed
        self._log(
            f"epoch {self.epoch}: {reason} removed {n_removed} points "
            f"({n_before} -> {len(self.point_params)})"
        )
        return n_removed

    def maybe_prune_points(self, epoch: int) -> dict:
        """Run whichever point-removal rules fire at `epoch` (TRIPS's, then trippy's).

        TRIPS runs its removal pass at the top of an epoch, before that
        epoch's training steps (`src/apps/train.cpp:670-674`,
        `AddAndRemovePoints(epoch_id)` inside `if (do_train && epoch_id > 0)`);
        `fit()` calls this in the same place. When both rules fire in the
        same epoch, TRIPS's runs first and trippy's `shade_prune` then sees
        the already-thinned cloud.

        Args:
            epoch: the epoch about to run.

        Returns:
            `{"point_removal": n, "shade_prune": n}` for the rules that
            fired (absent keys mean that rule did not fire). Empty when
            neither did.
        """
        result: dict = {}
        removal_cfg = self.cfg.point_removal
        if removal_cfg.fires_at(epoch):
            _xyz, _rgb, conf = self._point_arrays()
            init_conf = self.point_params.init_conf.detach().cpu().numpy().astype(np.float64)
            keep = prune.removal_keep_mask(
                conf,
                removal_cfg.conf_threshold,
                removal_cfg.min_points,
                mode=removal_cfg.mode,
                rel_factor=removal_cfg.rel_factor,
                init_conf=init_conf,
            )
            result["point_removal"] = self._apply_keep_mask(keep, "point_removal")

        shade_cfg = self.cfg.shade_prune
        if shade_cfg.fires_at(epoch):
            views = self.shade_views()
            if views is None:
                self._log(f"epoch {epoch}: shade_prune skipped ({self._shade_region_error})")
            else:
                xyz, rgb, conf = self._point_arrays()
                inside, _zfrac = prune.in_region(views, xyz)
                inside &= np.isfinite(xyz).all(axis=1)
                init_conf = self.point_params.init_conf.detach().cpu().numpy().astype(np.float64)
                keep = prune.shade_prune_keep_mask(
                    inside,
                    prune.luminance(rgb),
                    conf,
                    shade_cfg.lum_threshold,
                    shade_cfg.conf_threshold,
                    shade_cfg.min_points,
                    mode=shade_cfg.mode,
                    rel_factor=shade_cfg.rel_factor,
                    init_conf=init_conf,
                )
                result["shade_prune"] = self._apply_keep_mask(keep, "shade_prune")
        return result

    def _resize_point_params(self, n: int) -> None:
        """Rebuild every per-point parameter/buffer at length `n`, keeping optimiser groups aligned.

        Needed by `load_state`: a checkpoint written after a removal pass
        holds fewer points than this Trainer's freshly built point source
        does, so the parameters must be resized *before* `load_state_dict`
        can fill them. Values are placeholders (the caller overwrites them
        immediately); what matters is that the optimiser's param groups end
        up pointing at the same objects the module now holds, in the same
        order, so `optimizer.load_state_dict` maps its saved state onto them
        by position.
        """
        if n == len(self.point_params):
            return
        for attr, group_name in self._POINT_PARAM_GROUPS:
            old_param = getattr(self.point_params, attr)
            shape = (n, *old_param.shape[1:])
            new_param = nn.Parameter(
                torch.zeros(shape, dtype=old_param.dtype, device=old_param.device),
                requires_grad=old_param.requires_grad,
            )
            self.optimizer.state.pop(old_param, None)
            group = self.optimizer.param_groups[self._group_index[group_name]]
            group["params"] = [new_param if p is old_param else p for p in group["params"]]
            setattr(self.point_params, attr, new_param)
        provenance = self.point_params.provenance
        self.point_params.provenance = torch.zeros(n, dtype=provenance.dtype, device=provenance.device)
        init_conf = self.point_params.init_conf
        self.point_params.init_conf = torch.zeros(n, dtype=init_conf.dtype, device=init_conf.device)

    # --- checkpointing / export ---

    def save_checkpoint(self, epoch: int | None = None) -> Path:
        """Save a checkpoint for `epoch`, update aliases, then apply the retention policy.

        Always writes `checkpoint_ep<NNNN>.pt` and `checkpoint_latest.pt`
        (unconditional, as before). Additionally:
          - If `epoch` is currently the best-known held-out-PSNR epoch (see
            `evaluate`'s `self._best_epoch` bookkeeping -- only set when this
            exact epoch's eval happened to be the best seen so far; a
            `save_checkpoint` call for an epoch `evaluate()` never scored,
            e.g. a manual `save_checkpoint(epoch=3)` in a test, leaves
            `checkpoint_best.pt`/`best.json` untouched), also writes
            `checkpoint_best.pt` (a second full copy, not a rename/symlink --
            symlinks don't survive `scripts/deliver.sh`/rsync-style copies
            reliably) and `best.json` ({"epoch", "psnr"}).
          - Deletes every other `checkpoint_ep<NNNN>.pt` in `checkpoint_dir`
            the policy doesn't protect (`trippy.train.retention.
            select_checkpoints_to_delete`, keeping `cfg.checkpoint_keep_every`
            multiples, the `cfg.checkpoint_keep_last` most recent, and the
            best epoch). `checkpoint_latest.pt`/`checkpoint_best.pt` are never
            in that glob, so they are never candidates for deletion.

        Args:
            epoch: defaults to `self.epoch`.

        Returns:
            The `checkpoint_ep<NNNN>.pt` path just written (not
            `checkpoint_latest.pt`/`checkpoint_best.pt`).
        """
        epoch = self.epoch if epoch is None else epoch
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "cfg": self.cfg.to_dict(),
            "point_params": self.point_params.state_dict(),
            "pose_params": self.pose_params.state_dict(),
            "net": self.net.state_dict(),
            "camera": self.camera.state_dict(),
            "background": self.background.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "points_removed_total": self.points_removed_total,
        }
        path = self.checkpoint_dir / TRAIN_CHECKPOINT_FILENAME_FMT.format(epoch=epoch)
        checkpoint_io.save_checkpoint(path, payload)
        checkpoint_io.save_checkpoint(self.checkpoint_dir / TRAIN_CHECKPOINT_LATEST_FILENAME, payload)
        self._log(f"epoch {epoch}: checkpoint saved to {path}")

        if self._best_epoch == epoch:
            checkpoint_io.save_checkpoint(self.checkpoint_dir / TRAIN_CHECKPOINT_BEST_FILENAME, payload)
            best_json = {"epoch": epoch, "psnr": self._best_psnr}
            (self.checkpoint_dir / TRAIN_CHECKPOINT_BEST_JSON_FILENAME).write_text(json.dumps(best_json, indent=2))
            self._log(f"epoch {epoch}: new best held-out psnr={self._best_psnr:.3f} -> checkpoint_best.pt")

        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        """Delete checkpoint_ep*.pt files the retention policy no longer wants (see save_checkpoint).

        Runs with `protect_newer_than_s=0.0`: everything already on disk in
        this same process is fair game the instant it stops being the
        best/a keep_every multiple/within keep_last -- unlike `trippy
        prune-run` (a separate, later process against a possibly
        still-running job), there is no concurrent writer to race here.
        """
        existing = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        to_delete = retention.select_checkpoints_to_delete(
            existing,
            best_epoch=self._best_epoch,
            keep_every=self.cfg.checkpoint_keep_every,
            keep_last=self.cfg.checkpoint_keep_last,
            protect_newer_than_s=0.0,
        )
        for victim in to_delete:
            victim.unlink(missing_ok=True)
            self._log(f"pruned {victim} (retention policy)")

    def load_state(self, payload: dict) -> None:
        """Load a checkpoint payload (as produced by `save_checkpoint`) into this Trainer.

        A checkpoint saved after a point-removal pass holds fewer points
        than this Trainer's point source rebuilt at construction, so the
        per-point parameters are resized to the checkpoint's own count
        first (`_resize_point_params`) -- otherwise `load_state_dict` would
        raise a size mismatch and no removal run could ever be resumed,
        re-evaluated or reported.
        """
        checkpoint_n = int(payload["point_params"]["xyz"].shape[0])
        if checkpoint_n != len(self.point_params):
            self._log(f"checkpoint holds {checkpoint_n} points (source built {len(self.point_params)}); resizing")
            self._resize_point_params(checkpoint_n)
        self.point_params.load_state_dict(payload["point_params"])
        self.pose_params.load_state_dict(payload["pose_params"])
        self.net.load_state_dict(payload["net"])
        self.camera.load_state_dict(payload["camera"])
        with torch.no_grad():
            self.background.copy_(payload["background"].to(self.device))
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))
        self.points_removed_total = int(payload.get("points_removed_total", 0))

    def resume(self, path: str | Path) -> None:
        """Load a checkpoint file and continue from its epoch."""
        payload = checkpoint_io.load_checkpoint(path, map_location=self.device)
        self.load_state(payload)
        self._log(f"resumed from {path} at epoch {self.epoch}")

    def export_ply(self, path: str | Path | None = None) -> Path:
        """Export the current trained point cloud as a 3DGS-compatible PLY (trippy.train.export).

        The exported colour is the trained feature vector's first 3
        channels (clamped to [0, 1]) -- an approximation, not the true
        rendered appearance, which requires the trained U-Net decoding all
        `cfg.layers` pyramid levels together (something no 3DGS viewer can
        run). Channels 0:3 are seeded from `rgb0` (see `PointParams`), so
        this stays a meaningful colour preview even after training moves
        it away from the initial value.
        """
        path = Path(path) if path is not None else self.run_dir / TRAIN_EXPORT_FILENAME
        xyz = self.point_params.xyz.detach().cpu().numpy()
        size = self.point_params.size().detach().cpu().numpy()
        conf = self.point_params.conf().detach().cpu().numpy()
        rgb = np.clip(self.point_params.feat[:, :3].detach().cpu().numpy(), 0.0, 1.0)
        provenance = self.point_params.provenance.detach().cpu().numpy()
        export.write_gaussian_ply(path, xyz, rgb, conf, size, provenance=provenance)
        self._log(f"exported {len(self.point_params)} points to {path}")
        return path

    # --- logging ---

    def _append_metrics(self, record: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log(self, message: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    # --- top-level loop ---

    def fit(self, max_minutes: float | None = None) -> dict:
        """Run the full epoch schedule: locks, vgg start, eval/checkpoint cadence, time budget.

        Args:
            max_minutes: wall-clock budget; None uses `cfg.max_minutes`
                (itself possibly None, meaning "run to `cfg.epochs`"). When
                the budget is hit, the current epoch's checkpoint/eval are
                still written before returning, so a queue job ends cleanly
                (docs/EXPERIMENTS.md "Training runs").

        Returns:
            The most recent `evaluate()` metrics dict (empty if no eval ran
            -- only possible if `max_minutes` expires before the first
            `eval_every` boundary).
        """
        max_minutes = self.cfg.max_minutes if max_minutes is None else max_minutes
        start_time = time.monotonic()
        steps_per_epoch = self.cfg.steps_per_epoch(len(self.train_names))
        last_metrics: dict = {}

        def _time_up() -> bool:
            return max_minutes is not None and (time.monotonic() - start_time) / 60.0 >= max_minutes

        while self.epoch < self.cfg.epochs:
            self.maybe_prune_points(self.epoch)
            self._apply_locks(self.epoch)
            self.loss_fn.weights.vgg = self.cfg.loss_vgg if self.epoch >= self.cfg.vgg_start_epoch else 0.0

            for _ in range(steps_per_epoch):
                self.train_step()
                if _time_up():
                    break

            is_last = self.epoch == self.cfg.epochs - 1
            time_up = _time_up()

            if self.epoch % self.cfg.eval_every == 0 or is_last or time_up:
                last_metrics = self.evaluate(epoch=self.epoch)
                self.scheduler.step(last_metrics["psnr_mean"])
            if self.epoch % self.cfg.checkpoint_every == 0 or is_last or time_up:
                self.save_checkpoint(epoch=self.epoch)

            self.epoch += 1
            if time_up:
                self._log(f"time budget ({max_minutes} min) reached at epoch {self.epoch - 1}; stopping")
                break

        self.export_ply()
        return last_metrics
