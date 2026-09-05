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
from torch import nn

from trippy.config import load_settings, pick_device
from trippy.constants import (
    SCENE_CACHE_META_FILENAME,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_CHECKPOINT_FILENAME_FMT,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
    TRAIN_EVAL_DIRNAME_FMT,
    TRAIN_EVAL_MAX_SHEET_IMAGES,
    TRAIN_EVAL_METRICS_FILENAME,
    TRAIN_EVAL_SHEET_FILENAME,
    TRAIN_EXPORT_FILENAME,
    TRAIN_LOG_FILENAME,
    TRAIN_LPIPS_METRIC_NET,
    TRAIN_METRICS_FILENAME,
    TRAIN_PSNR_EPS,
)
from trippy.geom import xform_b
from trippy.hybrid.gaussian_input import GaussianInputs
from trippy.net.camera_model import NeuralCamera
from trippy.net.losses import LossWeights, TripsLoss, _LazyLPIPS, mse_loss, ssim
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.raster.pyramid import render_pyramid
from trippy.render.sheets import colorize, contact_sheet, save_png
from trippy.scene import splits
from trippy.scene.dataset import SceneDataset
from trippy.scene.dataset import crop as dataset_crop
from trippy.train import checkpoint_io, export
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
            self.dataset.names, cfg.forced_heldout, k=cfg.heldout_k
        )
        if not self.train_names:
            raise ValueError("split produced an empty train set -- check heldout_k/forced_heldout vs dataset size")

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
        """
        meta = json.loads((self.dataset.cache_dir / SCENE_CACHE_META_FILENAME).read_text())["images"]
        values = []
        for name in self.dataset.names:
            info = meta.get(name, {})
            exposure_time = info.get("exposure_time")
            iso = info.get("iso")
            if exposure_time and exposure_time > 0 and iso and iso > 0:
                ev = math.log2(1.0 / exposure_time) + math.log2(iso / 100.0)
            else:
                ev = 0.0
            values.append(ev)
        ev_tensor = torch.tensor(values, dtype=torch.float32)
        if ev_tensor.numel() == 0:
            return ev_tensor
        return ev_tensor - ev_tensor.mean()

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

    def evaluate(self, names: list[str] | None = None, epoch: int | None = None) -> dict:
        """Full-frame held-out evaluation: PSNR/SSIM(/LPIPS) + an honesty contact sheet.

        Args:
            names: image names to evaluate; defaults to `self.heldout_names`.
            epoch: label used for the output directory / metrics record;
                defaults to `self.epoch`.

        Returns:
            {"epoch", "n_images", "psnr_mean", "ssim_mean", "lpips_mean"
            (None if `cfg.eval_lpips` is False), "names"}. Also writes
            `<run_dir>/eval_ep<NNNN>/metrics.json` and, for up to
            `TRAIN_EVAL_MAX_SHEET_IMAGES` images (forced-held-out shade
            frames first), `sheet.png`: photo | render | raw level-0 |
            coverage, one row per image (docs/EXPERIMENTS.md "Mandatory
            honesty sheet").
        """
        self.net.eval()
        self.camera.eval()
        names = list(self.heldout_names) if names is None else list(names)
        epoch = self.epoch if epoch is None else epoch

        forced = set(self.cfg.forced_heldout)
        sheet_names = sorted(names, key=lambda n: (n not in forced, n))[:TRAIN_EVAL_MAX_SHEET_IMAGES]

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        lpips_vals: list[float] = []
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
                psnr = -10.0 * torch.log10(mse + TRAIN_PSNR_EPS)
                psnr_vals.append(float(psnr.item()))
                ssim_vals.append(float(ssim(pred, target_c, mask_c).item()))
                if self._eval_lpips is not None:
                    lpips_vals.append(float(self._eval_lpips(pred, target_c, mask_c).item()))

                if name in sheet_names:
                    raw = layers[0][:3].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    coverage = (1.0 - aux["t_final"][0]).clamp(0.0, 1.0).cpu().numpy()
                    coverage_rgb = colorize(coverage, 0.0, 1.0)
                    photo_np = target_c[0].permute(1, 2, 0).cpu().numpy()
                    pred_np = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    sheet_images += [photo_np, pred_np, raw, coverage_rgb]
                    sheet_labels += [f"{name} photo", "render", "raw L0", "coverage"]

        metrics = {
            "epoch": epoch,
            "n_images": len(names),
            "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
            "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
            "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
            "names": names,
        }

        eval_dir = self.run_dir / TRAIN_EVAL_DIRNAME_FMT.format(epoch=epoch)
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / TRAIN_EVAL_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))
        if sheet_images:
            sheet = contact_sheet(sheet_images, sheet_labels, cols=4)
            save_png(eval_dir / TRAIN_EVAL_SHEET_FILENAME, sheet)

        self._append_metrics({"eval": True, **{k: v for k, v in metrics.items() if k != "names"}})
        self._log(f"epoch {epoch}: eval psnr={metrics['psnr_mean']:.3f} ssim={metrics['ssim_mean']:.4f}")
        return metrics

    # --- checkpointing / export ---

    def save_checkpoint(self, epoch: int | None = None) -> Path:
        """Save a checkpoint for `epoch` (default `self.epoch`) and update the "latest" alias."""
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
        }
        path = self.checkpoint_dir / TRAIN_CHECKPOINT_FILENAME_FMT.format(epoch=epoch)
        checkpoint_io.save_checkpoint(path, payload)
        checkpoint_io.save_checkpoint(self.checkpoint_dir / TRAIN_CHECKPOINT_LATEST_FILENAME, payload)
        self._log(f"epoch {epoch}: checkpoint saved to {path}")
        return path

    def load_state(self, payload: dict) -> None:
        """Load a checkpoint payload (as produced by `save_checkpoint`) into this Trainer."""
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
