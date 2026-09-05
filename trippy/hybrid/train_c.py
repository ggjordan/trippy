"""HybridCTrainer: the Design C render->photo U-Net trainer.

Module: trippy.hybrid.train_c
Invariants:
    - Deliberately independent of `trippy.train.trainer.Trainer` (per this task's brief:
      "the Trainer is point-based, so write a small separate trainer for image->image"). No
      point cloud, no rasteriser, no pose refinement: the input is a *fixed* rendered image
      (rgb + alpha + optional depth, from `trippy.hybrid.render_splat_views`), the target is
      the photo, and the only trainable state is the U-Net (`trippy.net.unet`) and the
      per-image tone mapper (`trippy.net.camera_model.NeuralCamera`) -- reused unmodified from
      trippy.train.
    - `frame_index` for `NeuralCamera`/EXIF-init purposes indexes into
      `self.dataset.names` (the FULL scene's registered images), not the paired subset --
      matches `Trainer`'s own convention, so `NeuralCamera(num_frames=len(dataset.names))`
      always has a slot for any name in `self.dataset.names`, and a checkpoint's per-image
      exposure/response state is stable even if the paired subset changes between renders.
    - Loss mask is deliberately full-frame: `(render alpha > 0) | ones_like(...)` always
      evaluates to all-ones (an OR with True is always True) -- this is intentional, not a
      bug, see `train_step`'s comment. Design C's whole point is a network that can also
      repair the render's own holes toward the photo, not just refine already-covered pixels.
    - `evaluate()` reports PSNR/SSIM/LPIPS twice per frame -- "baseline" (raw render vs photo,
      no U-Net) and "refined" (U-Net + tone-mapper output vs photo) -- split into "all",
      "shade" (`cfg.forced_heldout`, `SHADE_FRAMES_KK` by default), and "nonshade" buckets, so
      a shade-region verdict is never averaged away by the easy frames (docs/SPEC.md D10).
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement";
    experiments/EXP-0005-hybrid-c/README.md; trippy.hybrid.dataset_c (pairing + pyramids).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from trippy.config import load_settings, pick_device
from trippy.constants import (
    HYBRID_C_EVAL_MAX_SHEET_IMAGES,
    SCENE_CACHE_META_FILENAME,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_CHECKPOINT_FILENAME_FMT,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
    TRAIN_EVAL_DIRNAME_FMT,
    TRAIN_EVAL_METRICS_FILENAME,
    TRAIN_EVAL_SHEET_FILENAME,
    TRAIN_LOG_FILENAME,
    TRAIN_LPIPS_METRIC_NET,
    TRAIN_LR_DECAY_FACTOR,
    TRAIN_LR_DECAY_PATIENCE,
    TRAIN_METRICS_FILENAME,
    TRAIN_PSNR_EPS,
)
from trippy.hybrid import dataset_c
from trippy.hybrid.config_c import HybridCConfig
from trippy.net.camera_model import NeuralCamera
from trippy.net.losses import LossWeights, TripsLoss, _LazyLPIPS, ssim
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.render.sheets import colorize, contact_sheet, save_png
from trippy.scene import splits
from trippy.scene.dataset import SceneDataset
from trippy.train import checkpoint_io
from trippy.train.config import steps_per_epoch


def _center_crop_like(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Centre-crop the last two dims of `x` down to (target_h, target_w).

    Small, deliberately-duplicated twin of `trippy.train.trainer._center_crop_like` (see
    this module's docstring: Design C's trainer stays independent of Trainer internals). The
    U-Net's output can be a few pixels smaller than its input when the input is not exactly
    divisible by `2 ** (num_layers - 1)` (see `trippy.net.unet` "CombineBridge / odd-size
    handling"); the loss target/mask/raw-render comparisons must shrink to match.
    """
    h, w = x.shape[-2], x.shape[-1]
    dh = max(0, (h - target_h) // 2)
    dw = max(0, (w - target_w) // 2)
    return x[..., dh : dh + target_h, dw : dw + target_w]


class HybridCTrainer:
    """Owns the paired render/photo dataset, the U-Net, the tone mapper, and the optimiser."""

    def __init__(self, cfg: HybridCConfig) -> None:
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        settings = load_settings()
        cache_root = Path(cfg.cache_root) if cfg.cache_root else settings.trippy_output / "cache"

        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / TRAIN_CHECKPOINT_DIRNAME
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / TRAIN_LOG_FILENAME
        self.metrics_path = self.run_dir / TRAIN_METRICS_FILENAME

        self.renders_dir = Path(cfg.renders_dir)
        self.dataset = SceneDataset(cfg.scene_root, cfg.width, cache_root, device=self.device, limit=cfg.limit_images)
        self._name_to_index = {name: i for i, name in enumerate(self.dataset.names)}

        paired = dataset_c.paired_names(self.renders_dir, self.dataset.names)
        if not paired:
            raise ValueError(f"no paired render/photo frames found under {self.renders_dir}")
        self.train_names, self.heldout_names = splits.split_with_forced_heldout(
            paired, cfg.forced_heldout, k=cfg.heldout_k
        )
        if not self.train_names:
            raise ValueError("split produced an empty train set -- check heldout_k/forced_heldout vs paired frames")

        network_cfg = NetworkConfig(num_input_channels=cfg.channels, num_layers=cfg.layers)
        self.net = MultiScaleUnet2dDecOnlySmallFixed(network_cfg).to(self.device)

        first_item = self.dataset[0]
        full_h, full_w = int(first_item["rgb"].shape[0]), int(first_item["rgb"].shape[1])
        self.camera = NeuralCamera(
            image_height=full_h,
            image_width=full_w,
            num_frames=len(self.dataset.names),
            initial_exposure=self._initial_exposure(),
        ).to(self.device)
        # Same fix_vignette=true / fix_wb=true freezing Trainer applies (docs/TRIPS_REFERENCE.md
        # Sec. 7) -- both modules stay present (their forward pass still runs) but untrained.
        if self.camera.vignette_net is not None:
            for p in self.camera.vignette_net.parameters():
                p.requires_grad_(False)
        if self.camera.white_balance_values is not None:
            self.camera.white_balance_values.requires_grad_(False)

        self.loss_fn = TripsLoss(
            LossWeights(vgg=0.0, l1=cfg.loss_l1, ssim=cfg.loss_ssim, lpips=cfg.loss_lpips)
        ).to(self.device)
        self._eval_lpips = _LazyLPIPS(net=TRAIN_LPIPS_METRIC_NET) if cfg.eval_lpips else None

        group_specs: list[tuple[str, list[torch.nn.Parameter], float]] = [
            ("network", list(self.net.parameters()), cfg.lr_network)
        ]
        if self.camera.exposures_values is not None:
            group_specs.append(("exposure", [self.camera.exposures_values], cfg.lr_exposure))
        if self.camera.camera_response is not None:
            group_specs.append(("response", list(self.camera.camera_response.parameters()), cfg.lr_response))
        self.optimizer = torch.optim.Adam([{"params": params, "lr": lr} for _, params, lr in group_specs])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=TRAIN_LR_DECAY_FACTOR, patience=TRAIN_LR_DECAY_PATIENCE
        )

        self._rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self.epoch = 0
        self.global_step = 0

    # --- construction helpers ---

    def _initial_exposure(self) -> torch.Tensor:
        """Per-image EV_log2 from cached EXIF, 0.0 where EXIF is missing.

        Identical formula to `trippy.train.trainer.Trainer._initial_exposure` (deliberately
        duplicated, not imported -- see module docstring).
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
        return torch.tensor(values, dtype=torch.float32)

    def _load_pair(self, name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load one name's full-resolution (render, photo, frame_index), aligned in (H, W)."""
        frame_index = self._name_to_index[name]
        stem = Path(name).stem
        render_arrays = dataset_c.load_render_arrays(self.renders_dir, stem)
        render_full = dataset_c.render_to_tensor(render_arrays, channels=self.cfg.channels).to(self.device)
        item = self.dataset[frame_index]
        photo_full = dataset_c.photo_to_tensor(item["rgb"]).to(self.device)
        if render_full.shape[-2:] != photo_full.shape[-2:]:
            raise ValueError(
                f"{name}: render {tuple(render_full.shape[-2:])} and photo "
                f"{tuple(photo_full.shape[-2:])} are misaligned -- re-render at cfg.width"
            )
        return render_full, photo_full, frame_index

    def _forward(self, render: torch.Tensor) -> torch.Tensor:
        """U-Net over `render`'s pyramid -> (1, C, H, W) untone-mapped output."""
        levels = dataset_c.build_pyramid(render, self.cfg.layers)
        inputs = [lvl.unsqueeze(0) for lvl in levels]
        return self.net(inputs)

    def _tone_map(self, net_out: torch.Tensor, frame_index: int) -> torch.Tensor:
        frame_index_t = torch.tensor([frame_index], device=self.device, dtype=torch.long)
        return self.camera(net_out, frame_index_t)

    # --- training ---

    def train_step(self, name: str | None = None, y0: int | None = None, x0: int | None = None) -> dict:
        """Sample one crop, run the U-Net + tone mapper, backprop, step.

        Args:
            name: override the random image pick (tests use this for determinism).
            y0, x0: override the random crop origin (tests use this for determinism); both
                must be given together, or both left None.

        Returns:
            A dict of scalar metrics for this step (also appended to `metrics.jsonl`).
        """
        self.net.train()
        self.camera.train()

        if name is None:
            pick = torch.randint(0, len(self.train_names), (), generator=self._rng).item()
            name = self.train_names[pick]
        render_full, photo_full, frame_index = self._load_pair(name)
        height, width = photo_full.shape[-2], photo_full.shape[-1]

        if y0 is None or x0 is None:
            y0, x0 = dataset_c.sample_crop_origin(height, width, self.cfg.crop, self._rng)
        render_crop, photo_crop = dataset_c.crop_pair(render_full, photo_full, self.cfg.crop, y0, x0)
        alpha_crop = render_crop[3:4]  # channel 3 is alpha, see dataset_c.render_to_tensor.

        net_out = self._forward(render_crop)
        pred = self._tone_map(net_out, frame_index)

        target = _center_crop_like(photo_crop.unsqueeze(0), pred.shape[-2], pred.shape[-1])
        # Mask is deliberately full-frame: OR-ing `alpha > 0` with an all-ones tensor always
        # yields all-ones (see module docstring "Loss mask is deliberately full-frame"). Kept
        # explicit (not simplified to `torch.ones_like`) so the render's own coverage is
        # visibly part of the design decision being documented, not silently dropped.
        alpha_valid = alpha_crop > 0.0
        always_valid = torch.ones_like(alpha_valid, dtype=torch.bool)
        mask = (alpha_valid | always_valid).to(torch.float32).unsqueeze(0)
        mask = _center_crop_like(mask, pred.shape[-2], pred.shape[-1])

        image_loss = self.loss_fn(pred, target, mask)
        camera_reg = self.camera.regularizer()
        total = image_loss + camera_reg

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self.camera.apply_constraints()

        self.global_step += 1
        record = {
            "step": self.global_step,
            "epoch": self.epoch,
            "image": name,
            "loss": float(total.detach().item()),
            "image_loss": float(image_loss.detach().item()),
            "camera_reg": float(camera_reg.detach().item()),
        }
        self._append_metrics(record)
        return record

    # --- evaluation ---

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        mse = ((pred - target) ** 2).mean()
        psnr = -10.0 * torch.log10(mse + TRAIN_PSNR_EPS)
        out = {"psnr": float(psnr.item()), "ssim": float(ssim(pred, target).item())}
        if self._eval_lpips is not None:
            out["lpips"] = float(self._eval_lpips(pred, target).item())
        return out

    @staticmethod
    def _summarize_bucket(records: list[dict]) -> dict:
        if not records:
            return {"n": 0, "psnr_mean": None, "ssim_mean": None, "lpips_mean": None}
        n = len(records)
        lpips_vals = [r["lpips"] for r in records if "lpips" in r]
        return {
            "n": n,
            "psnr_mean": sum(r["psnr"] for r in records) / n,
            "ssim_mean": sum(r["ssim"] for r in records) / n,
            "lpips_mean": (sum(lpips_vals) / len(lpips_vals)) if lpips_vals else None,
        }

    def evaluate(self, names: list[str] | None = None, epoch: int | None = None) -> dict:
        """Held-out PSNR/SSIM/LPIPS for the raw render (baseline) and the U-Net output (refined).

        Args:
            names: image names to evaluate; defaults to `self.heldout_names`.
            epoch: label used for the output directory / metrics record; defaults to
                `self.epoch`.

        Returns:
            `{"epoch", "n_images", "baseline": {"all"|"shade"|"nonshade": {"n", "psnr_mean",
            "ssim_mean", "lpips_mean"}}, "refined": {...same shape...}, "names"}`. Also writes
            `<run_dir>/eval_ep<NNNN>/metrics.json`, an up-to-`HYBRID_C_EVAL_MAX_SHEET_IMAGES`
            `sheet.png` (photo | render | refined | |diff|, shade frames first), and standalone
            `shade_frames/<stem>_refined.png` for every `cfg.forced_heldout` name present in
            `names` (the delivery artifact for Jordan).
        """
        self.net.eval()
        self.camera.eval()
        names = list(self.heldout_names) if names is None else list(names)
        epoch = self.epoch if epoch is None else epoch
        forced = set(self.cfg.forced_heldout)

        sheet_names = sorted(names, key=lambda n: (n not in forced, n))[:HYBRID_C_EVAL_MAX_SHEET_IMAGES]
        records: dict[str, dict[str, list[dict]]] = {
            "baseline": {"all": [], "shade": [], "nonshade": []},
            "refined": {"all": [], "shade": [], "nonshade": []},
        }
        sheet_images: list = []
        sheet_labels: list[str] = []
        shade_refined_pngs: dict[str, object] = {}

        with torch.no_grad():
            for name in names:
                render_full, photo_full, frame_index = self._load_pair(name)
                net_out = self._forward(render_full)
                pred = self._tone_map(net_out, frame_index)

                target = _center_crop_like(photo_full.unsqueeze(0), pred.shape[-2], pred.shape[-1])
                raw = _center_crop_like(render_full[:3].clamp(0.0, 1.0).unsqueeze(0), pred.shape[-2], pred.shape[-1])

                bucket = "shade" if name in forced else "nonshade"
                m_base = self._compute_metrics(raw, target)
                m_ref = self._compute_metrics(pred, target)
                records["baseline"][bucket].append(m_base)
                records["baseline"]["all"].append(m_base)
                records["refined"][bucket].append(m_ref)
                records["refined"]["all"].append(m_ref)

                if name in sheet_names:
                    photo_np = target[0].permute(1, 2, 0).cpu().numpy()
                    raw_np = raw[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    pred_np = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    diff = (pred[0] - target[0]).abs().mean(dim=0).clamp(0.0, 1.0).cpu().numpy()
                    diff_rgb = colorize(diff, 0.0, 1.0)
                    sheet_images += [photo_np, raw_np, pred_np, diff_rgb]
                    sheet_labels += [f"{name} photo", "render", "refined", "|diff|"]

                if name in forced:
                    shade_refined_pngs[name] = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

        metrics = {
            "epoch": epoch,
            "n_images": len(names),
            "baseline": {k: self._summarize_bucket(v) for k, v in records["baseline"].items()},
            "refined": {k: self._summarize_bucket(v) for k, v in records["refined"].items()},
            "names": names,
        }

        eval_dir = self.run_dir / TRAIN_EVAL_DIRNAME_FMT.format(epoch=epoch)
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / TRAIN_EVAL_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))
        if sheet_images:
            sheet = contact_sheet(sheet_images, sheet_labels, cols=4)
            save_png(eval_dir / TRAIN_EVAL_SHEET_FILENAME, sheet)
        if shade_refined_pngs:
            shade_dir = eval_dir / "shade_frames"
            for name, arr in shade_refined_pngs.items():
                save_png(shade_dir / f"{Path(name).stem}_refined.png", arr)

        self._append_metrics(
            {
                "eval": True,
                "epoch": epoch,
                "n_images": metrics["n_images"],
                "baseline": metrics["baseline"],
                "refined": metrics["refined"],
            }
        )
        self._log(
            f"epoch {epoch}: baseline psnr={metrics['baseline']['all']['psnr_mean']:.3f} "
            f"refined psnr={metrics['refined']['all']['psnr_mean']:.3f}"
        )
        return metrics

    # --- checkpointing ---

    def save_checkpoint(self, epoch: int | None = None) -> Path:
        epoch = self.epoch if epoch is None else epoch
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "cfg": self.cfg.to_dict(),
            "net": self.net.state_dict(),
            "camera": self.camera.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        path = self.checkpoint_dir / TRAIN_CHECKPOINT_FILENAME_FMT.format(epoch=epoch)
        checkpoint_io.save_checkpoint(path, payload)
        checkpoint_io.save_checkpoint(self.checkpoint_dir / TRAIN_CHECKPOINT_LATEST_FILENAME, payload)
        self._log(f"epoch {epoch}: checkpoint saved to {path}")
        return path

    def load_state(self, payload: dict) -> None:
        self.net.load_state_dict(payload["net"])
        self.camera.load_state_dict(payload["camera"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.epoch = int(payload.get("epoch", 0))
        self.global_step = int(payload.get("global_step", 0))

    def resume(self, path: str | Path) -> None:
        payload = checkpoint_io.load_checkpoint(path, map_location=self.device)
        self.load_state(payload)
        self._log(f"resumed from {path} at epoch {self.epoch}")

    # --- logging ---

    def _append_metrics(self, record: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log(self, message: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    # --- top-level loop ---

    def fit(self, max_minutes: float | None = None) -> dict:
        """Run the full epoch schedule (see `trippy.train.trainer.Trainer.fit` for the same
        eval/checkpoint-cadence/time-budget pattern, deliberately mirrored here in miniature).
        """
        max_minutes = self.cfg.max_minutes if max_minutes is None else max_minutes
        start_time = time.monotonic()
        n_steps_per_epoch = steps_per_epoch(self.cfg.train_factor, len(self.train_names))
        last_metrics: dict = {}

        def _time_up() -> bool:
            return max_minutes is not None and (time.monotonic() - start_time) / 60.0 >= max_minutes

        while self.epoch < self.cfg.epochs:
            for _ in range(n_steps_per_epoch):
                self.train_step()
                if _time_up():
                    break

            is_last = self.epoch == self.cfg.epochs - 1
            time_up = _time_up()

            if self.epoch % self.cfg.eval_every == 0 or is_last or time_up:
                last_metrics = self.evaluate(epoch=self.epoch)
                self.scheduler.step(last_metrics["refined"]["all"]["psnr_mean"])
            if self.epoch % self.cfg.checkpoint_every == 0 or is_last or time_up:
                self.save_checkpoint(epoch=self.epoch)

            self.epoch += 1
            if time_up:
                self._log(f"time budget ({max_minutes} min) reached at epoch {self.epoch - 1}; stopping")
                break

        return last_metrics


def build_trainer_from_checkpoint(checkpoint_path: str | Path, device: str | None = None) -> HybridCTrainer:
    """Rebuild a HybridCTrainer (dataset, net, camera) from a checkpoint's own saved config."""
    payload = checkpoint_io.load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = HybridCConfig.from_dict(payload["cfg"])
    if device is not None:
        cfg.device = device
    trainer = HybridCTrainer(cfg)
    trainer.load_state(payload)
    return trainer


def evaluate_checkpoint(
    checkpoint_path: str | Path, images: list[str] | None = None, device: str | None = None
) -> dict:
    """Evaluate a checkpoint on `images` (default: its own held-out split). See `HybridCTrainer.evaluate`."""
    trainer = build_trainer_from_checkpoint(checkpoint_path, device=device)
    return trainer.evaluate(names=images)
