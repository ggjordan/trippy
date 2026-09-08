"""`trippy profile-step`: where the seconds of one training step actually go.

Module: trippy.train.profile_step
Purpose: run a real `Trainer.train_step` on a real config and report, per
    stage, the milliseconds it costs on the device -- so that optimisation
    work is directed by measurement rather than by which stage looks
    expensive in the source. The stages are instrumented in the production
    code path (`trippy.train.steptimer`), not in a copy of it, so this can
    never profile something the trainer does not do.
Invariants:
    - **No imagery is read, written or displayed.** The trainer loads the
      scene's cached pixels because a crop's intrinsics come from them, and
      the loss needs the target, but nothing here writes or returns an image
      (AGENTS.md section 6). The output is a table, a JSON file and a loss
      curve.
    - **Two numbers per run, deliberately.** `untimed` is the honest
      seconds-per-step (no synchronisation the trainer would not do);
      `timed` is the same steps with a `torch.mps.synchronize()` between
      stages, which is what makes the breakdown attributable and also makes
      the total larger. Reporting one without the other invites reading a
      profiling artefact as a regression.
    - **The epoch matters and is explicit.** A Karekare-v2 step at epoch 0
      is a different step from one at epoch 60: `Trainer._apply_locks` frees
      the pose deltas and xyz/size at `lock_*_epochs`, and the perceptual
      (VGG/LPIPS) loss term switches on at `vgg_start_epoch`. `--epoch`
      selects which regime is profiled; the default is the run's steady
      state (every lock released, every loss term on), because that is what
      a 300-epoch run spends its time in.
    - The loss of every profiled step is recorded, so two variants profiled
      from the same seed can be compared step by step. That is the parity
      gate for any change that could move numerics.
Units: every printed and stored time is **milliseconds**; `sec_per_step` and
    `min_per_epoch` are seconds and minutes.
Related docs: docs/ARCHITECTURE.md ("Where a training step's time goes");
    research/trips-metal.md.
"""

from __future__ import annotations

import copy
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from trippy.train import steptimer
from trippy.train.config import TrainConfig
from trippy.train.trainer import Trainer

# Printed in this order; anything measured but not listed is appended after
# these, so a newly instrumented stage shows up without editing this list.
STAGE_ORDER = (
    "data",
    "raster_project_cull",
    "raster_emit",
    "raster_sort",
    "raster_segment",
    "raster_gather",
    "raster_blend_fwd",
    "raster_split",
    "unet_fwd",
    "tone_map",
    "loss",
    "zero_grad",
    "backward",
    "optimizer",
    "metrics_sync",
)

# Backward sub-stages derived from the autograd hooks; they add up to
# `backward`, so they are printed indented under it rather than summed again.
BACKWARD_PARTS = ("bwd_loss", "bwd_tone_map", "bwd_unet", "bwd_raster")

# Non-time diagnostics carried on the same rows.
NOTE_KEYS = ("fragments", "fragments_kept", "points_total", "points_valid")


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def steady_state_epoch(cfg: TrainConfig) -> int:
    """First epoch at which no lock and no loss term is still waiting.

    A profile taken before this epoch understates a run: `_apply_locks`
    still freezes the pose deltas (and, earlier, xyz/size), so their
    gradients are never computed, and `fit()` has not yet switched the
    perceptual loss on. Both are large.
    """
    return max(cfg.lock_cameras_epochs, cfg.lock_structure_epochs, cfg.vgg_start_epoch)


def _prepare_epoch(trainer: Trainer, epoch: int) -> None:
    """Put the trainer in the state `fit()` would have it in at `epoch`.

    Mirrors the three things `Trainer.fit`'s epoch preamble does before its
    steps -- and only those three, since point removal (`maybe_prune_points`)
    changes the point count and would make two profiles incomparable.
    """
    trainer.epoch = epoch
    trainer._apply_locks(epoch)
    trainer.loss_fn.weights.vgg = trainer.cfg.loss_vgg if epoch >= trainer.cfg.vgg_start_epoch else 0.0


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _snapshot(trainer: Trainer) -> dict[str, Any]:
    """Everything a profiled step mutates, cloned so the next arm can start here.

    Without this, a `--raster-cap both` or `--amp both` sweep would run its
    second arm on the model its first arm left behind, and the two `losses`
    lists -- the parity evidence for anything that touches numerics -- would
    not be comparable. Kept in device memory rather than written out: at
    Karekare-v2's 7.5M points this is ~0.8 GB of parameters plus Adam moments,
    which is cheap next to a checkpoint round trip.
    """
    return {
        "net": {k: v.detach().clone() for k, v in trainer.net.state_dict().items()},
        "camera": {k: v.detach().clone() for k, v in trainer.camera.state_dict().items()},
        "point": {k: v.detach().clone() for k, v in trainer.point_params.state_dict().items()},
        "pose": {k: v.detach().clone() for k, v in trainer.pose_params.state_dict().items()},
        "background": trainer.background.detach().clone(),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "rng": trainer._rng.get_state().clone(),
    }


def _restore(trainer: Trainer, snapshot: dict[str, Any]) -> None:
    """Undo everything `_snapshot` recorded (see there)."""
    trainer.net.load_state_dict(snapshot["net"])
    trainer.camera.load_state_dict(snapshot["camera"])
    trainer.point_params.load_state_dict(snapshot["point"])
    trainer.pose_params.load_state_dict(snapshot["pose"])
    with torch.no_grad():
        trainer.background.copy_(snapshot["background"])
    trainer.optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    trainer.epoch = snapshot["epoch"]
    trainer.global_step = snapshot["global_step"]
    trainer._rng.set_state(snapshot["rng"])


def profile_step(
    cfg: TrainConfig,
    steps: int = 30,
    warmup: int = 3,
    epoch: int | None = None,
    trainer: Trainer | None = None,
    raster_cap: bool = True,
    amp: bool | None = None,
    fast_crop: bool | None = None,
    sanitise_sync: bool | None = None,
) -> dict[str, Any]:
    """Time `steps` real training steps, twice: untimed, then stage by stage.

    Args:
        cfg: the training config to profile. Its `run_dir` is created and
            written to (`metrics.jsonl`, `log.txt`) exactly as a real run
            would, so point at a scratch directory when profiling a config
            whose real run matters.
        steps: timed steps per phase (medians are reported).
        warmup: untimed steps before each phase. At least 1 is required on
            MPS: the first step of any new tensor shape pays for compiling
            it, and a training run amortises that over thousands of steps.
        epoch: which epoch's regime to profile (see `steady_state_epoch`);
            None uses the steady state.
        trainer: an already-built Trainer for this exact `cfg`, so several
            regimes can be profiled in one process. Building one on the
            Karekare-v2 config reads a 2.1 GB PLY and computes 7.5M kNN
            sizes; doing that once per profiled epoch would dominate the
            job. None builds one here.
        raster_cap: `Trainer.raster_cap_to_max_frags` for this measurement.
            False reproduces the pre-perf/train-step rasteriser (the full
            sorted fragment list reaches the kernels), which is what the
            before/after table compares against.
        amp: `Trainer.set_amp` for this measurement (None leaves the config's
            own setting alone). float16 changes numerics, so the `losses` this
            returns are the evidence for or against it.
        fast_crop, sanitise_sync: `Trainer.use_fast_crop` /
            `Trainer.sanitise_conditional` for this measurement (None leaves
            them alone). Both select between code paths that produce identical
            values; sweeping them here -- one process, one snapshot per arm --
            resolves differences that a job of one-process-per-arm cannot,
            because the machine drifts several ms between process launches.

    Returns:
        A JSON-safe dict with `config`, `points`, `untimed`, `timed`,
        `stages_ms`, `losses` and `environment` keys.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    build_start = time.perf_counter()
    if trainer is None:
        trainer = Trainer(cfg)
    build_s = time.perf_counter() - build_start
    device = trainer.device
    profile_epoch = steady_state_epoch(cfg) if epoch is None else int(epoch)
    _prepare_epoch(trainer, profile_epoch)
    trainer.raster_cap_to_max_frags = bool(raster_cap)
    if amp is not None:
        trainer.set_amp(amp)
    if fast_crop is not None:
        trainer.use_fast_crop = bool(fast_crop)
    if sanitise_sync is not None:
        trainer.sanitise_conditional = bool(sanitise_sync)

    n_points = int(trainer.point_params.xyz.shape[0])
    n_train = len(trainer.train_names)
    per_epoch = cfg.steps_per_epoch(n_train)

    # Every arm of a sweep starts from the same model, optimiser and crop RNG,
    # so the loss curves below are comparable and the trainer is handed back
    # unchanged.
    snapshot = _snapshot(trainer)
    try:

        # --- phase 1: untimed, the honest seconds per step ---
        losses: list[float] = []
        untimed: list[float] = []
        for index in range(warmup + steps):
            _sync(device)
            start = time.perf_counter()
            record = trainer.train_step()
            _sync(device)
            elapsed = time.perf_counter() - start
            if index >= warmup:
                untimed.append(elapsed)
            losses.append(float(record["loss"]))

        # --- phase 1b: the same step with the crop FROZEN ---
        #
        # Every array between the cull and the blend is sized by data: how many
        # points landed in this crop, how many fragments they emitted. Those two
        # numbers move every step, so MPS sees a tensor shape it has not compiled
        # for on every step -- a tax measured at 6-9x for a single elementwise
        # kernel on this machine (docs/ARCHITECTURE.md "Emission cost"). Repeating
        # ONE crop makes every shape repeat, so the difference between this number
        # and `untimed` is that tax at whole-step scale, and it is the evidence
        # for or against padding the fragment buffers to bucketed sizes. It is not
        # a speed-up on its own: a real run never repeats a crop.
        frozen_name = trainer.train_names[0]
        frozen_height, frozen_width = trainer.dataset.frame_size(trainer._name_to_index[frozen_name])
        frozen_zoom = 1.0
        frozen_center = (frozen_width / 2.0, frozen_height / 2.0)
        frozen: list[float] = []
        for index in range(warmup + steps):
            _sync(device)
            start = time.perf_counter()
            trainer.train_step(name=frozen_name, zoom=frozen_zoom, center=frozen_center)
            _sync(device)
            elapsed = time.perf_counter() - start
            if index >= warmup:
                frozen.append(elapsed)

        # --- phase 2: the same steps with a synchronisation between stages ---
        timer = steptimer.StepTimer(device)
        previous = steptimer.set_active(timer)
        timed: list[float] = []
        try:
            for index in range(warmup + steps):
                timer.begin_step()
                _sync(device)
                start = time.perf_counter()
                trainer.train_step()
                _sync(device)
                elapsed = time.perf_counter() - start
                row = timer.end_step()
                if index >= warmup:
                    timed.append(elapsed)
                    row["_total"] = elapsed
                else:
                    row["_warmup"] = True
        finally:
            steptimer.set_active(previous)

    finally:
        _restore(trainer, snapshot)

    rows = [row for row in timer.rows if not row.get("_warmup")]
    names: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_") or key in NOTE_KEYS or key in names:
                continue
            names.append(key)
    ordered = [name for name in STAGE_ORDER if name in names]
    ordered += [name for name in names if name not in STAGE_ORDER and name not in BACKWARD_PARTS]
    ordered += [name for name in BACKWARD_PARTS if name in names]

    stages_ms = {
        name: _median([float(row[name]) for row in rows if name in row]) * 1e3 for name in ordered
    }
    leaf_total_ms = sum(
        value for name, value in stages_ms.items() if name not in BACKWARD_PARTS
    )
    untimed_med = _median(untimed)
    result: dict[str, Any] = {
        "config": {
            "path": getattr(cfg, "_source_path", None),
            "run_dir": cfg.run_dir,
            "width": cfg.width,
            "crop": cfg.crop,
            "mode": cfg.mode,
            "layers": cfg.layers,
            "feature_channels": cfg.feature_channels,
            "use_masks": cfg.use_masks,
            "epochs": cfg.epochs,
            "device": str(device),
            "optimizer_fused": bool(cfg.optimizer_fused),
            "amp": bool(trainer.amp_enabled),
            "fast_crop": bool(trainer.use_fast_crop),
            "sanitise_sync": bool(trainer.sanitise_conditional),
        },
        "epoch_profiled": profile_epoch,
        "raster_cap": bool(raster_cap),
        "regime": {
            "vgg_weight": float(trainer.loss_fn.weights.vgg),
            "poses_trainable": bool(trainer.pose_params.delta.requires_grad),
            "structure_trainable": bool(trainer.point_params.xyz.requires_grad),
        },
        "points": n_points,
        "train_images": n_train,
        "steps_per_epoch": per_epoch,
        "trainer_build_s": build_s,
        "steps": steps,
        "warmup": warmup,
        "frozen_crop": {
            "sec_per_step": _median(frozen),
            "min_per_epoch": _median(frozen) * per_epoch / 60.0,
            "name": frozen_name,
        },
        "untimed": {
            "sec_per_step": untimed_med,
            "sec_per_step_mean": (sum(untimed) / len(untimed)) if untimed else float("nan"),
            "sec_per_step_min": min(untimed) if untimed else float("nan"),
            "sec_per_step_max": max(untimed) if untimed else float("nan"),
            "min_per_epoch": untimed_med * per_epoch / 60.0,
        },
        "timed": {
            "sec_per_step": _median(timed),
            "min_per_epoch": _median(timed) * per_epoch / 60.0,
            "leaf_sum_ms": leaf_total_ms,
        },
        "stages_ms": stages_ms,
        "stage_order": ordered,
        "fragments": _median([float(row["fragments"]) for row in rows if "fragments" in row]),
        "fragments_kept": _median(
            [float(row["fragments_kept"]) for row in rows if "fragments_kept" in row]
        ),
        "points_valid": _median([float(row["points_valid"]) for row in rows if "points_valid" in row]),
        "losses": losses,
        "environment": {
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    return result


def format_table(result: dict[str, Any]) -> str:
    """Human-readable stage table; the thing that goes in the research log."""
    cfg = result["config"]
    lines: list[str] = []
    lines.append(
        f"profile-step: {cfg['run_dir']}  device={cfg['device']}  "
        f"width={cfg['width']} crop={cfg['crop']} mode={cfg['mode']} layers={cfg['layers']}"
    )
    lines.append(
        f"  points={result['points']:,}  train_images={result['train_images']}  "
        f"steps/epoch={result['steps_per_epoch']}  fragments/step={result['fragments']:,.0f}  "
        f"points after cull={result['points_valid']:,.0f}  "
        f"fragments composited-eligible={result['fragments_kept']:,.0f}"
    )
    regime = result["regime"]
    lines.append(
        f"  epoch profiled={result['epoch_profiled']}  raster_cap={result['raster_cap']}  "
        f"fused_adam={cfg['optimizer_fused']}  amp={cfg['amp']}  "
        f"fast_crop={cfg['fast_crop']}  sanitise_sync={cfg['sanitise_sync']}  "
        f"vgg_weight={regime['vgg_weight']}  "
        f"poses_trainable={regime['poses_trainable']}  structure_trainable={regime['structure_trainable']}"
    )
    untimed = result["untimed"]
    lines.append(
        f"  UNTIMED  {untimed['sec_per_step'] * 1e3:8.1f} ms/step   "
        f"{untimed['min_per_epoch']:6.2f} min/epoch   "
        f"(min {untimed['sec_per_step_min'] * 1e3:.1f}, max {untimed['sec_per_step_max'] * 1e3:.1f})"
    )
    frozen = result.get("frozen_crop")
    if frozen:
        ratio = untimed["sec_per_step"] / frozen["sec_per_step"] if frozen["sec_per_step"] else 0.0
        lines.append(
            f"  FROZEN   {frozen['sec_per_step'] * 1e3:8.1f} ms/step   "
            f"{frozen['min_per_epoch']:6.2f} min/epoch   "
            f"(one repeated crop: every tensor shape repeats; random/frozen = {ratio:.2f}x)"
        )
    lines.append(
        f"  TIMED    {result['timed']['sec_per_step'] * 1e3:8.1f} ms/step   "
        f"{result['timed']['min_per_epoch']:6.2f} min/epoch   "
        f"(per-stage synchronisation adds overhead)"
    )
    lines.append("")
    lines.append(f"  {'stage':<22} {'ms':>9} {'% of timed step':>16}")
    lines.append(f"  {'-' * 22} {'-' * 9} {'-' * 16}")
    total = result["timed"]["sec_per_step"] * 1e3
    for name in result["stage_order"]:
        value = result["stages_ms"][name]
        share = 100.0 * value / total if total > 0 else float("nan")
        label = f"  {name}" if name in BACKWARD_PARTS else name
        lines.append(f"  {label:<22} {value:9.2f} {share:15.1f}%")
    lines.append(f"  {'-' * 22} {'-' * 9} {'-' * 16}")
    leaf = result["timed"]["leaf_sum_ms"]
    lines.append(f"  {'sum of stages':<22} {leaf:9.2f} {100.0 * leaf / total if total else 0:15.1f}%")
    lines.append(
        f"  {'unattributed':<22} {total - leaf:9.2f} "
        f"{100.0 * (total - leaf) / total if total else 0:15.1f}%"
    )
    return "\n".join(lines)



def format_comparison(results: list[dict[str, Any]]) -> str:
    """One row per profiled arm: cost, speed-up and loss-curve agreement.

    This is the before/after table. The first row is the reference; `x` is its
    seconds-per-step divided by this row's, and `max|dloss|` is the largest
    absolute difference between the two arms' step-by-step training losses.
    Because every arm starts from the same snapshot (see `_snapshot`), a
    `max|dloss|` of exactly 0 means the change did not move a single number,
    and anything else is the size of what it did move.

    Args:
        results: `profile_step` results, in the order they were measured.

    Returns:
        A printable table. Empty string for fewer than two results.
    """
    if len(results) < 2:
        return ""
    base = results[0]
    base_s = base["untimed"]["sec_per_step"]
    base_losses = base["losses"]
    header = (
        f"  {'epoch':>5} {'cap':>5} {'amp':>5} {'crop':>5} {'sync':>5} "
        f"{'ms/step':>9} {'min/epoch':>10} {'x':>6} {'frozen ms':>10} {'max|dloss|':>12}"
    )
    lines = ["comparison (row 1 is the reference):", header, "  " + "-" * (len(header) - 2)]
    for result in results:
        seconds = result["untimed"]["sec_per_step"]
        losses = result["losses"]
        if len(losses) == len(base_losses):
            delta = max((abs(a - b) for a, b in zip(losses, base_losses, strict=True)), default=0.0)
            delta_text = f"{delta:12.3e}"
        else:
            delta_text = f"{'n/a':>12}"
        lines.append(
            f"  {result['epoch_profiled']:>5} {result['raster_cap']!s:>5} "
            f"{result['config']['amp']!s:>5} {result['config']['fast_crop']!s:>5} "
            f"{result['config']['sanitise_sync']!s:>5} "
            f"{seconds * 1e3:9.1f} {result['untimed']['min_per_epoch']:10.2f} "
            f"{(base_s / seconds if seconds else float('nan')):6.2f} "
            f"{result['frozen_crop']['sec_per_step'] * 1e3:10.1f} {delta_text}"
        )
    return "\n".join(lines)


def write_json(result: dict[str, Any], path: str | Path) -> Path:
    """Write `result` as JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return out
