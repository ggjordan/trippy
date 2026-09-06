#!/usr/bin/env python
"""Stage-by-stage timing of the pyramid rasteriser on one real training crop.

Module: tools.profile_raster
Purpose: answer "where does a training step's time go, and why is mode
    `trips` so much more expensive than mode `broadcast`?" with numbers
    instead of guesses. Four measurements, each selectable from the CLI:

    1. **Stage table** (default). Reproduces one `Trainer.train_step` render
       exactly -- the same point source, the same 384-px K-adjusted crop, the
       same `image_hw = (crop, crop)` -- and times project / cull / emit /
       sort / segment / blend-forward / backward separately, with a
       `torch.mps.synchronize()` around each, for every (mode, emission
       implementation) pair asked for. Each pair is timed twice: with the
       camera frozen and with it moving, because only the second one sees a
       backend's cost for an unfamiliar tensor shape.
    2. **`--micro`**: the individual torch primitives the rasteriser is built
       from, at the sizes the real render uses them at.
    3. **`--shape-probe`** (implied by `--micro`): the same op on a repeated
       shape vs on a shape the process has never used. This is what
       identified the cause of the `trips`/`broadcast` gap
       (docs/ARCHITECTURE.md "Emission cost").
    4. **`--train-steps N`**: whole `Trainer.train_step` calls, bisected into
       rasteriser / +U-Net / whole step, which is the only number directly
       comparable to a training log's minutes per epoch.
Invariants:
    - NO IMAGERY IS EVER DISPLAYED OR WRITTEN. The scene item's `rgb` is read
      only so `trippy.scene.dataset.crop` can hand back the crop's adjusted
      intrinsics, and it is dropped immediately (AGENTS.md section 6).
    - MPS work only ever runs through `scripts/gpu_submit.sh`; this script
      does not acquire any lock and must not be run directly on the GPU.
    - The point set is built once on the CPU and cached as an .npz under
      `<TRIPPY_OUTPUT>/cache/profile_raster/`, so repeated queue jobs do not
      re-read a 5.7M-point PLY or recompute kNN sizes.
    - Timings are medians over `--repeat` runs after `--warmup` untimed runs
      (median, not mean: the queue machine has other processes on it). The
      first MPS run of any shape pays for compiling it, so a warmup of at
      least 1 is mandatory for the frozen-camera numbers to mean anything --
      and the moving-camera number is deliberately the one that does NOT get
      that benefit, because a training step never repeats a shape either.
    - Every stage is timed on the SAME fragment list, and the fragment count
      is printed with it: a mode is only "slow" relative to the work it does.
Units: all times are milliseconds (ms) of wall clock with the device idle at
    both ends. Fragment counts are absolute.
Related docs: docs/ARCHITECTURE.md ("Emission cost"); research/trips-metal.md.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from trippy.config import load_settings
from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.geom import xform_b
from trippy.raster.blend_autograd import blend_fragments
from trippy.raster.emit import (
    SortedFragments,
    cull_points,
    emit_fragments,
    layer_grid,
    project_points,
)
from trippy.raster.pyramid import render_pyramid
from trippy.raster.sort import segment_offsets, sort_fragments
from trippy.scene.dataset import SceneDataset
from trippy.scene.dataset import crop as dataset_crop
from trippy.train.config import PointSourceConfig, TrainConfig

DEFAULT_CONFIG = "experiments/EXP-0003-kk-trips-train/config_full2_trips.yaml"
MODES = ("broadcast", "trips", "trilinear")
IMPLS = ("vectorised", "loop")


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    """Block until the device has finished everything queued so far."""
    if device.type == "mps":
        torch.mps.synchronize()


def time_ms(fn: Callable[[], Any], device: torch.device, repeat: int, warmup: int) -> tuple[float, Any]:
    """Median wall-clock ms of `fn`, plus the last value it returned.

    Args:
        fn: nullary callable; its device work must be complete once `_sync`
            returns (i.e. it must not hand back a lazily-evaluated handle).
        device: device to synchronise on.
        repeat: timed runs (median is reported -- the queue machine has other
            processes on it, so a mean would be dragged by outliers).
        warmup: untimed runs first; >= 1 is required on MPS so MPSGraph
            compilation is not charged to the measurement.

    Returns:
        (median_ms, last_result).
    """
    result = None
    for _ in range(warmup):
        result = fn()
        _sync(device)
    samples: list[float] = []
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), result


# --------------------------------------------------------------------------
# scene / point-set loading (CPU only)
# --------------------------------------------------------------------------


def cached_point_set_path(cfg: TrainConfig) -> Path:
    """Path to a cached .npz of the config's point set, building it if absent.

    Keyed on a content hash of `PointSource.describe()`, so two configs that
    name the same PLY with the same options share one cache entry and the
    5.7M-point kNN pass runs once per machine instead of once per profile.

    Args:
        cfg: training config whose `point_source` to materialise.

    Returns:
        Path to the .npz (written by `PointSet.save_npz`).
    """
    settings = load_settings()
    cache_dir = settings.trippy_output / "cache" / "profile_raster"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cfg.point_source.to_source()
    # A content hash of the source description, NOT Python's `hash()`: string
    # hashing is salted per process, so that would miss the cache every run.
    key = json.dumps(source.describe(), sort_keys=True, default=str)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"points_{digest}.npz"
    if not cache_path.exists():
        start = time.perf_counter()
        point_set = source.build()
        print(f"point set: built in {time.perf_counter() - start:.1f} s -> {cache_path}")
        point_set.save_npz(cache_path)
    return cache_path


def load_points(cfg: TrainConfig, subsample: int | None, seed: int) -> dict[str, torch.Tensor]:
    """Build (or load from cache) the run's point set as trainer-shaped tensors.

    The returned `size`/`conf` are the *effective* values a `PointParams`
    forward produces at initialisation (softplus(inverse_softplus(size0)) ==
    size0 and likewise for conf), so the rasteriser sees exactly what step 1
    of the real run sees. `feat` is the same shape the trainer uses but is
    filled deterministically here -- the rasteriser's cost does not depend on
    feature values, only on their shape.

    Args:
        cfg: the training config whose `point_source` to build.
        subsample: keep this many points (uniform, seeded) or None for all.
        seed: RNG seed for the subsample and the feature fill.

    Returns:
        dict of CPU float32 tensors: xyz (N, 3), size (N,), conf (N,),
        feat (N, C).
    """
    from trippy.points.source import PointSet

    point_set = PointSet.load_npz(cached_point_set_path(cfg))

    xyz = torch.from_numpy(point_set.xyz).to(torch.float32)
    size = torch.from_numpy(point_set.size0).to(torch.float32).clamp(min=torch.finfo(torch.float32).eps)
    conf = torch.from_numpy(point_set.conf0).to(torch.float32)
    rgb = torch.from_numpy(point_set.rgb0).to(torch.float32)

    if subsample is not None and subsample < xyz.shape[0]:
        generator = torch.Generator().manual_seed(seed)
        pick = torch.randperm(xyz.shape[0], generator=generator)[:subsample]
        xyz, size, conf, rgb = xyz[pick], size[pick], conf[pick], rgb[pick]

    channels = cfg.feature_channels
    feat = torch.zeros(xyz.shape[0], channels, dtype=torch.float32)
    feat[:, :3] = rgb
    if channels > 3:
        generator = torch.Generator().manual_seed(seed + 1)
        feat[:, 3:] = torch.randn(xyz.shape[0], channels - 3, generator=generator) * 0.01
    return {"xyz": xyz, "size": size, "conf": conf, "feat": feat}


def load_camera(cfg: TrainConfig, image: str | None, zoom: float) -> dict[str, torch.Tensor]:
    """One real training crop's (K, R, t) -- intrinsics only, pixels discarded.

    Args:
        cfg: training config (scene_root, width, crop).
        image: image name to use, or None for the first in the dataset.
        zoom: crop zoom factor, as `Trainer._sample_zoom` would produce.

    Returns:
        dict with K (3, 3), R (3, 3), t (3,) float32 CPU tensors and the
        image name. The crop centre is the frame centre, which is the median
        case for `Trainer._sample_crop_center`.
    """
    settings = load_settings()
    cache_root = Path(cfg.cache_root) if cfg.cache_root else settings.trippy_output / "cache"
    dataset = SceneDataset(cfg.scene_root, cfg.width, cache_root, device="cpu")
    index = 0 if image is None else dataset.names.index(image)
    item = dataset[index]
    cropped = dataset_crop(item, size=cfg.crop, zoom=zoom)
    K = cropped["K"].clone()
    R = xform_b.qvec2R(item["qvec"]).clone()
    t = item["tvec"].clone()
    del item, cropped  # drop the pixels immediately; nothing here ever looks at them
    return {"K": K, "R": R, "t": t, "name": dataset.names[index]}


# --------------------------------------------------------------------------
# the profile itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scene:
    """Everything a render needs that is not a trainable tensor."""

    K: Tensor
    R: Tensor
    t: Tensor
    grid: Any
    mode: str
    impl: str
    alpha_min: float


@dataclass(frozen=True)
class _GradState:
    """The four trainable tensors, cloned so each measurement owns its graph."""

    xyz: Tensor
    size: Tensor
    conf: Tensor
    feat: Tensor

    @classmethod
    def of(
        cls, xyz: Tensor, size: Tensor, conf: Tensor, feat: Tensor, geometry_grad: bool
    ) -> _GradState:
        """Clone the inputs; `feat` always requires grad, the geometry optionally.

        `geometry_grad=False` isolates `blend_bwd` plus the `index_add_`
        reduction; True adds the whole emission graph on top.
        """
        return cls(
            xyz=xyz.detach().clone().requires_grad_(geometry_grad),
            size=size.detach().clone().requires_grad_(geometry_grad),
            conf=conf.detach().clone().requires_grad_(geometry_grad),
            feat=feat.detach().clone().requires_grad_(True),
        )

    def params(self) -> list[Tensor]:
        return [p for p in (self.feat, self.xyz, self.size, self.conf) if p.requires_grad]


def render_loss(scene: _Scene, state: _GradState) -> tuple[Tensor, int]:
    """One crop's project -> cull -> emit -> sort -> segment -> blend, as a scalar.

    The scalar is deliberately a plain function of both outputs that carry
    gradient (`out` and `t_final`) rather than a real image loss: this script
    measures the rasteriser, and a U-Net would drown it.

    Returns:
        (loss, fragment count).
    """
    uv, depth, size_px = project_points(
        state.xyz, state.size, scene.K, scene.R, scene.t, znear=RASTER_ZNEAR
    )
    valid = cull_points(uv, depth, size_px, scene.grid, znear=RASTER_ZNEAR)
    frags = emit_fragments(
        uv,
        depth,
        size_px,
        state.conf,
        scene.grid,
        mode=scene.mode,
        valid=valid,
        alpha_min=scene.alpha_min,
        impl=scene.impl,
    )
    perm = sort_fragments(frags.layer_pixel, frags.depth, max_layer_pixel=scene.grid.total - 1)
    layer_pixel = frags.layer_pixel.index_select(0, perm)
    sorted_frags = SortedFragments(
        layer_pixel=layer_pixel,
        layer=frags.layer.index_select(0, perm),
        pixel=frags.pixel.index_select(0, perm),
        depth=frags.depth.index_select(0, perm),
        point_id=frags.point_id.index_select(0, perm),
        alpha=frags.alpha.index_select(0, perm),
        offsets=segment_offsets(layer_pixel, scene.grid.total),
        grid=scene.grid,
    )
    out, t_final, _n_used, _depth_sum = blend_fragments(
        sorted_frags, state.feat, max_frags=RASTER_MAX_FRAGS, t_cutoff=RASTER_T_CUTOFF
    )
    return out.square().mean() + t_final.mean(), len(frags)


def _grad_of(loss: Tensor, state: _GradState) -> tuple[Tensor | None, ...]:
    """Backward for whichever of `state`'s tensors require grad."""
    return torch.autograd.grad(loss, state.params(), allow_unused=True)


def profile_mode(
    tensors: dict[str, torch.Tensor],
    camera: dict[str, torch.Tensor],
    crop: int,
    num_layers: int,
    mode: str,
    impl: str,
    device: torch.device,
    repeat: int,
    warmup: int,
    alpha_min: float,
) -> dict[str, Any]:
    """Time every rasteriser stage for one (mode, emission impl) pair.

    Stages are run in pipeline order and each one's inputs are the previous
    one's real outputs, so the numbers add up to a real forward pass. The
    backward is measured twice -- once with only `feat` requiring grad (which
    exercises `blend_bwd` plus the `index_add_` reduction and nothing else)
    and once with the geometry parameters requiring grad as well. The
    difference is the cost of the emission graph.

    Returns:
        dict of stage name -> ms, plus fragment counts and the point counts
        that survived the cull.
    """
    grid = layer_grid(crop, crop, num_layers)
    xyz = tensors["xyz"].to(device)
    size = tensors["size"].to(device)
    conf = tensors["conf"].to(device)
    feat = tensors["feat"].to(device)
    K = camera["K"].to(device)
    R = camera["R"].to(device)
    t = camera["t"].to(device)

    out: dict[str, Any] = {"mode": mode, "impl": impl, "num_points": int(xyz.shape[0])}

    # --- forward, stage by stage (no autograd graph: pure kernel cost) ---
    with torch.no_grad():
        ms, projected = time_ms(
            lambda: project_points(xyz, size, K, R, t, znear=RASTER_ZNEAR), device, repeat, warmup
        )
        out["project"] = ms
        uv, depth, size_px = projected

        ms, valid = time_ms(
            lambda: cull_points(uv, depth, size_px, grid, znear=RASTER_ZNEAR), device, repeat, warmup
        )
        out["cull"] = ms
        out["points_after_cull"] = int(valid.sum().item())

        ms, frags = time_ms(
            lambda: emit_fragments(
                uv, depth, size_px, conf, grid, mode=mode, valid=valid, alpha_min=alpha_min, impl=impl
            ),
            device,
            repeat,
            warmup,
        )
        out["emit"] = ms
        out["fragments"] = len(frags)

        ms, perm = time_ms(
            lambda: sort_fragments(
                frags.layer_pixel, frags.depth, max_layer_pixel=grid.total - 1
            ),
            device,
            repeat,
            warmup,
        )
        out["sort"] = ms

        # The documented fallback: two stable 32-bit-keyed sorts instead of one
        # int64 argsort. `tests/test_raster_sort.py` pins that the two produce
        # the SAME permutation, so if this is faster it can simply be switched on.
        ms, _ = time_ms(
            lambda: sort_fragments(
                frags.layer_pixel, frags.depth, method="two_pass", max_layer_pixel=grid.total - 1
            ),
            device,
            repeat,
            warmup,
        )
        out["sort_two_pass"] = ms

        ms, layer_pixel_sorted = time_ms(
            lambda: frags.layer_pixel.index_select(0, perm), device, repeat, warmup
        )
        # build_sorted_fragments applies the permutation to six tensors.
        out["gather_one"] = ms
        out["gather"] = ms * 6.0

        ms, offsets = time_ms(
            lambda: segment_offsets(layer_pixel_sorted, grid.total), device, repeat, warmup
        )
        out["segments"] = ms

        sorted_frags = SortedFragments(
            layer_pixel=layer_pixel_sorted,
            layer=frags.layer.index_select(0, perm),
            pixel=frags.pixel.index_select(0, perm),
            depth=frags.depth.index_select(0, perm),
            point_id=frags.point_id.index_select(0, perm),
            alpha=frags.alpha.index_select(0, perm),
            offsets=offsets,
            grid=grid,
        )
        ms, _ = time_ms(
            lambda: blend_fragments(
                sorted_frags, feat, max_frags=RASTER_MAX_FRAGS, t_cutoff=RASTER_T_CUTOFF
            ),
            device,
            repeat,
            warmup,
        )
        out["blend_fwd"] = ms

    # --- forward + backward, end to end ---
    scene = _Scene(K=K, R=R, t=t, grid=grid, mode=mode, impl=impl, alpha_min=alpha_min)
    for label, geometry_grad in (("feat_only", False), ("full", True)):
        state = _GradState.of(xyz, size, conf, feat, geometry_grad)
        # `st=state` binds the current iteration's tensors into the lambda
        # rather than closing over the loop variable.
        ms_fwd, _ = time_ms(lambda st=state: render_loss(scene, st)[0], device, repeat, warmup)
        out[f"fwd_graph_{label}"] = ms_fwd
        ms_both, _ = time_ms(
            lambda st=state: _grad_of(render_loss(scene, st)[0], st), device, repeat, warmup
        )
        out[f"fwd_bwd_{label}"] = ms_both
        out[f"bwd_{label}"] = ms_both - ms_fwd

    # --- the realistic case: a DIFFERENT camera every iteration ---
    #
    # Everything above is measured at a frozen shape, so the backend's
    # shape-keyed kernel/graph cache is warm by the second run. A real training
    # step never repeats a shape: the crop centre, the zoom and the pose all
    # move, so the number of points the cull keeps -- and, in the loop
    # implementation, the number selected on each individual layer -- is
    # different every step. If a backend pays for an unseen shape, only this
    # measurement sees it, and it is the one that has to explain the observed
    # seconds per step.
    state = _GradState.of(xyz, size, conf, feat, True)
    offsets_m = [0.05 * k for k in range(-((repeat + warmup) // 2), (repeat + warmup + 1) // 2)]
    kept_counts: list[int] = []
    samples: list[float] = []
    for step, shift in enumerate(offsets_m):
        moved = _Scene(
            K=K,
            t=t + torch.tensor([shift, 0.0, 0.0], device=device, dtype=t.dtype),
            R=R,
            grid=grid,
            mode=mode,
            impl=impl,
            alpha_min=alpha_min,
        )
        _sync(device)
        start = time.perf_counter()
        loss, num_fragments = render_loss(moved, state)
        _grad_of(loss, state)
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1e3
        if step >= warmup:
            samples.append(elapsed)
            kept_counts.append(num_fragments)
    out["fwd_bwd_varying"] = statistics.median(samples)
    out["fwd_bwd_varying_max"] = max(samples)
    out["varying_fragments"] = kept_counts

    return out


# --------------------------------------------------------------------------
# micro-benchmarks: the individual torch ops the emission path leans on
# --------------------------------------------------------------------------


def micro_benchmarks(
    num_points: int, num_valid: int, num_fragments: int, num_pixels: int,
    channels: int, device: torch.device, repeat: int, warmup: int,
) -> list[dict[str, Any]]:
    """Time the primitives the rasteriser is built from, at the real sizes.

    This is what separates "mode X does more work" from "mode X uses an op
    that is pathological on this backend": every row here is a single torch
    call at the size the real render uses it at.
    """
    rows: list[dict[str, Any]] = []

    def add(name: str, size: int, fn: Callable[[], Any]) -> None:
        ms, _ = time_ms(fn, device, repeat, warmup)
        rows.append({"op": name, "n": size, "ms": ms})

    generator = torch.Generator(device="cpu").manual_seed(0)
    mask = torch.zeros(num_points, dtype=torch.bool)
    mask[torch.randperm(num_points, generator=generator)[:num_valid]] = True
    mask = mask.to(device)
    big = torch.rand(num_points, 2, generator=generator).to(device)
    frag_mask = (torch.rand(num_fragments, generator=generator) < 0.6).to(device)
    keys = torch.randint(0, num_pixels, (num_fragments,), generator=generator).to(device)
    keys64 = keys.to(torch.int64) * (1 << 32) + torch.randint(
        0, 1 << 30, (num_fragments,), generator=generator
    ).to(device)
    keys_f32 = torch.rand(num_fragments, generator=generator).to(device)
    idx = torch.randint(0, num_points, (num_fragments,), generator=generator).to(device)
    values = torch.rand(num_fragments, channels, generator=generator).to(device)
    bins = torch.arange(num_pixels + 1, dtype=torch.int64, device=device)
    keys_sorted = torch.sort(keys.to(torch.int64)).values

    add("nonzero(bool)", num_points, lambda: torch.nonzero(mask, as_tuple=False))
    add("nonzero(bool)", num_fragments, lambda: torch.nonzero(frag_mask, as_tuple=False))
    add("mul (N,2) contiguous", num_points, lambda: big * 0.5)
    add("floor (N,2)", num_points, lambda: torch.floor(big))
    add("ge on strided col [:,0]", num_points, lambda: big[:, 0] >= 0.0)
    add("ge on contiguous col", num_points, lambda: big.select(1, 0).contiguous() >= 0.0)
    add("argsort int64 stable", num_fragments, lambda: torch.argsort(keys64, stable=True))
    add("argsort int32 stable", num_fragments, lambda: torch.argsort(keys.to(torch.int32), stable=True))
    add("argsort int64 unstable", num_fragments, lambda: torch.argsort(keys64, stable=False))
    add("argsort float32 stable", num_fragments, lambda: torch.argsort(keys_f32, stable=True))
    add("searchsorted P bins", num_pixels, lambda: torch.searchsorted(keys_sorted, bins))
    add("bincount P bins", num_pixels, lambda: torch.bincount(keys_sorted, minlength=num_pixels))
    add(
        "index_add_ (N,C) <- F",
        num_fragments,
        lambda: torch.zeros(num_points, channels, device=device).index_add_(0, idx, values),
    )
    add("index_select (N,2) by F", num_fragments, lambda: big.index_select(0, idx))
    add("pow(2^int tensor)", num_valid, lambda: torch.pow(
        torch.full((num_valid,), 2.0, device=device), torch.zeros(num_valid, device=device)
    ))
    return rows


def shape_cache_probe(
    base_size: int, device: torch.device, repeat: int, warmup: int
) -> list[dict[str, Any]]:
    """Does this backend charge for a tensor shape it has not seen before?

    The decisive experiment behind "mode trips is 10x a broadcast step even
    though it emits FEWER fragments". Every other measurement in this script
    repeats one shape, so a shape-keyed kernel/graph cache is warm by the
    second run -- but a training step never repeats a shape, and the loop
    emission produces five *different* per-layer sizes per step in mode
    "trips" against one repeated size in mode "broadcast".

    Each op is timed twice: `warm` runs it on the same tensor every time,
    `cold` runs it on a tensor of a size the process has never used. The
    ratio is the per-shape cost.

    Args:
        base_size: rows to use for the warm case; the cold case walks
            `base_size + 1000 * k`.
        device, repeat, warmup: as elsewhere.

    Returns:
        one row per op with "warm_ms", "cold_ms" and their ratio.
    """
    rows: list[dict[str, Any]] = []
    total = repeat + warmup

    def probe(name: str, make: Callable[[int], Any], run: Callable[[Any], Any]) -> None:
        warm_input = make(base_size)
        warm_ms, _ = time_ms(lambda: run(warm_input), device, repeat, warmup)
        cold_inputs = [make(base_size + 1000 * k) for k in range(total)]
        samples: list[float] = []
        for step, item in enumerate(cold_inputs):
            _sync(device)
            start = time.perf_counter()
            run(item)
            _sync(device)
            if step >= warmup:
                samples.append((time.perf_counter() - start) * 1e3)
        cold_ms = statistics.median(samples)
        rows.append(
            {"op": name, "warm_ms": warm_ms, "cold_ms": cold_ms, "ratio": cold_ms / max(warm_ms, 1e-9)}
        )

    probe(
        "floor(x * 0.5) on (n,2)",
        lambda n: torch.rand(n, 2, device=device),
        lambda x: torch.floor(x * 0.5),
    )
    probe(
        "nonzero(bool n)",
        lambda n: torch.rand(n, device=device) < 0.25,
        lambda x: torch.nonzero(x, as_tuple=False),
    )
    probe(
        "index_select (n,2) by n/4",
        lambda n: (
            torch.rand(n, 2, device=device),
            torch.randint(0, n, (max(n // 4, 1),), device=device),
        ),
        lambda pair: pair[0].index_select(0, pair[1]),
    )
    probe(
        "argsort int64 stable",
        lambda n: torch.randint(0, 1 << 40, (n,), device=device),
        lambda x: torch.argsort(x, stable=True),
    )
    probe(
        "sum() reduction",
        lambda n: torch.rand(n, device=device),
        lambda x: x.sum(),
    )
    return rows


@contextlib.contextmanager
def forced_emit_impl(impl: str):
    """Make every `emit_fragments` call use `impl`, for the duration.

    A profiling-only monkeypatch. `render_pyramid` deliberately has no
    `emit_impl` parameter -- production must only ever run the default -- so
    an end-to-end A/B of the two emission implementations has to be arranged
    from outside. `build_sorted_fragments` resolves `emit_fragments` through
    the module global, so swapping that global is enough, and the wrapper
    overrides the caller's own `impl=` keyword rather than merely defaulting
    it.
    """
    from trippy.raster import emit as emit_module

    original = emit_module.emit_fragments

    def forced(*args: Any, **kwargs: Any) -> Any:
        kwargs["impl"] = impl
        return original(*args, **kwargs)

    emit_module.emit_fragments = forced
    try:
        yield
    finally:
        emit_module.emit_fragments = original


def _profile_one_trainer(
    config_path: str, impl: str, steps: int, warmup: int, run_root: Path, device_override: str | None
) -> dict[str, Any]:
    """Build one `Trainer` and time three nested slices of its training step.

    The three phases are cumulative, so their differences name the cost:
    `raster_s` is `render_pyramid` forward+backward on the run's real
    parameters, `raster_net_s` adds the U-Net decode (`Trainer._render`), and
    `step_s` is the whole `Trainer.train_step` -- crop sampling, tone mapper,
    loss, gradient sanitising, Adam, metrics write.

    Args:
        config_path: training config YAML.
        impl: emission implementation to force for the whole measurement.
        steps: timed calls per phase.
        warmup: untimed calls per phase first (>= 1; the first call compiles
            kernels and pages the point set in).
        run_root: parent directory for this run's scratch run dir.
        device_override: force a device, or None to use the config's.

    Returns:
        one row with the median and max seconds of each phase.
    """
    from trippy.train.trainer import Trainer

    cfg = TrainConfig.load_yaml(config_path)
    cfg.run_dir = str(run_root / f"{Path(config_path).stem}-{impl}")
    # Same point set, loaded instead of rebuilt: four Trainers would otherwise
    # each re-read a 1.7 GB PLY and re-run kNN over 5.7M points while holding
    # the GPU lock. Only the checkpoint's recorded `point_source` metadata
    # differs, and these run dirs are scratch.
    cfg.point_source = PointSourceConfig(type="npz", path=str(cached_point_set_path(cfg)))
    # LPIPS is an eval-time metric and would pull a VGG backbone in; the
    # training step itself never uses it.
    cfg.eval_lpips = False
    if device_override is not None:
        cfg.device = device_override
    trainer = Trainer(cfg)
    device = trainer.device
    crop_hw = (cfg.crop, cfg.crop)

    def sample_crop() -> tuple[Tensor, Tensor, Tensor]:
        """One crop drawn exactly the way `Trainer.train_step` draws it."""
        pick = int(torch.randint(0, len(trainer.train_names), (), generator=trainer._rng).item())
        frame_index = trainer._name_to_index[trainer.train_names[pick]]
        item = trainer.dataset[frame_index]
        height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
        zoom = trainer._sample_zoom()
        center = trainer._sample_crop_center(height, width, zoom)
        cropped = dataset_crop(item, size=cfg.crop, zoom=zoom, center=center)
        R, t = trainer._pose_for(item, frame_index)
        return cropped["K"], R, t

    def raster_only() -> None:
        """Rasteriser forward + backward on the run's real parameters."""
        trainer.optimizer.zero_grad(set_to_none=True)
        K, R, t = sample_crop()
        layers, _aux = render_pyramid(
            trainer.point_params.xyz,
            trainer.point_params.size(),
            trainer.point_params.feat,
            trainer.point_params.conf(),
            K,
            R,
            t,
            crop_hw,
            num_layers=cfg.layers,
            mode=cfg.mode,
            bg=trainer.background,
            pixel_center=cfg.pixel_center,
            pyramid_halving=cfg.pyramid_halving,
        )
        sum(layer.square().mean() for layer in layers).backward()

    def raster_and_net() -> None:
        """The same, plus the U-Net decode."""
        trainer.optimizer.zero_grad(set_to_none=True)
        K, R, t = sample_crop()
        net_out, _layers, _aux = trainer._render(K, R, t, crop_hw)
        net_out.square().mean().backward()

    phases: dict[str, Callable[[], Any]] = {
        "raster_s": raster_only,
        "raster_net_s": raster_and_net,
        "step_s": trainer.train_step,
    }
    row: dict[str, Any] = {
        "config": config_path,
        "mode": cfg.mode,
        "impl": impl,
        "steps": steps,
        "points": len(trainer.point_params),
    }
    with forced_emit_impl(impl):
        for phase, fn in phases.items():
            for _ in range(warmup):
                fn()
            samples: list[float] = []
            for _ in range(steps):
                _sync(device)
                start = time.perf_counter()
                fn()
                _sync(device)
                samples.append(time.perf_counter() - start)
            row[phase] = statistics.median(samples)
            row[phase + "_max"] = max(samples)
    print(
        f"  {cfg.mode}/{impl}: raster {row['raster_s'] * 1e3:.0f} ms, "
        f"+U-Net {row['raster_net_s'] * 1e3:.0f} ms, whole step "
        f"{row['step_s'] * 1e3:.0f} ms (max {row['step_s_max'] * 1e3:.0f})"
    )
    # The Trainer (5.7M points x parameters + Adam state) goes out of scope with
    # this frame; the MPS cache is what has to be told about it.
    if device.type == "mps":
        torch.mps.empty_cache()
    return row


def profile_train_steps(
    config_path: str,
    steps: int,
    warmup: int,
    impls: list[str],
    run_root: Path,
    device_override: str | None,
) -> list[dict[str, Any]]:
    """`_profile_one_trainer` once per emission implementation."""
    return [
        _profile_one_trainer(config_path, impl, steps, warmup, run_root, device_override)
        for impl in impls
    ]


def memory_report(device: torch.device) -> dict[str, float]:
    """Current MPS allocator numbers in MB (empty dict off MPS)."""
    if device.type != "mps":
        return {}
    return {
        "current_allocated_mb": torch.mps.current_allocated_memory() / 1e6,
        "driver_allocated_mb": torch.mps.driver_allocated_memory() / 1e6,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

STAGE_COLUMNS = (
    "project", "cull", "emit", "sort", "sort_two_pass", "gather", "segments", "blend_fwd",
    "fwd_graph_full", "bwd_feat_only", "bwd_full", "fwd_bwd_full", "fwd_bwd_varying",
)


def print_table(rows: list[dict[str, Any]]) -> None:
    """One line per (mode, impl); ms per stage, then the fragment counts."""
    header = f"{'mode':<11}{'impl':<12}" + "".join(f"{c:>16}" for c in STAGE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['mode']:<11}{row['impl']:<12}"
        line += "".join(f"{row.get(c, float('nan')):>16.2f}" for c in STAGE_COLUMNS)
        print(line)
    print()
    print(
        f"{'mode':<11}{'impl':<12}{'points kept':>14}{'fragments':>14}{'frag/point':>14}"
        f"{'moving-camera fragments':>28}"
    )
    for row in rows:
        kept = row["points_after_cull"]
        varying = row.get("varying_fragments") or [0]
        print(
            f"{row['mode']:<11}{row['impl']:<12}{kept:>14d}{row['fragments']:>14d}"
            f"{row['fragments'] / max(kept, 1):>14.2f}"
            f"{f'{min(varying)}..{max(varying)}':>28}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="training config to mirror")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--crop", type=int, default=None, help="override the config's crop size")
    parser.add_argument("--layers", type=int, default=None, help="override the config's layer count")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--points", type=int, default=None, help="subsample to this many points")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="cap the POINT SOURCE (kNN sizes are then computed on the subsample, so this "
        "is a different -- and much cheaper to build -- point set than --points)",
    )
    parser.add_argument("--image", default=None, help="scene image whose pose/K to use")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--alpha-min", type=float, default=RASTER_ALPHA_MIN)
    parser.add_argument("--micro", action="store_true", help="also time the individual torch ops")
    parser.add_argument(
        "--shape-probe",
        action="store_true",
        help="also measure the cost of a tensor shape the backend has not seen before",
    )
    parser.add_argument("--json", default=None, help="write the raw numbers here")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="also time this many whole Trainer.train_step calls (0 = skip)",
    )
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-impls", default=",".join(IMPLS))
    parser.add_argument(
        "--train-configs",
        default=None,
        help="comma-separated configs for --train-steps (default: --config)",
    )
    parser.add_argument(
        "--skip-stages", action="store_true", help="skip the per-stage profile entirely"
    )
    args = parser.parse_args()

    cfg = TrainConfig.load_yaml(args.config)
    if args.max_points is not None:
        cfg.point_source.max_points = args.max_points
    crop = args.crop or cfg.crop
    num_layers = args.layers or cfg.layers
    device = torch.device(args.device)

    grid = layer_grid(crop, crop, num_layers)
    rows: list[dict[str, Any]] = []
    image_name: str | None = None
    if not args.skip_stages:
        # Deliberately inside the guard: building the point source reads a
        # 1.7 GB PLY and runs kNN over 5.7M points, which a --skip-stages run
        # (whole train steps only, where the Trainer builds its own) must not pay.
        tensors = load_points(cfg, args.points, seed=cfg.seed)
        camera = load_camera(cfg, args.image, args.zoom)
        image_name = camera["name"]
        print(
            f"config={args.config} scene={cfg.scene_root} image={camera['name']} "
            f"crop={crop} layers={num_layers} zoom={args.zoom} device={device} "
            f"points={tensors['xyz'].shape[0]} channels={tensors['feat'].shape[1]} "
            f"repeat={args.repeat} warmup={args.warmup}"
        )
        print(f"pyramid shapes={grid.shapes} total_layer_pixels={grid.total}")

    for mode in [] if args.skip_stages else args.modes.split(","):
        for impl in args.impls.split(","):
            row = profile_mode(
                tensors, camera, crop, num_layers, mode, impl, device,
                args.repeat, args.warmup, args.alpha_min,
            )
            rows.append(row)
            print(
                f"  done {mode}/{impl}: emit {row['emit']:.1f} ms, "
                f"{row['fragments']} fragments, fwd+bwd fixed-shape "
                f"{row['fwd_bwd_full']:.1f} ms / moving camera "
                f"{row['fwd_bwd_varying']:.1f} ms"
            )
    if rows:
        print()
        print_table(rows)

    train_rows: list[dict[str, Any]] = []
    if args.train_steps > 0:
        settings = load_settings()
        run_root = settings.trippy_output / "runs" / "perf" / "profile_raster"
        print()
        print(f"whole train steps (scratch run dirs under {run_root}):")
        for train_config in (args.train_configs or args.config).split(","):
            train_rows += profile_train_steps(
                train_config,
                args.train_steps,
                args.train_warmup,
                args.train_impls.split(","),
                run_root,
                args.device if args.device else None,
            )
        print()
        header = (
            f"{'config':<34}{'mode':<11}{'impl':<12}"
            f"{'raster s':>12}{'+U-Net s':>12}{'whole step s':>14}{'step max':>11}"
        )
        print(header)
        print("-" * len(header))
        for row in train_rows:
            print(
                f"{Path(row['config']).name:<34}{row['mode']:<11}{row['impl']:<12}"
                f"{row['raster_s']:>12.3f}{row['raster_net_s']:>12.3f}"
                f"{row['step_s']:>14.3f}{row['step_s_max']:>11.3f}"
            )

    micro: list[dict[str, Any]] = []
    if args.micro and rows:
        reference = rows[0]
        micro = micro_benchmarks(
            reference["num_points"], max(reference["points_after_cull"], 1),
            max(reference["fragments"], 1), grid.total, tensors["feat"].shape[1],
            device, args.repeat, args.warmup,
        )
        print()
        print(f"{'op':<28}{'n':>12}{'ms':>10}")
        print("-" * 50)
        for row in micro:
            print(f"{row['op']:<28}{row['n']:>12d}{row['ms']:>10.3f}")

    probe: list[dict[str, Any]] = []
    # The shape probe is cheap and it is the measurement that separates "more
    # work" from "a shape the backend has not compiled for", so --micro implies it.
    if (args.shape_probe or args.micro) and rows:
        probe = shape_cache_probe(
            max(rows[0]["points_after_cull"], 1), device, args.repeat, args.warmup
        )
        print()
        print(f"{'op (n = points kept)':<30}{'warm ms':>10}{'cold ms':>10}{'cold/warm':>12}")
        print("-" * 62)
        for row in probe:
            print(
                f"{row['op']:<30}{row['warm_ms']:>10.3f}{row['cold_ms']:>10.3f}"
                f"{row['ratio']:>12.2f}"
            )

    memory = memory_report(device)
    if memory:
        print()
        print("MPS memory: " + ", ".join(f"{k}={v:.1f}" for k, v in memory.items()))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "config": args.config,
                    "crop": crop,
                    "layers": num_layers,
                    "device": str(device),
                    "image": image_name,
                    "zoom": args.zoom,
                    "stages": rows,
                    "train_steps": train_rows,
                    "micro": micro,
                    "shape_probe": probe,
                    "memory": memory,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
