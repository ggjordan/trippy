"""Orchestration behind `trippy render`: scene + points -> pyramid -> sheets.

Module: trippy.render.pyramid_render
Purpose: wire the pieces documented in docs/ARCHITECTURE.md's "Forward pass
    data flow" together for a CLI user: load a COLMAP scene restricted to an
    explicit list of frames, build a point set (currently GaussianPlySource
    only), rasterise the TRIPS pyramid for each frame (RGB features, no
    U-Net -- that stage does not exist yet), and write per-frame PNGs, a
    per-frame contact sheet, one summary sheet across frames, and a
    metrics.json with an emit/sort/blend timing breakdown.
Invariants:
    - Never builds a SceneDataset over every registered image: `_NamedSceneDataset`
      below duplicates `trippy.scene.dataset.SceneDataset.__init__` with an
      explicit `names` filter instead of a `limit` prefix, because
      trippy/scene/dataset.py is out of scope for this change and its cache
      builder runs eagerly (in `__init__`, not lazily per `__getitem__`) over
      whatever `self._names` holds.
    - No background compositing (`bg=None` throughout): this is the raw,
      un-networked level-0/1/2/3/4 splat AGENTS.md's honesty rule asks for
      ("raw composite ... no U-Net"), so holes are rendered black, not
      papered over with a background colour.
    - `coverage_stats()` (mean coverage, full frame and a central crop) is
      computed directly from the `T_final` tensor, never from a rendered
      image -- a coverage verdict must never require opening an image
      derived from one of Jordan's scenes (AGENTS.md privacy rule).
      `coverage.png` itself is a `colorize()` output with no photo pixels
      blended in, so it is safe to open, but the numeric stats exist so a
      caller never has to.
    - Timing is measured by calling the same building blocks
      `trippy.raster.pyramid.render_pyramid` calls internally (project ->
      cull -> emit -> sort -> segment -> blend_fwd/composite_sorted),
      instrumented with `torch.mps.synchronize()` barriers between stages.
      trippy/raster/pyramid.py is out of scope for this change, so this is a
      deliberate duplication of its dispatch logic rather than an edit to it;
      tests/test_raster_ref.py-style agreement is not re-verified here (the
      raster module's own tests already do that) -- this module only adds
      instrumentation around calls to functions that module already exports.
Related docs: docs/ARCHITECTURE.md (forward pass data flow); docs/EXPERIMENTS.md
    (contact-sheet / run-location conventions, mandatory honesty sheet);
    experiments/EXP-0001-forward-pyramid/README.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from trippy.config import load_settings
from trippy.constants import (
    DEFAULT_MIN_OPACITY,
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_NUM_LAYERS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
    RENDER_CACHE_SUBDIR,
    RENDER_CENTER_REGION_FRAC,
    RENDER_COVERAGE_EPS,
    RENDER_DEPTH_PERCENTILE_HIGH,
    RENDER_DEPTH_PERCENTILE_LOW,
    SCENE_CACHE_META_FILENAME,
    TRAIN_DEFAULT_MODE,
)
from trippy.geom.xform_b import qvec2R
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.raster import metal_lib
from trippy.raster.emit import cull_points, emit_fragments, layer_grid, project_points
from trippy.raster.ref_torch import composite_sorted, split_layers
from trippy.raster.sort import segment_offsets, sort_fragments
from trippy.render.sheets import colorize, contact_sheet, save_png, side_by_side
from trippy.scene import colmap_io
from trippy.scene.dataset import SceneDataset, resolve_sparse_dir


class _NamedSceneDataset(SceneDataset):
    """SceneDataset restricted to an explicit set of frame names.

    See the module docstring: this exists only because SceneDataset's
    `limit` parameter keeps a sorted *prefix* of the registered images,
    and its cache builder runs eagerly in `__init__`, so there is no way to
    ask the base class for a handful of specific, possibly-non-contiguous
    frame names without either processing the whole scene first or
    duplicating the relevant slice of `__init__` here.
    """

    def __init__(
        self,
        scene_root: str | Path,
        width: int,
        cache_root: str | Path,
        names: list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        self.scene_root = Path(scene_root)
        self.width = int(width)
        self.cache_root = Path(cache_root)
        self.device = torch.device(device)

        sparse_dir = resolve_sparse_dir(self.scene_root)
        self._scene = colmap_io.load_colmap_model(sparse_dir)
        self._images_by_name = self._scene.images_by_name()

        missing = sorted(n for n in set(names) if n not in self._images_by_name)
        if missing:
            raise KeyError(
                f"frame(s) not found in the registered COLMAP images of {self.scene_root}: {missing}"
            )
        self._names = sorted(set(names))

        self.cache_dir = self.cache_root / self.scene_root.name / f"w{self.width}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.cache_dir / SCENE_CACHE_META_FILENAME

        self._meta = self._load_or_build_cache()


def _sync(device: torch.device) -> None:
    """Block until pending device work finishes (no-op on CPU)."""
    if device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class FrameTiming:
    """Wall-clock milliseconds for one frame's forward pass, by stage."""

    emit_ms: float
    sort_ms: float
    blend_ms: float
    total_ms: float


def render_one_frame(
    xyz: Tensor,
    size: Tensor,
    feat: Tensor,
    conf: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    image_hw: tuple[int, int],
    num_layers: int,
    mode: str,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
    alpha_min: float = RASTER_ALPHA_MIN,
    znear: float = RASTER_ZNEAR,
) -> tuple[list[Tensor], dict, FrameTiming, int, int]:
    """Render one pyramid, with an emit/sort/blend timing breakdown.

    Args: same meaning as trippy.raster.render_pyramid's positional
        arguments (see that function's docstring for units/frames); `feat`
        must already be on the target device.

    Returns:
        layers, aux: as trippy.raster.render_pyramid.
        timing: FrameTiming (milliseconds).
        num_fragments: total fragments emitted (pre-cap; matches
            aux's "num_fragments" convention in render_pyramid).
        points_visible: points that survived the conservative view-frustum
            cull (trippy.raster.cull_points) -- i.e. candidates handed to
            emission, not the (more expensive to compute) count of points
            that actually survived the per-pixel fragment cap/transmittance
            cutoff inside compositing.
    """
    device = feat.device
    grid = layer_grid(int(image_hw[0]), int(image_hw[1]), num_layers)

    _sync(device)
    t0 = time.perf_counter()
    uv, depth, size_px = project_points(xyz, size, K, R, t, znear=znear)
    valid = cull_points(uv, depth, size_px, grid, znear=znear)
    frags = emit_fragments(uv, depth, size_px, conf, grid, mode=mode, valid=valid, alpha_min=alpha_min)
    _sync(device)
    t1 = time.perf_counter()

    perm = sort_fragments(frags.layer_pixel, frags.depth, method="composite", stable=True)
    layer_pixel = frags.layer_pixel.index_select(0, perm)
    offsets = segment_offsets(layer_pixel, grid.total, method="searchsorted")
    point_id = frags.point_id.index_select(0, perm)
    alpha = frags.alpha.index_select(0, perm)
    frag_depth = frags.depth.index_select(0, perm)
    _sync(device)
    t2 = time.perf_counter()

    if device.type == "mps":
        out, t_final, n_used, depth_sum = metal_lib.blend_fwd(
            offsets.to(torch.int32).contiguous(),
            point_id.to(torch.int32).contiguous(),
            alpha.detach().to(torch.float32).contiguous(),
            frag_depth.detach().to(torch.float32).contiguous(),
            feat.detach().to(torch.float32).contiguous(),
            max_frags=max_frags,
            t_cutoff=t_cutoff,
        )
    elif device.type == "cpu":
        out, t_final, n_used, depth_sum = composite_sorted(
            layer_pixel,
            frag_depth,
            point_id,
            alpha,
            offsets,
            feat,
            max_frags=max_frags,
            t_cutoff=t_cutoff,
        )
    else:
        raise ValueError(f"render_one_frame supports device 'cpu' and 'mps', got {device.type!r}")
    layers, aux = split_layers(out, t_final, n_used, depth_sum, grid)
    aux["num_fragments"] = len(frags)
    _sync(device)
    t3 = time.perf_counter()

    timing = FrameTiming(
        emit_ms=(t1 - t0) * 1000.0,
        sort_ms=(t2 - t1) * 1000.0,
        blend_ms=(t3 - t2) * 1000.0,
        total_ms=(t3 - t0) * 1000.0,
    )
    return layers, aux, timing, len(frags), int(valid.sum().item())


def _level_to_uint8(level: Tensor) -> np.ndarray:
    """(C, h, w) float layer in [0, 1] -> (h, w, C) uint8, clamped."""
    arr = level.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.round(arr * 255.0).astype(np.uint8)


def _nearest_upsample(arr_uint8: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a (h, w, 3) uint8 array to `out_hw`.

    Used only when composing a contact sheet (see module docstring): coarse
    pyramid levels are blocky by construction, and nearest keeps that
    visible instead of smoothing it away the way contact_sheet's own
    bilinear thumbnail resize would.
    """
    from PIL import Image as PILImage

    height, width = out_hw
    img = PILImage.fromarray(arr_uint8, mode="RGB")
    return np.array(img.resize((width, height), PILImage.NEAREST))


def coverage_tensor(aux: dict) -> Tensor:
    """Level-0 coverage, `1 - T_final`, clamped to [0, 1]. No image involved."""
    return (1.0 - aux["t_final"][0]).clamp(0.0, 1.0)


def coverage_stats(coverage: Tensor, center_frac: float = RENDER_CENTER_REGION_FRAC) -> dict:
    """Numeric coverage summary computed directly from the T_final tensor.

    Deliberately never touches a rendered image (photo, sheet, or coverage
    PNG): AGENTS.md's privacy rule means no image derived from one of
    Jordan's scenes may be opened to answer "is this region covered", so
    the shade-region verdict is a plain mean over a numeric array instead.

    Args:
        coverage: (h0, w0) float tensor, `1 - T_final` (see coverage_tensor).
        center_frac: fraction of height/width kept for the central crop
            (e.g. 0.5 keeps the middle 50% x 50% of the frame).

    Returns:
        {"mean_full": float, "mean_center": float, "center_frac": float}
        -- both means are plain averages of coverage in [0, 1] (0 = no
        point ever reached that pixel, 1 = fully opaque).
    """
    height, width = coverage.shape
    ch = max(1, round(height * center_frac))
    cw = max(1, round(width * center_frac))
    y0 = (height - ch) // 2
    x0 = (width - cw) // 2
    center = coverage[y0 : y0 + ch, x0 : x0 + cw]
    return {
        "mean_full": float(coverage.mean().item()),
        "mean_center": float(center.mean().item()),
        "center_frac": center_frac,
    }


def _coverage_and_depth(aux: dict, coverage: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Level-0 coverage (colorized) and expected-depth (colorized) maps.

    Returns:
        coverage_color: (h0, w0, 3) uint8 -- colorize() output only, no
            photo pixels are ever blended in (safe to open per AGENTS.md).
        depth_color: (h0, w0, 3) uint8; uncovered pixels are exactly black
            (never a fabricated depth value -- AGENTS.md honesty rule).
        covered_mask: (h0, w0) bool, pixels with coverage > RENDER_COVERAGE_EPS.
    """
    coverage_np = coverage.detach().cpu().numpy()
    coverage_color = colorize(coverage_np, 0.0, 1.0)

    covered = coverage > RENDER_COVERAGE_EPS
    depth_sum0 = aux["depth_sum"][0]
    depth_expected = torch.zeros_like(depth_sum0)
    depth_expected[covered] = depth_sum0[covered] / coverage[covered]
    depth_np = depth_expected.detach().cpu().numpy()
    covered_np = covered.detach().cpu().numpy()

    if covered_np.any():
        vmin = float(np.percentile(depth_np[covered_np], RENDER_DEPTH_PERCENTILE_LOW))
        vmax = float(np.percentile(depth_np[covered_np], RENDER_DEPTH_PERCENTILE_HIGH))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    depth_color = colorize(depth_np, vmin, vmax).copy()
    depth_color[~covered_np] = 0

    return coverage_color, depth_color, covered_np


def _write_readme(out_dir: Path, metrics: dict, command: str | None) -> None:
    lines = [
        "# trippy render run",
        "",
        f"Command: `{command}`" if command else "Command: (not recorded)",
        "",
        f"- Scene: {metrics['scene']}",
        f"- PLY: {metrics['ply']}",
        (
            f"- Device: {metrics['device']}  Mode: {metrics['mode']}  "
            f"Layers: {metrics['num_layers']}  Width: {metrics['width']}"
        ),
        (
            f"- Points (post-opacity-filter{', subsampled' if metrics['max_points'] else ''}): "
            f"{metrics['num_points_total']}"
        ),
        "",
        (
            "| frame | emit ms | sort ms | blend ms | total ms | fragments | points visible "
            "| coverage (full) | coverage (center) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f in metrics["frames"]:
        tm = f["timing_ms"]
        cov = f["coverage"]
        lines.append(
            f"| {f['name']} | {tm['emit']:.2f} | {tm['sort']:.2f} | {tm['blend']:.2f} | "
            f"{tm['total']:.2f} | {f['num_fragments']} | {f['points_visible']} | "
            f"{cov['mean_full']:.4f} | {cov['mean_center']:.4f} |"
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def render_frames(
    scene_root: str | Path,
    ply_path: str | Path,
    frame_names: list[str],
    width: int,
    out_dir: str | Path,
    device: torch.device,
    mode: str = TRAIN_DEFAULT_MODE,
    num_layers: int = RASTER_NUM_LAYERS,
    min_opacity: float = DEFAULT_MIN_OPACITY,
    size_mode: str = "scale",
    max_points: int | None = None,
    cache_root: str | Path | None = None,
    command: str | None = None,
) -> dict:
    """Render `frame_names` from `scene_root` with a GaussianPlySource point set.

    Writes, under `out_dir`:
        - `<frame_stem>/photo.png`, `level_{0..num_layers-1}.png` (native
          per-level resolution), `coverage.png`, `depth.png`, `sheet.png`
          (photo | L0..L{n-1} | coverage | depth, one row).
        - `summary_sheet.png` (all frames, photo | L0 | coverage).
        - `metrics.json`, `README.md` (command + the timing/fragment table).

    Args:
        scene_root: COLMAP scene root (images/ + sparse/0 or sparse_txt).
        ply_path: binary 3DGS PLY for GaussianPlySource.
        frame_names: image filenames to render, in the order they should
            appear in the summary sheet (need not be sorted).
        width: SceneDataset's undistortion width (layer-0 image width).
        out_dir: output directory (created if missing).
        device: torch.device("cpu") or torch.device("mps").
        mode: one of trippy.constants.RASTER_MODES -- "trips" (TRIPS's own
            layer rule, the default), "trilinear" or "broadcast". See
            trippy.raster.emit.emit_fragments.
        num_layers: pyramid layer count.
        min_opacity, size_mode, max_points: forwarded to GaussianPlySource.
        cache_root: SceneDataset cache root; defaults to
            `trippy.config.load_settings().trippy_output / "cache"`.
        command: the shell command that produced this run, recorded verbatim
            in README.md for reproducibility (optional).

    Returns:
        The metrics dict also written to `<out_dir>/metrics.json`.
    """
    scene_root = Path(scene_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_root is None:
        cache_root = load_settings().trippy_output / RENDER_CACHE_SUBDIR

    dataset = _NamedSceneDataset(scene_root, width, cache_root, names=frame_names, device="cpu")
    name_to_index = {name: i for i, name in enumerate(dataset.names)}

    source = GaussianPlySource(ply_path, min_opacity=min_opacity, size_mode=size_mode, max_points=max_points)
    point_set = source.build()
    xyz = torch.from_numpy(point_set.xyz).to(device)
    size = torch.from_numpy(point_set.size0).to(device)
    feat = torch.from_numpy(point_set.rgb0).to(device)
    conf = torch.from_numpy(point_set.conf0).to(device)

    frame_metrics: list[dict] = []
    summary_images: list[np.ndarray] = []
    summary_labels: list[str] = []

    for name in frame_names:
        item = dataset[name_to_index[name]]
        photo = item["rgb"].numpy()
        K = item["K"].to(device)
        qvec = item["qvec"].to(device)
        tvec = item["tvec"].to(device)
        R = qvec2R(qvec)
        image_hw = (photo.shape[0], photo.shape[1])

        layers, aux, timing, num_fragments, points_visible = render_one_frame(
            xyz, size, feat, conf, K, R, tvec, image_hw, num_layers, mode
        )

        level_native = [_level_to_uint8(layers[i]) for i in range(num_layers)]
        level_upsampled = [_nearest_upsample(level_native[i], image_hw) for i in range(num_layers)]
        coverage_t = coverage_tensor(aux)
        cov_stats = coverage_stats(coverage_t)
        coverage_color, depth_color, _covered = _coverage_and_depth(aux, coverage_t)

        stem = Path(name).stem
        frame_dir = out_dir / stem
        save_png(frame_dir / "photo.png", photo)
        for i in range(num_layers):
            save_png(frame_dir / f"level_{i}.png", level_native[i])
        save_png(frame_dir / "coverage.png", coverage_color)
        save_png(frame_dir / "depth.png", depth_color)

        images = [photo, *level_upsampled, coverage_color, depth_color]
        labels = ["photo", *[f"L{i}" for i in range(num_layers)], "coverage", "depth"]
        save_png(frame_dir / "sheet.png", side_by_side(images, labels))

        summary_images.extend([photo, level_upsampled[0], coverage_color])
        summary_labels.extend([f"{name}:photo", f"{name}:L0", f"{name}:coverage"])

        frame_metrics.append(
            {
                "name": name,
                "image_hw": [int(image_hw[0]), int(image_hw[1])],
                "timing_ms": {
                    "emit": timing.emit_ms,
                    "sort": timing.sort_ms,
                    "blend": timing.blend_ms,
                    "total": timing.total_ms,
                },
                "num_fragments": num_fragments,
                "points_visible": points_visible,
                "coverage": cov_stats,
            }
        )
        print(
            f"{name}: emit={timing.emit_ms:.2f}ms sort={timing.sort_ms:.2f}ms "
            f"blend={timing.blend_ms:.2f}ms total={timing.total_ms:.2f}ms "
            f"fragments={num_fragments} points_visible={points_visible} "
            f"coverage_full={cov_stats['mean_full']:.4f} coverage_center={cov_stats['mean_center']:.4f}"
        )

    save_png(out_dir / "summary_sheet.png", contact_sheet(summary_images, summary_labels, cols=3))

    metrics = {
        "scene": str(scene_root),
        "ply": str(ply_path),
        "device": str(device),
        "mode": mode,
        "num_layers": num_layers,
        "width": width,
        "min_opacity": min_opacity,
        "size_mode": size_mode,
        "max_points": max_points,
        "num_points_total": len(point_set),
        "frames": frame_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_readme(out_dir, metrics, command)
    return metrics
