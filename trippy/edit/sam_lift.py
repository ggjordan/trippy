"""SAM-3 mask lift: a 2D mask on one photo becomes a 3D `pointset` Region.

Module: trippy.edit.sam_lift
Purpose: the E5 milestone of docs/EDITOR.md Sec 3 "4. SAM 3 lift" -- Jordan
    points at an object in ONE registered photograph (a click, a box, or a
    text phrase), the local SAM 3 segments it (`trippy.edit.sam_runner`),
    and this module lifts that mask onto the bundle's own TRIPS points by
    projection, then repeats the lift in the N nearest capture views and
    keeps the points a majority of the views that can see them agree on.
    Output is a `pointset` Region -- the same shape the shade-cloud finder
    and click-to-cluster produce, so the Inspector does not need to know
    which tool made it.

The three steps, and why each is there:

1. **Project.** Every point of `points.npz` goes through the prompted
   view's own `(R, t, K, distortion)` from `bundle.json` into that view's
   pixels, then through one uniform scale into the PHOTOGRAPH's pixels
   (a bundle view is the same camera at a smaller raster: on kk-coherent,
   view `fx=757.36`, `cx=504` at 1008x756 is exactly COLMAP's `fx=3029.46`,
   `cx=2016` at 4032x3024 divided by 4). A point is a candidate when it
   lands inside the mask.
2. **Depth-gate.** A mask is 2D: everything behind the object along the same
   ray is inside it too. Points are binned into `SAM_LIFT_DEPTH_CELL_PX`
   cells and each cell's NEAREST supported depth mode (log-spaced bins of
   relative width `SAM_LIFT_DEPTH_TOL`, a bin counting as a surface at
   `SAM_LIFT_DEPTH_SUPPORT_FRAC` of the cell's fullest bin) is taken as the
   surface there; a candidate survives only within a relative band of it. This
   is the same "the mode of the depths under the cursor is the surface"
   seed rule the click-to-cluster tool uses, applied per cell instead of
   per click.
3. **Vote.** The lift repeats in the `--views-around` nearest capture views
   that can see the selection's centroid, prompting SAM there with that
   centroid's own projected pixel. A point is selected when a majority
   (`SAM_LIFT_VOTE_FRACTION`) of the views that can actually SEE it voted
   for it -- EDITOR.md Sec 3's "several independent signals must agree",
   the discipline `~/Splats/tools/make_masks3.py` already uses for masks.

Invariants:
    - Nothing here opens, decodes, renders or displays a photograph. The
      photo is touched by exactly one thing, the SAM child process
      (`trippy.edit.sam_runner`), which is not a model and returns an
      array; this module additionally reads the image FILE HEADER for its
      pixel size (`photo_size`, PIL's lazy `.size`, no pixel decode).
      Masks are arrays. The optional `--preview` is drawn FROM SCRATCH
      from projected point counts and contains no photographic content
      (AGENTS.md Sec 6 "never send scene imagery to a model").
    - Mask polarity is `True` = the prompted object, fixed by
      `sam_runner`'s own contract and re-checked here (`_check_mask`).
    - The segmenter is injected (`segmenter=` / `SegmenterFn`), so every
      step except SAM itself runs on CPU in the test suite with a fake
      (tests/test_edit_sam.py). SAM 3 on MPS is GPU work and only ever
      runs inside a `scripts/gpu_submit.sh` job.
    - `point_ids` index `points.npz`'s own row order, exactly like
      `trippy.edit.shade_finder`'s (docs/EDITOR.md Sec 1 `pointset`).
    - Camera maths is `trippy.geom.xform_a`'s convention: `R` row-major
      world-to-camera, `x_cam = R @ x_world + t`, `+Z` forward, `+Y` down
      (docs/GEOMETRY.md). Saiga 8-coefficient distortion is applied with
      `trippy.render.parity.distort_normalized` -- the existing, parity-
      tested implementation, never a second copy of it.
Units: world units for `xyz`/`t`/depths; pixels for everything named `u`,
    `v`, `cx`, `cy`, `fx`, `fy`, boxes and masks (photo pixels unless the
    name says `view`); `mix` and mask-area fractions are dimensionless.
Related docs: docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)", Sec 1 (`pointset`),
    Sec 7 ("Segmentation quality (E5) is the least certain estimate");
    docs/GEOMETRY.md; trippy.edit.sam_runner (the SAM 3 subprocess);
    trippy.edit.shade_finder (the other `pointset` producer this mirrors).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    SAM_LIFT_DEFAULT_MIX,
    SAM_LIFT_DEFAULT_OP,
    SAM_LIFT_DEFAULT_VIEWS_AROUND,
    SAM_LIFT_DEPTH_CELL_PX,
    SAM_LIFT_DEPTH_SUPPORT_FRAC,
    SAM_LIFT_DEPTH_TOL,
    SAM_LIFT_MIN_DEPTH,
    SAM_LIFT_PREVIEW_CELL_PX,
    SAM_LIFT_PREVIEW_HIGH_RGB,
    SAM_LIFT_PREVIEW_LOW_RGB,
    SAM_LIFT_VOTE_FRACTION,
)
from trippy.edit.model import Region, new_region_id
from trippy.edit.sam_runner import SamPrompt

__all__ = [
    "LiftView",
    "SegmenterFn",
    "depth_mode_keep",
    "lift_mask_in_view",
    "load_lift_inputs",
    "neighbour_views",
    "photo_size",
    "project_to_photo",
    "sam_lift",
    "scale_prompt",
    "scene_distortions",
    "write_selection_preview",
]

#: A segmenter takes `(image_path, prompt)` and returns `(mask, info)`:
#: `mask` is `(H, W)` bool in the photo's own pixel size, True = the object.
SegmenterFn = Callable[[Path, SamPrompt], "tuple[np.ndarray, dict[str, Any]]"]

#: Relative tolerance on the photo/view aspect-ratio agreement. A bundle view
#: is the capture camera at a smaller raster, so the two aspects match to
#: within one pixel of rounding; anything worse means the `--scene` photos are
#: not the images this bundle was built from, which is a caller error, not
#: something to silently scale away.
_ASPECT_TOL = 0.01


@dataclass(frozen=True)
class LiftView:
    """One `bundle.json` view, as this module needs it.

    Attributes:
        name: image file name, e.g. "IMG_3703.jpg".
        index: the view's array position in `bundle.json`'s `views`.
        width, height: the VIEW's raster size in pixels (not the photo's).
        fx, fy, cx, cy: pinhole intrinsics in view pixels.
        R: `(3, 3)` row-major world-to-camera rotation.
        t: `(3,)` world-to-camera translation, world units.
        distortion: `(8,)` Saiga coefficients `k1 k2 k3 k4 k5 k6 p1 p2`;
            all-zero on a trippy-native bundle (undistorted on ingest).
    """

    name: str
    index: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray
    distortion: np.ndarray

    @staticmethod
    def from_json(doc: dict[str, Any], index: int) -> LiftView:
        """Build a `LiftView` from one `bundle.json` `views[]` entry."""
        return LiftView(
            name=str(doc["name"]),
            index=index,
            width=int(doc["width"]),
            height=int(doc["height"]),
            fx=float(doc["fx"]),
            fy=float(doc["fy"]),
            cx=float(doc["cx"]),
            cy=float(doc["cy"]),
            R=np.asarray(doc["R"], dtype=np.float64).reshape(3, 3),
            t=np.asarray(doc["t"], dtype=np.float64).reshape(3),
            distortion=np.asarray(doc.get("distortion", [0.0] * 8), dtype=np.float64).reshape(-1),
        )

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates, `C = -R.T @ t`."""
        return -self.R.T @ self.t


def load_lift_inputs(bundle_dir: str | Path) -> tuple[dict[str, Any], list[LiftView], np.ndarray]:
    """Read a bundle's document, views and point positions.

    Args:
        bundle_dir: a bundle directory (`bundle.json` + `points.npz`).

    Returns:
        `(bundle_doc, views, xyz)` -- `views` in `bundle.json` order,
        `xyz` `(N, 3)` float64 in the COLMAP world frame.

    Raises:
        FileNotFoundError: no `bundle.json`/`points.npz` there.
    """
    bundle_dir = Path(bundle_dir)
    bundle_json = bundle_dir / "bundle.json"
    if not bundle_json.exists():
        raise FileNotFoundError(f"no bundle.json in {bundle_dir}")
    doc = json.loads(bundle_json.read_text())
    points_path = bundle_dir / doc.get("points", "points.npz")
    if not points_path.exists():
        raise FileNotFoundError(f"no {points_path.name} in {bundle_dir}")
    with np.load(points_path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
    views = [LiftView.from_json(view, i) for i, view in enumerate(doc.get("views", []))]
    return doc, views, xyz


def photo_size(path: str | Path) -> tuple[int, int]:
    """`(width, height)` of an image file, from its HEADER only.

    PIL's `Image.open(...).size` parses the header and does not decode a
    single pixel; nothing here reads, renders or displays the photograph
    (module docstring, AGENTS.md Sec 6).

    Raises:
        FileNotFoundError: no such file.
    """
    from PIL import Image

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no photo at {path}")
    with Image.open(path) as handle:
        return int(handle.size[0]), int(handle.size[1])


def photo_scale(view: LiftView, photo_wh: tuple[int, int]) -> float:
    """View pixels -> photo pixels, as ONE uniform scale.

    Args:
        view: the bundle view.
        photo_wh: `(width, height)` of the photograph on disk.

    Returns:
        `photo_width / view.width`.

    Raises:
        ValueError: the two aspect ratios disagree by more than `_ASPECT_TOL`
            (the photos are not this bundle's images).
    """
    width, height = photo_wh
    sx = float(width) / float(view.width)
    sy = float(height) / float(view.height)
    if abs(sx - sy) > _ASPECT_TOL * max(sx, sy):
        raise ValueError(
            f"photo {width}x{height} and bundle view {view.name} {view.width}x{view.height} "
            f"have different aspect ratios (scale {sx:.4f} vs {sy:.4f})"
        )
    return sx


#: The two coordinate systems a prompt can arrive in. `"photo"` is the
#: as-captured image's own pixel grid (what `trippy edits sam` has always
#: taken, and what SAM 3 is prompted in); `"view"` is the BUNDLE view's
#: smaller raster, which is what the Rust viewer measures a drag-box in --
#: it renders `bundle.json`'s views and never opens a photograph, so it
#: cannot know the photo's size (docs/EDITOR.md Sec 3 "4. SAM 3 lift").
PROMPT_SPACES = ("photo", "view")


def scale_prompt(prompt: SamPrompt, scale: float) -> SamPrompt:
    """The same prompt with its pixel coordinates multiplied by `scale`.

    Args:
        prompt: a point or box prompt (a `text` prompt has no pixels and is
            returned unchanged).
        scale: view pixels -> photo pixels, from `photo_scale`.

    Returns:
        A new `SamPrompt`; the input is frozen and never mutated.
    """
    if prompt.kind == "point":
        u, v = prompt.point  # type: ignore[misc]
        return SamPrompt(kind="point", point=(u * scale, v * scale))
    if prompt.kind == "box":
        x0, y0, x1, y1 = prompt.box  # type: ignore[misc]
        return SamPrompt(
            kind="box", box=(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        )
    return prompt


def scene_distortions(scene_root: str | Path) -> dict[str, tuple[float, float, float, float]]:
    """`(k1, k2, p1, p2)` per registered image name, reading the model ONCE.

    Needed because a trippy-native bundle's views are UNDISTORTED (their
    `distortion` is all zeros -- `trippy.scene.dataset` undistorts on
    ingest) while `--scene`'s photographs on disk are still as-captured.
    Projecting into the photo therefore has to re-apply the lens the photo
    was taken with, or the mask and the projected points drift apart
    towards the frame edges (~19 px at 4032 px wide on kk-coherent's
    `k1 = 0.057`).

    A whole-model read per view would be a COLMAP parse per SAM call, so
    the caller asks once and looks up per view.

    Args:
        scene_root: scene directory (`sparse/0` or `sparse_txt` under it).

    Returns:
        `name -> (k1, k2, p1, p2)`; `(0, 0, 0, 0)` for a distortion-free
        camera model. Images whose camera model `trippy.scene.colmap_io.
        distortion` does not understand are omitted (the caller then
        projects with no distortion at all, which is the honest fallback).

    Raises:
        FileNotFoundError: no sparse model under `scene_root`.
    """
    from trippy.scene.colmap_io import distortion, load_colmap_model
    from trippy.scene.dataset import resolve_sparse_dir

    scene = load_colmap_model(resolve_sparse_dir(scene_root))
    table: dict[str, tuple[float, float, float, float]] = {}
    for name, image in scene.images_by_name().items():
        try:
            table[name] = distortion(scene.cameras[image.camera_id])
        except ValueError:
            continue
    return table


def scene_opencv_distortion(scene_root: str | Path, name: str) -> tuple[float, float, float, float]:
    """`(k1, k2, p1, p2)` for ONE image; `scene_distortions` for many.

    Raises:
        ValueError: `name` is not registered in the scene's sparse model.
    """
    table = scene_distortions(scene_root)
    if name not in table:
        raise ValueError(f"{name} is not registered in {scene_root}")
    return table[name]


def _distort_saiga(xn: np.ndarray, yn: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Saiga 8-coefficient distortion, via the parity-tested torch implementation.

    `trippy.render.parity.distort_normalized` is the ONE implementation of
    this in the repo (checked against Saiga's own `distortNormalizedPoint`);
    calling it with CPU tensors keeps that single source of truth instead of
    adding a numpy copy that could drift from it.
    """
    import torch

    from trippy.render.parity import distort_normalized

    xy = torch.from_numpy(np.stack([xn, yn], axis=1))
    out = distort_normalized(xy, torch.from_numpy(np.asarray(coefficients, dtype=np.float64)))
    return out.numpy()


def _distort_opencv(xn: np.ndarray, yn: np.ndarray, k: tuple[float, float, float, float]) -> np.ndarray:
    """COLMAP/OpenCV `(k1, k2, p1, p2)` distortion, via `trippy.geom.camera`."""
    from trippy.geom.camera import OpenCVDistortion

    return OpenCVDistortion(k1=k[0], k2=k[1], p1=k[2], p2=k[3]).distort(
        np.stack([xn, yn], axis=1)
    )


def project_to_photo(
    view: LiftView,
    xyz: np.ndarray,
    scale: float,
    opencv_distortion: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into one view's PHOTOGRAPH pixels.

    Args:
        view: the bundle view (pose + intrinsics + Saiga distortion).
        xyz: `(N, 3)` world positions.
        scale: view pixels -> photo pixels (`photo_scale`).
        opencv_distortion: the as-captured lens `(k1, k2, p1, p2)` to apply
            when the VIEW itself carries no distortion (a trippy-native
            bundle projecting into a raw photo). Ignored when the view has
            its own nonzero Saiga coefficients, which already describe that
            lens (a TRIPS/ADOP bundle).

    Returns:
        `(uv, depth)` -- `uv` `(N, 2)` float64 photo pixels, `depth` `(N,)`
        camera-space z in world units (<= 0 means behind the camera; `uv`
        is meaningless there and the caller must gate on `depth`).
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    cam = xyz @ view.R.T + view.t.reshape(1, 3)
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < SAM_LIFT_MIN_DEPTH, np.nan, depth)
    xn = cam[:, 0] / safe
    yn = cam[:, 1] / safe

    if np.any(view.distortion != 0.0):
        distorted = _distort_saiga(np.nan_to_num(xn), np.nan_to_num(yn), view.distortion)
        xn, yn = distorted[:, 0], distorted[:, 1]
    elif opencv_distortion is not None and any(k != 0.0 for k in opencv_distortion):
        distorted = _distort_opencv(np.nan_to_num(xn), np.nan_to_num(yn), opencv_distortion)
        xn, yn = distorted[:, 0], distorted[:, 1]

    u = (view.fx * xn + view.cx) * scale
    v = (view.fy * yn + view.cy) * scale
    return np.stack([u, v], axis=1), depth


def depth_mode_keep(
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    depth: np.ndarray,
    cell_px: int = SAM_LIFT_DEPTH_CELL_PX,
    tol: float = SAM_LIFT_DEPTH_TOL,
    support_fraction: float = SAM_LIFT_DEPTH_SUPPORT_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the points on their pixel cell's NEAREST supported surface.

    The mask cannot say which of the points along a ray is the object, so
    the points themselves are asked. Within a `cell_px` x `cell_px` cell,
    depths go into log bins of relative width `tol` (a depth "mode" per
    cell); the cell's surface is the NEAREST bin holding at least
    `support_fraction` of the fullest bin's population, and a point
    survives when its depth is within a factor `(1 + tol)` of that surface.

    Nearest-with-support, rather than simply the fullest bin, is the whole
    point: a mask over a nearby object almost always has more background
    behind it than object in it (a distant wall projects many more points
    per pixel than a close object does), so "the modal depth" alone would
    lock onto the background and delete the object. The support fraction
    is what still rejects a lone floater in front of the object -- one
    stray point is not a surface.

    Args:
        u_idx, v_idx: `(M,)` integer photo pixel indices of the candidates.
        depth: `(M,)` positive camera-space depths of the same candidates.
        cell_px: cell size in photo pixels.
        tol: relative depth tolerance (also the log bin width).
        support_fraction: a bin counts as a surface at this fraction of its
            cell's fullest bin (1.0 = "the fullest bin only").

    Returns:
        `(keep, surface_depth)` -- `(M,)` bool and `(M,)` the depth of each
        candidate's own cell's surface (equal within a cell).

    Raises:
        ValueError: `cell_px < 1`, `tol <= 0`, `support_fraction` outside
            `(0, 1]`, or a non-positive depth.
    """
    if cell_px < 1:
        raise ValueError(f"cell_px must be >= 1, got {cell_px}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if not 0.0 < support_fraction <= 1.0:
        raise ValueError(f"support_fraction must be in (0, 1], got {support_fraction}")
    u_idx = np.asarray(u_idx).reshape(-1)
    v_idx = np.asarray(v_idx).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if depth.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    if np.any(depth <= 0.0):
        raise ValueError("depth_mode_keep needs strictly positive depths")

    cols = (u_idx // cell_px).astype(np.int64)
    rows = (v_idx // cell_px).astype(np.int64)
    ncols = int(cols.max()) + 1
    cell = rows * ncols + cols

    log_step = math.log1p(tol)
    bin_index = np.floor(np.log(depth) / log_step).astype(np.int64)

    # `np.unique(axis=0)` returns rows lexicographically sorted, i.e. grouped
    # by cell and ascending in bin (= ascending in depth) within a cell --
    # which is what makes "the first supported bin" the nearest one.
    pairs = np.stack([cell, bin_index], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    starts = np.flatnonzero(
        np.concatenate(([True], unique_pairs[1:, 0] != unique_pairs[:-1, 0]))
    )
    group_sizes = np.diff(np.concatenate((starts, [unique_pairs.shape[0]])))
    cell_peak = np.repeat(np.maximum.reduceat(counts, starts), group_sizes)
    supported = counts >= support_fraction * cell_peak
    positions = np.where(supported, np.arange(counts.size), counts.size)
    first_supported = np.minimum.reduceat(positions, starts)

    surface_cells = unique_pairs[starts, 0]
    surface_bins = unique_pairs[first_supported, 1]
    lookup = np.searchsorted(surface_cells, cell)
    surface_depth = np.exp((surface_bins[lookup] + 0.5) * log_step)
    keep = np.abs(np.log(depth / surface_depth)) <= log_step
    return keep, surface_depth


def _check_mask(mask: np.ndarray, photo_wh: tuple[int, int], name: str) -> np.ndarray:
    """Validate a segmenter's mask: `(H, W)` bool matching the photo."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask for {name} must be 2-D (H, W), got shape {mask.shape}")
    width, height = photo_wh
    if mask.shape != (height, width):
        raise ValueError(
            f"mask for {name} is {mask.shape} but the photo is {(height, width)} (H, W)"
        )
    return mask.astype(bool)


def lift_mask_in_view(
    view: LiftView,
    xyz: np.ndarray,
    mask: np.ndarray,
    photo_wh: tuple[int, int],
    opencv_distortion: tuple[float, float, float, float] | None = None,
    cell_px: int = SAM_LIFT_DEPTH_CELL_PX,
    depth_tol: float = SAM_LIFT_DEPTH_TOL,
) -> dict[str, Any]:
    """Lift ONE mask onto the point cloud through ONE view.

    Args:
        view: the bundle view the mask belongs to.
        xyz: `(N, 3)` world positions (the bundle's own row order).
        mask: `(H, W)` bool, True = the prompted object.
        photo_wh: `(width, height)` of the photograph.
        opencv_distortion: see `project_to_photo`.
        cell_px, depth_tol: the depth gate (`depth_mode_keep`).

    Returns:
        dict with `selected` `(N,)` bool (in the mask AND depth-consistent),
        `visible` `(N,)` bool (in front of the camera and inside the frame
        -- i.e. entitled to vote), `uv` `(N, 2)` photo pixels, `depth`,
        and `stats` (counts and the mask's area fraction).
    """
    mask = _check_mask(mask, photo_wh, view.name)
    width, height = photo_wh
    scale = photo_scale(view, photo_wh)
    uv, depth = project_to_photo(view, xyz, scale, opencv_distortion)

    # Cast through a clamped, finite sentinel. A non-finite coordinate would
    # make the int64 cast platform-defined, and so would a huge one: the
    # distortion polynomial explodes for a point almost in the pinhole (a
    # point at depth 1e-3 reaches |u| ~ 1e19, past int64). Everything outside
    # `[-1, max(W, H)]` is out of frame by definition, so clamping there
    # changes no in-frame index and makes the cast exact.
    sentinel = float(max(width, height))
    finite_uv = np.clip(np.where(np.isfinite(uv), uv, -1.0), -1.0, sentinel)
    u_idx = np.floor(finite_uv[:, 0]).astype(np.int64)
    v_idx = np.floor(finite_uv[:, 1]).astype(np.int64)
    in_front = np.isfinite(depth) & (depth > SAM_LIFT_MIN_DEPTH)
    in_frame = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
    visible = in_front & in_frame & np.isfinite(uv).all(axis=1)

    in_mask = np.zeros(visible.shape, dtype=bool)
    candidates = np.flatnonzero(visible)
    if candidates.size:
        in_mask[candidates] = mask[v_idx[candidates], u_idx[candidates]]

    selected = np.zeros(visible.shape, dtype=bool)
    picked = np.flatnonzero(in_mask)
    n_before_depth = int(picked.size)
    if picked.size:
        keep, _mode = depth_mode_keep(u_idx[picked], v_idx[picked], depth[picked], cell_px, depth_tol)
        selected[picked[keep]] = True

    return {
        "selected": selected,
        "visible": visible,
        "uv": uv,
        "depth": depth,
        "stats": {
            "view": view.name,
            "mask_area_px": int(mask.sum()),
            "mask_area_fraction": float(mask.mean()) if mask.size else 0.0,
            "photo_width": int(width),
            "photo_height": int(height),
            "view_to_photo_scale": float(scale),
            "n_visible": int(visible.sum()),
            "n_in_mask": n_before_depth,
            "n_selected": int(selected.sum()),
        },
    }


def neighbour_views(
    views: list[LiftView],
    primary: LiftView,
    target: np.ndarray,
    count: int,
    photo_sizes: dict[str, tuple[int, int]] | None = None,
) -> list[LiftView]:
    """The `count` nearest capture views that can also SEE `target`.

    "Nearest" is by camera centre distance from `primary` (the neighbouring
    captures of a walk-around are the views most likely to show the same
    object from a usefully different angle); a view only qualifies if
    `target` projects inside its frame in front of its camera, so a view
    facing the other way is never prompted with a point that is not in it.

    Args:
        views: all bundle views.
        primary: the prompted view (excluded from the result).
        target: `(3,)` world point that must be visible (the selection's
            centroid).
        count: how many neighbours to return (0 returns nothing).
        photo_sizes: optional `name -> (width, height)`; only the aspect
            matters here, so the view's own raster is used when absent.

    Returns:
        Up to `count` views, nearest first.
    """
    if count <= 0:
        return []
    target = np.asarray(target, dtype=np.float64).reshape(1, 3)
    ranked = sorted(
        (v for v in views if v.name != primary.name),
        key=lambda v: float(np.linalg.norm(v.centre - primary.centre)),
    )
    chosen: list[LiftView] = []
    for view in ranked:
        size = (photo_sizes or {}).get(view.name, (view.width, view.height))
        try:
            scale = photo_scale(view, size)
        except ValueError:
            continue
        uv, depth = project_to_photo(view, target, scale)
        if not np.isfinite(depth[0]) or depth[0] <= SAM_LIFT_MIN_DEPTH:
            continue
        u, v = uv[0]
        if not (0.0 <= u < size[0] and 0.0 <= v < size[1]):
            continue
        chosen.append(view)
        if len(chosen) >= count:
            break
    return chosen


def write_selection_preview(
    path: str | Path,
    uv: np.ndarray,
    selected: np.ndarray,
    photo_wh: tuple[int, int],
    cell_px: int = SAM_LIFT_PREVIEW_CELL_PX,
) -> Path:
    """Draw a FROM-SCRATCH heatmap of where the selected points project.

    This is a picture of a point selection, not of a photograph and not of
    a mask: counts of selected points per `cell_px` cell, ramped between
    two flat colours. It contains no photographic content by construction,
    which is what makes it safe to look at (AGENTS.md Sec 6 "Allowed to
    view: ... abstract heatmaps with no photographic content").

    Args:
        path: PNG to write.
        uv: `(N, 2)` projected photo pixels.
        selected: `(N,)` bool, which points are in the region.
        photo_wh: `(width, height)` of the frame being previewed.
        cell_px: heatmap cell size in photo pixels.

    Returns:
        The written path.
    """
    from PIL import Image

    width, height = photo_wh
    cols = max(1, width // cell_px)
    rows = max(1, height // cell_px)
    grid = np.zeros((rows, cols), dtype=np.float64)

    picked = np.flatnonzero(np.asarray(selected, dtype=bool))
    if picked.size:
        u = np.floor(np.asarray(uv)[picked, 0] / cell_px).astype(np.int64)
        v = np.floor(np.asarray(uv)[picked, 1] / cell_px).astype(np.int64)
        keep = (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        np.add.at(grid, (v[keep], u[keep]), 1.0)

    peak = float(grid.max())
    ramp = grid / peak if peak > 0 else grid
    low = np.asarray(SAM_LIFT_PREVIEW_LOW_RGB, dtype=np.float64)
    high = np.asarray(SAM_LIFT_PREVIEW_HIGH_RGB, dtype=np.float64)
    rgb = low[None, None, :] + ramp[:, :, None] * (high - low)[None, None, :]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)
    return path


def sam_lift(
    bundle_dir: str | Path,
    scene_root: str | Path,
    view_name: str,
    prompt: SamPrompt,
    segmenter: SegmenterFn,
    views_around: int = SAM_LIFT_DEFAULT_VIEWS_AROUND,
    op: str = SAM_LIFT_DEFAULT_OP,
    mix: float = SAM_LIFT_DEFAULT_MIX,
    name: str | None = None,
    region_id: str | None = None,
    cell_px: int = SAM_LIFT_DEPTH_CELL_PX,
    depth_tol: float = SAM_LIFT_DEPTH_TOL,
    vote_fraction: float = SAM_LIFT_VOTE_FRACTION,
    apply_scene_distortion: bool = True,
    preview: str | Path | None = None,
    prompt_space: str = "photo",
    progress: Callable[[str], None] | None = None,
) -> tuple[Region, dict[str, Any]]:
    """Segment one photo, lift it onto the bundle's points, vote across views.

    Args:
        bundle_dir: bundle directory (`bundle.json` + `points.npz`).
        scene_root: scene directory; photos are `<scene_root>/images/<name>`
            and the COLMAP model under it supplies the as-captured lens.
        view_name: which registered view to prompt (a `bundle.json` view
            name, i.e. an image file name).
        prompt: the click/box/phrase, in that PHOTO's pixels.
        segmenter: `(image_path, prompt) -> (mask, info)`; inject
            `trippy.edit.sam_runner.Sam3Segmenter()` for the real thing, or
            a fake in tests (the whole rest of this function is CPU numpy).
        views_around: how many neighbouring capture views also get
            segmented and vote (0 = the prompted view alone).
        op, mix: the Region's action (docs/EDITOR.md Sec 1).
        name: Region name; defaults to "sam: <prompt>".
        region_id: stable id (default: a fresh `new_region_id()`).
        cell_px, depth_tol: the depth gate (`depth_mode_keep`).
        vote_fraction: fraction of a point's *eligible* views that must
            select it (0.5 = simple majority).
        apply_scene_distortion: re-apply the COLMAP camera's own
            `(k1, k2, p1, p2)` when the bundle view is undistorted
            (`scene_opencv_distortion`); set False to project with the
            bundle's own optics only.
        preview: optional PNG path for a from-scratch selection heatmap in
            the prompted view (`write_selection_preview`).
        prompt_space: which pixel grid `prompt` is measured in -- "photo"
            (the default, the as-captured image) or "view" (the bundle
            view's own smaller raster, which is what the Rust viewer can
            measure; see `PROMPT_SPACES`). A "view" prompt is multiplied by
            this view's `photo_scale` before SAM ever sees it.
        progress: optional one-line-at-a-time callback, called before and
            after every segmented view. The viewer's SAM tool runs this
            command as a child process and shows these lines while it waits
            (docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)").

    Returns:
        `(region, summary)`. `region` is a `pointset` Region over
        `points.npz`'s row order. `summary` records the prompt, the
        per-view stats (mask area fraction, counts), the vote, and the
        segmenter `info` for every view -- enough to reproduce the run.

    Raises:
        ValueError: `view_name` is not in the bundle, `prompt_space` is not
            one of `PROMPT_SPACES`, the photo and the view disagree about
            aspect, a mask has the wrong shape, or no point survived the
            lift in the prompted view.
        FileNotFoundError: missing bundle, points, or photograph.
    """
    if prompt_space not in PROMPT_SPACES:
        raise ValueError(f"prompt_space must be one of {PROMPT_SPACES}, got {prompt_space!r}")
    note = progress if progress is not None else (lambda _message: None)
    bundle_dir = Path(bundle_dir)
    scene_root = Path(scene_root)
    _doc, views, xyz = load_lift_inputs(bundle_dir)
    by_name = {v.name: v for v in views}
    if view_name not in by_name:
        raise ValueError(f"{view_name} is not a view of {bundle_dir} ({len(views)} views)")
    primary = by_name[view_name]

    def photo_path(view: LiftView) -> Path:
        return scene_root / "images" / view.name

    scene_lens: dict[str, tuple[float, float, float, float]] = {}
    if apply_scene_distortion:
        try:
            scene_lens = scene_distortions(scene_root)
        except (FileNotFoundError, ValueError):
            scene_lens = {}

    def distortion_for(view: LiftView) -> tuple[float, float, float, float] | None:
        """The as-captured lens to re-apply, or None (see `scene_distortions`).

        A view that carries its own Saiga coefficients already describes
        that lens, so the COLMAP one must not be applied on top of it.
        """
        if np.any(view.distortion != 0.0):
            return None
        return scene_lens.get(view.name)

    per_view: list[dict[str, Any]] = []
    votes = np.zeros(xyz.shape[0], dtype=np.int64)
    eligible = np.zeros(xyz.shape[0], dtype=np.int64)

    primary_photo = photo_path(primary)
    primary_wh = photo_size(primary_photo)
    view_prompt = prompt
    if prompt_space == "view":
        # The viewer measures its drag-box on the RENDER, whose pixel grid is
        # the bundle view's, so the photo scale is applied here rather than in
        # a caller that would have to open the photograph to learn it.
        prompt = scale_prompt(prompt, photo_scale(primary, primary_wh))
    note(
        f"segmenting {primary.name} ({prompt.kind} prompt) at "
        f"{primary_wh[0]}x{primary_wh[1]}"
    )
    mask, info = segmenter(primary_photo, prompt)
    lifted = lift_mask_in_view(
        primary, xyz, mask, primary_wh, distortion_for(primary), cell_px, depth_tol
    )
    note(
        f"{primary.name}: {lifted['stats']['n_in_mask']} points inside the mask, "
        f"{int(lifted['selected'].sum())} after the depth gate"
    )
    votes += lifted["selected"].astype(np.int64)
    eligible += lifted["visible"].astype(np.int64)
    stats = dict(lifted["stats"])
    stats["prompt"] = prompt.to_json()
    stats["segmenter"] = {k: v for k, v in info.items() if k != "command"}
    stats["command"] = info.get("command")
    per_view.append(stats)

    if int(lifted["selected"].sum()) == 0:
        raise ValueError(
            f"the mask on {primary.name} lifted onto 0 points "
            f"({stats['n_in_mask']} projected inside it, none depth-consistent); "
            "check the prompt, --depth-tol, or that --scene matches --bundle"
        )

    centroid = np.median(xyz[lifted["selected"]], axis=0)
    neighbours = neighbour_views(views, primary, centroid, views_around)
    note(f"{len(neighbours)} neighbour view(s) will vote")
    for position, view in enumerate(neighbours, start=1):
        try:
            neighbour_wh = photo_size(photo_path(view))
        except FileNotFoundError:
            continue
        scale = photo_scale(view, neighbour_wh)
        uv, _depth = project_to_photo(view, centroid.reshape(1, 3), scale, distortion_for(view))
        neighbour_prompt = SamPrompt(kind="point", point=(float(uv[0, 0]), float(uv[0, 1])))
        n_mask, n_info = segmenter(photo_path(view), neighbour_prompt)
        n_lift = lift_mask_in_view(
            view, xyz, n_mask, neighbour_wh, distortion_for(view), cell_px, depth_tol
        )
        votes += n_lift["selected"].astype(np.int64)
        eligible += n_lift["visible"].astype(np.int64)
        n_stats = dict(n_lift["stats"])
        n_stats["prompt"] = neighbour_prompt.to_json()
        n_stats["segmenter"] = {k: v for k, v in n_info.items() if k != "command"}
        n_stats["command"] = n_info.get("command")
        per_view.append(n_stats)
        note(
            f"neighbour {position}/{len(neighbours)} {view.name}: "
            f"{int(n_lift['selected'].sum())} points"
        )

    # STRICT majority: more than `vote_fraction` of the eligible views, not
    # "at least". With two eligible views, `ceil(0.5 * 2) = 1` would let a
    # single view's sloppy mask carry a point that the other view rejected --
    # exactly the failure the vote exists to prevent.
    needed = np.floor(vote_fraction * eligible).astype(np.int64) + 1
    final = (eligible > 0) & (votes >= needed)
    point_ids = np.flatnonzero(final).tolist()
    note(
        f"vote over {len(per_view)} view(s): {len(point_ids)} of {xyz.shape[0]} points selected"
    )

    region = Region(
        id=region_id or new_region_id(),
        name=name or f"sam: {prompt.text or prompt.kind}",
        kind="pointset",
        params={"point_ids": point_ids},
        mix=float(mix),
        op=op,
        enabled=True,
    )

    summary: dict[str, Any] = {
        "n_points": len(point_ids),
        "n_points_total": int(xyz.shape[0]),
        "n_points_primary_view": int(lifted["selected"].sum()),
        "centroid": [float(c) for c in centroid],
        "view": primary.name,
        "views_used": [v["view"] for v in per_view],
        "views_around_requested": int(views_around),
        "prompt": prompt.to_json(),
        "per_view": per_view,
        "vote": {
            "fraction": float(vote_fraction),
            "n_views": len(per_view),
            "n_points_any_vote": int((votes > 0).sum()),
        },
        "settings": {
            "depth_cell_px": int(cell_px),
            "depth_tol": float(depth_tol),
            "apply_scene_distortion": bool(apply_scene_distortion),
            "op": op,
            "mix": float(mix),
            "prompt_space": prompt_space,
        },
    }
    if prompt_space == "view":
        summary["prompt_as_given"] = view_prompt.to_json()

    if preview is not None:
        summary["preview"] = str(
            write_selection_preview(preview, lifted["uv"], final, primary_wh)
        )
    return region, summary
