"""Click-to-cluster: a clicked pixel to a `pointset` Region, by 3D k-NN growth.

Module: trippy.edit.cluster
Purpose: the E4 milestone (docs/EDITOR.md Sec 3 "1. Click-to-cluster (E4)")
    -- turn a bundle, a view (or an explicit camera), and a clicked pixel
    into a `pointset` Region that "visibly matches the clicked object's
    extent, without needing a depth buffer". The algorithm, precisely:

    1. Project every point into the clicked view with
       `trippy.geom.xform_a` (`world_to_cam` + `project_pinhole`; numbers
       only, no distortion -- see the Invariants below for why).
    2. `candidates` = points whose projection lands within `radius_px` of
       the click AND are in front of the camera (`depth > 0`).
    3. Among `candidates`, cluster camera-space depth by gap: the group
       nearest the camera (smallest depth) is the SEED depth mode. This is
       what stops a click from selecting straight through a gap onto a
       same-coloured surface behind the clicked object -- the two form two
       depth modes, and only the nearer one seeds.
    4. `seed` = the candidates in that nearest depth mode.
    5. Grow outward from `seed` by repeated k-NN queries
       (`scipy.spatial.cKDTree`, `trippy.points.knn_size`'s own query
       pattern) in 3D, admitting a neighbour only if its base colour
       (`feat[:, :3]`, clipped to `[0, 1]` -- the same slice
       `trippy.edit.shade_finder` reads) is within `colour_tol` of the
       seed's own mean colour AND its distance from the seed centroid is
       within `max_radius`. Growth stops when the frontier is exhausted or
       `max_points` is reached.

Invariants:
    - No distortion is applied when projecting: `trippy.geom.xform_a` has
      no distortion model (that lives in `trippy.render.parity`, torch-only,
      and in `trippy.geom.camera`'s OpenCV convention, a different
      convention again from a bundle view's Saiga coefficients). A
      trippy-native bundle's views carry all-zero distortion by construction
      (`trippy.render.bundle.native_views`), so this is exact there; on a
      TRIPS/ADOP bundle it is a first-order approximation of where a point
      lands -- acceptable for "which points are near this click", not
      claimed accurate to sub-pixel precision. Recorded here rather than
      silently matching TRIPS's own distortion, which this module never
      imports (no torch).
    - The depth-gap threshold and the growth's own `max_radius` share ONE
      scene scale by design (`trippy.constants.CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR`
      /`CLICK_DEFAULT_DEPTH_GAP_FACTOR`): see those constants' own comments.
    - `click_to_cluster` is pure numpy (+ `scipy.spatial.cKDTree`); no
      torch, no MPS, no Rust -- same posture as the rest of `trippy.edit`.
    - The returned Region's `point_ids` are indices into the SAME `xyz`
      array passed in (points.npz's own row order, per
      `trippy.edit.model`'s own `pointset` contract).
    - `render_click_preview` never opens or reads a photograph -- it
      rasterises a synthetic point-density heatmap from scratch (a scatter
      of the selected points' own projections), which `AGENTS.md` Sec 6
      explicitly allows ("abstract heatmaps with no photographic content").
Units: `xyz`/camera `t`/`max_radius` are world units (COLMAP world frame,
    docs/GEOMETRY.md); `radius_px` is screen pixels; `colour_tol` is a
    Euclidean distance in `[0, 1]^3` colour space.
Related docs: docs/EDITOR.md Sec 3 "1. Click-to-cluster (E4)";
    trippy.geom.xform_a (the projection); trippy.points.knn_size (the
    cKDTree query pattern this reuses); trippy.edit.shade_finder (the
    sibling `pointset`-producing selector, same base-colour convention);
    trippy.edit.model (the `Region`/`pointset` contract).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from trippy.constants import (
    CLICK_DEFAULT_COLOUR_TOL,
    CLICK_DEFAULT_DEPTH_GAP_FACTOR,
    CLICK_DEFAULT_MAX_POINTS,
    CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR,
    CLICK_DEFAULT_MIX,
    CLICK_DEFAULT_OP,
    CLICK_DEFAULT_RADIUS_PX,
    CLICK_FALLBACK_MAX_RADIUS_FACTOR,
    CLICK_GROW_KNN_K,
    CLICK_PREVIEW_MAX_DIM,
)
from trippy.edit.model import Region, auto_region_name, new_region_id
from trippy.geom.xform_a import project_pinhole, world_to_cam
from trippy.points.knn_size import median_nn_distance

__all__ = [
    "CameraView",
    "camera_center",
    "camera_view_from_bundle_view",
    "click_to_cluster",
    "click_to_cluster_in_bundle",
    "default_max_radius_from_bundle",
    "find_bundle_view",
    "project_points",
    "render_click_preview",
]


# --- camera -------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraView:
    """The pieces of a `bundle.json` view a projection needs.

    Attributes:
        R: `(3, 3)` world-to-camera rotation (row-major, `x_cam = R @ x_world + t`).
        t: `(3,)` world-to-camera translation, world units.
        fx, fy, cx, cy: pinhole intrinsics, pixels.
        width, height: image size, pixels (used only by the preview PNG's canvas).
        name: image filename, for messages/summaries only.
    """

    R: np.ndarray
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""


def camera_view_from_bundle_view(view: dict[str, Any]) -> CameraView:
    """Build a `CameraView` from one `bundle.json` `views[]` entry.

    Raises:
        KeyError: `view` is missing `R`/`t`/`fx`/`fy`/`cx`/`cy`.
        ValueError: `R` is not 9 numbers or `t` is not 3.
    """
    R = np.asarray(view["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(view["t"], dtype=np.float64).reshape(3)
    return CameraView(
        R=R,
        t=t,
        fx=float(view["fx"]),
        fy=float(view["fy"]),
        cx=float(view["cx"]),
        cy=float(view["cy"]),
        width=int(view.get("width", 0)),
        height=int(view.get("height", 0)),
        name=str(view.get("name", "")),
    )


def find_bundle_view(bundle_doc: dict[str, Any], view_name: str) -> dict[str, Any]:
    """The `views[]` entry of `bundle_doc` named `view_name`.

    Raises:
        ValueError: no view of `bundle_doc["views"]` has that `name`.
    """
    for view in bundle_doc.get("views", []):
        if view.get("name") == view_name:
            return view
    available = [v.get("name") for v in bundle_doc.get("views", [])]
    raise ValueError(f"no view named {view_name!r} in bundle (have: {available})")


def camera_center(view: dict[str, Any]) -> np.ndarray:
    """A view's world-frame camera centre, `C = -R^T @ t` (world-to-camera `R`, `t`)."""
    R = np.asarray(view["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(view["t"], dtype=np.float64).reshape(3)
    return -R.T @ t


def default_max_radius_from_bundle(bundle_doc: dict[str, Any], xyz: np.ndarray) -> float:
    """The click-to-cluster default `max_radius`: median nearest-CAMERA spacing.

    See `trippy.constants.CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR`'s comment for
    why camera baseline, not point-cloud density, is the scale used. Falls back
    to a multiple of the point cloud's own median nearest-neighbour spacing
    (`trippy.points.knn_size.median_nn_distance`) when the bundle has fewer
    than two views (camera spacing is then undefined).

    Args:
        bundle_doc: the loaded `bundle.json` document.
        xyz: `(N, 3)` world-frame points -- only used for the fallback.

    Returns:
        A positive float, world units. `0.0` only when both the camera
        count and the point count are too small to measure any scale at
        all (fewer than 2 cameras AND fewer than 2 points) -- callers
        should treat that as "let the caller supply max_radius explicitly".
    """
    views = bundle_doc.get("views", [])
    if len(views) >= 2:
        centers = np.stack([camera_center(v) for v in views], axis=0)
        spacing = median_nn_distance(centers)
        if spacing > 0.0:
            return float(CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR * spacing)
    spacing = median_nn_distance(np.asarray(xyz, dtype=np.float64))
    return float(CLICK_FALLBACK_MAX_RADIUS_FACTOR * spacing)


# --- projection -----------------------------------------------------------------------


def project_points(xyz: np.ndarray, camera: CameraView) -> tuple[np.ndarray, np.ndarray]:
    """Project every point into `camera`'s pixel space (`trippy.geom.xform_a`, no distortion).

    Args:
        xyz: `(N, 3)` world-frame positions.
        camera: the view to project into.

    Returns:
        `(uv, depth)`: `uv` is `(N, 2)` float64 pixel coordinates (undefined,
        but finite, for `depth <= 0`); `depth` is `(N,)` float64 camera-space
        z (positive = in front of the camera).
    """
    xyz_c = world_to_cam(camera.R, camera.t, xyz)
    uv, depth = project_pinhole(xyz_c, camera.fx, camera.fy, camera.cx, camera.cy)
    return uv, depth


# --- depth-mode seed selection ---------------------------------------------------------


def _nearest_depth_mode_mask(depth: np.ndarray, gap_threshold: float) -> np.ndarray:
    """Boolean mask over `depth` selecting its nearest-camera contiguous run.

    Sorts `depth` ascending and cuts the run at the first gap wider than
    `gap_threshold`; the returned mask covers everything from the smallest
    depth up to (not including) that cut -- "the depth mode nearest the
    camera" among the candidates, per the module docstring's step 3.

    Args:
        depth: `(M,)` float64 camera-space depths, all `> 0`.
        gap_threshold: a gap in sorted depth wider than this starts a new
            mode. `<= 0` degenerates to "every candidate is its own mode",
            i.e. only the single nearest point seeds.

    Returns:
        `(M,)` bool mask, indices into `depth`'s own (unsorted) order.
    """
    order = np.argsort(depth)
    sorted_depth = depth[order]
    gaps = np.diff(sorted_depth)
    breaks = np.flatnonzero(gaps > gap_threshold)
    end = int(breaks[0]) + 1 if breaks.size else sorted_depth.size
    mask = np.zeros(depth.shape[0], dtype=bool)
    mask[order[:end]] = True
    return mask


# --- growth -----------------------------------------------------------------------------


def click_to_cluster(
    xyz: np.ndarray,
    feat: np.ndarray,
    camera: CameraView,
    px: tuple[float, float],
    radius_px: float = CLICK_DEFAULT_RADIUS_PX,
    colour_tol: float = CLICK_DEFAULT_COLOUR_TOL,
    max_radius: float | None = None,
    max_points: int = CLICK_DEFAULT_MAX_POINTS,
    knn_k: int = CLICK_GROW_KNN_K,
    depth_gap_factor: float = CLICK_DEFAULT_DEPTH_GAP_FACTOR,
    op: str = CLICK_DEFAULT_OP,
    mix: float = CLICK_DEFAULT_MIX,
    region_id: str | None = None,
    name: str | None = None,
    existing_names: Any = (),
) -> tuple[Region, dict[str, Any]]:
    """Click `px` in `camera`'s view; grow a `pointset` Region from the nearest surface there.

    See the module docstring for the full algorithm. `max_radius` should
    normally come from `default_max_radius_from_bundle` when a bundle is
    available (`click_to_cluster_in_bundle` does this); a direct caller
    with no bundle context may pass `None` to fall back to
    `CLICK_FALLBACK_MAX_RADIUS_FACTOR * median_nn_distance(xyz)`.

    Args:
        xyz: `(N, 3)` world-frame positions -- `point_ids` index this array.
        feat: `(N, C)` per-point features; `feat[:, :3]` (clipped to
            `[0, 1]`) is the base colour the colour gate reads.
        camera: the view the click was made in.
        px: `(u, v)` clicked pixel, in `camera`'s own pixel coordinates.
        radius_px: click catchment radius, screen pixels.
        colour_tol: max Euclidean colour distance (in `[0, 1]^3`) from the
            seed's own mean colour a grown point may have.
        max_radius: max Euclidean distance (world units) from the seed
            centroid a grown point may have. `None` uses the fallback
            above.
        max_points: hard cap on the selection's size.
        knn_k: neighbours queried per growth step (see `CLICK_GROW_KNN_K`).
        depth_gap_factor: multiplies `max_radius` to get the depth-mode
            gap threshold (see `trippy.constants.CLICK_DEFAULT_DEPTH_GAP_FACTOR`).
        op, mix: the new Region's own fields (docs/EDITOR.md Sec 1).
        region_id: stable id for the new region (default: a fresh one).
        name: shown in the Regions panel. `None` (the default) auto-names
            the region `"click-<n>"` (`trippy.edit.model.auto_region_name`,
            docs/EDITOR.md Sec 1 "Named regions").
        existing_names: the target document's current region names, used
            ONLY to number the auto name above; ignored when `name` is
            given explicitly.

    Returns:
        `(region, summary)`. `summary` always has `n_candidates`,
        `n_seed`, `n_selected`, `max_radius`, `radius_px`, `colour_tol`,
        `click_px`, `hit_max_points` (bool); `n_candidates == 0` additionally
        sets `summary["warning"]` and returns an EMPTY pointset region
        rather than raising (a miss is a normal outcome of a click tool,
        not an error).

    Raises:
        ValueError: `xyz`/`feat` row counts disagree, or `feat` has fewer
            than 3 columns.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2 or feat.shape[1] < 3:
        raise ValueError(f"feat must be (N, C>=3), got {feat.shape}")
    if feat.shape[0] != xyz.shape[0]:
        raise ValueError(f"xyz has {xyz.shape[0]} rows but feat has {feat.shape[0]}")
    n = int(xyz.shape[0])
    base_rgb = np.clip(feat[:, :3], 0.0, 1.0)

    if max_radius is None:
        spacing = median_nn_distance(xyz)
        max_radius = float(CLICK_FALLBACK_MAX_RADIUS_FACTOR * spacing)
    max_radius = float(max_radius)

    u0, v0 = float(px[0]), float(px[1])
    uv, depth = project_points(xyz, camera)
    px_dist = np.hypot(uv[:, 0] - u0, uv[:, 1] - v0)
    candidate_mask = (depth > 0.0) & (px_dist <= float(radius_px))
    candidate_idx = np.flatnonzero(candidate_mask)

    resolved_name = name if name is not None else auto_region_name(existing_names, "click")
    source = {
        "tool": "click",
        "params": {
            "view": camera.name,
            "px": [u0, v0],
            "radius_px": float(radius_px),
            "colour_tol": float(colour_tol),
            "max_radius": max_radius,
            "op": op,
            "mix": float(mix),
        },
    }

    summary: dict[str, Any] = {
        "click_px": [u0, v0],
        "radius_px": float(radius_px),
        "colour_tol": float(colour_tol),
        "max_radius": max_radius,
        "n_candidates": int(candidate_idx.size),
    }

    if candidate_idx.size == 0 or n == 0:
        region = Region(
            id=region_id or new_region_id(),
            name=resolved_name,
            kind="pointset",
            params={"point_ids": []},
            mix=mix,
            op=op,
            source=source,
        )
        summary.update({"n_seed": 0, "n_selected": 0, "hit_max_points": False})
        summary["warning"] = "no points projected within radius_px of the click"
        return region, summary

    depth_candidates = depth[candidate_idx]
    gap_threshold = float(depth_gap_factor) * max_radius
    seed_mask = _nearest_depth_mode_mask(depth_candidates, gap_threshold)
    seed_idx = candidate_idx[seed_mask]

    seed_centroid = xyz[seed_idx].mean(axis=0)
    region_colour = base_rgb[seed_idx].mean(axis=0)

    tree = cKDTree(xyz)
    selected: set[int] = {int(i) for i in seed_idx}
    frontier = list(selected)
    hit_max_points = False
    max_points = int(max_points)

    while frontier and len(selected) < max_points:
        k_eff = min(int(knn_k), n)
        _, neighbour_idx = tree.query(xyz[frontier], k=k_eff)
        # `frontier` is always indexed as a list, so `xyz[frontier]` is always (M, 3) --
        # cKDTree.query then returns (M,) only when k_eff == 1 (never (M, k) squeezed to
        # (M,) any other way), so this reshape is the one case `np.atleast_2d` would get
        # wrong (it would produce (1, M), not (M, 1)).
        if k_eff == 1:
            neighbour_idx = neighbour_idx.reshape(-1, 1)
        next_frontier: list[int] = []
        for idx in np.unique(neighbour_idx.reshape(-1)):
            idx = int(idx)
            if idx in selected:
                continue
            if float(np.linalg.norm(base_rgb[idx] - region_colour)) > colour_tol:
                continue
            if float(np.linalg.norm(xyz[idx] - seed_centroid)) > max_radius:
                continue
            selected.add(idx)
            next_frontier.append(idx)
            if len(selected) >= max_points:
                hit_max_points = True
                break
        frontier = next_frontier

    point_ids = sorted(selected)
    region = Region(
        id=region_id or new_region_id(),
        name=resolved_name,
        kind="pointset",
        params={"point_ids": point_ids},
        mix=mix,
        op=op,
        source=source,
    )
    summary.update(
        {
            "n_seed": int(seed_idx.size),
            "n_selected": len(point_ids),
            "seed_depth_mean": float(depth_candidates[seed_mask].mean()),
            "hit_max_points": hit_max_points,
        }
    )
    return region, summary


def click_to_cluster_in_bundle(
    bundle_dir: str | Path,
    view_name: str,
    px: tuple[float, float],
    max_radius: float | None = None,
    preview_path: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Region, dict[str, Any]]:
    """`click_to_cluster`, sourcing `xyz`/`feat`/the camera from a bundle directory.

    Args:
        bundle_dir: bundle directory (`bundle.json` + `points.npz`).
        view_name: a `views[]` entry's own `name` (e.g. an image filename).
        px: `(u, v)` clicked pixel.
        max_radius: forwarded to `click_to_cluster`; `None` computes
            `default_max_radius_from_bundle`'s own default from this
            bundle's camera spacing (docs/EDITOR.md Sec 3's "1.").
        preview_path: if given, also writes `render_click_preview`'s
            heatmap PNG there; the summary then carries a `"preview"` key.
        **kwargs: forwarded to `click_to_cluster` (`radius_px`,
            `colour_tol`, `max_points`, `knn_k`, `depth_gap_factor`, `op`,
            `mix`, `region_id`, `name`).

    Returns:
        Same as `click_to_cluster`.

    Raises:
        ValueError: `view_name` is not a registered view.
        FileNotFoundError: `bundle_dir` has no `bundle.json`/`points.npz`.
    """
    bundle_dir = Path(bundle_dir)
    bundle_doc = json.loads((bundle_dir / "bundle.json").read_text())
    points_path = bundle_dir / bundle_doc.get("points", "points.npz")
    with np.load(points_path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        feat = np.asarray(data["feat"], dtype=np.float64)

    view = find_bundle_view(bundle_doc, view_name)
    camera = camera_view_from_bundle_view(view)

    if max_radius is None:
        max_radius = default_max_radius_from_bundle(bundle_doc, xyz)

    region, summary = click_to_cluster(xyz, feat, camera, px, max_radius=max_radius, **kwargs)
    summary["view"] = view_name

    if preview_path is not None:
        point_ids = region.params["point_ids"]
        summary["preview"] = render_click_preview(xyz, camera, point_ids, preview_path, click_px=px)

    return region, summary


# --- preview: a from-scratch heatmap PNG, no photo content --------------------------


def _hot_colormap(value: np.ndarray) -> np.ndarray:
    """`value` in `[0, 1]` -> `(..., 3)` uint8 RGB, the standard "hot" ramp (black -> red -> yellow -> white).

    Computed from a plain piecewise-linear formula (no matplotlib dependency,
    per `pyproject.toml`'s dependency list -- `pillow` is already present,
    `matplotlib` is not).
    """
    v = np.clip(value, 0.0, 1.0)
    r = np.clip(3.0 * v, 0.0, 1.0)
    g = np.clip(3.0 * v - 1.0, 0.0, 1.0)
    b = np.clip(3.0 * v - 2.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def render_click_preview(
    xyz: np.ndarray,
    camera: CameraView,
    point_ids: list[int],
    out_path: str | Path,
    click_px: tuple[float, float] | None = None,
    max_dim: int = CLICK_PREVIEW_MAX_DIM,
) -> dict[str, Any]:
    """A from-scratch heatmap PNG of `point_ids`' projections into `camera` -- no photo content.

    Purely a synthetic density scatter of the SELECTED points' own pixel
    projections (`trippy.geom.xform_a`, same as `click_to_cluster`'s own
    projection) rendered on a blank canvas -- never a photograph, never a
    render of the scene's actual appearance (`AGENTS.md` Sec 6's "abstract
    heatmaps with no photographic content" is the allowed case this is).

    Args:
        xyz: `(N, 3)` world-frame positions (the same array `point_ids`
            indexes).
        camera: the view the click was made in.
        point_ids: indices into `xyz` to plot (typically the selected
            region's own `point_ids`).
        out_path: PNG path (parent directories created if missing).
        click_px: if given, draws a small crosshair there (scaled into the
            canvas), so the preview shows the click alongside the selection.
        max_dim: longest canvas edge, pixels (`CLICK_PREVIEW_MAX_DIM`).

    Returns:
        `{"path": str, "n_selected": int, "canvas": [w, h]}`.
    """
    out_path = Path(out_path)
    width = max(int(camera.width), 1)
    height = max(int(camera.height), 1)
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    canvas_w = max(1, round(width * scale))
    canvas_h = max(1, round(height * scale))

    density = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    ids = np.asarray(point_ids, dtype=np.int64) if len(point_ids) else np.zeros(0, dtype=np.int64)
    if ids.size:
        uv, depth = project_points(xyz[ids], camera)
        visible = depth > 0.0
        bx = np.clip(np.floor(uv[visible, 0] * scale).astype(np.int64), 0, canvas_w - 1)
        by = np.clip(np.floor(uv[visible, 1] * scale).astype(np.int64), 0, canvas_h - 1)
        np.add.at(density, (by, bx), 1.0)

    peak = float(density.max())
    normalized = density / peak if peak > 0.0 else density
    rgb = _hot_colormap(normalized)
    image = Image.fromarray(rgb, mode="RGB")

    if click_px is not None:
        draw = ImageDraw.Draw(image)
        cx = float(click_px[0]) * scale
        cy = float(click_px[1]) * scale
        arm = 4
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(0, 200, 255))
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(0, 200, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {"path": str(out_path), "n_selected": int(ids.size), "canvas": [canvas_w, canvas_h]}
