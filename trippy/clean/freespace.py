"""Is a low-confidence point in FRONT of a surface, on it, or behind it?

Module: trippy.clean.freespace
Purpose: the honesty check behind `trippy.clean.select`. The claim that
    justifies Design B is "TRIPS renders the ground under the tree, so it
    knows which points are surface and which are fog". This module tests
    that claim as a number instead of asserting it. For each frame it
    builds a coarse first-hit depth buffer out of the CONFIDENT points
    only -- the surface TRIPS believes in -- and classifies every point the
    deletion rule would remove as `front` (free space between the camera
    and that surface: fog), `on` (within a tolerance of the surface:
    geometry we would be damaging) or `behind` (occluded: harmless).
    Separately -- and this is the number that actually guards Jordan's
    complaint that the earlier Splats-side prune "removed the ground
    behind the cloud too" -- it counts PIXELS that held a point before the
    deletion and hold none after (`pixels_emptied`). A deletion can be
    entirely `front` and still be wrong if it was the only thing covering
    a pixel; a deletion that empties no pixel cannot have punched a hole.
Invariants:
    - No photograph is read, decoded or rendered; the depth buffers are
      built from point positions only, and the only thing that leaves this
      module is a dict of counts (AGENTS.md Sec 6).
    - The camera model, the depth convention (`(p - C) @ R[2]`) and the
      half-open pixel test are `trippy.train.prune.in_region`'s, so this
      and the Splats shade audit are looking through the same cameras.
    - The two buffers are NOT the same buffer. Classification uses the
      confident-points buffer; hole detection uses an ALL-points buffer,
      before and after the deletion. Measuring holes against the confident
      buffer would be vacuous: every deletion candidate is by definition
      below the confidence cutoff, so it was never in that buffer to begin
      with and removing it can never change it.
    - The buffers are built at `scale` (a divisor on the image size) purely
      for cost; the tolerance is expressed as a FRACTION of each frame's
      own median observed depth `d`, so it scales with the frame the way
      the audit's `znear`/`zfar` do.
Units: depths and tolerances are COLMAP world units; `scale` is
    dimensionless; `d` is the audit's per-frame median observed depth.
Related docs: docs/EXPERIMENTS.md "Shade audit"; trippy.train.prune
    (`ShadeView`, `in_region`); docs/SPEC.md D2.
"""

from __future__ import annotations

import numpy as np

from trippy.constants import (
    CLEAN_FREESPACE_DEPTH_SCALE,
    CLEAN_FREESPACE_SURFACE_CONF,
    CLEAN_FREESPACE_SURFACE_TOL_FRAC,
)
from trippy.train.prune import ShadeView


def project(
    view: ShadeView, xyz: np.ndarray, scale: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Project points into `view` at 1/`scale` resolution.

    Args:
        view: a `ShadeView` from `trippy.train.prune.build_shade_region`.
        xyz: (N, 3) world positions.
        scale: integer divisor on the image size (1 = full resolution).

    Returns:
        `(idx, pix, z, (w, h))`: `idx` are the indices of the points that
        land inside the image at positive depth, `pix` their flat pixel
        index into a `h * w` buffer, `z` their camera-space depth, and
        `(w, h)` the reduced buffer size.

    Raises:
        ValueError: `scale` is not a positive integer.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    w = max(1, view.width // scale)
    h = max(1, view.height // scale)
    sx = w / view.width
    sy = h / view.height

    xyz = np.asarray(xyz, dtype=np.float64)
    rel = xyz - view.C
    z = rel @ view.R[2]
    ahead = np.flatnonzero(z > 0)
    if ahead.size == 0:
        return ahead, ahead, z[ahead], (w, h)
    za = z[ahead]
    u = (view.fx * (rel[ahead] @ view.R[0]) / za + view.cx) * sx
    v = (view.fy * (rel[ahead] @ view.R[1]) / za + view.cy) * sy
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    vis = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    idx = ahead[vis]
    pix = vi[vis] * w + ui[vis]
    return idx, pix, za[vis], (w, h)


def first_hit_depth(pix: np.ndarray, z: np.ndarray, n_pixels: int) -> np.ndarray:
    """Nearest depth per pixel, `inf` where nothing landed.

    Implemented as a lexsort + first-of-run rather than `np.minimum.at`,
    which is an unbuffered ufunc call per element and is roughly an order
    of magnitude slower at the millions-of-points scale this runs at.

    Args:
        pix: (M,) flat pixel indices.
        z: (M,) depths, same order.
        n_pixels: buffer size.

    Returns:
        (n_pixels,) float64 depth buffer.
    """
    buf = np.full(n_pixels, np.inf, dtype=np.float64)
    if pix.size == 0:
        return buf
    order = np.lexsort((z, pix))
    ps = pix[order]
    zs = z[order]
    first = np.empty(ps.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(ps[1:], ps[:-1], out=first[1:])
    buf[ps[first]] = zs[first]
    return buf


def classify_against_surface(
    views: list[ShadeView],
    xyz: np.ndarray,
    conf: np.ndarray,
    candidates: np.ndarray,
    surface_conf: float = CLEAN_FREESPACE_SURFACE_CONF,
    tol_frac: float = CLEAN_FREESPACE_SURFACE_TOL_FRAC,
    scale: int = CLEAN_FREESPACE_DEPTH_SCALE,
) -> dict:
    """Where do the candidate deletions sit, and do they leave a hole?

    A candidate is classified under a "closest view wins" rule: the view in
    which it is nearest to the camera is the view whose opinion is taken,
    because that is the view it most obstructs.

    Args:
        views: frames, from `trippy.train.prune.build_shade_region`.
        xyz: (N, 3) world positions of ALL points (the surface is built
            from the confident ones among them).
        conf: (N,) learned confidence.
        candidates: (N,) bool, the points the deletion rule would remove.
        surface_conf: a point counts as surface at or above this confidence.
        tol_frac: a candidate within `tol_frac * view.d` of the surface
            depth counts as `on` it, not in front of or behind it; the same
            tolerance decides whether a pixel's first hit "moved".
        scale: depth-buffer resolution divisor.

    Returns:
        `{"n_candidates", "n_seen", "front", "on", "behind", "no_surface",
        "front_frac", "on_frac", "surface_conf", "tol_frac", "scale",
        "n_views", "surface_pixels", "pixels_covered", "pixels_emptied",
        "pixels_emptied_frac", "pixels_revealed",
        "pixels_revealed_confident"}`, pixel counts summed over views.
        `no_surface` counts candidates whose pixel held no confident point
        at all in the view that saw them closest. `pixels_emptied` is the
        hole count: pixels that held SOME point before the deletion and
        none after -- the failure mode this check exists to rule out.
        `pixels_revealed` counts pixels whose nearest point moved further
        away (the deletion opened a line of sight) and
        `pixels_revealed_confident` how many of those have a confident
        point to reveal -- together, the "we removed the fog and the ground
        was there all along" number.

    Raises:
        ValueError: `views` is empty.
    """
    if not views:
        raise ValueError("classify_against_surface needs at least one view")
    xyz = np.asarray(xyz, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=bool)
    is_surface = conf >= surface_conf

    n = xyz.shape[0]
    best_z = np.full(n, np.inf, dtype=np.float64)
    verdict = np.full(n, -1, dtype=np.int8)  # -1 unseen, 0 front, 1 on, 2 behind, 3 no surface
    surface_pixels = 0
    pixels_covered = 0
    pixels_emptied = 0
    pixels_revealed = 0
    revealed_confident = 0

    for view in views:
        idx, pix, z, (w, h) = project(view, xyz, scale)
        if idx.size == 0:
            continue
        n_pixels = w * h
        tol = tol_frac * view.d

        surf_here = is_surface[idx]
        depth = first_hit_depth(pix[surf_here], z[surf_here], n_pixels)
        has_surface = np.isfinite(depth)
        surface_pixels += int(has_surface.sum())

        any_before = first_hit_depth(pix, z, n_pixels)
        kept = ~candidates[idx]
        any_after = first_hit_depth(pix[kept], z[kept], n_pixels)
        had = np.isfinite(any_before)
        has = np.isfinite(any_after)
        pixels_covered += int(had.sum())
        pixels_emptied += int((had & ~has).sum())
        moved = had & has & (any_after > any_before + tol)
        pixels_revealed += int(moved.sum())
        revealed_confident += int((moved & has_surface).sum())

        cand_here = candidates[idx]
        if not cand_here.any():
            continue
        c_idx = idx[cand_here]
        c_pix = pix[cand_here]
        c_z = z[cand_here]
        closer = c_z < best_z[c_idx]
        if not closer.any():
            continue
        c_idx = c_idx[closer]
        c_pix = c_pix[closer]
        c_z = c_z[closer]
        best_z[c_idx] = c_z
        zs = depth[c_pix]
        verdicts = np.full(c_idx.shape[0], 3, dtype=np.int8)
        finite = np.isfinite(zs)
        verdicts[finite & (c_z < zs - tol)] = 0
        verdicts[finite & (np.abs(c_z - zs) <= tol)] = 1
        verdicts[finite & (c_z > zs + tol)] = 2
        verdict[c_idx] = verdicts

    n_seen = int((verdict >= 0).sum())
    counts = {
        label: int((verdict == code).sum())
        for code, label in ((0, "front"), (1, "on"), (2, "behind"), (3, "no_surface"))
    }
    return {
        "n_candidates": int(candidates.sum()),
        "n_seen": n_seen,
        **counts,
        "front_frac": counts["front"] / max(n_seen, 1),
        "on_frac": counts["on"] / max(n_seen, 1),
        "surface_conf": float(surface_conf),
        "tol_frac": float(tol_frac),
        "scale": int(scale),
        "n_views": len(views),
        "surface_pixels": surface_pixels,
        "pixels_covered": pixels_covered,
        "pixels_emptied": pixels_emptied,
        "pixels_emptied_frac": pixels_emptied / max(pixels_covered, 1),
        "pixels_revealed": pixels_revealed,
        "pixels_revealed_confident": revealed_confident,
    }
