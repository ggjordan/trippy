"""Top-down density map of what was deleted -- the honesty artifact for Design B.

Module: trippy.clean.heatmap
Purpose: AGENTS.md Sec 7's honesty rule ("every experiment leaves an
    artifact Jordan can open") without breaking Sec 6's privacy rule. A
    contact sheet of a Karekare render is forbidden; a 2D histogram of
    Gaussian centres is not photographic content and can be opened, by
    Jordan or by an agent, safely. It answers the one question the counts
    cannot: is the deletion CONCENTRATED where the fog is, or is it a thin
    uniform shave off the whole scene (which would mean the rule is not
    finding anything, just lowering density everywhere)?
Invariants:
    - Drawn from scratch out of point coordinates. No photograph, render,
      depth map or mask is read, and no pixel of this image derives from
      one.
    - Two panels side by side on one canvas: LEFT the deleted-Gaussian
      count per cell, RIGHT the deleted FRACTION of that cell's Gaussians.
      The fraction panel is the honest one -- a raw count map is bright
      wherever the scene is dense, whether or not the rule targeted it.
    - Both panels are normalised by their own p99.5 (not their max), so one
      freak cell cannot flatten the image; cells with fewer than
      `CLEAN_HEATMAP_MIN_CELL_POINTS` Gaussians are left at the floor
      colour in the fraction panel, because a 1-of-1 cell is 100% and means
      nothing.
Units: the axes are two COLMAP world axes (default X and Z, i.e. the
    ground plane for a +Y-down scene); cell size is in world units.
Related docs: AGENTS.md Sec 6 ("abstract heatmaps with no photographic
    content" are the allowed image class), Sec 7 (honesty rule);
    trippy.render.sheets (the colour ramp reused here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import (
    CLEAN_HEATMAP_CELLS,
    CLEAN_HEATMAP_GUTTER_PX,
    CLEAN_HEATMAP_MIN_CELL_POINTS,
    CLEAN_HEATMAP_PERCENTILE,
    COLORMAP_VIRIDIS_STOPS,
)


def _ramp(t: np.ndarray) -> np.ndarray:
    """Map t in [0, 1] onto `COLORMAP_VIRIDIS_STOPS`, returning (..., 3) uint8."""
    stops = np.asarray(COLORMAP_VIRIDIS_STOPS, dtype=np.float64)
    n = stops.shape[0] - 1
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * n
    lo = np.clip(np.floor(t).astype(np.int64), 0, n - 1)
    frac = (t - lo)[..., None]
    return np.clip(stops[lo] * (1 - frac) + stops[lo + 1] * frac, 0, 255).astype(np.uint8)


def _grid(xyz: np.ndarray, axes: tuple[int, int], lo: np.ndarray, hi: np.ndarray, cells: int) -> np.ndarray:
    """2D histogram of `xyz` over `axes`, `cells` x `cells`, clipped to [lo, hi]."""
    a, b = axes
    span = np.maximum(hi - lo, np.finfo(np.float64).eps)
    ia = np.clip(((xyz[:, a] - lo[0]) / span[0] * cells).astype(np.int64), 0, cells - 1)
    ib = np.clip(((xyz[:, b] - lo[1]) / span[1] * cells).astype(np.int64), 0, cells - 1)
    flat = np.bincount(ib * cells + ia, minlength=cells * cells)
    return flat.reshape(cells, cells).astype(np.float64)


def deleted_density_png(
    all_xyz: np.ndarray,
    deleted: np.ndarray,
    dest: str | Path,
    axes: tuple[int, int] = (0, 2),
    cells: int = CLEAN_HEATMAP_CELLS,
    percentile: float = CLEAN_HEATMAP_PERCENTILE,
) -> dict:
    """Write the two-panel top-down deletion map (see module docstring).

    Args:
        all_xyz: (n_ply, 3) every Gaussian centre in the source splat.
        deleted: (n_ply,) bool, which ones this variant removes.
        dest: output `.png` path (parents created).
        axes: which two coordinate axes to plot (default X, Z).
        cells: grid resolution per side.
        percentile: normalisation percentile for both panels.

    Returns:
        `{"path", "cells", "axes", "extent", "max_cell_deleted",
        "max_cell_fraction", "occupied_cells", "cells_touched"}` -- numbers
        that describe the picture, so the verdict never depends on anyone
        looking at it.

    Raises:
        ValueError: `all_xyz` and `deleted` disagree on length.
    """
    # Deferred: PIL import costs ~200 ms and only this function needs it.
    from PIL import Image

    all_xyz = np.asarray(all_xyz, dtype=np.float64)
    deleted = np.asarray(deleted, dtype=bool)
    if deleted.shape[0] != all_xyz.shape[0]:
        raise ValueError(f"deleted has {deleted.shape[0]} entries, all_xyz has {all_xyz.shape[0]}")

    a, b = axes
    lo = np.array([np.percentile(all_xyz[:, a], 0.5), np.percentile(all_xyz[:, b], 0.5)])
    hi = np.array([np.percentile(all_xyz[:, a], 99.5), np.percentile(all_xyz[:, b], 99.5)])

    total = _grid(all_xyz, axes, lo, hi, cells)
    gone = _grid(all_xyz[deleted], axes, lo, hi, cells)

    count_norm = max(float(np.percentile(gone[gone > 0], percentile)) if (gone > 0).any() else 1.0, 1.0)
    count_panel = _ramp(gone / count_norm)

    frac = np.zeros_like(total)
    enough = total >= CLEAN_HEATMAP_MIN_CELL_POINTS
    frac[enough] = gone[enough] / total[enough]
    frac_norm = max(float(np.percentile(frac[enough], percentile)) if enough.any() else 1.0, 1e-6)
    frac_panel = _ramp(frac / frac_norm)

    gutter = CLEAN_HEATMAP_GUTTER_PX
    canvas = np.zeros((cells, cells * 2 + gutter, 3), dtype=np.uint8)
    canvas[:, :cells] = count_panel[::-1]
    canvas[:, cells + gutter :] = frac_panel[::-1]

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(dest)
    return {
        "path": str(dest),
        "cells": int(cells),
        "axes": list(axes),
        "extent": {"lo": lo.tolist(), "hi": hi.tolist()},
        "max_cell_deleted": int(gone.max()),
        "max_cell_fraction": float(frac.max()),
        "occupied_cells": int((total > 0).sum()),
        "cells_touched": int((gone > 0).sum()),
    }
