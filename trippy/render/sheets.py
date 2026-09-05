"""Contact-sheet composer and a matplotlib-free depth/coverage colour map.

Module: trippy.render.sheets
Invariants: PIL only for image compositing/resizing/text (no matplotlib,
    per AGENTS.md "no new dependencies"); every public function accepts
    both uint8 HxWx3 and float [0, 1] HxW/HxWx3 arrays and normalises to
    uint8 RGB internally via _to_uint8_rgb. Used for the honesty sheet
    (docs/SPEC.md: raw composite | network output | coverage/provenance
    map) and for quick-look multi-image reviews.
Related docs: docs/SPEC.md "Technical design" (honesty sheet); AGENTS.md
    section 7 "Honesty rule" (every candidate exported with a 3-panel
    honesty sheet).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from trippy.constants import (
    COLORMAP_VIRIDIS_STOPS,
    CONTACT_SHEET_BG,
    CONTACT_SHEET_CELL_MAX,
    CONTACT_SHEET_LABEL_BAND_PX,
    CONTACT_SHEET_PAD,
)

_VIRIDIS_STOPS = np.array(COLORMAP_VIRIDIS_STOPS, dtype=np.float64)


def colorize(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a scalar array to RGB with a matplotlib-free viridis-like ramp.

    Args:
        depth: (H, W) (or any shape) float array, e.g. depth or coverage.
        vmin: value mapped to the first colour stop.
        vmax: value mapped to the last colour stop; must be > vmin.

    Returns:
        uint8 array, same leading shape as `depth` plus a trailing 3
        (RGB), linearly interpolated between COLORMAP_VIRIDIS_STOPS
        (5 stops at t = 0, 0.25, 0.5, 0.75, 1.0). Values outside
        [vmin, vmax] are clamped before mapping, so the output is
        monotonically non-decreasing (in perceived luminance) with the
        input value.

    Raises:
        ValueError: vmax <= vmin.
    """
    depth = np.asarray(depth, dtype=np.float64)
    span = vmax - vmin
    if span <= 0:
        raise ValueError(f"vmax must be > vmin, got vmin={vmin}, vmax={vmax}")

    t = np.clip((depth - vmin) / span, 0.0, 1.0)
    n_stops = _VIRIDIS_STOPS.shape[0]
    scaled = t * (n_stops - 1)
    lo = np.clip(np.floor(scaled).astype(np.int64), 0, n_stops - 2)
    frac = (scaled - lo)[..., None]
    rgb = _VIRIDIS_STOPS[lo] + frac * (_VIRIDIS_STOPS[lo + 1] - _VIRIDIS_STOPS[lo])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalise an image to uint8 (H, W, 3), accepting float [0,1] or grayscale.

    Args:
        img: (H, W) or (H, W, 3), uint8 or floating point.

    Returns:
        uint8 (H, W, 3) array (grayscale is broadcast to 3 channels;
        floating point is assumed to be in [0, 1] and scaled to [0, 255]).

    Raises:
        ValueError: unsupported shape (not 2-D or (*, *, 3)).
    """
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.round(arr * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be (H, W) or (H, W, 3), got shape {arr.shape}")
    return arr


def _resize_to_cell(arr: np.ndarray, cell_max: int) -> Image.Image:
    """Resize (preserving aspect ratio) so the longer side equals cell_max."""
    img = Image.fromarray(arr, mode="RGB")
    w, h = img.size
    scale = cell_max / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.BILINEAR)


def contact_sheet(
    images: list[np.ndarray],
    labels: list[str],
    cols: int,
    cell_max: int = CONTACT_SHEET_CELL_MAX,
    pad: int = CONTACT_SHEET_PAD,
    bg: tuple[int, int, int] = CONTACT_SHEET_BG,
) -> np.ndarray:
    """Compose a labelled grid contact sheet from a list of images.

    Args:
        images: list of (H, W, 3) uint8 or (H, W) uint8/float[0,1] arrays
            (mixed shapes/dtypes allowed; each is independently resized
            to fit within a cell_max x cell_max box, aspect preserved).
        labels: text label drawn under each image, same length as images.
        cols: number of columns; rows = ceil(len(images) / cols).
        cell_max: longest side (pixels) each thumbnail is resized to fit.
        pad: gutter (pixels) around every cell and the sheet border.
        bg: (R, G, B) fill colour for gutters and letterboxed thumbnails.

    Returns:
        uint8 (sheet_h, sheet_w, 3) array.

    Raises:
        ValueError: images/labels length mismatch, empty input, or
            non-positive cols.
    """
    if len(images) != len(labels):
        raise ValueError(f"images and labels must be the same length, got {len(images)} and {len(labels)}")
    if len(images) == 0:
        raise ValueError("images must be non-empty")
    if cols <= 0:
        raise ValueError(f"cols must be positive, got {cols}")

    n = len(images)
    rows = math.ceil(n / cols)
    cell_w = cell_max
    cell_h = cell_max + CONTACT_SHEET_LABEL_BAND_PX
    stride_w = cell_w + pad
    stride_h = cell_h + pad

    sheet_w = pad + cols * stride_w
    sheet_h = pad + rows * stride_h
    sheet_img = Image.new("RGB", (sheet_w, sheet_h), color=bg)
    draw = ImageDraw.Draw(sheet_img)

    for i, (img_arr, label) in enumerate(zip(images, labels, strict=True)):
        r, c = divmod(i, cols)
        x0 = pad + c * stride_w
        y0 = pad + r * stride_h

        thumb = _resize_to_cell(_to_uint8_rgb(img_arr), cell_max)
        tw, th = thumb.size
        ox = x0 + (cell_w - tw) // 2
        oy = y0 + (cell_max - th) // 2
        sheet_img.paste(thumb, (ox, oy))

        text_y = y0 + cell_max + CONTACT_SHEET_LABEL_BAND_PX // 4
        draw.text((x0, text_y), label, fill=(255, 255, 255))

    return np.array(sheet_img)


def side_by_side(
    images: list[np.ndarray],
    labels: list[str],
    cell_max: int = CONTACT_SHEET_CELL_MAX,
    pad: int = CONTACT_SHEET_PAD,
    bg: tuple[int, int, int] = CONTACT_SHEET_BG,
) -> np.ndarray:
    """contact_sheet() laid out as a single row (e.g. the 3-panel honesty sheet).

    Args:
        images: see contact_sheet.
        labels: see contact_sheet.
        cell_max: see contact_sheet.
        pad: see contact_sheet.
        bg: see contact_sheet.

    Returns:
        uint8 (H, W, 3) array, one row, len(images) columns.
    """
    return contact_sheet(images, labels, cols=len(images), cell_max=cell_max, pad=pad, bg=bg)


def save_png(path: str | Path, arr: np.ndarray) -> Path:
    """Save an image array (uint8 or float [0,1], grayscale or RGB) as a PNG.

    Args:
        path: output .png path; parent directories are created if missing.
        arr: (H, W) or (H, W, 3) array, uint8 or floating point in [0, 1].

    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_rgb(arr), mode="RGB").save(path)
    return path
