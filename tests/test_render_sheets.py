"""Tests for trippy.render.sheets: contact_sheet, side_by_side, colorize, save_png.

Module: tests.test_render_sheets
Invariants under test: contact_sheet() produces the expected grid shape
    and actually draws non-background pixels in each cell's label band
    (i.e. labels are not silently skipped); colorize() is monotonically
    non-decreasing in perceived luminance as the input value increases
    (the property a viridis-like ramp is chosen for); save_png round-trips
    a synthetic image to disk.
Fixture: synthetic random/gradient arrays only (no photos).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trippy.constants import CONTACT_SHEET_BG, CONTACT_SHEET_LABEL_BAND_PX, CONTACT_SHEET_PAD
from trippy.render.sheets import colorize, contact_sheet, save_png, side_by_side


def _solid_image(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.array(color, dtype=np.uint8), (h, w, 1))


def test_contact_sheet_shape_for_grid() -> None:
    images = [_solid_image(64, 32, (255, 0, 0)) for _ in range(5)]
    labels = [f"img{i}" for i in range(5)]
    cell_max = 50
    sheet = contact_sheet(images, labels, cols=3, cell_max=cell_max, pad=4)

    rows = 2  # ceil(5 / 3)
    cols = 3
    cell_h = cell_max + CONTACT_SHEET_LABEL_BAND_PX
    expected_w = 4 + cols * (cell_max + 4)
    expected_h = 4 + rows * (cell_h + 4)

    assert sheet.dtype == np.uint8
    assert sheet.shape == (expected_h, expected_w, 3)


def test_contact_sheet_labels_draw_nonbackground_pixels() -> None:
    images = [_solid_image(40, 40, (10, 10, 10)) for _ in range(2)]
    labels = ["alpha", "beta"]
    cell_max = 40
    pad = CONTACT_SHEET_PAD
    sheet = contact_sheet(images, labels, cols=2, cell_max=cell_max, pad=pad, bg=CONTACT_SHEET_BG)

    # Label band for cell 0 sits just below its thumbnail.
    x0 = pad
    y0 = pad
    band_top = y0 + cell_max
    band_bottom = band_top + CONTACT_SHEET_LABEL_BAND_PX
    band = sheet[band_top:band_bottom, x0 : x0 + cell_max]

    bg = np.array(CONTACT_SHEET_BG, dtype=np.uint8)
    non_bg = np.any(band != bg, axis=-1)
    assert non_bg.sum() > 0, "expected label text to draw at least one non-background pixel"


def test_contact_sheet_handles_float_and_grayscale_images() -> None:
    float_img = np.random.default_rng(0).uniform(0.0, 1.0, size=(20, 20, 3)).astype(np.float32)
    gray_img = np.random.default_rng(1).integers(0, 256, size=(20, 20), dtype=np.uint8)
    sheet = contact_sheet([float_img, gray_img], ["float", "gray"], cols=2, cell_max=20)
    assert sheet.dtype == np.uint8
    assert sheet.ndim == 3 and sheet.shape[2] == 3


def test_side_by_side_is_single_row() -> None:
    images = [_solid_image(30, 30, (1, 2, 3)) for _ in range(4)]
    labels = [str(i) for i in range(4)]
    sheet = side_by_side(images, labels, cell_max=30, pad=5)
    cell_h = 30 + CONTACT_SHEET_LABEL_BAND_PX
    expected_h = 5 + 1 * (cell_h + 5)
    assert sheet.shape[0] == expected_h


def test_colorize_shape_and_dtype() -> None:
    depth = np.linspace(0.0, 10.0, num=100).reshape(10, 10)
    rgb = colorize(depth, vmin=0.0, vmax=10.0)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8


def test_colorize_monotonic_in_luminance() -> None:
    values = np.linspace(0.0, 1.0, num=256).reshape(1, -1)
    rgb = colorize(values, vmin=0.0, vmax=1.0).astype(np.float64)[0]
    luma = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    diffs = np.diff(luma)
    # colorize() is exactly monotonic in the underlying float ramp (each of
    # the 5 control stops has strictly higher luma than the last, and
    # interpolation between them is linear); uint8 rounding can introduce
    # at most ~1 LSB of jitter per channel, so allow a small negative
    # tolerance rather than requiring bit-exact non-decreasing output.
    quantization_tolerance = -2.0
    assert np.all(diffs >= quantization_tolerance), "luminance must be non-decreasing as the input increases"
    # But the overall trend across the full range must be strongly increasing.
    assert luma[-1] - luma[0] > 100.0


def test_colorize_clamps_outside_range() -> None:
    depth = np.array([[-100.0, 0.0, 1.0, 100.0]])
    rgb = colorize(depth, vmin=0.0, vmax=1.0)
    np.testing.assert_array_equal(rgb[0, 0], rgb[0, 1])  # below range clamps to vmin's colour
    np.testing.assert_array_equal(rgb[0, 3], rgb[0, 2])  # above range clamps to vmax's colour


def test_save_png_round_trip(tmp_path: Path) -> None:
    arr = _solid_image(16, 24, (12, 34, 56))
    path = tmp_path / "out.png"
    save_png(path, arr)
    assert path.exists()
    loaded = np.array(Image.open(path).convert("RGB"))
    np.testing.assert_array_equal(loaded, arr)
