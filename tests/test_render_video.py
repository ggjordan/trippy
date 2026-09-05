"""Tests for trippy.render.video: write_video (ffmpeg pipe) and frames_from_dir.

Module: tests.test_render_video
Invariants under test: write_video pipes raw RGB24 frames to ffmpeg and
    produces a playable MP4 with the right frame count (checked via
    ffprobe when present); odd-dimension frames are silently edge-padded
    to even rather than rejected; write_video raises a clear RuntimeError
    (not a cryptic subprocess failure) when ffmpeg is missing.
Fixture: synthetic solid-colour frames only (no photos/scenes).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from trippy.render.video import frames_from_dir, write_video

N_FRAMES = 10
FPS = 10


def _synthetic_frames(n: int = N_FRAMES, h: int = 32, w: int = 48):
    for i in range(n):
        color = (i * 20 % 256, 30, 200)
        yield np.tile(np.array(color, dtype=np.uint8), (h, w, 1))


def test_write_video_missing_ffmpeg_raises_clear_error(tmp_path: Path) -> None:
    with (
        mock.patch("trippy.render.video.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        write_video(tmp_path / "out.mp4", _synthetic_frames(n=1))


def test_write_video_empty_frames_raises(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available on this machine")
    with pytest.raises(ValueError, match="at least one frame"):
        write_video(tmp_path / "out.mp4", iter([]))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this machine")
def test_write_video_produces_file_with_expected_frame_count(tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    result = write_video(out, _synthetic_frames(), fps=FPS)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available to verify frame count")

    probe_cmd = [
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json",
        str(out),
    ]  # fmt: skip
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30, check=True)
    info = json.loads(probe.stdout)
    nb_frames = int(info["streams"][0]["nb_read_frames"])
    assert nb_frames == N_FRAMES


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this machine")
def test_write_video_pads_odd_dimensions(tmp_path: Path) -> None:
    odd_frames = [np.zeros((33, 47, 3), dtype=np.uint8) for _ in range(3)]
    out = tmp_path / "odd.mp4"
    write_video(out, odd_frames, fps=FPS, use_videotoolbox=False)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this machine")
def test_write_video_rejects_bad_frame_dtype(tmp_path: Path) -> None:
    bad_frames = [np.zeros((10, 10, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="uint8"):
        write_video(tmp_path / "bad.mp4", bad_frames)


def test_frames_from_dir_reads_sorted_pngs(tmp_path: Path) -> None:
    from PIL import Image

    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        img = Image.new("RGB", (8, 6), color=color)
        img.save(tmp_path / f"frame_{i:03d}.png")

    frames = list(frames_from_dir(tmp_path, "*.png"))
    assert len(frames) == 3
    for frame in frames:
        assert frame.dtype == np.uint8
        assert frame.shape == (6, 8, 3)
    np.testing.assert_array_equal(frames[0][0, 0], [255, 0, 0])
    np.testing.assert_array_equal(frames[2][0, 0], [0, 0, 255])


def test_frames_from_dir_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files matching"):
        list(frames_from_dir(tmp_path, "*.png"))
