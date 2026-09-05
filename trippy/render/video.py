"""ffmpeg-backed video writer: pipe raw RGB frames to an MP4.

Module: trippy.render.video
Invariants: no video-encoding dependency is added to pyproject.toml --
    frames are piped as raw bytes over stdin to the system `ffmpeg`
    binary (AGENTS.md: no new dependencies). If ffmpeg is not on PATH,
    write_video() raises a clear RuntimeError rather than silently
    failing or falling back to a slower pure-Python encoder. Every frame
    must be uint8 (H, W, 3); odd dimensions are edge-padded to even
    (required by yuv420p / most H.264 decoders) before encoding.
Related docs: docs/EXPERIMENTS.md "Dolly camera paths" (MP4 output,
    typically 2-5s at 24 fps); Splats' PROJECT.md note that
    h264_videotoolbox (Apple Silicon hardware encode) is a free win on
    this machine -- used when available and requested, libx264 otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from trippy.constants import (
    FFMPEG_LIBX264_ENCODER,
    FFMPEG_OUTPUT_PIX_FMT,
    FFMPEG_RAW_PIX_FMT,
    FFMPEG_VIDEOTOOLBOX_ENCODER,
    VIDEO_DEFAULT_FPS,
)

# Timeout (seconds) for the one-shot `ffmpeg -encoders` probe used to check
# h264_videotoolbox availability; this is a local subprocess listing static
# build info, not network I/O, so a short timeout is safe.
_ENCODER_PROBE_TIMEOUT_S = 10.0


def _ffmpeg_path() -> str:
    """Resolve the ffmpeg binary on PATH, or raise a clear error."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install it (e.g. `brew install ffmpeg`) to write videos"
        )
    return ffmpeg


def _videotoolbox_available(ffmpeg: str) -> bool:
    """Probe `ffmpeg -encoders` once for h264_videotoolbox support."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=_ENCODER_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return FFMPEG_VIDEOTOOLBOX_ENCODER in result.stdout


def _pad_even(frame: np.ndarray) -> np.ndarray:
    """Edge-pad a (H, W, 3) uint8 frame so both H and W are even."""
    h, w = frame.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _check_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 (H, W, 3), got shape {frame.shape} dtype {frame.dtype}")
    return frame


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: int = VIDEO_DEFAULT_FPS,
    use_videotoolbox: bool = True,
) -> Path:
    """Encode a sequence of RGB frames to an H.264 MP4 via ffmpeg.

    Args:
        path: output .mp4 path; parent directories are created if missing.
        frames: iterable of uint8 (H, W, 3) RGB frames, all the same
            shape (after even-dimension padding is applied here).
        fps: output frame rate.
        use_videotoolbox: prefer the h264_videotoolbox hardware encoder
            when ffmpeg reports it available; falls back to libx264
            otherwise (or always, if this is False).

    Returns:
        The output path.

    Raises:
        RuntimeError: ffmpeg is not on PATH, or ffmpeg exits non-zero.
        ValueError: frames is empty, a frame isn't uint8 (H, W, 3), or
            frames have inconsistent shapes.
    """
    path = Path(path)
    ffmpeg = _ffmpeg_path()

    frame_iter: Iterator[np.ndarray] = iter(frames)
    try:
        first = _check_frame(next(frame_iter))
    except StopIteration:
        raise ValueError("frames must contain at least one frame") from None
    first = _pad_even(first)
    h, w = first.shape[:2]

    encoder = FFMPEG_LIBX264_ENCODER
    if use_videotoolbox and _videotoolbox_available(ffmpeg):
        encoder = FFMPEG_VIDEOTOOLBOX_ENCODER

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", FFMPEG_RAW_PIX_FMT,
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", encoder,
        "-pix_fmt", FFMPEG_OUTPUT_PIX_FMT,
        str(path),
    ]  # fmt: skip

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(first.tobytes())
        for raw_frame in frame_iter:
            frame = _pad_even(_check_frame(raw_frame))
            if frame.shape[:2] != (h, w):
                raise ValueError(f"all frames must share one shape; got {frame.shape[:2]} vs {(h, w)}")
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{stderr.decode(errors='replace')}")
    return path


def frames_from_dir(directory: str | Path, pattern: str = "*.png") -> Iterator[np.ndarray]:
    """Yield RGB frames from image files in `directory`, sorted by filename.

    Args:
        directory: directory to glob.
        pattern: glob pattern (default all PNGs).

    Yields:
        uint8 (H, W, 3) arrays, one per matched file, in sorted filename order.

    Raises:
        ValueError: no files match `pattern` in `directory`.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise ValueError(f"no files matching {pattern!r} in {directory}")
    for p in paths:
        with Image.open(p) as img:
            yield np.array(img.convert("RGB"), dtype=np.uint8)
