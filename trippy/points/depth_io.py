"""I/O glue between trippy and Splats' DepthPro batch tool (depth_batch.py).

Module: trippy.points.depth_io
Invariants: this module never runs DepthPro itself (that only ever happens
    inside a GPU-queue job, see scripts/gpu_submit.sh); it only prepares the
    inputs DepthPro needs (a manifest + undistorted RGB frames) and parses
    the outputs it wrote. All array I/O here is numpy; torch is used only
    for the one-time undistortion step (CPU-only `grid_sample`, mirroring
    trippy.scene.dataset), never for anything MPS.

Undistortion cache duplication note: `trippy.scene.dataset.SceneDataset`
already builds an identical on-disk undistortion cache, but its
constructor only accepts a `limit`-first-N-sorted-images subset, not an
arbitrary curated name list. This experiment needs 6 shade frames plus 6
frames spread across the whole 219-image sequence, which would force
`SceneDataset(limit=...)` to undistort most of the scene just to reach the
later frames -- exactly the "never process all 219 images" case the task
brief forbids. `undistort_and_cache` below reimplements that one
per-image step (using the same public helpers:
`trippy.scene.colmap_io` + `trippy.geom.camera.undistort_maps`) for
exactly the requested names, writing the *same* cache directory layout
(`<cache_root>/<scene_root.name>/w<width>/<name>.npy` + `meta.json`) so a
later full `SceneDataset` build over the same `cache_root`/width would see
these images as already cached and skip them.
`tests/test_points_monodepth.py::test_undistort_and_cache_matches_scene_dataset`
cross-checks the two implementations produce byte-identical output on a
synthetic scene.

Resolution/EXIF note: DepthPro is run on the undistorted image written
here, not the original distorted capture -- this way DepthPro's pixel grid
and `SceneDataset`'s pinhole intrinsics K describe the exact same pinhole
camera, so no separate undistort-the-pixel-coordinates step is needed at
unprojection time (see trippy.points.monodepth). Like
`trippy.scene.dataset` and Splats' own `depth_batch.py`, EXIF orientation
is deliberately left untouched (never `ImageOps.exif_transpose`'d): both
producers already treat the raw on-disk pixel orientation as correct for
this project's photo folders, so leaving it alone keeps depth and
intrinsics in agreement. `depth_batch.py` itself may still downscale if an
image's long side exceeds its own WORK_MAX_SIDE=1600px cap -- an image
whose undistorted long side is under that cap (true for width<=1008 4:3/
3:4 photos) comes back unchanged; `read_depth_output` asserts the returned
depth map's shape matches the meta it shipped with, and
`trippy.points.monodepth` additionally checks that shape against the
undistortion cache's own (width, height) so a silent resolution mismatch
is never possible.
Related docs: docs/SPEC.md D4 point source 2 (monocular depth);
    trippy.points.monodepth (consumes this module's outputs).
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from trippy.constants import (
    DEPTHPRO_DEPTH_SUFFIX,
    DEPTHPRO_MASK_SUFFIX,
    DEPTHPRO_META_SUFFIX,
    DEPTHPRO_SCRIPT_REL,
    DEPTHPRO_VENV_PYTHON_REL,
    SCENE_CACHE_META_FILENAME,
)
from trippy.geom import camera as camera_geom
from trippy.scene import colmap_io
from trippy.scene.dataset import _dst_size, resolve_sparse_dir

_DEPTHPRO_INPUT_DIRNAME = "depthpro_input"


def cache_dir_for(cache_root: str | Path, scene_root: str | Path, width: int) -> Path:
    """Undistortion cache directory for `scene_root` at `width`.

    Matches `trippy.scene.dataset.SceneDataset.cache_dir`'s convention
    exactly (`<cache_root>/<scene_root.name>/w<width>/`), so this cache is
    interchangeable with one `SceneDataset` would build.
    """
    return Path(cache_root) / Path(scene_root).name / f"w{int(width)}"


def undistort_and_cache(
    scene_root: str | Path,
    image_names: list[str],
    width: int,
    cache_root: str | Path,
) -> Path:
    """Undistort exactly `image_names` (not the whole scene) at `width`.

    Idempotent: an image already present in the cache with matching
    intrinsics is skipped (same staleness check as `SceneDataset`: a
    mismatch raises rather than silently serving stale pixels for the
    cached width).

    Args:
        scene_root: scene directory with `images/` and `sparse/0` or
            `sparse_txt` (see `trippy.scene.dataset.resolve_sparse_dir`).
        image_names: exact registered image filenames to undistort.
        width: destination pinhole image width in pixels (per-camera
            height is derived to preserve that camera's aspect ratio).
        cache_root: root directory for the on-disk cache (see
            `cache_dir_for`).

    Returns:
        The cache directory written to (`cache_dir_for(...)`).
    """
    scene_root = Path(scene_root)
    sparse_dir = resolve_sparse_dir(scene_root)
    scene = colmap_io.load_colmap_model(sparse_dir)
    images_by_name = scene.images_by_name()

    cache_dir = cache_dir_for(cache_root, scene_root, width)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / SCENE_CACHE_META_FILENAME
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("images", {})
    wrote_anything = False

    for name in image_names:
        im = images_by_name.get(name)
        if im is None:
            raise KeyError(f"{name!r} is not a registered image under {sparse_dir}")
        cam = scene.cameras[im.camera_id]
        fx, fy, cx, cy = colmap_io.intrinsics(cam)
        scale = width / cam.width
        width_dst, height_dst = _dst_size(cam.width, cam.height, width)
        k_dst = [
            [fx * scale, 0.0, cx * scale],
            [0.0, fy * scale, cy * scale],
            [0.0, 0.0, 1.0],
        ]

        npy_path = cache_dir / f"{name}.npy"
        cached = meta["images"].get(name)
        if cached is not None and npy_path.exists():
            cached_k = np.array(cached["K"], dtype=np.float64)
            fresh_k = np.array(k_dst, dtype=np.float64)
            if not np.allclose(cached_k, fresh_k, atol=1e-4):
                raise AssertionError(
                    f"cache is stale: {meta_path} intrinsics for {name!r} "
                    f"({cached_k.tolist()}) do not match the live COLMAP model's "
                    f"({fresh_k.tolist()}) at width={width}"
                )
            continue

        rgb = _undistort_one(scene_root, name, cam, fx, fy, cx, cy, scale, width_dst, height_dst)
        np.save(npy_path, rgb)
        meta["images"][name] = {
            "camera_id": im.camera_id,
            "orig_width": cam.width,
            "orig_height": cam.height,
            "width": width_dst,
            "height": height_dst,
            "K": k_dst,
            "qvec": im.qvec.tolist(),
            "tvec": im.tvec.tolist(),
        }
        wrote_anything = True

    if wrote_anything or not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2))
    return cache_dir


def _undistort_one(
    scene_root: Path,
    name: str,
    cam: colmap_io.Camera,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scale: float,
    width_dst: int,
    height_dst: int,
) -> np.ndarray:
    """Undistort+resize one source image (same math as `SceneDataset._undistort_image`)."""
    image_path = scene_root / "images" / name
    with PILImage.open(image_path) as pil_img:
        src = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    fx_dst, fy_dst, cx_dst, cy_dst = fx * scale, fy * scale, cx * scale, cy * scale
    k1, k2, p1, p2 = colmap_io.distortion(cam)
    dist = camera_geom.OpenCVDistortion(k1=k1, k2=k2, p1=p1, p2=p2) if any((k1, k2, p1, p2)) else None

    grid_np = camera_geom.undistort_maps(
        fx_src=fx,
        fy_src=fy,
        cx_src=cx,
        cy_src=cy,
        width_src=cam.width,
        height_src=cam.height,
        fx_dst=fx_dst,
        fy_dst=fy_dst,
        cx_dst=cx_dst,
        cy_dst=cy_dst,
        width_dst=width_dst,
        height_dst=height_dst,
        distortion=dist,
    )

    src_t = torch.from_numpy(src).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    grid_t = torch.from_numpy(grid_np).unsqueeze(0)
    out = F.grid_sample(src_t, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False)
    out_rgb = out.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
    return out_rgb.numpy()


def load_cache_meta(cache_dir: str | Path) -> dict:
    """Load the `meta.json` sidecar written by `undistort_and_cache` (or `SceneDataset`)."""
    return json.loads((Path(cache_dir) / SCENE_CACHE_META_FILENAME).read_text())


def load_cached_rgb(cache_dir: str | Path, name: str) -> np.ndarray:
    """Load one image's cached undistorted RGB: (H, W, 3) uint8."""
    return np.load(Path(cache_dir) / f"{name}.npy")


def write_depthpro_manifest(
    cache_dir: str | Path,
    image_names: list[str],
    depth_dir: str | Path,
    manifest_path: str | Path,
) -> list[dict]:
    """Write a depth_batch.py-format manifest for `image_names`' cached undistorted frames.

    For each name, writes (or reuses) a PNG copy of the cached undistorted
    RGB under `cache_dir/depthpro_input/<stem>.png` -- depth_batch.py reads
    an image path, not our `.npy` cache -- and adds one manifest record
    `{"id": stem, "image": <png path>, "mask": None, "out_dir": depth_dir}`,
    matching ~/Splats/tools/ldi/depth_batch.py's documented manifest format
    exactly (id/image/mask/out_dir, all absolute paths).

    Args:
        cache_dir: undistortion cache directory (see `cache_dir_for`);
            must already contain `<name>.npy` for every name in
            `image_names` (call `undistort_and_cache` first).
        image_names: registered image filenames to include.
        depth_dir: directory depth_batch.py should write its outputs into.
        manifest_path: where to write the manifest JSON.

    Returns:
        The list of manifest record dicts written (also written to disk).
    """
    cache_dir = Path(cache_dir)
    depth_dir = Path(depth_dir)
    input_dir = cache_dir / _DEPTHPRO_INPUT_DIRNAME
    input_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name in image_names:
        stem = Path(name).stem
        png_path = input_dir / f"{stem}.png"
        if not png_path.exists():
            rgb = load_cached_rgb(cache_dir, name)
            PILImage.fromarray(rgb, mode="RGB").save(png_path)
        records.append(
            {
                "id": stem,
                "image": str(png_path.resolve()),
                "mask": None,
                "out_dir": str(depth_dir.resolve()),
            }
        )

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2))
    return records


def resolve_depthpro_paths(splats_root: str | Path) -> tuple[Path, Path]:
    """(python executable, depth_batch.py path) for the DepthPro GPU job, under `splats_root`."""
    splats_root = Path(splats_root)
    return splats_root / DEPTHPRO_VENV_PYTHON_REL, splats_root / DEPTHPRO_SCRIPT_REL


def format_depthpro_command(python_path: str | Path, script_path: str | Path, manifest_path: str | Path) -> str:
    """Shell-quoted `<python> <depth_batch.py> <manifest.json>` command string."""
    return " ".join(shlex.quote(str(p)) for p in (python_path, script_path, manifest_path))


@dataclass
class DepthOutput:
    """One image's parsed DepthPro output.

    Attributes:
        depth: (H, W) float32, metric depth in metres at working resolution.
        mask: (H, W) bool, True = trust this depth value.
        meta: {"width", "height", "orig_width", "orig_height",
            "focal_length_px", "has_person_mask"} as written by
            depth_batch.py's `<id>_meta.json`.
    """

    depth: np.ndarray
    mask: np.ndarray
    meta: dict


def read_depth_output(depth_dir: str | Path, image_id: str) -> DepthOutput:
    """Parse one depth_batch.py output triple: `<id>_depth.npy/_mask.npy/_meta.json`.

    Raises:
        FileNotFoundError: any of the three expected files is missing
            (message names the missing file(s), used by
            `depth_output_missing` to build the CLI's "go run this job
            first" check).
        ValueError: the depth array's shape does not match its own
            `meta.json` (width, height) -- a corrupt or partially-written
            output, never silently trusted.
    """
    depth_dir = Path(depth_dir)
    depth_path = depth_dir / f"{image_id}{DEPTHPRO_DEPTH_SUFFIX}"
    mask_path = depth_dir / f"{image_id}{DEPTHPRO_MASK_SUFFIX}"
    meta_path = depth_dir / f"{image_id}{DEPTHPRO_META_SUFFIX}"
    for p in (depth_path, mask_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"missing DepthPro output file: {p}")

    depth = np.load(depth_path).astype(np.float32)
    mask = np.load(mask_path).astype(bool)
    meta = json.loads(meta_path.read_text())

    expected_shape = (int(meta["height"]), int(meta["width"]))
    if depth.shape != expected_shape:
        raise ValueError(
            f"{depth_path}: depth shape {depth.shape} != its own meta.json (height, width) "
            f"{expected_shape} -- corrupt or partially written DepthPro output"
        )
    if mask.shape != depth.shape:
        raise ValueError(f"{mask_path}: mask shape {mask.shape} != depth shape {depth.shape}")

    return DepthOutput(depth=depth, mask=mask, meta=meta)


def depth_output_missing(depth_dir: str | Path, image_ids: list[str]) -> list[str]:
    """Image ids among `image_ids` missing any of their three DepthPro output files."""
    depth_dir = Path(depth_dir)
    missing = []
    for image_id in image_ids:
        paths = (
            depth_dir / f"{image_id}{DEPTHPRO_DEPTH_SUFFIX}",
            depth_dir / f"{image_id}{DEPTHPRO_MASK_SUFFIX}",
            depth_dir / f"{image_id}{DEPTHPRO_META_SUFFIX}",
        )
        if not all(p.exists() for p in paths):
            missing.append(image_id)
    return missing
