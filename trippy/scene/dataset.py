"""One-time COLMAP image undistortion + multi-resolution disk cache.

Module: trippy.scene.dataset
Invariants: undistortion happens exactly once per (scene, width, image) --
    the second SceneDataset construction over the same cache_root/width
    must find every image already cached and skip recompute (this is what
    makes repeated training runs at 1008/2016 wide affordable, per
    docs/SPEC.md v0.1.0). Cache validity is checked by recomputing each
    image's scaled intrinsics from the live COLMAP model and asserting
    they match what is stored in meta.json; a mismatch means the cache is
    stale (e.g. scene re-run through COLMAP) and raises loudly rather than
    silently serving wrong intrinsics for cached pixels.
Coordinate frame / pixel convention: docs/GEOMETRY.md "Image coordinates"
    (pixel (row i, col j) spans [j, j+1) x [i, i+1), centre (j+0.5, i+0.5)).
    `crop()` below guards against historical bug class 3 (docs/GEOMETRY.md
    "Padded pixels unprojected as scene"): pixels where a crop window
    overshoots the source image are never given real image content --
    rgb is exactly 0 and mask is exactly 0 there, so a caller cannot
    accidentally learn from (or backprop into) padding.
Related docs: docs/ARCHITECTURE.md "Module overview" (trippy/scene);
    docs/SPEC.md v0.1.0 milestone ("dataset (undistort+cache 1008/2016
    wide)"); trippy.geom.camera.undistort_maps (grid_sample coordinate
    convention); trippy.scene.colmap_io (ColmapScene, intrinsics,
    distortion).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from trippy.constants import (
    EXIF_TAG_EXIF_IFD_POINTER,
    EXIF_TAG_EXPOSURE_TIME,
    EXIF_TAG_ISO,
    SCENE_CACHE_META_FILENAME,
)
from trippy.geom import camera as camera_geom
from trippy.scene import colmap_io


def resolve_sparse_dir(scene_root: str | Path) -> Path:
    """Pick a scene's sparse model directory: `sparse/0` (binary) over `sparse_txt`.

    Args:
        scene_root: scene directory containing `images/` and either
            `sparse/0/` (COLMAP binary export) and/or `sparse_txt/`
            (COLMAP text export).

    Returns:
        The directory to hand to `trippy.scene.colmap_io.load_colmap_model`.

    Raises:
        FileNotFoundError: neither `sparse/0` nor `sparse_txt` exists.
    """
    scene_root = Path(scene_root)
    bin_dir = scene_root / "sparse" / "0"
    if bin_dir.exists():
        return bin_dir
    txt_dir = scene_root / "sparse_txt"
    if txt_dir.exists():
        return txt_dir
    raise FileNotFoundError(f"no sparse/0 or sparse_txt directory under {scene_root}")


def _dst_size(width_src: int, height_src: int, width_dst: int) -> tuple[int, int]:
    """Destination (width, height) in pixels, keeping the source aspect ratio."""
    scale = width_dst / width_src
    height_dst = max(1, round(height_src * scale))
    return width_dst, height_dst


def _read_exif(image_path: Path) -> dict[str, float | int | None]:
    """Best-effort EXIF exposure time (seconds) and ISO from `image_path`.

    Used to seed the tone-mapper's per-image exposure init later (see
    docs/ARCHITECTURE.md "Tone mapper"). Returns {None, None} for images
    with no EXIF, an unreadable file, or a non-JPEG/TIFF format -- missing
    exposure/ISO is expected and fine, never raises.
    """
    exposure_time: float | None = None
    iso: int | None = None
    try:
        with PILImage.open(image_path) as img:
            exif = img.getexif()
            if exif:
                try:
                    exif_ifd = exif.get_ifd(EXIF_TAG_EXIF_IFD_POINTER)  # holds ExposureTime/ISO.
                except (KeyError, ValueError, AttributeError):
                    exif_ifd = {}
                raw_exposure = exif_ifd.get(EXIF_TAG_EXPOSURE_TIME, exif.get(EXIF_TAG_EXPOSURE_TIME))
                raw_iso = exif_ifd.get(EXIF_TAG_ISO, exif.get(EXIF_TAG_ISO))
                if raw_exposure is not None:
                    exposure_time = float(raw_exposure)
                if raw_iso is not None:
                    iso = int(raw_iso)
    except (OSError, ValueError):
        pass
    return {"exposure_time": exposure_time, "iso": iso}


class SceneDataset(torch.utils.data.Dataset):
    """A COLMAP scene's images, undistorted once and cached at a fixed width.

    Attributes:
        names: sorted image names in this dataset (deterministic order).
    """

    def __init__(
        self,
        scene_root: str | Path,
        width: int,
        cache_root: str | Path,
        device: str | torch.device = "cpu",
        limit: int | None = None,
    ) -> None:
        """Build (or load) the undistortion cache for a COLMAP scene.

        Args:
            scene_root: directory with `images/` and `sparse/0` and/or
                `sparse_txt` (see `resolve_sparse_dir`).
            width: destination pinhole image width in pixels; height is
                derived per-camera to keep that camera's aspect ratio.
            cache_root: root directory for the disk cache; images are
                written under `cache_root/<scene_root.name>/w<width>/`.
            device: torch device (or device string) items are returned on.
                Cache building itself always runs on CPU.
            limit: if given, only the first `limit` images (by sorted
                name) are loaded/cached -- use this to avoid processing an
                entire real scene in tests (never process all of a real
                scene's images in a test).
        """
        self.scene_root = Path(scene_root)
        self.width = int(width)
        self.cache_root = Path(cache_root)
        self.device = torch.device(device)

        sparse_dir = resolve_sparse_dir(self.scene_root)
        self._scene = colmap_io.load_colmap_model(sparse_dir)
        self._images_by_name = self._scene.images_by_name()

        names = sorted(self._images_by_name.keys())
        if limit is not None:
            names = names[:limit]
        self._names = names

        self.cache_dir = self.cache_root / self.scene_root.name / f"w{self.width}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.cache_dir / SCENE_CACHE_META_FILENAME

        self._meta = self._load_or_build_cache()

    @property
    def names(self) -> list[str]:
        """Sorted image names in this dataset (see `__init__`'s `limit`)."""
        return list(self._names)

    def _load_or_build_cache(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        if self._meta_path.exists():
            meta = json.loads(self._meta_path.read_text())
        meta.setdefault("images", {})
        wrote_anything = False

        for name in self._names:
            im = self._images_by_name[name]
            cam = self._scene.cameras[im.camera_id]
            fx, fy, cx, cy = colmap_io.intrinsics(cam)
            scale = self.width / cam.width
            width_dst, height_dst = _dst_size(cam.width, cam.height, self.width)
            k_dst = [
                [fx * scale, 0.0, cx * scale],
                [0.0, fy * scale, cy * scale],
                [0.0, 0.0, 1.0],
            ]

            npy_path = self.cache_dir / f"{name}.npy"
            cached = meta["images"].get(name)
            if cached is not None and npy_path.exists():
                cached_k = np.array(cached["K"], dtype=np.float64)
                fresh_k = np.array(k_dst, dtype=np.float64)
                if not np.allclose(cached_k, fresh_k, atol=1e-4):
                    raise AssertionError(
                        f"cache is stale: {self._meta_path} intrinsics for {name!r} "
                        f"({cached_k.tolist()}) do not match the live COLMAP model's "
                        f"({fresh_k.tolist()}) at width={self.width}"
                    )
                continue

            rgb = self._undistort_image(name, cam, fx, fy, cx, cy, scale, width_dst, height_dst)
            np.save(npy_path, rgb)
            exif = _read_exif(self.scene_root / "images" / name)
            meta["images"][name] = {
                "camera_id": im.camera_id,
                "orig_width": cam.width,
                "orig_height": cam.height,
                "width": width_dst,
                "height": height_dst,
                "K": k_dst,
                "qvec": im.qvec.tolist(),
                "tvec": im.tvec.tolist(),
                "exposure_time": exif["exposure_time"],
                "iso": exif["iso"],
            }
            wrote_anything = True

        if wrote_anything or not self._meta_path.exists():
            self._meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    def _undistort_image(
        self,
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
        """Undistort+resize one source image to (height_dst, width_dst, 3) uint8 RGB.

        Runs once per (image, width); the caller is responsible for caching
        the result. Pinhole source cameras (all-zero distortion) skip the
        distortion step -- the sampling grid degenerates to a plain resize.
        """
        image_path = self.scene_root / "images" / name
        with PILImage.open(image_path) as pil_img:
            src = np.array(pil_img.convert("RGB"), dtype=np.uint8)  # (H_src, W_src, 3), owns its buffer

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

        src_t = torch.from_numpy(src).to(torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        grid_t = torch.from_numpy(grid_np).unsqueeze(0)  # (1, height_dst, width_dst, 2)
        out = F.grid_sample(src_t, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False)
        out_rgb = out.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        return out_rgb.numpy()

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one cached, undistorted image.

        Returns:
            dict with:
                "rgb": (H, W, 3) uint8 tensor, undistorted RGB pixels.
                "K": (3, 3) float32 tensor, pinhole intrinsics at this
                    dataset's `width` (see docs/GEOMETRY.md "Camera
                    intrinsics").
                "qvec": (4,) float32 tensor, (qw, qx, qy, qz), world->camera
                    rotation (see trippy.geom.xform_a.qvec2R).
                "tvec": (3,) float32 tensor, world->camera translation.
                "name": str, image filename.
                "index": int, this dataset's index for `name`.
        """
        name = self._names[index]
        meta = self._meta["images"][name]
        rgb_np = np.load(self.cache_dir / f"{name}.npy")
        return {
            "rgb": torch.from_numpy(rgb_np).to(self.device),
            "K": torch.tensor(meta["K"], dtype=torch.float32, device=self.device),
            "qvec": torch.tensor(meta["qvec"], dtype=torch.float32, device=self.device),
            "tvec": torch.tensor(meta["tvec"], dtype=torch.float32, device=self.device),
            "name": name,
            "index": index,
        }


def crop(
    item: dict[str, Any],
    size: int,
    zoom: float = 1.0,
    center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Crop a dataset item to a `size` x `size` window, never faking padding as content.

    Historical bug class 3 (docs/GEOMETRY.md): a crop window overshooting
    the source image must not be filled with real-looking pixels that get
    treated as scene content downstream. Overshoot pixels get rgb == 0 and
    mask == 0, exactly.

    Args:
        item: a dataset item (as returned by `SceneDataset.__getitem__`):
            "rgb" (H, W, 3) uint8 tensor, and optionally "K" (3, 3) float32
            tensor.
        size: output crop side length in pixels (square).
        zoom: > 1 zooms in (samples a smaller `size/zoom` source window,
            nearest-neighbour resampled up to `size`); 1.0 = no zoom.
        center: (x, y) crop-window centre in source pixel coordinates
            (continuous, pixel-centre convention -- see docs/GEOMETRY.md);
            defaults to the source image centre.

    Returns:
        dict with:
            "rgb": (size, size, 3) uint8 tensor, same dtype as `item["rgb"]`,
                0 wherever the crop overshoots the source image.
            "mask": (size, size) float32 tensor, 1.0 = real image content,
                0.0 = crop overshoot (padding).
            "K": (3, 3) float32 tensor, intrinsics adjusted for the crop
                offset and zoom -- present only if `item` has a "K".
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")

    rgb = item["rgb"]
    height_src, width_src = rgb.shape[0], rgb.shape[1]
    if center is None:
        center = (width_src / 2.0, height_src / 2.0)
    center_x, center_y = center

    window = size / zoom
    half = window / 2.0
    x0 = center_x - half
    y0 = center_y - half

    # Index math in float64 on the CPU (MPS has no float64); only the int64 gather indices move to
    # the image's device.
    device = rgb.device
    out_idx = torch.arange(size, dtype=torch.float64)
    src_x = x0 + (out_idx + 0.5) / size * window  # (size,) continuous source pixel coords
    src_y = y0 + (out_idx + 0.5) / size * window

    src_col = torch.floor(src_x).to(torch.int64).to(device)  # nearest source pixel (spans [i, i+1))
    src_row = torch.floor(src_y).to(torch.int64).to(device)

    valid_col = (src_col >= 0) & (src_col < width_src)
    valid_row = (src_row >= 0) & (src_row < height_src)
    mask2d = valid_row[:, None] & valid_col[None, :]  # (size, size)

    row_idx = src_row.clamp(0, height_src - 1)
    col_idx = src_col.clamp(0, width_src - 1)
    out_rgb = rgb[row_idx][:, col_idx]  # (size, size, 3), gathered before masking

    mask_same_dtype = mask2d.to(out_rgb.dtype)
    out_rgb = out_rgb * mask_same_dtype.unsqueeze(-1)

    result: dict[str, Any] = {"rgb": out_rgb, "mask": mask2d.to(torch.float32)}

    K = item.get("K")
    if K is not None:
        new_k = K.clone()
        new_k[0, 0] = K[0, 0] * zoom
        new_k[1, 1] = K[1, 1] * zoom
        new_k[0, 2] = (K[0, 2] - x0) * zoom
        new_k[1, 2] = (K[1, 2] - y0) * zoom
        result["K"] = new_k

    return result
