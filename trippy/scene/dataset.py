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
    Person masks (`masks_dir`, auto-discovered as `<scene_root>/masks`) ride
    the same cache and the same undistortion grid as the pixels they mask --
    nearest interpolation, because a mask is a decision, not a signal -- and
    are folded into that same validity mask by `crop()`. One mask, two reasons
    to be zero (padding, or a person), so every existing consumer of "mask"
    honours them without changing.
Related docs: docs/ARCHITECTURE.md "Module overview" (trippy/scene);
    docs/SPEC.md v0.1.0 milestone ("dataset (undistort+cache 1008/2016
    wide)"); trippy.geom.camera.undistort_maps (grid_sample coordinate
    convention); trippy.scene.colmap_io (ColmapScene, intrinsics,
    distortion).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    SCENE_MASK_CACHE_SUFFIX,
    SCENE_MASK_KEEP_THRESHOLD,
    SCENE_MASK_SUFFIX,
    SCENE_MASKS_DIRNAME,
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


def default_masks_dir(scene_root: str | Path) -> Path | None:
    """`<scene_root>/masks` when it exists, else None (the scene has no masks).

    Auto-discovery for `SceneDataset`/`TrainConfig`: Splats' own mask writers,
    COLMAP's `--ImageReader.mask_path` and Brush's default masks folder all use
    this one directory name (`trippy.constants.SCENE_MASKS_DIRNAME`), so "does
    this scene have person masks?" is answerable without configuration. Returns
    None rather than raising -- a scene without masks trains unmasked, which is
    a legitimate arm, not an error (see `SCENE_MASK_KEEP_THRESHOLD`).
    """
    candidate = Path(scene_root) / SCENE_MASKS_DIRNAME
    return candidate if candidate.is_dir() else None


def mask_path_for(masks_dir: str | Path, image_name: str) -> Path:
    """The mask file for `image_name`: the image's stem with `SCENE_MASK_SUFFIX`.

    `images/IMG_3683.jpg` -> `masks/IMG_3683.png` (the mask writers always emit
    PNG, whatever the photo's own extension is).
    """
    return Path(masks_dir) / (Path(image_name).stem + SCENE_MASK_SUFFIX)


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
        masks_dir: resolved person-mask directory, or None when this dataset
            is unmasked (see `__init__`'s `masks_dir`/`use_masks`).
    """

    #: Class-level default so a subclass that reimplements `__init__` rather than
    #: calling it -- `trippy.render.pyramid_render._NamedSceneDataset` does, to
    #: select non-contiguous frame names -- is simply an unmasked dataset instead
    #: of an AttributeError. Masking is opt-in machinery; not opting in must never
    #: be a crash.
    masks_dir: Path | None = None

    def __init__(
        self,
        scene_root: str | Path,
        width: int,
        cache_root: str | Path,
        device: str | torch.device = "cpu",
        limit: int | None = None,
        masks_dir: str | Path | None = None,
        use_masks: bool = True,
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
            masks_dir: directory of person masks, one `<stem>.png` per image
                (`mask_path_for`). None (the default) auto-discovers
                `<scene_root>/masks` via `default_masks_dir`; a scene without
                that directory is simply unmasked.
            use_masks: False turns masking off outright, even for a scene that
                has a `masks/` directory (the "unmasked arm" of an experiment).
                Masks are a comparability device, not a rule -- see
                `trippy.constants.SCENE_MASK_KEEP_THRESHOLD`.

        Raises:
            FileNotFoundError: `masks_dir` resolves to a directory but one of
                `names` has no mask file in it. Silently training a subset of
                the frames unmasked would make the run's numbers a mixture of
                two protocols, so a partial mask set fails loudly.
        """
        self.scene_root = Path(scene_root)
        self.width = int(width)
        self.cache_root = Path(cache_root)
        self.device = torch.device(device)
        if not use_masks:
            self.masks_dir: Path | None = None
        elif masks_dir is not None:
            self.masks_dir = Path(masks_dir)
        else:
            self.masks_dir = default_masks_dir(self.scene_root)

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
                # A cache built before masks were configured (or before this feature
                # existed) has valid pixels but no mask sidecar; fill it in rather than
                # rebuilding the whole cache, so turning masks on costs one mask pass.
                if self._ensure_mask_cached(name, cam, fx, fy, cx, cy, scale, width_dst, height_dst, cached):
                    wrote_anything = True
                continue

            rgb = self._undistort_image(name, cam, fx, fy, cx, cy, scale, width_dst, height_dst)
            np.save(npy_path, rgb)
            exif = _read_exif(self.scene_root / "images" / name)
            entry = {
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
            meta["images"][name] = entry
            self._ensure_mask_cached(name, cam, fx, fy, cx, cy, scale, width_dst, height_dst, entry)
            wrote_anything = True

        if wrote_anything or not self._meta_path.exists():
            self._meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    def _sampling_grid(
        self,
        cam: colmap_io.Camera,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float,
        width_dst: int,
        height_dst: int,
    ) -> torch.Tensor:
        """The (1, height_dst, width_dst, 2) `grid_sample` grid undistorting `cam`.

        Shared by the photo and its person mask so both are resampled through
        *exactly* the same geometry -- a mask built from a second, independently
        constructed grid could drift by a fraction of a pixel and start
        excluding the wrong pixels. Pinhole source cameras (all-zero distortion)
        skip the distortion step; the grid degenerates to a plain resize.
        """
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
        return torch.from_numpy(grid_np).unsqueeze(0)  # (1, height_dst, width_dst, 2)

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
        the result.
        """
        image_path = self.scene_root / "images" / name
        with PILImage.open(image_path) as pil_img:
            src = np.array(pil_img.convert("RGB"), dtype=np.uint8)  # (H_src, W_src, 3), owns its buffer

        grid_t = self._sampling_grid(cam, fx, fy, cx, cy, scale, width_dst, height_dst)
        src_t = torch.from_numpy(src).to(torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        out = F.grid_sample(src_t, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False)
        out_rgb = out.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
        return out_rgb.numpy()

    def _undistort_mask(
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
        """Undistort+resize one person mask to (height_dst, width_dst) uint8 {0, 1}.

        Same grid as the photo (`_sampling_grid`) but **nearest** interpolation:
        a mask is a decision, not a signal, and bilinear resampling would
        produce fractional edge values that are neither "person" nor "keep".
        Grid points outside the source image sample 0 (`padding_mode="zeros"`),
        i.e. excluded -- consistent with `crop`'s padding rule.

        Returns 1 where the source mask is >= `SCENE_MASK_KEEP_THRESHOLD`
        (WHITE = keep) and 0 where it is below (BLACK = person, ignore); see
        `trippy.constants.SCENE_MASK_KEEP_THRESHOLD` for the polarity's
        provenance.
        """
        assert self.masks_dir is not None  # only called when masking is on
        path = mask_path_for(self.masks_dir, name)
        if not path.exists():
            raise FileNotFoundError(
                f"masks_dir {self.masks_dir} has no mask for image {name!r} (expected {path}); "
                "training some frames masked and others unmasked would mix two protocols -- "
                "generate the missing masks or set use_masks: false"
            )
        with PILImage.open(path) as pil_mask:
            src = np.array(pil_mask.convert("L"), dtype=np.uint8)  # (H_src, W_src)

        keep = (src >= SCENE_MASK_KEEP_THRESHOLD).astype(np.float32)
        grid_t = self._sampling_grid(cam, fx, fy, cx, cy, scale, width_dst, height_dst)
        src_t = torch.from_numpy(keep).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        out = F.grid_sample(src_t, grid_t, mode="nearest", padding_mode="zeros", align_corners=False)
        return out.squeeze(0).squeeze(0).to(torch.uint8).numpy()

    def _ensure_mask_cached(
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
        entry: dict[str, Any],
    ) -> bool:
        """Cache `name`'s undistorted mask if masking is on and it is not on disk yet.

        Records `mask` (bool) and `mask_keep_frac` (the fraction of undistorted
        pixels the loss may use, 1.0 = nobody masked out) in `entry` so a run can
        report per-image mask coverage without re-reading any pixels.

        Returns:
            True when `entry` was modified (the caller must rewrite meta.json).
        """
        if self.masks_dir is None:
            return False
        mask_path = self.cache_dir / f"{name}{SCENE_MASK_CACHE_SUFFIX}"
        if mask_path.exists() and entry.get("mask_keep_frac") is not None:
            return False
        mask = self._undistort_mask(name, cam, fx, fy, cx, cy, scale, width_dst, height_dst)
        np.save(mask_path, mask)
        entry["mask"] = True
        entry["mask_keep_frac"] = float(mask.mean())
        return True

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
                "mask": (H, W) float32 tensor, 1.0 = a pixel the loss may
                    use, 0.0 = a person pixel to exclude. **Present only
                    when this dataset is masked** (see `__init__`'s
                    `masks_dir`/`use_masks`); an unmasked dataset returns
                    exactly the keys it returned before masks existed, so
                    `item.get("mask")` is the safe way to read it.
        """
        name = self._names[index]
        meta = self._meta["images"][name]
        rgb_np = np.load(self.cache_dir / f"{name}.npy")
        item: dict[str, Any] = {
            "rgb": torch.from_numpy(rgb_np).to(self.device),
            "K": torch.tensor(meta["K"], dtype=torch.float32, device=self.device),
            "qvec": torch.tensor(meta["qvec"], dtype=torch.float32, device=self.device),
            "tvec": torch.tensor(meta["tvec"], dtype=torch.float32, device=self.device),
            "name": name,
            "index": index,
        }
        if self.masks_dir is not None:
            mask_np = np.load(self.cache_dir / f"{name}{SCENE_MASK_CACHE_SUFFIX}")
            item["mask"] = torch.from_numpy(mask_np).to(torch.float32).to(self.device)
        return item

    def frame_size(self, index: int) -> tuple[int, int]:
        """(H, W) of image `index` at this dataset's width, from the cache metadata.

        Reads no pixels, so a caller that only needs the frame's shape --
        `Trainer.train_step`, to sample a crop centre -- does not have to
        materialise and upload a 2.3 MB image first.
        """
        meta = self._meta["images"][self._names[index]]
        return int(meta["height"]), int(meta["width"])

    def crop_item(
        self,
        index: int,
        size: int,
        zoom: float = 1.0,
        center: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """`crop(self[index], size, zoom, center)` without uploading the whole frame.

        Identical output to that expression, key for key and value for value
        (tests/test_scene_dataset.py::test_crop_item_matches_crop_of_getitem),
        arrived at the other way round: the nearest-neighbour gather runs on
        the host against the memory-mapped cache file, and only the
        `size x size` result is moved to `self.device`.

        Why it exists (perf/train-step): `__getitem__` uploads the full
        undistorted frame -- at Karekare-v2's 1008x756 that is 2.3 MB of RGB
        plus a 3.0 MB float32 person mask -- and `crop` then gathers a 384x384
        window out of it on the device. Every training step pays for the
        whole frame to reach the GPU so that ~8% of it can be used. Here the
        step moves ~1 MB instead, and the two large device gathers disappear.

        Args:
            index: image index (same ordering as `names`).
            size: output crop side length in pixels (square).
            zoom: crop zoom factor (see `crop`).
            center: (x, y) window centre in source pixels, or None for the
                frame centre.

        Returns:
            dict with `crop`'s own keys -- "rgb" (size, size, 3) uint8,
            "mask" (size, size) float32, "K" (3, 3) float32 -- plus the
            per-image fields a caller would otherwise have taken from
            `__getitem__`: "qvec", "tvec", "name", "index". Every tensor is
            on `self.device`.
        """
        name = self._names[index]
        meta = self._meta["images"][name]
        height_src, width_src = int(meta["height"]), int(meta["width"])
        plan = crop_plan(height_src, width_src, size, zoom, center)
        row_np = plan.row_idx.numpy()
        col_np = plan.col_idx.numpy()

        # mmap: only the rows the window touches are faulted in, instead of
        # reading the whole cached frame on every step.
        rgb_src = np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
        gathered = np.ascontiguousarray(rgb_src[row_np][:, col_np])
        out_rgb = torch.from_numpy(gathered)
        mask2d = plan.mask2d
        out_rgb = out_rgb * mask2d.to(out_rgb.dtype).unsqueeze(-1)

        valid = mask2d.to(torch.float32)
        if self.masks_dir is not None:
            mask_src = np.load(
                self.cache_dir / f"{name}{SCENE_MASK_CACHE_SUFFIX}", mmap_mode="r"
            )
            person = torch.from_numpy(np.ascontiguousarray(mask_src[row_np][:, col_np]))
            valid = valid * person.to(torch.float32)

        device = self.device
        K = torch.tensor(meta["K"], dtype=torch.float32)
        return {
            "rgb": out_rgb.to(device),
            "mask": valid.to(device),
            "K": adjust_intrinsics(K, plan.x0, plan.y0, zoom).to(device),
            "qvec": torch.tensor(meta["qvec"], dtype=torch.float32, device=device),
            "tvec": torch.tensor(meta["tvec"], dtype=torch.float32, device=device),
            "name": name,
            "index": index,
        }

    def mask_keep_fracs(self) -> dict[str, float]:
        """`{name: fraction of pixels the loss may use}` for every image, or `{}` if unmasked.

        Read straight out of the cache sidecar (`_ensure_mask_cached` writes it),
        so a run can report mask coverage -- and prove the polarity is the right
        way round, since a "person" mask that kept ~0% of a frame would show up
        here immediately -- without opening any imagery.
        """
        if self.masks_dir is None:
            return {}
        return {name: float(self._meta["images"][name]["mask_keep_frac"]) for name in self._names}


@dataclass(frozen=True)
class CropPlan:
    """Where a crop window reads from, and which of its pixels are real.

    The *only* place the crop's index arithmetic lives, so `crop` (which
    gathers on whatever device the item is on) and
    `SceneDataset.crop_item` (which gathers on the host, before anything is
    uploaded) cannot drift apart by a pixel.

    Attributes:
        row_idx: (size,) int64 CPU, source row per output row, clamped into
            the image.
        col_idx: (size,) int64 CPU, source column per output column, clamped.
        mask2d: (size, size) bool CPU, True where the window is inside the
            source image (False = crop overshoot, i.e. padding that must
            never be treated as content -- docs/GEOMETRY.md bug class 3).
        x0, y0: float, the window's top-left corner in continuous source
            pixel coordinates, needed for the intrinsics adjustment.
    """

    row_idx: torch.Tensor
    col_idx: torch.Tensor
    mask2d: torch.Tensor
    x0: float
    y0: float


def crop_plan(
    height_src: int,
    width_src: int,
    size: int,
    zoom: float = 1.0,
    center: tuple[float, float] | None = None,
) -> CropPlan:
    """Index arithmetic for one crop window (see CropPlan).

    Args:
        height_src, width_src: source image size in pixels.
        size: output crop side length in pixels (square).
        zoom: > 1 zooms in (samples a smaller `size / zoom` source window,
            nearest-neighbour resampled up to `size`); 1.0 = no zoom.
        center: (x, y) window centre in continuous source pixel coordinates;
            None centres it on the image.

    Returns:
        CropPlan, all tensors on the CPU.

    Raises:
        ValueError: on a non-positive `size` or `zoom`.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    if center is None:
        center = (width_src / 2.0, height_src / 2.0)
    center_x, center_y = center

    window = size / zoom
    half = window / 2.0
    x0 = center_x - half
    y0 = center_y - half

    # Index math in float64 on the CPU (MPS has no float64); only the int64
    # gather indices ever move to a device.
    out_idx = torch.arange(size, dtype=torch.float64)
    src_x = x0 + (out_idx + 0.5) / size * window  # (size,) continuous source pixel coords
    src_y = y0 + (out_idx + 0.5) / size * window

    src_col = torch.floor(src_x).to(torch.int64)  # nearest source pixel (spans [i, i+1))
    src_row = torch.floor(src_y).to(torch.int64)

    valid_col = (src_col >= 0) & (src_col < width_src)
    valid_row = (src_row >= 0) & (src_row < height_src)
    return CropPlan(
        row_idx=src_row.clamp(0, height_src - 1),
        col_idx=src_col.clamp(0, width_src - 1),
        mask2d=valid_row[:, None] & valid_col[None, :],
        x0=float(x0),
        y0=float(y0),
    )


def adjust_intrinsics(K: torch.Tensor, x0: float, y0: float, zoom: float) -> torch.Tensor:
    """`K` re-expressed in a crop window's own pixel coordinates.

    Args:
        K: (3, 3) float32 intrinsics at the source resolution.
        x0, y0: the window's top-left corner in source pixels (CropPlan).
        zoom: the crop's zoom factor.

    Returns:
        (3, 3) float32 on `K`'s device: focal lengths scaled by `zoom`, the
        principal point moved to the window origin and then scaled.
    """
    new_k = K.clone()
    new_k[0, 0] = K[0, 0] * zoom
    new_k[1, 1] = K[1, 1] * zoom
    new_k[0, 2] = (K[0, 2] - x0) * zoom
    new_k[1, 2] = (K[1, 2] - y0) * zoom
    return new_k


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

    A person mask travels with the item ("mask", from a masked
    `SceneDataset`) and is multiplied into the returned validity mask through
    the *same* nearest-neighbour gather as the pixels, so a masked pixel and
    the pixel it masks can never come from different source locations. The
    result is one mask with two reasons to be 0 -- crop overshoot, or a person
    -- and every consumer (L1, SSIM, LPIPS-by-zeroing, PSNR, the exposure
    diagnostics) already honours it, so nothing downstream needs to know which
    reason applied. `rgb` itself is left alone where the person mask is 0: the
    photographed pixels are real content (unlike padding, which is not), they
    simply do not count towards the loss.

    Args:
        item: a dataset item (as returned by `SceneDataset.__getitem__`):
            "rgb" (H, W, 3) uint8 tensor, optionally "K" (3, 3) float32
            tensor, and optionally "mask" (H, W) float32 tensor (1 = usable
            pixel, 0 = person -- `trippy.constants.SCENE_MASK_KEEP_THRESHOLD`).
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
            "mask": (size, size) float32 tensor, 1.0 = a pixel the loss may
                use, 0.0 = crop overshoot (padding) **or** a person pixel when
                `item` carries a "mask".
            "K": (3, 3) float32 tensor, intrinsics adjusted for the crop
                offset and zoom -- present only if `item` has a "K".
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")

    rgb = item["rgb"]
    height_src, width_src = rgb.shape[0], rgb.shape[1]
    plan = crop_plan(int(height_src), int(width_src), size, zoom, center)
    x0, y0 = plan.x0, plan.y0

    device = rgb.device
    row_idx = plan.row_idx.to(device)
    col_idx = plan.col_idx.to(device)
    mask2d = plan.mask2d.to(device)
    out_rgb = rgb[row_idx][:, col_idx]  # (size, size, 3), gathered before masking

    mask_same_dtype = mask2d.to(out_rgb.dtype)
    out_rgb = out_rgb * mask_same_dtype.unsqueeze(-1)

    valid = mask2d.to(torch.float32)
    image_mask = item.get("mask")
    if image_mask is not None:
        # Gathered with the SAME clamped indices as the pixels, then multiplied in:
        # the padding mask has already zeroed everything outside the source image,
        # so the clamped lookup at those positions cannot resurrect anything.
        valid = valid * image_mask[row_idx][:, col_idx].to(torch.float32)

    result: dict[str, Any] = {"rgb": out_rgb, "mask": valid}

    K = item.get("K")
    if K is not None:
        result["K"] = adjust_intrinsics(K, x0, y0, zoom)

    return result
