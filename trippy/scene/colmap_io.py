"""COLMAP sparse model loading: text and binary readers into one ColmapScene.

Module: trippy.scene.colmap_io
Invariants: text parsing is never duplicated here -- it is delegated to
    trippy.geom.xform_a's read_cameras_txt/read_images_txt/read_points3d_txt.
    Binary parsing is implemented from scratch with `struct`, against
    COLMAP's documented cameras.bin/images.bin/points3D.bin layout (little-
    endian, as written by COLMAP's own scripts/python/read_write_model.py).
    The layout was verified byte-for-byte against a real
    ~/Splats/scenes/karekare/kk-coherent/sparse/0 model during development
    (6 OPENCV cameras, 219 registered images) -- see
    tests/test_scene_colmap_io.py's real-scene integration test.
    The writers (`write_cameras_txt`/`write_images_txt`/`write_points3d_txt`/
    `save_colmap_model_txt`, added for trippy.distill's design-B pipeline,
    docs/SPEC.md D2) are the exact textual inverse of the text readers above
    -- `write_*` then `load_colmap_model` round-trips every field the text
    format itself carries (not the binary-only point3D track, which the text
    format has no column for -- see read_points3d_txt's own docstring).
Coordinate frame: identical COLMAP world/camera convention as
    trippy.geom.xform_a/xform_b -- qvec is (qw, qx, qy, qz), x_cam =
    R(qvec) @ x_world + tvec. xys are 2D keypoint pixel coordinates in the
    *distorted* (as-captured) image; point3D_ids[i] == -1 means keypoint i
    has no triangulated 3D point.
Related docs: docs/ARCHITECTURE.md "Module overview" (trippy/scene);
    docs/SPEC.md v0.1.0 milestone ("colmap_io"); trippy.geom.camera
    (intrinsics_from_colmap_params, OpenCVDistortion); docs/EXPERIMENTS.md
    "Distillation (design B)".
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trippy.constants import COLMAP_CAMERA_MODEL_TABLE
from trippy.geom import camera as camera_geom
from trippy.geom import xform_a


@dataclass
class Camera:
    """One COLMAP camera (a physical lens/sensor config, shared by 1+ images).

    Attributes:
        model: COLMAP camera model name, e.g. "OPENCV", "PINHOLE".
        width: sensor image width in pixels (as captured, before undistort).
        height: sensor image height in pixels.
        params: raw COLMAP params list, meaning depends on `model` (e.g.
            OPENCV = [fx, fy, cx, cy, k1, k2, p1, p2]).
    """

    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class Image:
    """One registered COLMAP image: a pose plus its observed 2D keypoints.

    Attributes:
        name: image filename (e.g. "IMG_3703.jpg"), relative to the scene's
            images/ directory.
        camera_id: key into ColmapScene.cameras.
        qvec: shape (4,), float64, (qw, qx, qy, qz), unit norm, world->camera
            rotation (see trippy.geom.xform_a.qvec2R).
        tvec: shape (3,), float64, world->camera translation.
        xys: shape (N, 2), float64, keypoint pixel coordinates in the
            as-captured (distorted) image, (x right, y down).
        point3D_ids: shape (N,), int64, xys[i]'s triangulated point id, or
            -1 if keypoint i was never triangulated.
    """

    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point3D:
    """One triangulated sparse point and the image observations that built it.

    Attributes:
        xyz: shape (3,), float64, world-frame position.
        rgb: shape (3,), uint8, COLMAP's mean observed colour.
        error: float, COLMAP's mean reprojection error in pixels.
        track: list of (image_id, point2D_idx) pairs -- image_id keys into
            ColmapScene.images by id (see ColmapScene.images_by_id), and
            point2D_idx indexes that image's xys/point3D_ids arrays.
    """

    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: list[tuple[int, int]]


@dataclass
class ColmapScene:
    """A full COLMAP sparse reconstruction, keyed the way COLMAP keys it.

    Attributes:
        cameras: camera_id -> Camera.
        images: image_id -> Image (registered images only -- a capture that
            failed to register during SfM will not appear here even if its
            file exists under images/; see docs/PLAN-2026-09-05.md's
            "238 images" (raw captures) vs the smaller registered count).
        points3D: point3D_id -> Point3D.
    """

    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points3D: dict[int, Point3D] = field(default_factory=dict)

    def images_by_name(self) -> dict[str, Image]:
        """Convenience view: image filename -> Image (names are unique per scene)."""
        return {im.name: im for im in self.images.values()}


def _has_binary_model(sparse_dir: Path) -> bool:
    return (sparse_dir / "cameras.bin").exists() and (sparse_dir / "images.bin").exists()


def load_colmap_model(sparse_dir: str | Path) -> ColmapScene:
    """Load a COLMAP sparse model, auto-detecting binary vs text format.

    Binary is preferred when both `cameras.bin` and `images.bin` exist in
    `sparse_dir`; otherwise the text triple (`cameras.txt`, `images.txt`,
    `points3D.txt`) is read via trippy.geom.xform_a's parsers.

    Args:
        sparse_dir: directory containing either the COLMAP binary triple
            (cameras.bin, images.bin, points3D.bin) or text triple
            (cameras.txt, images.txt, points3D.txt).

    Returns:
        ColmapScene with cameras/images/point3D populated.

    Raises:
        FileNotFoundError: neither format is present in `sparse_dir`.
    """
    sparse_dir = Path(sparse_dir)
    if _has_binary_model(sparse_dir):
        cameras = _read_cameras_bin(sparse_dir / "cameras.bin")
        images = _read_images_bin(sparse_dir / "images.bin")
        points_path = sparse_dir / "points3D.bin"
        points3D = _read_points3d_bin(points_path) if points_path.exists() else {}
        return ColmapScene(cameras=cameras, images=images, points3D=points3D)

    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    points_txt = sparse_dir / "points3D.txt"
    if not (cameras_txt.exists() and images_txt.exists()):
        raise FileNotFoundError(
            f"no COLMAP model (binary or text) found under {sparse_dir}; "
            "expected cameras.bin+images.bin or cameras.txt+images.txt"
        )

    raw_cameras = xform_a.read_cameras_txt(str(cameras_txt))
    cameras = {
        cid: Camera(model=c["model"], width=c["width"], height=c["height"], params=list(c["params"]))
        for cid, c in raw_cameras.items()
    }

    raw_images = xform_a.read_images_txt(str(images_txt))
    images: dict[int, Image] = {}
    for im in raw_images.values():
        points2d = im["points2d"]
        if points2d:
            xys = np.array([(x, y) for x, y, _ in points2d], dtype=np.float64)
            point3D_ids = np.array([pid for _, _, pid in points2d], dtype=np.int64)
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            point3D_ids = np.zeros((0,), dtype=np.int64)
        images[im["image_id"]] = Image(
            name=im["name"],
            camera_id=im["camera_id"],
            qvec=np.asarray(im["qvec"], dtype=np.float64),
            tvec=np.asarray(im["tvec"], dtype=np.float64),
            xys=xys,
            point3D_ids=point3D_ids,
        )

    points3D: dict[int, Point3D] = {}
    if points_txt.exists():
        raw_points = xform_a.read_points3d_txt(str(points_txt))
        for pid, p in raw_points.items():
            points3D[pid] = Point3D(
                xyz=np.asarray(p["xyz"], dtype=np.float64),
                rgb=np.asarray(p["rgb"], dtype=np.uint8),
                error=float(p["error"]),
                track=[],  # text format (as parsed by xform_a) does not carry the track.
            )

    return ColmapScene(cameras=cameras, images=images, points3D=points3D)


def _read_struct(f, fmt: str, size: int) -> tuple:
    """Read exactly `size` bytes and unpack as little-endian `fmt`."""
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"expected {size} bytes, got {len(data)} (truncated COLMAP binary file)")
    return struct.unpack("<" + fmt, data)


def _read_cameras_bin(path: Path) -> dict[int, Camera]:
    """Parse a COLMAP cameras.bin.

    Layout (little-endian): uint64 num_cameras, then per camera:
    int32 camera_id, int32 model_id, uint64 width, uint64 height,
    float64[num_params(model_id)] params. num_params per model_id comes
    from trippy.constants.COLMAP_CAMERA_MODEL_TABLE.
    """
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num_cameras,) = _read_struct(f, "Q", 8)
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_struct(f, "iiQQ", 24)
            if model_id not in COLMAP_CAMERA_MODEL_TABLE:
                raise ValueError(f"unknown COLMAP camera model_id {model_id} in {path}")
            model_name, num_params = COLMAP_CAMERA_MODEL_TABLE[model_id]
            params = _read_struct(f, "d" * num_params, 8 * num_params)
            cameras[camera_id] = Camera(
                model=model_name, width=int(width), height=int(height), params=list(params)
            )
    return cameras


def _read_cstring(f) -> str:
    """Read a null-terminated UTF-8 string (COLMAP's image name encoding)."""
    chars = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            break
        chars += c
    return chars.decode("utf-8")


def _read_images_bin(path: Path) -> dict[int, Image]:
    """Parse a COLMAP images.bin.

    Layout (little-endian): uint64 num_reg_images, then per image:
    int32 image_id, float64[4] qvec (qw,qx,qy,qz), float64[3] tvec,
    int32 camera_id, null-terminated name string, uint64 num_points2D, then
    num_points2D * (float64 x, float64 y, int64 point3D_id [-1 if none]).
    """
    images: dict[int, Image] = {}
    with open(path, "rb") as f:
        (num_reg_images,) = _read_struct(f, "Q", 8)
        for _ in range(num_reg_images):
            (image_id,) = _read_struct(f, "i", 4)
            qvec = np.array(_read_struct(f, "dddd", 32), dtype=np.float64)
            tvec = np.array(_read_struct(f, "ddd", 24), dtype=np.float64)
            (camera_id,) = _read_struct(f, "i", 4)
            name = _read_cstring(f)
            (num_points2d,) = _read_struct(f, "Q", 8)
            if num_points2d:
                raw = _read_struct(f, "ddq" * num_points2d, 24 * num_points2d)
                arr = np.array(raw, dtype=np.float64).reshape(-1, 3)
                xys = arr[:, :2].copy()
                point3D_ids = arr[:, 2].astype(np.int64)
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                point3D_ids = np.zeros((0,), dtype=np.int64)
            images[image_id] = Image(
                name=name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def _read_points3d_bin(path: Path) -> dict[int, Point3D]:
    """Parse a COLMAP points3D.bin.

    Layout (little-endian): uint64 num_points, then per point: uint64
    point3D_id, float64[3] xyz, uint8[3] rgb, float64 error, uint64
    track_length, then track_length * (int32 image_id, int32 point2D_idx).
    """
    points: dict[int, Point3D] = {}
    with open(path, "rb") as f:
        (num_points,) = _read_struct(f, "Q", 8)
        for _ in range(num_points):
            (point3d_id,) = _read_struct(f, "Q", 8)
            xyz = np.array(_read_struct(f, "ddd", 24), dtype=np.float64)
            rgb = np.array(_read_struct(f, "BBB", 3), dtype=np.uint8)
            (error,) = _read_struct(f, "d", 8)
            (track_length,) = _read_struct(f, "Q", 8)
            track: list[tuple[int, int]] = []
            if track_length:
                raw = _read_struct(f, "ii" * track_length, 8 * track_length)
                track = [(raw[2 * i], raw[2 * i + 1]) for i in range(track_length)]
            points[point3d_id] = Point3D(xyz=xyz, rgb=rgb, error=float(error), track=track)
    return points


def intrinsics(cam: Camera) -> tuple[float, float, float, float]:
    """Distortion-free pinhole intrinsics (fx, fy, cx, cy) in pixels for `cam`.

    Thin wrapper over trippy.geom.camera.intrinsics_from_colmap_params.
    """
    return camera_geom.intrinsics_from_colmap_params(cam.model, cam.params)


# COLMAP camera models this module knows how to reduce to (k1, k2, p1, p2)
# radial-tangential distortion, and the slice of `cam.params` each term
# comes from. Models not listed here have no supported distortion readout.
_OPENCV_LIKE_MODELS = {"OPENCV", "OPENCV_FISHEYE"}
_SIMPLE_RADIAL_MODELS = {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}
_RADIAL_MODELS = {"RADIAL", "RADIAL_FISHEYE"}
_DISTORTION_FREE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def distortion(cam: Camera) -> tuple[float, float, float, float]:
    """Radial-tangential distortion (k1, k2, p1, p2) for `cam`.

    Returns (0.0, 0.0, 0.0, 0.0) for pinhole models (PINHOLE, SIMPLE_PINHOLE)
    and for the radial-only models (params carry only k1[, k2]; p1=p2=0.0).

    Raises:
        ValueError: `cam.model` is not one of the models this function
            knows how to reduce to (k1, k2, p1, p2).
    """
    if cam.model in _DISTORTION_FREE_MODELS:
        return (0.0, 0.0, 0.0, 0.0)
    if cam.model in _OPENCV_LIKE_MODELS:
        k1, k2, p1, p2 = cam.params[4:8]
        return (float(k1), float(k2), float(p1), float(p2))
    if cam.model in _SIMPLE_RADIAL_MODELS:
        (k1,) = cam.params[3:4]
        return (float(k1), 0.0, 0.0, 0.0)
    if cam.model in _RADIAL_MODELS:
        k1, k2 = cam.params[3:5]
        return (float(k1), float(k2), 0.0, 0.0)
    raise ValueError(f"unsupported COLMAP camera model for distortion(): {cam.model!r}")


# --- text writers: the inverse of read_cameras_txt/read_images_txt/read_points3d_txt ---


def write_cameras_txt(path: str | Path, cameras: dict[int, Camera]) -> Path:
    """Write a COLMAP cameras.txt from `cameras` (camera_id -> Camera), sorted by id.

    Args:
        path: output path; parent directories are created if missing.
        cameras: as `ColmapScene.cameras`.

    Returns:
        `path`.
    """
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera_id in sorted(cameras):
        cam = cameras[camera_id]
        params = " ".join(repr(float(p)) for p in cam.params)
        lines.append(f"{camera_id} {cam.model} {cam.width} {cam.height} {params}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_images_txt(path: str | Path, images: dict[int, Image]) -> Path:
    """Write a COLMAP images.txt from `images` (image_id -> Image), sorted by id.

    Every image line is followed by a POINTS2D line, blank for an image with
    no observations (`xys.shape[0] == 0`) -- a real, zero-length line, never
    omitted, matching this module's own text reader (see its docstring:
    "a zero-observation image still writes a genuine, blank POINTS2D line").

    Args:
        path: output path; parent directories are created if missing.
        images: as `ColmapScene.images`.

    Returns:
        `path`.
    """
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}",
    ]
    for image_id in sorted(images):
        im = images[image_id]
        qvec_str = " ".join(repr(float(v)) for v in im.qvec)
        tvec_str = " ".join(repr(float(v)) for v in im.tvec)
        lines.append(f"{image_id} {qvec_str} {tvec_str} {im.camera_id} {im.name}")
        if im.xys.shape[0] == 0:
            lines.append("")
        else:
            points2d = " ".join(
                f"{x!r} {y!r} {pid}"
                for (x, y), pid in zip(im.xys.tolist(), im.point3D_ids.tolist(), strict=True)
            )
            lines.append(points2d)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_points3d_txt(path: str | Path, points3D: dict[int, Point3D]) -> Path:
    """Write a COLMAP points3D.txt from `points3D` (point3D_id -> Point3D), sorted by id.

    The track is written when present (`Point3D.track`) but this module's own
    text reader ignores it on the way back in (see `read_points3d_txt`'s
    docstring) -- writing it anyway keeps a round-tripped file
    format-realistic for other COLMAP-reading tools.

    Args:
        path: output path; parent directories are created if missing.
        points3D: as `ColmapScene.points3D`.

    Returns:
        `path`.
    """
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(points3D)}",
    ]
    for point3d_id in sorted(points3D):
        p = points3D[point3d_id]
        xyz_str = " ".join(repr(float(v)) for v in p.xyz)
        rgb_str = " ".join(str(int(v)) for v in p.rgb)
        line = f"{point3d_id} {xyz_str} {rgb_str} {p.error!r}"
        if p.track:
            line += " " + " ".join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in p.track)
        lines.append(line)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def save_colmap_model_txt(sparse_dir: str | Path, scene: ColmapScene) -> Path:
    """Write `scene` as a COLMAP text triple (cameras.txt/images.txt/points3D.txt) under `sparse_dir`.

    Args:
        sparse_dir: output directory; created if missing.
        scene: the ColmapScene to write.

    Returns:
        `sparse_dir`.
    """
    sparse_dir = Path(sparse_dir)
    write_cameras_txt(sparse_dir / "cameras.txt", scene.cameras)
    write_images_txt(sparse_dir / "images.txt", scene.images)
    write_points3d_txt(sparse_dir / "points3D.txt", scene.points3D)
    return sparse_dir
