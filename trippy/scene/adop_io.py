"""Reader for an ADOP/TRIPS scene directory (the format `colmap2adop` emits).

Module: trippy.scene.adop_io
Purpose: parse everything trippy needs to reproduce a TRIPS render from a
    published scene directory -- `dataset.ini`, `camera<i>.ini`,
    `images.txt`, `camera_indices.txt`, `poses.txt`, `exposure.txt`,
    `white_balance.txt`, `masks.txt` and the zlib-compressed
    `point_cloud.bin` -- and hand back per-image `(K, R, t, H, W)` in
    trippy's own conventions.
Invariants:
    - On-disk `poses.txt` is **camera-to-world** with an **xyzw**
      quaternion (third_party/TRIPS/src/lib/data/SceneData.cpp:165,458-469;
      docs/TRIPS_REFERENCE.md Sec. 8). Everything this module returns is
      **world-to-camera** with a **wxyz** quaternion, trippy's internal
      convention (docs/GEOMETRY.md "Quaternions"). The conversion is
      exposed both ways (`pose_c2w_xyzw_to_w2c_wxyz` /
      `pose_w2c_wxyz_to_c2w_xyzw`) and round-trip tested.
    - `K` is returned in *layer-0 render pixels*, i.e. already multiplied
      by `render_scale` from `dataset.ini` together with `(H, W)`.
    - Distortion coefficients keep Saiga's own storage order
      `k1 k2 k3 k4 k5 k6 p1 p2` (NOT OpenCV's `k1 k2 p1 p2 k3 ...`); see
      `saiga/vision/cameraModel/Distortion.h:20-45,130-171`.
    - Pure numpy; no torch import, so this module can be used by the CPU
      readers and tests without pulling the raster stack in.
Units: positions and translations are in scene world units; `K`, `cx`,
    `cy` are in pixels; `exposure` is EV (log2 stops).
Related docs: docs/TRIPS_REFERENCE.md Sec. 8 (ADOP scene format), Sec. 8a
    (corrections found while writing this reader); docs/GEOMETRY.md.

-- point_cloud.bin --
`SceneData` caches its preprocessed cloud as `point_cloud.bin` via
`Saiga::UnifiedMesh::SaveCompressed` (SceneData.cpp:121;
saiga/core/model/UnifiedMesh.cpp:508-528). That is a Saiga zlib container
(saiga/core/util/zlib.cpp:15-88) -- a 24-byte header of three little-endian
`size_t`s `(magic=0x006712956A9725DE, compressed_size, decompressed_size)`
followed by a raw zlib stream -- wrapping a `BinaryOutputVector` dump
(saiga/core/util/BinaryFile.h:79-110) of, in order:

    position (vec3) | normal (vec3) | color (vec4) | texture_coordinates (vec2)
    | data (vec4) | bone_info | triangles (ivec3) | lines (ivec2) | material_id (int)

each `std::vector<T>` written as a `size_t` count followed by `count`
tightly-packed elements. `data.x` is the kNN radius `ComputeRadius` wrote
(SceneData.cpp:620-663) and `data.w` its randomised drop-out radius.

NOTE the `compressed_size` field in that header is unreliable: Saiga's
`compress3` accumulates `stream.total_out` without resetting it between
`deflate` calls (zlib.cpp:40-60), so it over-counts for multi-chunk
streams. This reader therefore feeds the rest of the file to zlib and
validates only against `decompressed_size`.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import (
    ADOP_COMPRESSED_HEADER_BYTES,
    ADOP_DISTORTION_COEFFS,
    ADOP_POINT_CLOUD_MAGIC,
    ADOP_UNIFIED_MESH_VERTEX_FIELDS,
    EPS_QUAT_AXIS,
)

# --- ini parsing ---------------------------------------------------------


def read_ini(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse a Saiga-style `.ini` into `{section: {key: value}}`.

    Saiga writes `[Section]` headers, `key = value` lines and `#` comments
    (`SceneData.cpp:1230-1290` via `SAIGA_PARAM`). Values may be empty or
    contain spaces (e.g. `K = 1164.46 1164.46 960 540 0`); they are returned
    verbatim, stripped of surrounding whitespace.

    Args:
        path: path to the ini file.

    Returns:
        Mapping of section name to key/value mapping. Keys outside any
        section land in the `""` section.

    Raises:
        FileNotFoundError: if `path` does not exist.
    """
    sections: dict[str, dict[str, str]] = {"": {}}
    current = ""
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        sections[current][key.strip()] = value.strip()
    return sections


def _flat(sections: dict[str, dict[str, str]]) -> dict[str, str]:
    """Flatten all sections into one key/value map (ADOP key names are unique)."""
    out: dict[str, str] = {}
    for values in sections.values():
        out.update(values)
    return out


def _floats(text: str) -> list[float]:
    return [float(tok) for tok in text.replace(",", " ").split()]


# --- quaternions and poses ----------------------------------------------


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from ADOP's on-disk xyzw to trippy's wxyz."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from trippy's wxyz to ADOP's on-disk xyzw."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return `q` scaled to unit norm (wxyz or xyzw -- order-agnostic)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < EPS_QUAT_AXIS:
        raise ValueError(f"quaternion has ~zero norm: {q}")
    return q / norm


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for a unit quaternion) of a wxyz quaternion."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def qvec2R(q: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion -> 3x3 rotation matrix.

    Independent of `trippy.geom.xform_a`/`xform_b` on purpose (AGENTS.md
    Sec. 7 "implement transforms twice"); `tests/test_scene_adop_io.py`
    asserts this agrees with `trippy.geom.xform_b.qvec2R` on random inputs.
    """
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_c2w_xyzw_to_w2c_wxyz(q_xyzw: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ADOP `poses.txt` line -> trippy's world-to-camera `(q_wxyz, t)`.

    ADOP stores `qx qy qz qw tx ty tz` for the **camera-to-world** SE3
    (SceneData.cpp:458-469, written from COLMAP's world-to-camera pose via
    `.inverse()`, colmap2adop.cpp:143-148). trippy wants world-to-camera:
    `x_cam = R @ x_world + t`.

    Args:
        q_xyzw: (4,) camera-to-world rotation, xyzw order, unit norm.
        t_c2w: (3,) camera-to-world translation (== camera centre in world).

    Returns:
        `(q_wxyz, t_w2c)`: (4,) world-to-camera rotation in wxyz order and
        (3,) world-to-camera translation.
    """
    q_c2w = quat_normalize(quat_xyzw_to_wxyz(q_xyzw))
    q_w2c = quat_conjugate_wxyz(q_c2w)
    r_w2c = qvec2R(q_w2c)
    t_w2c = -r_w2c @ np.asarray(t_c2w, dtype=np.float64).reshape(3)
    return q_w2c, t_w2c


def pose_w2c_wxyz_to_c2w_xyzw(q_wxyz: np.ndarray, t_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse of `pose_c2w_xyzw_to_w2c_wxyz` (used to write ADOP poses back)."""
    q_w2c = quat_normalize(q_wxyz)
    q_c2w = quat_conjugate_wxyz(q_w2c)
    r_c2w = qvec2R(q_c2w)
    t_c2w = -r_c2w @ np.asarray(t_w2c, dtype=np.float64).reshape(3)
    return quat_wxyz_to_xyzw(q_c2w), t_c2w


# --- dataclasses ---------------------------------------------------------


@dataclass(frozen=True)
class AdopCameraParams:
    """One `camera<i>.ini` (`SceneCameraParams`, SceneData.h:180-241).

    Attributes:
        width, height: calibrated image size in pixels.
        fx, fy, cx, cy, s: pinhole intrinsics in pixels (`K = fx fy cx cy s`,
            SceneData.h:209-222). Saiga's `normalizedToImage` is
            `(fx*x + s*y + cx, fy*y + cy)` (Intrinsics4.h:80).
        distortion: 8 coefficients in Saiga order `k1 k2 k3 k4 k5 k6 p1 p2`
            (Distortion.h:20-45).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    s: float
    distortion: tuple[float, ...]

    def K(self, scale: float = 1.0) -> np.ndarray:
        """3x3 intrinsics matrix, all pixel quantities multiplied by `scale`."""
        return np.array(
            [
                [self.fx * scale, self.s * scale, self.cx * scale],
                [0.0, self.fy * scale, self.cy * scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def size(self, scale: float = 1.0) -> tuple[int, int]:
        """`(height, width)` at the given render scale, rounded like TRIPS (`int()`)."""
        return int(self.height * scale), int(self.width * scale)


@dataclass(frozen=True)
class AdopPointCloud:
    """The vertex arrays inside `point_cloud.bin` (`Saiga::UnifiedMesh`).

    Attributes:
        position: (N, 3) float32 world positions.
        normal: (N, 3) float32 (may be empty, shape (0, 3)).
        color: (N, 4) float32 RGBA in [0, 1] (may be empty).
        data: (N, 4) float32; `data[:, 0]` is the kNN radius written by
            `SceneData::ComputeRadius` and `data[:, 3]` its randomised
            drop-out radius (SceneData.cpp:620-663).
    """

    position: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    data: np.ndarray

    def __len__(self) -> int:
        return int(self.position.shape[0])


@dataclass(frozen=True)
class AdopView:
    """Everything `trippy.raster.pyramid.render_pyramid` needs for one image."""

    index: int
    image_name: str
    image_path: Path
    mask_path: Path | None
    camera_index: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    height: int
    width: int
    distortion: np.ndarray
    exposure: float
    white_balance: np.ndarray


@dataclass(frozen=True)
class AdopScene:
    """A parsed ADOP scene directory."""

    root: Path
    image_dir: Path
    image_names: list[str]
    mask_names: list[str]
    camera_indices: np.ndarray
    cameras: list[AdopCameraParams]
    q_w2c: np.ndarray
    t_w2c: np.ndarray
    exposure: np.ndarray
    white_balance: np.ndarray
    render_scale: float
    znear: float
    zfar: float
    scene_exposure_value: float
    point_cloud_file: Path

    def __len__(self) -> int:
        return len(self.image_names)

    def view(self, index: int, render_scale: float | None = None) -> AdopView:
        """Build an `AdopView` for image `index` at `render_scale` (default: the scene's)."""
        scale = self.render_scale if render_scale is None else render_scale
        cam = self.cameras[int(self.camera_indices[index])]
        height, width = cam.size(scale)
        mask_name = self.mask_names[index] if index < len(self.mask_names) else ""
        return AdopView(
            index=index,
            image_name=self.image_names[index],
            image_path=self.image_dir / self.image_names[index],
            mask_path=(self.root / mask_name) if mask_name else None,
            camera_index=int(self.camera_indices[index]),
            K=cam.K(scale),
            R=qvec2R(self.q_w2c[index]),
            t=self.t_w2c[index].copy(),
            height=height,
            width=width,
            distortion=np.asarray(cam.distortion, dtype=np.float64),
            exposure=float(self.exposure[index]),
            white_balance=self.white_balance[index].copy(),
        )

    def index_of(self, image_name: str) -> int:
        """Index of `image_name` in `images.txt`.

        Raises:
            KeyError: if the name is not in the scene.
        """
        try:
            return self.image_names.index(image_name)
        except ValueError as exc:
            raise KeyError(f"{image_name!r} is not in images.txt") from exc


# --- readers -------------------------------------------------------------


def read_camera_ini(path: str | Path) -> AdopCameraParams:
    """Parse one `camera<i>.ini`."""
    values = _flat(read_ini(path))
    k = _floats(values["K"])
    if len(k) != 5:
        raise ValueError(f"{path}: expected 5 K values (fx fy cx cy s), got {len(k)}")
    dist = _floats(values.get("distortion", ""))
    if len(dist) not in (0, ADOP_DISTORTION_COEFFS):
        raise ValueError(f"{path}: expected {ADOP_DISTORTION_COEFFS} distortion values, got {len(dist)}")
    if not dist:
        dist = [0.0] * ADOP_DISTORTION_COEFFS
    return AdopCameraParams(
        width=int(float(values["w"])),
        height=int(float(values["h"])),
        fx=k[0],
        fy=k[1],
        cx=k[2],
        cy=k[3],
        s=k[4],
        distortion=tuple(dist),
    )


def _read_lines(path: Path, count: int | None = None, strip: bool = True) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if strip:
        lines = [line.strip() for line in lines]
    if count is not None and len(lines) > count:
        lines = lines[:count]
    return lines


def read_poses(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `poses.txt` and return world-to-camera `(q_wxyz (N,4), t (N,3))`.

    The file itself is camera-to-world with xyzw quaternions -- see
    `pose_c2w_xyzw_to_w2c_wxyz`.
    """
    rows = [line for line in _read_lines(Path(path)) if line]
    q_out = np.zeros((len(rows), 4), dtype=np.float64)
    t_out = np.zeros((len(rows), 3), dtype=np.float64)
    for i, line in enumerate(rows):
        vals = _floats(line)
        if len(vals) != 7:
            raise ValueError(f"{path}:{i + 1}: expected 7 numbers (qx qy qz qw tx ty tz), got {len(vals)}")
        q_out[i], t_out[i] = pose_c2w_xyzw_to_w2c_wxyz(np.array(vals[:4]), np.array(vals[4:]))
    return q_out, t_out


def read_point_cloud_bin(path: str | Path) -> AdopPointCloud:
    """Read a Saiga `UnifiedMesh::SaveCompressed` dump (see module docstring).

    Args:
        path: path to `point_cloud.bin`.

    Returns:
        AdopPointCloud with position/normal/color/data arrays.

    Raises:
        ValueError: if the magic number or the decompressed size is wrong.
    """
    raw = Path(path).read_bytes()
    if len(raw) < ADOP_COMPRESSED_HEADER_BYTES:
        raise ValueError(f"{path}: file shorter than the {ADOP_COMPRESSED_HEADER_BYTES}-byte Saiga zlib header")
    magic, _compressed_size, decompressed_size = struct.unpack_from("<QQQ", raw, 0)
    if magic != ADOP_POINT_CLOUD_MAGIC:
        raise ValueError(f"{path}: bad Saiga compress magic {magic:#x} (expected {ADOP_POINT_CLOUD_MAGIC:#x})")
    # _compressed_size is deliberately ignored -- Saiga's compress3 over-counts it
    # (zlib.cpp:40-60); the zlib stream is self-terminating anyway.
    blob = zlib.decompress(raw[ADOP_COMPRESSED_HEADER_BYTES:])
    if len(blob) != decompressed_size:
        raise ValueError(f"{path}: decompressed {len(blob)} bytes, header says {decompressed_size}")

    fields: dict[str, np.ndarray] = {}
    offset = 0
    for name, width in ADOP_UNIFIED_MESH_VERTEX_FIELDS:
        (count,) = struct.unpack_from("<Q", blob, offset)
        offset += 8
        nbytes = count * width * 4
        if offset + nbytes > len(blob):
            raise ValueError(f"{path}: truncated while reading '{name}' ({count} x {width} floats)")
        arr = np.frombuffer(blob, dtype=np.float32, count=count * width, offset=offset).reshape(count, width)
        fields[name] = np.array(arr, copy=True)
        offset += nbytes

    return AdopPointCloud(
        position=fields["position"],
        normal=fields["normal"],
        color=fields["color"],
        data=fields["data"],
    )


def write_point_cloud_bin(path: str | Path, cloud: AdopPointCloud) -> Path:
    """Write `cloud` back in Saiga's compressed `UnifiedMesh` format.

    Only exists so `tests/test_scene_adop_io.py` can build a synthetic
    fixture in exactly the on-disk format (AGENTS.md: synthetic fixtures
    only) and prove `read_point_cloud_bin` round-trips it. The trailing
    `bone_info`/`triangles`/`lines`/`material_id` fields are written empty,
    matching every ADOP point cloud.
    """
    parts: list[bytes] = []
    arrays = {
        "position": cloud.position,
        "normal": cloud.normal,
        "color": cloud.color,
        "texture_coordinates": np.zeros((0, 2), dtype=np.float32),
        "data": cloud.data,
    }
    for name, width in ADOP_UNIFIED_MESH_VERTEX_FIELDS:
        arr = np.ascontiguousarray(arrays[name], dtype=np.float32).reshape(-1, width)
        parts.append(struct.pack("<Q", arr.shape[0]))
        parts.append(arr.tobytes())
    # bone_info, triangles, lines: empty vectors; then material_id (int32).
    parts.append(struct.pack("<QQQ", 0, 0, 0))
    parts.append(struct.pack("<i", 0))
    blob = b"".join(parts)
    compressed = zlib.compress(blob)
    header = struct.pack("<QQQ", ADOP_POINT_CLOUD_MAGIC, len(compressed), len(blob))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + compressed)
    return out


def load_adop_scene(root: str | Path) -> AdopScene:
    """Read a whole ADOP scene directory.

    Args:
        root: directory containing `dataset.ini`, `images.txt`, ...

    Returns:
        An `AdopScene`. Missing optional files (`exposure.txt`,
        `white_balance.txt`, `masks.txt`) fall back to TRIPS's own
        defaults: exposure 0, white balance (1, 1, 1), no mask.

    Raises:
        FileNotFoundError: if `dataset.ini`, `images.txt` or `poses.txt` is missing.
        ValueError: on a per-image list whose length disagrees with `images.txt`.
    """
    root = Path(root)
    dataset = _flat(read_ini(root / "dataset.ini"))

    image_names = [line for line in _read_lines(root / "images.txt") if line]
    num_images = len(image_names)

    camera_files = dataset.get("camera_files", "camera0.ini").split()
    cameras = [read_camera_ini(root / name) for name in camera_files]

    cam_idx_lines = [line for line in _read_lines(root / "camera_indices.txt") if line]
    if cam_idx_lines:
        camera_indices = np.array([int(float(v)) for v in cam_idx_lines], dtype=np.int64)
    else:
        camera_indices = np.zeros(num_images, dtype=np.int64)
    if camera_indices.shape[0] != num_images:
        raise ValueError(f"camera_indices.txt has {camera_indices.shape[0]} rows, images.txt has {num_images}")

    q_w2c, t_w2c = read_poses(root / "poses.txt")
    if q_w2c.shape[0] != num_images:
        raise ValueError(f"poses.txt has {q_w2c.shape[0]} rows, images.txt has {num_images}")

    exposure_lines = [line for line in _read_lines(root / "exposure.txt") if line]
    exposure = (
        np.array([float(v) for v in exposure_lines], dtype=np.float64)
        if exposure_lines
        else np.zeros(num_images, dtype=np.float64)
    )
    if exposure.shape[0] != num_images:
        raise ValueError(f"exposure.txt has {exposure.shape[0]} rows, images.txt has {num_images}")

    wb_lines = [line for line in _read_lines(root / "white_balance.txt") if line]
    white_balance = (
        np.array([_floats(line) for line in wb_lines], dtype=np.float64)
        if wb_lines
        else np.ones((num_images, 3), dtype=np.float64)
    )
    if white_balance.shape != (num_images, 3):
        raise ValueError(f"white_balance.txt is {white_balance.shape}, expected ({num_images}, 3)")

    # masks.txt: one relative path per image; TRIPS's tt_* scenes ship it blank.
    mask_names = _read_lines(root / "masks.txt")
    mask_names = (mask_names + [""] * num_images)[:num_images]

    # dataset.ini's image_dir is written relative to the *scene collection* root
    # (e.g. "scenes/tt_horse/images/"), which does not resolve from inside the
    # scene dir; prefer the scene's own images/ subdir when it exists.
    image_dir = root / "images"
    if not image_dir.is_dir():
        image_dir = (root / dataset.get("image_dir", "images")).resolve()

    return AdopScene(
        root=root,
        image_dir=image_dir,
        image_names=image_names,
        mask_names=mask_names,
        camera_indices=camera_indices,
        cameras=cameras,
        q_w2c=q_w2c,
        t_w2c=t_w2c,
        exposure=exposure,
        white_balance=white_balance,
        render_scale=float(dataset.get("render_scale", 1.0)),
        znear=float(dataset.get("znear", 0.1)),
        zfar=float(dataset.get("zfar", 1000.0)),
        scene_exposure_value=float(dataset.get("scene_exposure_value", 0.0)),
        point_cloud_file=root / dataset.get("file_point_cloud_compressed", "point_cloud.bin"),
    )
