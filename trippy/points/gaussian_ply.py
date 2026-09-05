"""PointSource that reads a trained 3DGS binary PLY as splat centres.

Module: trippy.points.gaussian_ply
Invariants: reads the binary vertex body with a single np.fromfile call
    against a structured dtype built from the parsed header -- no per-row
    Python loop and no plyfile for the big read (plyfile is fine for
    writing small synthetic test fixtures, per AGENTS.md test-fixture
    rule). This matters: production PLYs run to ~7.36M rows / ~1.7 GB.
Related docs: docs/GEOMETRY.md "3DGS PLY export mapping" (field meanings,
    opacity is stored in logit space, scale is stored in log space);
    docs/SPEC.md D4 (point source 1: trained Gaussian centres).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import (
    DEFAULT_MIN_OPACITY,
    KNN_CHUNK,
    KNN_SIZE_K,
    PROVENANCE_GAUSSIAN,
    SH_C0,
)
from trippy.points.knn_size import knn_mean_distance
from trippy.points.source import PointSet, PointSource

_NP_DTYPE = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "<u1",
    "uint8": "<u1",
    "char": "<i1",
    "int8": "<i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, vectorised, float64 in / float64 out."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _read_ply_header(f) -> tuple[int, list[tuple[str, str]]]:
    """Parse a binary-little-endian PLY header for the `vertex` element.

    Args:
        f: binary file object positioned at byte 0.

    Returns:
        (vertex_count, [(prop_name, prop_type), ...]) in header order.

    Raises:
        ValueError: not a binary_little_endian PLY, no vertex element, or
            a list property is present (unsupported -- 3DGS vertex
            properties are all scalar).
    """
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file (missing magic 'ply' line)")
    fmt = f.readline().strip()
    if fmt != b"format binary_little_endian 1.0":
        raise ValueError(f"unsupported PLY format {fmt!r}; expected binary_little_endian 1.0")

    count: int | None = None
    props: list[tuple[str, str]] = []
    current_element: str | None = None
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected EOF while reading PLY header")
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == b"comment":
            continue
        if tokens[0] == b"element":
            current_element = tokens[1].decode()
            if current_element == "vertex":
                count = int(tokens[2])
            continue
        if tokens[0] == b"property" and current_element == "vertex":
            if tokens[1] == b"list":
                raise ValueError(f"list property unsupported in vertex element: {line!r}")
            prop_type, prop_name = tokens[1].decode(), tokens[2].decode()
            props.append((prop_name, prop_type))
            continue
        if tokens[0] == b"end_header":
            break

    if count is None:
        raise ValueError("PLY header has no 'vertex' element")
    return count, props


class GaussianPlySource(PointSource):
    """Trained 3DGS Gaussian centres as splat points (D4 point source 1).

    Args:
        path: path to a binary_little_endian 3DGS PLY (x, y, z, f_dc_0..2,
            opacity, scale_0..2, rot_0..3 vertex properties; higher-order
            SH and rotation are read but unused here).
        min_opacity: drop points with sigmoid(opacity) below this
            (default trippy.constants.DEFAULT_MIN_OPACITY).
        size_mode: "scale" (mean of exp(scale_0..2), the trained Gaussian
            extent) or "knn" (mean distance to the KNN_SIZE_K nearest
            neighbours, ignoring the trained scale entirely).
        max_points: if set, randomly subsample to at most this many points
            *after* the opacity filter and *before* size_mode="knn" (so
            kNN never runs on the full multi-million-point cloud unless
            max_points is left unset).
        seed: RNG seed for the max_points subsample (reproducible runs).
    """

    def __init__(
        self,
        path: str | Path,
        min_opacity: float = DEFAULT_MIN_OPACITY,
        size_mode: str = "scale",
        max_points: int | None = None,
        seed: int = 0,
    ) -> None:
        if size_mode not in ("scale", "knn"):
            raise ValueError(f"size_mode must be 'scale' or 'knn', got {size_mode!r}")
        self.path = Path(path)
        self.min_opacity = min_opacity
        self.size_mode = size_mode
        self.max_points = max_points
        self.seed = seed

    def describe(self) -> dict:
        return {
            "type": "GaussianPlySource",
            "path": str(self.path),
            "min_opacity": self.min_opacity,
            "size_mode": self.size_mode,
            "max_points": self.max_points,
            "seed": self.seed,
        }

    def build(self) -> PointSet:
        with open(self.path, "rb") as f:
            count, props = _read_ply_header(f)
            dtype = np.dtype([(name, _NP_DTYPE[ptype]) for name, ptype in props])
            raw = np.fromfile(f, dtype=dtype, count=count)

        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        f_dc = np.stack([raw[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)
        opacity_logit = raw["opacity"].astype(np.float64)
        scale_log = np.stack([raw[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)

        conf0 = _sigmoid(opacity_logit)
        keep = conf0 >= self.min_opacity

        xyz = xyz[keep]
        rgb0 = np.clip(0.5 + SH_C0 * f_dc[keep], 0.0, 1.0).astype(np.float32)
        conf0 = conf0[keep].astype(np.float32)
        scale_log = scale_log[keep]

        n = xyz.shape[0]
        if self.max_points is not None and n > self.max_points:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.max_points, replace=False)
            xyz, rgb0, conf0, scale_log = xyz[idx], rgb0[idx], conf0[idx], scale_log[idx]
            n = self.max_points

        if self.size_mode == "scale":
            size0 = np.exp(scale_log).mean(axis=1).astype(np.float32)
        else:
            size0 = knn_mean_distance(xyz, k=KNN_SIZE_K, chunk=KNN_CHUNK)

        provenance = np.full(n, PROVENANCE_GAUSSIAN, dtype=np.uint8)
        return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)
