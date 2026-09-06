"""Point removal: TRIPS's confidence rule, and trippy's audit-aligned shade prune.

Module: trippy.train.prune
Purpose: decide *which* points a training run drops, and measure the
    statistic Jordan's complaint is phrased in ("the shadow is an object I
    have to move through") directly from the live parameters. Everything
    here is pure numpy and side-effect free -- it returns boolean keep
    masks and plain dicts. The parameter/optimiser surgery that acts on a
    keep mask lives in `trippy.train.trainer.Trainer._apply_keep_mask`,
    because only the Trainer owns the optimiser.

TRIPS's rule, as found in the source (third_party/TRIPS @ a59a65b6):
    `src/apps/train.cpp:846-851`
        indices_to_remove = torch::where(
            tex->confidence_value_of_point.squeeze()
                < params->points_adding_params.removal_confidence_cutoff, 1, 0).nonzero();
        if (indices_to_remove.size(0) > 0) { scene->RemovePoints(indices_to_remove); ... }
    `src/lib/models/NeuralTexture.h:42`
        confidence_value_of_point = sigmoid((10.f + narrowing_param_times_epoch) * confidence_raw);
    `src/apps/train.cpp:533-538` builds the schedule: epoch == start, then
        every epoch with (epoch - start) % interval == 0.
    `src/lib/data/Settings.h:403-406,427` code defaults
        start_removing_points_epoch = 200, point_removal_epoch_interval = 50,
        removal_confidence_cutoff = 0.3.
    `configs/train_normalnet.ini:8,130-134` shipped values: num_epochs = 600,
        start_removing_points_epoch = 2000 (never reached => removal OFF),
        point_removal_epoch_interval = 100, removal_confidence_cutoff = 0.500000119.
    That is the whole rule: a fixed threshold on the *effective* (post
    sigmoid-x10) confidence, on a fixed epoch schedule. There is no
    gradient, opacity-mass, visibility or error term in it.

TRIPS's point *adding* is deliberately NOT ported; see
docs/EXPERIMENTS.md "EXP-0010" and `docs/TRIPS_REFERENCE.md` for the full
finding. In one line: its default path shells out to a NeAT CT
reconstruction binary (`NeuralScene::AddNewRandomPointsFromCTHdr`, guarded
by `#ifdef COMPILE_WITH_VET`), and the in-tree fallback
(`AddNewRandomPointsInValuefilledBB`, `NeuralScene.cpp:1330-1373`) is dead
code -- it samples points in proportion to `t_cell_value`, and nothing in
the shipped renderer ever writes that buffer (`SetValueForCell` /
`GetPointerForValueForCell`, `NeuralPointCloudCuda.h:201-203`, have zero
callers), so it always adds exactly zero points.

Invariants: `build_shade_region`/`in_region`/`dark_mass_stats` are a
    field-for-field port of `~/Splats/tools/depthprior_shade_audit.py`
    (`build_region`, `in_region`, `audit`) in float64, so trippy's
    in-process number and that script's number on the exported PLY are the
    same number, not two approximations of one. The colour/opacity mapping
    is exact through `trippy.train.export`: the exporter writes
    `opacity = logit(clamp(conf))` and `f_dc = (clip(feat[:, :3], 0, 1) - 0.5)/SH_C0`,
    and the audit inverts both, so `conf` and `clip(feat[:, :3], 0, 1)` here
    are literally what the audit reads back.
Units: `xyz`/`ShadeView.C` are COLMAP world units; `d`, `znear`, `zfar` are
    the same; `fx`/`fy`/`cx`/`cy`/`W`/`H` are pixels of the *as-captured*
    image (the audit uses the raw COLMAP camera, and the projection test is
    invariant to a uniform image rescale, so a downscaled training cache
    does not change the region).
Related docs: docs/EXPERIMENTS.md "Shade audit", "EXP-0010: point removal";
    docs/TRIPS_REFERENCE.md Sec. 2 (confidence), Sec. 7 (schedule);
    docs/ARCHITECTURE.md "train/".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import (
    REC709_LUMA_WEIGHTS,
    SHADE_DARK_MASS_KEY_FMT,
    SHADE_DARK_N_KEY_FMT,
)


@dataclass(frozen=True)
class ShadeView:
    """One shade frame's contribution to the audit region (see module docstring).

    Attributes:
        name: image filename, e.g. "IMG_3830.jpg".
        R: (3, 3) float64 world->camera rotation. Rows are the camera axes
            in world coordinates, so `(p - C) @ R[2]` is camera-space depth
            (exactly how `depthprior_shade_audit.py` uses it).
        C: (3,) float64 camera centre in world coordinates, `-R.T @ t`.
        d: median camera-space depth of this frame's own observed sparse
            points (world units); 1.0 when the frame observes none, which
            is the audit's own fallback.
        fx, fy, cx, cy: pinhole intrinsics in pixels of the as-captured image.
        width, height: as-captured image size in pixels.
        nobs: how many of this frame's observations had positive depth.
        znear, zfar: the depth slab, `znear_frac * d` and `zfar_frac * d`.
    """

    name: str
    R: np.ndarray
    C: np.ndarray
    d: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    nobs: int
    znear: float
    zfar: float

    def to_json(self) -> dict:
        """The same five fields `depthprior_shade_audit.py --json-out` records per view."""
        return {
            "name": self.name,
            "d": self.d,
            "nobs": self.nobs,
            "znear": self.znear,
            "zfar": self.zfar,
        }


def build_shade_region(
    sparse_dir: str | Path,
    frames: list[str],
    znear_frac: float,
    zfar_frac: float,
) -> list[ShadeView]:
    """Build the audit's shade region from a COLMAP sparse model.

    Port of `depthprior_shade_audit.py`'s `build_region`, reading the model
    through `trippy.scene.colmap_io` (which handles both the binary
    `sparse/0` and the text `sparse_txt` layouts) instead of that script's
    text-only parser. On kk-coherent the cameras are `OPENCV`, whose
    `params[:4]` are `(fx, fy, cx, cy)` -- exactly what the script reads --
    so the two agree; `colmap_io.intrinsics` additionally does the right
    thing for camera models where a blind `params[:4]` would not.

    Args:
        sparse_dir: a COLMAP model directory (`sparse/0` or `sparse_txt`).
        frames: image filenames defining the region; every one must be
            registered in the model.
        znear_frac: near plane as a fraction of each frame's own median
            observed depth.
        zfar_frac: far plane, same units.

    Returns:
        One `ShadeView` per entry of `frames`, in that order.

    Raises:
        ValueError: `frames` is empty, or a frame is not registered in the
            model (the audit raises `SystemExit` here; a library raises).
    """
    # Deferred: trippy.scene.colmap_io pulls in trippy.geom, which the
    # (numpy-only) rest of this module does not need.
    from trippy.scene.colmap_io import intrinsics, load_colmap_model

    if not frames:
        raise ValueError("build_shade_region needs at least one frame")

    scene = load_colmap_model(Path(sparse_dir))
    by_name = scene.images_by_name()
    missing = [f for f in frames if f not in by_name]
    if missing:
        raise ValueError(f"frames not registered in {sparse_dir}: {missing}")

    views: list[ShadeView] = []
    for name in frames:
        image = by_name[name]
        camera = scene.cameras[image.camera_id]
        R = _qvec2R(image.qvec)
        C = -R.T @ np.asarray(image.tvec, dtype=np.float64)

        ids = image.point3D_ids
        observed = np.array(
            [scene.points3D[int(i)].xyz for i in ids[ids >= 0] if int(i) in scene.points3D],
            dtype=np.float64,
        )
        if observed.size:
            zc = (observed - C) @ R[2]
            zc = zc[zc > 0]
        else:
            zc = np.zeros(0, dtype=np.float64)
        d = float(np.median(zc)) if zc.size else 1.0

        fx, fy, cx, cy = intrinsics(camera)
        views.append(
            ShadeView(
                name=name,
                R=R,
                C=C,
                d=d,
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                width=int(camera.width),
                height=int(camera.height),
                nobs=int(zc.size),
                znear=znear_frac * d,
                zfar=zfar_frac * d,
            )
        )
    return views


def _qvec2R(qvec: np.ndarray) -> np.ndarray:
    """(qw, qx, qy, qz) -> (3, 3) rotation, float64 (see trippy.geom.xform_a.qvec2R)."""
    from trippy.geom.xform_a import qvec2R

    return np.asarray(qvec2R(np.asarray(qvec, dtype=np.float64)), dtype=np.float64)


def in_region(views: list[ShadeView], xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which points sit in the union of the shade frames' near depth slabs.

    Port of `depthprior_shade_audit.py`'s `in_region`, including its
    half-open pixel test (`0 <= u < W`) and its strict depth test
    (`znear < z < zfar`).

    Args:
        views: from `build_shade_region`.
        xyz: (N, 3) world-frame positions.

    Returns:
        `(inside, zfrac)`: `inside` is an (N,) bool mask; `zfrac` is (N,)
        float64 holding, per point, the smallest `z / d` over the views
        that contain it (`inf` for points in no view) -- the audit's
        distance-band profile axis.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    inside = np.zeros(n, dtype=bool)
    zfrac = np.full(n, np.inf, dtype=np.float64)
    for v in views:
        rel = xyz - v.C
        z = rel @ v.R[2]
        ok = (z > v.znear) & (z < v.zfar)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        x = rel[idx] @ v.R[0]
        y = rel[idx] @ v.R[1]
        zi = z[idx]
        u = v.fx * x / zi + v.cx
        vv = v.fy * y / zi + v.cy
        vis = (u >= 0) & (u < v.width) & (vv >= 0) & (vv < v.height)
        hit = idx[vis]
        inside[hit] = True
        zfrac[hit] = np.minimum(zfrac[hit], zi[vis] / v.d)
    return inside, zfrac


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of (N, 3) linear RGB, matching the audit's own weights."""
    rgb = np.asarray(rgb, dtype=np.float64)
    w = np.asarray(REC709_LUMA_WEIGHTS, dtype=np.float64)
    return rgb @ w


def dark_mass_stats(
    views: list[ShadeView],
    xyz: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray,
    lum_threshold: float,
) -> dict:
    """The shade audit's headline numbers, computed in-process from live parameters.

    Args:
        views: from `build_shade_region`.
        xyz: (N, 3) world-frame positions.
        rgb: (N, 3) linear base colour in [0, 1] (the exporter's
            `clip(feat[:, :3], 0, 1)`; the audit reads exactly this back).
        conf: (N,) effective confidence in (0, 1) -- the exporter's
            `opacity` field is `logit(conf)`, which the audit inverts with
            a sigmoid, so this *is* the audit's opacity.
        lum_threshold: "dark" luminance cutoff.

    Returns:
        `{"n", "n_in_region", "mass_in_region", "dark_mass_lum<t>",
        "dark_n_lum<t>", "dark_mass_fraction"}`. `dark_mass_fraction` is
        `dark_mass / max(mass_in_region, 1e-9)` -- the percentage the audit
        prints and the number EXP-0010 reports over training. Non-finite
        positions are excluded from the region, as in the audit.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    inside, _zfrac = in_region(views, xyz)
    inside &= np.isfinite(xyz).all(axis=1)

    lum = luminance(rgb)
    dark = inside & (lum < lum_threshold)
    mass_in_region = float(conf[inside].sum())
    dark_mass = float(conf[dark].sum())
    key = f"{lum_threshold:g}"
    return {
        "n": int(xyz.shape[0]),
        "n_in_region": int(inside.sum()),
        "mass_in_region": mass_in_region,
        SHADE_DARK_MASS_KEY_FMT.format(t=key): dark_mass,
        SHADE_DARK_N_KEY_FMT.format(t=key): int(dark.sum()),
        "dark_mass_fraction": dark_mass / max(mass_in_region, 1e-9),
    }


def apply_min_points(keep: np.ndarray, conf: np.ndarray, min_points: int) -> np.ndarray:
    """Widen `keep` so at least `min_points` points survive (trippy addition, not TRIPS).

    When the proposed mask would leave fewer than `min_points` points, the
    `min_points` highest-confidence points overall are kept instead -- a
    deterministic, order-independent fallback (ties broken by index, via
    `np.argsort`'s stable "kind" on the negated confidence). When
    `min_points` exceeds the cloud size, everything is kept.

    Args:
        keep: (N,) bool, the proposed keep mask.
        conf: (N,) effective confidence, used to rank survivors.
        min_points: floor on the surviving count.

    Returns:
        An (N,) bool mask with `sum >= min(min_points, N)`.
    """
    n = keep.shape[0]
    if int(keep.sum()) >= min(min_points, n):
        return keep
    order = np.argsort(-np.asarray(conf, dtype=np.float64), kind="stable")
    widened = np.zeros(n, dtype=bool)
    widened[order[: min(min_points, n)]] = True
    return widened


def removal_keep_mask(conf: np.ndarray, conf_threshold: float, min_points: int) -> np.ndarray:
    """TRIPS's rule: keep every point whose effective confidence is NOT below the cutoff.

    `train.cpp:846-851` selects `confidence_value_of_point < cutoff` for
    removal, so the keep mask is the complement, `conf >= cutoff`. The
    `min_points` floor is trippy's, applied afterwards.

    Args:
        conf: (N,) effective confidence, i.e. `sigmoid(10 * raw_conf)`.
        conf_threshold: TRIPS's `removal_confidence_cutoff`.
        min_points: floor on the surviving count (see `apply_min_points`).

    Returns:
        (N,) bool keep mask.
    """
    conf = np.asarray(conf, dtype=np.float64)
    keep = conf >= conf_threshold
    return apply_min_points(keep, conf, min_points)


def shade_prune_keep_mask(
    inside: np.ndarray,
    lum: np.ndarray,
    conf: np.ndarray,
    lum_threshold: float,
    conf_threshold: float,
    min_points: int,
) -> np.ndarray:
    """trippy's audit-aligned rule: drop in-region AND dark AND low-confidence points.

    This deliberately removes the thing the shade audit measures. It is a
    heuristic aimed at a metric, not a claim that those points are wrong;
    a run using it must be reported next to its held-out shade PSNR (see
    `trippy.train.prune_config.ShadePruneConfig` and docs/EXPERIMENTS.md
    "EXP-0010").

    Args:
        inside: (N,) bool, from `in_region`.
        lum: (N,) Rec.709 luminance of the base colour.
        conf: (N,) effective confidence.
        lum_threshold: "dark" cutoff.
        conf_threshold: confidence cutoff.
        min_points: floor on the surviving count (see `apply_min_points`).

    Returns:
        (N,) bool keep mask.
    """
    conf = np.asarray(conf, dtype=np.float64)
    drop = inside & (np.asarray(lum, dtype=np.float64) < lum_threshold) & (conf < conf_threshold)
    return apply_min_points(~drop, conf, min_points)
