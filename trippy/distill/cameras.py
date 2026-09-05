"""Near-path camera interpolation for the design-B distillation pipeline.

Module: trippy.distill.cameras
Purpose: build the camera set trippy.distill.render_set renders the TRIPS
    network output at: every registered training camera ("anchor") plus a
    small number of cameras interpolated between each *consecutive*
    (capture-order) pair of anchors -- slerp for rotation, lerp for the
    camera centre. This is the task brief's "near-path interpolated
    cameras" step of the design-B pipeline (docs/SPEC.md D2).
Honesty guard: AGENTS.md's honesty rule ("photographed vs inferred pixels
    must remain distinguishable") extends here to camera *poses*, not just
    pixels -- "only cameras close to the capture path; no far off-path
    invention". Two registered images are only bridged with interpolated
    poses when (a) they share one physical camera (camera_id) and (b) their
    centre-to-centre distance does not exceed `max_jump_multiplier` times
    the scene's own median consecutive-pair distance (DISTILL_MAX_JUMP_
    MULTIPLIER). A pair failing either check is not "consecutive along one
    continuous walk" (different lens, a registration gap, two separate
    sweeps of the same scene) and is skipped, recorded in
    `DistillCameraPlan.skipped_pairs` rather than silently interpolated
    through. Every interpolated pose is a linear blend between two real,
    photographed anchors, so by construction it can never be farther from
    the nearest anchor than the anchor-to-anchor distance itself.
Invariants: numpy only (no torch import), matching trippy.render.dolly/
    trippy.render.offpath's own convention for pure camera-geometry helpers.
    Consecutive order is capture order: `SceneDataset.names` / a scene's
    registered image names sorted lexicographically, which for a
    sequentially-numbered capture (IMG_3700.jpg, IMG_3701.jpg, ...) is the
    order the camera was actually walked.
Related docs: docs/EXPERIMENTS.md "Distillation (design B)"; docs/SPEC.md D2;
    trippy.render.dolly (CameraPose, camera_center, scaled_intrinsics --
    reused here, not reimplemented).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trippy.constants import (
    DISTILL_DEFAULT_INTERP_K,
    DISTILL_INTERP_NAME_FMT,
    DISTILL_MAX_JUMP_MULTIPLIER,
    DISTILL_SLERP_NEAR_IDENTICAL_DOT,
)
from trippy.geom import xform_a
from trippy.render.dolly import CameraPose, camera_center, scaled_intrinsics
from trippy.scene import colmap_io
from trippy.scene.dataset import resolve_sparse_dir


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (qw, qx, qy, qz), COLMAP convention.

    The inverse of `trippy.geom.xform_a.qvec2R`. Not added to xform_a itself:
    AGENTS.md section 7's "implement transforms twice independently" rule is
    about xform_a/xform_b (the numpy/torch camera-geometry twins with their
    own agreement test), and this distill-only helper has no torch
    counterpart to keep in sync with -- it belongs with the rest of this
    module's camera-pose construction instead.

    Uses the standard robust branch selection on the rotation matrix's trace
    (Shepperd's method), so it stays numerically stable for every proper
    rotation, not just ones close to identity.

    Args:
        R: shape (3, 3), a proper rotation matrix (orthonormal, det=+1).

    Returns:
        shape (4,), float64, unit-norm (qw, qx, qy, qz) with qw >= 0 (a
        deterministic sign choice -- q and -q represent the same rotation,
        and fixing the sign avoids a spurious "long way round" in `slerp`
        when this output feeds back into it).
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions at fraction `t`.

    Always takes the "short way round" (negates `q1` if the dot product with
    `q0` is negative -- q and -q are the same rotation, and the long way
    round would spin the interpolated camera the wrong direction). Falls
    back to a normalised linear interpolation when the two quaternions are
    almost identical, avoiding a 0/0 in `sin(theta)/theta`.

    Args:
        q0, q1: shape (4,), (qw, qx, qy, qz), need not be pre-normalised.
        t: interpolation fraction in [0, 1] (0 -> q0, 1 -> q1).

    Returns:
        shape (4,), float64, unit-norm.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > DISTILL_SLERP_NEAR_IDENTICAL_DOT:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta0 = np.arccos(dot)
    theta = theta0 * t
    q_perp = q1 - q0 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def image_filename(pose_name: str) -> str:
    """The rendered-image filename `trippy.distill.render_set` writes for one pose.

    Anchor poses carry the original registered image name (e.g.
    "IMG_3830.jpg") as `CameraPose.name`; interpolated poses already carry
    an extension-free synthetic name (`DISTILL_INTERP_NAME_FMT`). Either
    way the on-disk render is always a PNG (`trippy.render.candidate`'s own
    `net.png` output, copied/renamed here), so this always returns
    `f"{stem}.png"` regardless of the pose's own name having an extension.
    """
    return f"{Path(pose_name).stem}.png"


@dataclass(frozen=True)
class SkippedPair:
    """One consecutive anchor pair the honesty guard refused to interpolate.

    Attributes:
        name_a, name_b: the two (consecutive, by capture order) registered
            image names.
        distance: world-unit centre-to-centre distance between them.
        reason: human-readable reason (different camera_id, or the jump-
            distance guard).
    """

    name_a: str
    name_b: str
    distance: float
    reason: str


@dataclass(frozen=True)
class DistillCameraPlan:
    """The full camera set for one distillation render pass.

    Attributes:
        anchors: one CameraPose per registered training image, at its own
            real (COLMAP) pose -- no pose-refinement delta applied, the
            same convention `trippy.render.dolly`/`trippy.render.offpath`
            already use for arbitrary poses.
        interpolated: CameraPoses between consecutive anchor pairs that
            passed the honesty guard (`image_name=None` -- no natural
            source photo, so renderers fall back to mean exposure/white-
            balance, same as any off-path pose).
        skipped_pairs: consecutive anchor pairs the honesty guard refused.
        median_consecutive_distance: median centre-to-centre distance over
            every consecutive anchor pair (0.0 if fewer than 2 anchors).
        jump_threshold: `median_consecutive_distance * max_jump_multiplier`
            (`inf` when the median is 0, e.g. a single-anchor scene).
    """

    anchors: list[CameraPose] = field(default_factory=list)
    interpolated: list[CameraPose] = field(default_factory=list)
    skipped_pairs: list[SkippedPair] = field(default_factory=list)
    median_consecutive_distance: float = 0.0
    jump_threshold: float = 0.0

    @property
    def all_poses(self) -> list[CameraPose]:
        """Anchors followed by interpolated poses, in that order."""
        return list(self.anchors) + list(self.interpolated)


def build_distill_camera_plan(
    scene_root: str | Path,
    width: int,
    names: list[str] | None = None,
    k: int = DISTILL_DEFAULT_INTERP_K,
    max_jump_multiplier: float = DISTILL_MAX_JUMP_MULTIPLIER,
) -> DistillCameraPlan:
    """Build every anchor + honesty-guarded interpolated CameraPose for a scene.

    Args:
        scene_root: COLMAP scene root (`images/` + `sparse/0` or `sparse_txt`).
        width: destination pinhole image width in pixels (see
            `trippy.render.dolly.scaled_intrinsics`); height keeps each
            source camera's own aspect ratio.
        names: registered image names to use as anchors, in this order
            (consecutive pairs in this list are interpolation candidates);
            None uses every registered image, sorted by name (capture
            order for a sequentially-numbered capture).
        k: interpolated cameras generated between each consecutive pair
            that passes the honesty guard.
        max_jump_multiplier: honesty-guard threshold, see module docstring.

    Returns:
        A `DistillCameraPlan`.

    Raises:
        KeyError: any name in `names` is not registered under `scene_root`.
    """
    sparse_dir = resolve_sparse_dir(Path(scene_root))
    colmap_scene = colmap_io.load_colmap_model(sparse_dir)
    images_by_name = colmap_scene.images_by_name()

    all_names = sorted(images_by_name.keys()) if names is None else list(names)
    missing = sorted(n for n in set(all_names) if n not in images_by_name)
    if missing:
        raise KeyError(f"names not registered under {scene_root}: {missing}")

    anchors: list[CameraPose] = []
    centers: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    camera_ids: dict[str, int] = {}

    for name in all_names:
        im = images_by_name[name]
        cam = colmap_scene.cameras[im.camera_id]
        K, height, out_width = scaled_intrinsics(cam, width)
        R = xform_a.qvec2R(im.qvec)
        C = camera_center(R, im.tvec)
        t = -R @ C
        anchors.append(
            CameraPose(name=name, R=R.copy(), t=t.copy(), K=K.copy(), image_hw=(height, out_width), image_name=name)
        )
        centers[name] = C
        rotations[name] = R
        camera_ids[name] = im.camera_id

    consecutive_pairs = list(itertools.pairwise(all_names))
    pair_distances = [float(np.linalg.norm(centers[b] - centers[a])) for a, b in consecutive_pairs]
    median_d = float(np.median(pair_distances)) if pair_distances else 0.0
    threshold = median_d * max_jump_multiplier if median_d > 0.0 else float("inf")

    interpolated: list[CameraPose] = []
    skipped: list[SkippedPair] = []

    for (name_a, name_b), distance in zip(consecutive_pairs, pair_distances, strict=True):
        if camera_ids[name_a] != camera_ids[name_b]:
            skipped.append(SkippedPair(name_a, name_b, distance, "different camera_id"))
            continue
        if distance > threshold:
            skipped.append(
                SkippedPair(
                    name_a,
                    name_b,
                    distance,
                    f"distance exceeds honesty guard ({max_jump_multiplier:g}x median {median_d:.4g})",
                )
            )
            continue
        if k <= 0:
            continue

        qa = rotmat_to_qvec(rotations[name_a])
        qb = rotmat_to_qvec(rotations[name_b])
        Ca, Cb = centers[name_a], centers[name_b]
        cam = colmap_scene.cameras[camera_ids[name_a]]
        K, height, out_width = scaled_intrinsics(cam, width)
        stem_a, stem_b = Path(name_a).stem, Path(name_b).stem

        for j in range(1, k + 1):
            frac = j / (k + 1)
            R = xform_a.qvec2R(slerp(qa, qb, frac))
            C = Ca + frac * (Cb - Ca)
            t = -R @ C
            pose_name = DISTILL_INTERP_NAME_FMT.format(a=stem_a, b=stem_b, j=j)
            interpolated.append(
                CameraPose(name=pose_name, R=R, t=t, K=K.copy(), image_hw=(height, out_width), image_name=None)
            )

    return DistillCameraPlan(
        anchors=anchors,
        interpolated=interpolated,
        skipped_pairs=skipped,
        median_consecutive_distance=median_d,
        jump_threshold=threshold,
    )
