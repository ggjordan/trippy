"""PointParams / PoseParams: the trainable state a TRIPS-style trainer owns.

Module: trippy.train.params
Invariants: `PointParams` stores the *raw* (pre-activation) parameters --
    `xyz`, `raw_size`, `raw_conf`, `feat` -- as `nn.Parameter`s, exposing the
    effective (post-activation) values only through `size()`/`conf()`
    methods, exactly mirroring TRIPS's own split between
    `NeuralPointCloudCuda`'s raw `t_point_size` and the softplus'd value
    used at render time (docs/TRIPS_REFERENCE.md Sec. 2). `provenance` is a
    buffer (uint8), never a Parameter -- it is metadata, not something
    gradient descent should touch. `PoseParams` holds one 6-vector twist per
    image, composed onto that image's fixed COLMAP pose via
    `trippy.geom.xform_b.compose` at render time -- gradients flow into the
    delta through the projection, never into the frozen COLMAP pose itself.
Related docs: docs/TRIPS_REFERENCE.md Sec. 2 (point parametrisation: size
    init via inverse-softplus, confidence via sigmoid(10*raw) -- the x10
    decision is trippy.constants.CONF_SIGMOID_SCALE); trippy.geom.xform_b
    (se3_exp/compose, the SE(3) pose-delta convention).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from trippy.constants import (
    CONF_SIGMOID_SCALE,
    EXPORT_OPACITY_CLAMP_EPS,
    TRAIN_DEFAULT_FEATURE_CHANNELS,
    TRAIN_DEFAULT_SEED,
    TRAIN_FEAT_EXTRA_INIT_STD,
    TRAIN_SOFTPLUS_THRESHOLD,
)
from trippy.geom import xform_b
from trippy.points.source import PointSet


def inverse_softplus(x: torch.Tensor, threshold: float = TRAIN_SOFTPLUS_THRESHOLD) -> torch.Tensor:
    """Inverse of `torch.nn.functional.softplus` (beta=1, matching its `threshold` default).

    `softplus(inverse_softplus(x)) == x` for `x > 0`. Above `threshold`,
    softplus is (by construction, both here and in `functional.softplus`)
    the identity, so the inverse is too -- this must use the exact same
    threshold `functional.softplus` does or the round-trip breaks for large
    inputs (docs/TRIPS_REFERENCE.md Sec. 2: NeuralPointCloudCuda.cpp:19-24,
    "beta=1, threshold=20").

    Args:
        x: strictly positive tensor (world-unit point size).
        threshold: above this value, softplus/inverse-softplus is treated
            as the identity (avoids `log(expm1(x))` overflowing for large x).

    Returns:
        Tensor of the same shape, the raw pre-softplus value.
    """
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def logit(p: torch.Tensor, eps: float = EXPORT_OPACITY_CLAMP_EPS) -> torch.Tensor:
    """Inverse sigmoid, clamping `p` away from {0, 1} first (finite result)."""
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


class PointParams(nn.Module):
    """Trainable per-point state: position, size, confidence, feature vector.

    Attributes (all `nn.Parameter`, all float32):
        xyz: (N, 3) world-frame position, initialised from `point_set.xyz`.
        raw_size: (N,) pre-softplus size, initialised so
            `softplus(raw_size) == point_set.size0` (see `inverse_softplus`).
        raw_conf: (N,) pre-sigmoid*10 confidence, initialised so
            `sigmoid(CONF_SIGMOID_SCALE * raw_conf) == point_set.conf0`.
        feat: (N, feature_channels) per-point feature vector. Channels
            `0:3` are initialised to `point_set.rgb0` (so the untrained U-Net
            immediately sees real colour); any remaining channels are small
            Gaussian noise (`TRAIN_FEAT_EXTRA_INIT_STD`), NOT TRIPS's own
            full-range `Uniform(0,1)` texture init (docs/TRIPS_REFERENCE.md
            Sec. 2) -- a deliberate trippy choice per this task's brief.
        provenance: (N,) uint8 buffer (not trained), from `point_set.provenance`.
        init_conf: (N,) float32 buffer (not trained), a snapshot of `conf()`
            at construction (equal to `point_set.conf0`). Never touched by
            training or by an optimiser step; it exists so
            `trippy.train.prune`'s `mode: relative` confidence test
            (`trippy.train.prune_config.PointRemovalConfig.mode`) can
            threshold a point's CURRENT confidence against its OWN starting
            value rather than a fixed absolute cutoff -- see that module
            for why trippy needs this and TRIPS does not. Like
            `provenance`, it is index-selected (never optimizer-shrunk)
            when `Trainer._apply_keep_mask` drops points, so entry `i`
            keeps meaning "point `i`'s confidence when it was created", and
            it round-trips through `state_dict`/`load_state_dict` like any
            other buffer.
        bbox_min, bbox_max: (3,) buffers (not trained), the *initial* xyz
            bounding box -- used by the trainer's extent penalty.
    """

    def __init__(
        self,
        point_set: PointSet,
        feature_channels: int = TRAIN_DEFAULT_FEATURE_CHANNELS,
        seed: int = TRAIN_DEFAULT_SEED,
    ) -> None:
        super().__init__()
        if feature_channels < 3:
            raise ValueError(f"feature_channels must be >= 3 (rgb0 seeds channels 0:3), got {feature_channels}")

        xyz = torch.from_numpy(point_set.xyz.copy()).to(torch.float32)
        size0 = torch.from_numpy(point_set.size0.copy()).to(torch.float32).clamp(min=torch.finfo(torch.float32).eps)
        conf0 = torch.from_numpy(point_set.conf0.copy()).to(torch.float32)
        rgb0 = torch.from_numpy(point_set.rgb0.copy()).to(torch.float32)
        n = xyz.shape[0]

        self.xyz = nn.Parameter(xyz)
        self.raw_size = nn.Parameter(inverse_softplus(size0))
        self.raw_conf = nn.Parameter(logit(conf0) / CONF_SIGMOID_SCALE)

        feat = torch.zeros(n, feature_channels, dtype=torch.float32)
        feat[:, :3] = rgb0
        if feature_channels > 3:
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(n, feature_channels - 3, generator=generator) * TRAIN_FEAT_EXTRA_INIT_STD
            feat[:, 3:] = noise
        self.feat = nn.Parameter(feat)

        self.register_buffer("provenance", torch.from_numpy(point_set.provenance.copy()))
        # Snapshot BEFORE any optimiser step ever touches raw_conf, so this is exactly
        # point_set.conf0 (up to the logit/sigmoid round-trip's clamp epsilon) -- the
        # "initial confidence" trippy's relative point-removal rule needs (see docstring).
        self.register_buffer("init_conf", conf0.clone())
        self.register_buffer("bbox_min", xyz.min(dim=0).values.clone())
        self.register_buffer("bbox_max", xyz.max(dim=0).values.clone())

    def size(self) -> torch.Tensor:
        """Effective (post-softplus) world-unit point size, (N,)."""
        return functional.softplus(self.raw_size)

    def conf(self) -> torch.Tensor:
        """Effective (post-sigmoid) confidence in (0, 1), (N,).

        `sigmoid(CONF_SIGMOID_SCALE * raw_conf)`, matching TRIPS's
        `NeuralTexture.h:42` exactly (docs/TRIPS_REFERENCE.md Sec. 2's
        "keep the x10" decision -- see trippy.constants.CONF_SIGMOID_SCALE).
        """
        return torch.sigmoid(CONF_SIGMOID_SCALE * self.raw_conf)

    def __len__(self) -> int:
        return self.xyz.shape[0]


class PoseParams(nn.Module):
    """One learnable SE(3) pose-refinement twist per training image.

    Attributes:
        delta: (n_images, 6) `nn.Parameter`, zero-initialised (identity
            refinement at the start of training, matching TRIPS: pose
            refinement starts from the COLMAP pose unmodified).
    """

    def __init__(self, n_images: int) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_images, 6))

    def compose_pose(self, index: int, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply image `index`'s learned delta to a fixed COLMAP world->camera pose.

        Args:
            index: row into `self.delta`.
            R: (3, 3) world->camera rotation (the fixed COLMAP pose).
            t: (3,) world->camera translation (the fixed COLMAP pose).

        Returns:
            (R_refined, t_refined), each composed via
            `trippy.geom.xform_b.compose` so gradients flow from the
            rendered image, through the projection, into `self.delta[index]`
            -- never into `R`/`t` themselves (docs/SPEC.md "Technical
            design": learnable SE(3) pose delta).
        """
        pose = torch.eye(4, dtype=R.dtype, device=R.device)
        pose[:3, :3] = R
        pose[:3, 3] = t
        refined = xform_b.compose(pose, self.delta[index].to(dtype=R.dtype))
        return refined[:3, :3], refined[:3, 3]

    def __len__(self) -> int:
        return self.delta.shape[0]
