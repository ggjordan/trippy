"""HybridConfig: the `hybrid:` block of a `TrainConfig` (design A, splats + TRIPS).

Module: trippy.hybrid.config_a
Purpose: describe, in one YAML-round-trippable dataclass, how a Gaussian-splat
    render is fed to the TRIPS U-Net *alongside* the TRIPS point pyramid. Kept
    in its own module (dataclasses + trippy.constants only, no torch/numpy/PIL)
    so `trippy.train.config` -- which is imported by `trippy eval --help` and by
    every checkpoint load -- stays as cheap to import as it was before design A
    existed. The torch machinery that consumes this config lives in
    `trippy.hybrid.gaussian_input`.
Invariants:
    - `enabled = False` (the default) must leave every downstream consumer on
      exactly its pre-design-A code path; `trippy.train.trainer` asserts this by
      building no hybrid state at all in that case.
    - `channels` is canonicalised to `HYBRID_A_CHANNEL_ORDER`'s order on
      construction, so the trained channel layout depends only on the *set* of
      groups requested. Reordering the YAML list can never invalidate a
      checkpoint.
    - `depth_scale` is a *resolved* value: None in a hand-written config means
      "measure the scene's median camera-to-Gaussian depth at Trainer
      construction and write the number back here", so the checkpoint records
      the exact normaliser its weights were trained with.
Units: `depth_scale` is in COLMAP world units (the same units gsrender's depth
    output and `PointSet.xyz` use). The normalised depth channel is unitless.
Related docs: docs/EXPERIMENTS.md "Hybrid design A"; docs/ARCHITECTURE.md
    "hybrid/"; experiments/EXP-0009-hybrid-a/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trippy.constants import (
    HYBRID_A_CHANNEL_ORDER,
    HYBRID_A_CHANNEL_WIDTHS,
    HYBRID_A_DEFAULT_DROPOUT_P,
    HYBRID_A_DEFAULT_MASK_BY_ALPHA,
    HYBRID_A_MODES,
    HYBRID_C_GSRENDER_MAX_HW,
    HYBRID_C_GSRENDER_MIN_OPACITY,
)

#: How a name with no render triple on disk is handled (`HybridConfig.missing`).
HYBRID_A_MISSING_POLICIES = ("zeros", "error")


def gaussian_channel_count(channels: list[str] | tuple[str, ...]) -> int:
    """Total width, in channels, of the Gaussian block described by `channels`.

    Args:
        channels: group names, each in `HYBRID_A_CHANNEL_ORDER`.

    Returns:
        `sum(HYBRID_A_CHANNEL_WIDTHS[g] for g in channels)` -- e.g. 5 for
        `["rgb", "alpha", "depth"]`, 4 for `["rgb", "alpha"]`.
    """
    return sum(HYBRID_A_CHANNEL_WIDTHS[group] for group in channels)


@dataclass
class HybridConfig:
    """The `hybrid:` block of a training config (see module docstring).

    Attributes:
        enabled: master switch. False (the default) is a hard no-op: no
            renders are read, `NetworkConfig.num_input_channels` is
            unchanged, and the trainer's behaviour is bit-identical to a
            build without design A.
        renders_dir: directory of `<stem>.png` / `<stem>.depth.npy` /
            `<stem>.alpha.npy` triples written by
            `trippy.hybrid.render_splat_views` (design C's renderer, reused
            verbatim). Relative paths resolve against the process CWD, the
            same rule `TrainConfig.run_dir` already uses.
        channels: which Gaussian groups to concatenate; canonicalised to
            `HYBRID_A_CHANNEL_ORDER` order on construction.
        mode: `"all_levels"` (default) or `"concat_level0"` -- see
            `trippy.constants.HYBRID_A_MODES` for why all_levels is the
            default.
        dropout_gaussian_p: probability that a *training* crop has its
            Gaussian channels zeroed wholesale (ablation 1). Never applied
            at eval/report time.
        mask_by_alpha: multiply the Gaussian rgb by its own alpha before
            concatenation (ablation 2).
        depth_scale: world-unit divisor for the depth channel; None means
            "measure it" (see module docstring). Ignored when "depth" is
            not in `channels`.
        missing: what to do when a frame has no render triple on disk --
            "zeros" (treat as a fully-dropped-out frame; the default, and
            what lets a half-finished render shard still train) or "error".
        ply_path: the Gaussian PLY to render *live* at poses that have no
            precomputed render (candidate report / dolly / off-path). Empty
            means "no live rendering": such poses get zeros.
        gsrender_tools_dir: directory holding Splats' `gsrender.py`; empty
            uses `trippy.hybrid.render_splat_views.DEFAULT_GSRENDER_TOOLS_DIR`.
        gsrender_max_hw, gsrender_min_opacity: forwarded to `gsrender.render`
            for live renders; defaults match design C's, so a live render
            and a precomputed one are produced by the same call.
    """

    enabled: bool = False
    renders_dir: str = ""
    channels: list[str] = field(default_factory=lambda: list(HYBRID_A_CHANNEL_ORDER))
    mode: str = HYBRID_A_MODES[0]
    dropout_gaussian_p: float = HYBRID_A_DEFAULT_DROPOUT_P
    mask_by_alpha: bool = HYBRID_A_DEFAULT_MASK_BY_ALPHA
    depth_scale: float | None = None
    missing: str = HYBRID_A_MISSING_POLICIES[0]
    ply_path: str = ""
    gsrender_tools_dir: str = ""
    gsrender_max_hw: int = HYBRID_C_GSRENDER_MAX_HW
    gsrender_min_opacity: float = HYBRID_C_GSRENDER_MIN_OPACITY

    def __post_init__(self) -> None:
        unknown = [g for g in self.channels if g not in HYBRID_A_CHANNEL_WIDTHS]
        if unknown:
            raise ValueError(
                f"hybrid.channels entries must be in {list(HYBRID_A_CHANNEL_ORDER)}, got {unknown}"
            )
        requested = set(self.channels)
        if not requested:
            raise ValueError("hybrid.channels must not be empty")
        # Canonical order, deduplicated (see module docstring "Invariants").
        self.channels = [g for g in HYBRID_A_CHANNEL_ORDER if g in requested]
        if self.mode not in HYBRID_A_MODES:
            raise ValueError(f"hybrid.mode must be one of {list(HYBRID_A_MODES)}, got {self.mode!r}")
        if not 0.0 <= self.dropout_gaussian_p <= 1.0:
            raise ValueError(
                f"hybrid.dropout_gaussian_p must be in [0, 1], got {self.dropout_gaussian_p}"
            )
        if self.missing not in HYBRID_A_MISSING_POLICIES:
            raise ValueError(
                f"hybrid.missing must be one of {list(HYBRID_A_MISSING_POLICIES)}, got {self.missing!r}"
            )
        if self.depth_scale is not None and not self.depth_scale > 0.0:
            raise ValueError(f"hybrid.depth_scale must be positive, got {self.depth_scale}")
        if self.enabled and not self.renders_dir:
            raise ValueError("hybrid.enabled is true but hybrid.renders_dir is empty")

    @property
    def num_channels(self) -> int:
        """Width of the Gaussian block appended to every U-Net input level."""
        return gaussian_channel_count(self.channels)

    @property
    def wants_depth(self) -> bool:
        """True when a normalised depth channel is part of the block."""
        return "depth" in self.channels

    def channel_slice(self, group: str) -> slice:
        """Where `group` sits *within the Gaussian block* (0 == first Gaussian channel).

        Add `TrainConfig.feature_channels` to both ends to get the slice
        inside a full U-Net input level (`[TRIPS features | Gaussian block]`).
        """
        if group not in self.channels:
            raise ValueError(f"{group!r} is not in hybrid.channels ({self.channels})")
        start = 0
        for name in self.channels:
            width = HYBRID_A_CHANNEL_WIDTHS[name]
            if name == group:
                return slice(start, start + width)
            start += width
        raise AssertionError("unreachable: group was checked to be in self.channels")
