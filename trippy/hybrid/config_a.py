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
    HYBRID_A_GATE_PRIOR_DEFAULT_TARGET,
    HYBRID_A_GATE_PRIOR_DEFAULT_WEIGHT,
    HYBRID_A_GATE_SCALE_DEFAULT,
    HYBRID_A_GATE_SCALE_MAX,
    HYBRID_A_GATE_SCALE_MIN,
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
class GateConfig:
    """The `hybrid.gate:` block: the explicit splat-vs-TRIPS blend weight.

    `enabled: false` (the default) is a hard no-op -- the U-Net keeps its three
    output channels and no blend is ever computed, so every queued run and every
    existing hybrid checkpoint is bit-identical to a build without the gate. See
    `trippy.hybrid.gate` for the maths and why the blend happens after the tone
    mapper.

    Attributes:
        enabled: master switch for the extra output channel and the blend.
    """

    enabled: bool = False


@dataclass
class GatePriorConfig:
    """The `hybrid.gate_prior:` block: an optional pull on the gate's MEAN.

    Off by default (`weight = 0`). When on, `weight * (mean(g) - target) ** 2` is
    added to every training step's loss. It constrains the average only -- the
    per-pixel map, which is the whole point of the gate, is left to the image
    losses. Use it to ask "what does this scene look like if it has to lean 20%
    on the splat?", never to decide the answer.

    Attributes:
        target: the mean gate to pull towards, in [0, 1] (1 = all splat).
        weight: penalty weight; <= 0 disables the term entirely.
    """

    target: float = HYBRID_A_GATE_PRIOR_DEFAULT_TARGET
    weight: float = HYBRID_A_GATE_PRIOR_DEFAULT_WEIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.target <= 1.0:
            raise ValueError(f"hybrid.gate_prior.target must be in [0, 1], got {self.target}")

    @property
    def active(self) -> bool:
        """True when the term actually contributes to the loss."""
        return self.weight > 0.0


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
        gate: the blend gate (`GateConfig`); `enabled: false` by default and a
            hard no-op then. Requires `enabled` and `"rgb" in channels`.
        gate_prior: optional mean-gate regulariser (`GatePriorConfig`); off by
            default (`weight: 0`).
        gate_scale: the post-training knob, in
            `[HYBRID_A_GATE_SCALE_MIN, HYBRID_A_GATE_SCALE_MAX]`. Multiplies the
            trained gate at eval/render time (the product is clamped back into
            [0, 1]): 0 = pure TRIPS, 1 = as trained, 2 = push towards the splat.
            Recorded in the checkpoint so a run's default mix is reproducible;
            `trippy eval --gate-scale` / `candidate-report --gate-scale` and the
            viewer's Blend panel override it per call.
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
    gate: GateConfig = field(default_factory=GateConfig)
    gate_prior: GatePriorConfig = field(default_factory=GatePriorConfig)
    gate_scale: float = HYBRID_A_GATE_SCALE_DEFAULT

    def __post_init__(self) -> None:
        # YAML round-trip: `TrainConfig.from_dict` hands nested blocks over as plain
        # dicts (and `to_dict`/`asdict` turns them back into dicts), so rebuild the
        # dataclasses here exactly as `TrainConfig.__post_init__` does for `hybrid:`.
        if isinstance(self.gate, dict):
            self.gate = GateConfig(**self.gate)
        if isinstance(self.gate_prior, dict):
            self.gate_prior = GatePriorConfig(**self.gate_prior)
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
        if not HYBRID_A_GATE_SCALE_MIN <= self.gate_scale <= HYBRID_A_GATE_SCALE_MAX:
            raise ValueError(
                f"hybrid.gate_scale must be in [{HYBRID_A_GATE_SCALE_MIN}, "
                f"{HYBRID_A_GATE_SCALE_MAX}], got {self.gate_scale}"
            )
        if self.gate.enabled:
            # The gate blends *towards* the Gaussian colour, so there has to be one.
            if not self.enabled:
                raise ValueError("hybrid.gate.enabled needs hybrid.enabled")
            if "rgb" not in self.channels:
                raise ValueError(
                    "hybrid.gate.enabled needs 'rgb' in hybrid.channels: the gate blends "
                    f"towards the Gaussian colour and this run has channels={self.channels}"
                )

    @property
    def num_channels(self) -> int:
        """Width of the Gaussian block appended to every U-Net input level."""
        return gaussian_channel_count(self.channels)

    @property
    def gate_enabled(self) -> bool:
        """True when the network carries the extra gate output channel.

        The single predicate every consumer asks. It is deliberately an AND with
        `enabled`: a `gate: {enabled: true}` on a non-hybrid run is refused at
        construction, so this can never be true without a Gaussian block to blend.
        """
        return bool(self.enabled and self.gate.enabled)

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
