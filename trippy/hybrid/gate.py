"""The blend gate: an explicit, adjustable splat-vs-TRIPS mix (hybrid design A).

Module: trippy.hybrid.gate
Purpose: hybrid design A feeds the Gaussian-splat render into the U-Net as extra
    input channels, so *how much* of the finished image comes from the splat and
    how much from the TRIPS point pyramid is implicit in the weights -- it cannot
    be measured, shown to Jordan, or turned up and down after training. This
    module makes it explicit. The network grows ONE extra output channel; its
    sigmoid is a per-pixel weight `g` in [0, 1], and the displayed image is

        final = g * splat_rgb + (1 - g) * trips_rgb

    with `trips_rgb` the tone-mapped 3-channel network output and `splat_rgb` the
    alpha-masked Gaussian render at the same pixel. `g` is trained end to end by
    the ordinary image losses (nothing supervises it directly); `gate_prior_loss`
    optionally pulls its *mean* towards a target, and `gate_scale` re-weights it
    at eval/render time.

Invariants:
    - **Gate off is a hard no-op.** With `HybridConfig.gate.enabled` false the
      U-Net has its original 3 output channels, `split_output` returns
      `(net_out, None)`, and `blend` is never called. Every existing hybrid
      checkpoint and test therefore takes exactly the code path it took before
      this module existed.
    - **The blend is applied AFTER the tone mapper, not before.** Both operands
      have to live in the same space or the mix is meaningless, and the two are
      only commensurable in display-referred space: `splat_rgb` comes out of
      gsrender as a finished [0, 1] render of a 3DGS that was itself fitted to
      these photos, so it has already "had" its exposure applied. Blending
      pre-tone-map would push the splat through a second exposure/response curve
      it was never missing. It also makes the extreme testable and exact: at
      `g == 1` the output *is* the splat, pixel for pixel.
    - **Colour channels stay first.** The gate logit is channel
      `HYBRID_A_GATE_CHANNEL_INDEX` (== 3), appended after rgb, so `net_out[:,
      :3]` is bit-identical to what a gate-less network of the same weights would
      emit and the tone mapper, `Trainer.calibrate_frame` and every honesty
      artifact need no change at all.
    - **The gate is forced to zero where there is no splat.** A frame with no
      render triple, a crop that ablation 1 dropped out, or a pose with no live
      renderer, all present as `gaussian is None`; `splat_rgb_from_block` returns
      None and the caller must not blend. Blending against an all-zero
      `splat_rgb` there would paint black holes into the image *and* would be a
      lie -- there is no splat evidence at that pixel to mix in. Where the splat
      exists but is uncovered (alpha ~ 0) the network can and does learn `g -> 0`
      from the alpha channel it is fed; that is a learned decision, not a
      fabricated one.
    - **`gate_scale` clamps twice, deliberately.** The scale itself is clamped to
      `[HYBRID_A_GATE_SCALE_MIN, HYBRID_A_GATE_SCALE_MAX]` and the product `g * s`
      is clamped back to [0, 1]. A viewer slider can therefore never produce an
      out-of-range weight, and `s = 0` is exactly the TRIPS path while a large `s`
      saturates to exactly the splat wherever the trained gate already leaned that
      way.
    - Nothing here reads a file, touches MPS, or imports the trainer. It is pure
      tensor arithmetic plus one from-scratch heatmap, so all of it is CPU-testable.

Units: `g`, `splat_rgb`, `trips_rgb` and the blend are all unitless, in [0, 1]
    (`trips_rgb` is only *nominally* bounded -- the tone mapper's response curve
    can leak slightly outside, and the blend does not clamp it back, so a gate-off
    and a `gate_scale=0` render stay numerically identical).
Related docs: docs/EXPERIMENTS.md "Hybrid design A" / "The blend gate";
    docs/ARCHITECTURE.md "hybrid/"; docs/USER_GUIDE.md "Blend panel";
    rust/crates/brush-unet (the Rust reader of the extra output channel).
"""

from __future__ import annotations

import numpy as np
import torch

from trippy.constants import (
    HYBRID_A_GATE_CHANNEL_INDEX,
    HYBRID_A_GATE_PERCENTILES,
    HYBRID_A_GATE_SCALE_MAX,
    HYBRID_A_GATE_SCALE_MIN,
)
from trippy.hybrid.config_a import HybridConfig


def clamp_scale(scale: float) -> float:
    """Clamp a requested `gate_scale` into `[HYBRID_A_GATE_SCALE_MIN, ..._MAX]`.

    Clamped rather than rejected on purpose: this value comes from a viewer
    slider and a CLI flag, and neither should be able to fail a render.
    """
    return float(min(max(float(scale), HYBRID_A_GATE_SCALE_MIN), HYBRID_A_GATE_SCALE_MAX))


def split_output(net_out: torch.Tensor, gate_enabled: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a U-Net output into `(rgb, gate)`.

    Args:
        net_out: `(B, C, H, W)` -- `C == 3` when the gate is off, `C == 4`
            (rgb + gate logit) when it is on.
        gate_enabled: whether this network was built with the gate head.

    Returns:
        `(rgb, gate)`: `rgb` is `net_out[:, :3]`, and `gate` is
        `sigmoid(net_out[:, 3:4])` in [0, 1] -- or None when the gate is off.

    The sigmoid is applied here, on top of whatever `NetworkConfig.last_act`
    already did. With the shipped `last_act = "id"` (an `nn.Identity`) that is
    exactly "sigmoid of the final conv's logit", which is what the schema
    documents and what `brush-unet` reimplements.
    """
    if not gate_enabled:
        return net_out, None
    index = HYBRID_A_GATE_CHANNEL_INDEX
    if net_out.shape[1] <= index:
        raise ValueError(
            f"gate is enabled but the network output has only {net_out.shape[1]} channels; "
            f"channel {index} (the gate logit) is missing -- was the checkpoint trained with "
            "hybrid.gate.enabled?"
        )
    return net_out[:, :index], torch.sigmoid(net_out[:, index : index + 1])


def splat_rgb_from_block(block: torch.Tensor | None, cfg: HybridConfig) -> torch.Tensor | None:
    """The alpha-masked Gaussian rgb carried inside a `(G, H, W)`/`(B, G, H, W)` block.

    The block is exactly what `trippy.hybrid.gaussian_input` already hands the
    network, so the gate mixes in *the same pixels the network was shown* --
    there is no second render, no second normalisation, and no way for the two to
    drift apart.

    Args:
        block: the Gaussian block, or None (missing render / dropped-out crop /
            no live renderer). None propagates: see the module docstring on why
            a missing splat must not be blended as black.
        cfg: the run's `hybrid:` block, for the channel layout.

    Returns:
        `(B, 3, H, W)` float tensor in [0, 1], or None. Already multiplied by
        alpha: when `cfg.mask_by_alpha` is set the block's rgb *is* the masked
        render, and when it is not, alpha is applied here (so the gate always
        mixes in "the splat where the splat exists", regardless of ablation 2).

    Raises:
        ValueError: if `cfg.channels` has no "rgb" group -- the gate needs a
            colour to blend towards, so `HybridConfig` refuses that combination
            at construction and this is the belt-and-braces check.
    """
    if block is None:
        return None
    if "rgb" not in cfg.channels:
        raise ValueError("hybrid.gate needs 'rgb' in hybrid.channels: there is nothing to blend towards")
    x = block if block.dim() == 4 else block.unsqueeze(0)
    rgb = x[:, cfg.channel_slice("rgb")]
    if not cfg.mask_by_alpha and "alpha" in cfg.channels:
        rgb = rgb * x[:, cfg.channel_slice("alpha")]
    return rgb


def effective_gate(gate: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """`clamp(gate * clamp_scale(scale), 0, 1)` -- the weight actually used in `blend`."""
    return (gate * clamp_scale(scale)).clamp(0.0, 1.0)


def blend(
    splat_rgb: torch.Tensor, trips_rgb: torch.Tensor, gate: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """`g * splat_rgb + (1 - g) * trips_rgb` with `g = effective_gate(gate, scale)`.

    Args:
        splat_rgb: `(B, 3, H, W)` alpha-masked Gaussian render, display-referred.
        trips_rgb: `(B, 3, H, W)` tone-mapped network output, same size.
        gate: `(B, 1, H, W)` in [0, 1] (`split_output`'s second return value).
        scale: the post-training `gate_scale` knob; 1.0 is "as trained".

    Returns:
        `(B, 3, H, W)`. At `scale == 0` this is `trips_rgb` unchanged, to the
        last bit -- `1 - 0` is exactly 1 and `0 * splat` is exactly 0 in IEEE
        arithmetic, which is what makes the "gate_scale 0 == the TRIPS path"
        test an equality rather than an approximation.
    """
    if splat_rgb.shape[-2:] != trips_rgb.shape[-2:]:
        raise ValueError(
            f"blend needs matching spatial sizes, got splat {tuple(splat_rgb.shape[-2:])} "
            f"vs trips {tuple(trips_rgb.shape[-2:])}"
        )
    g = effective_gate(gate, scale).to(dtype=trips_rgb.dtype, device=trips_rgb.device)
    splat = splat_rgb.to(dtype=trips_rgb.dtype, device=trips_rgb.device)
    return g * splat + (1.0 - g) * trips_rgb


def gate_prior_loss(gate: torch.Tensor, target: float, weight: float) -> torch.Tensor:
    """`weight * (mean(gate) - target) ** 2` -- the optional mean-gate regulariser.

    A *mean* prior, not a per-pixel one, on purpose: the question it exists to
    ask is "how much of this scene wants to be splat overall?", and a per-pixel
    penalty would answer a different one by flattening the map the whole feature
    exists to expose. Returns a zero scalar (still connected to `gate`'s graph is
    unnecessary here, so it is a plain zero) when `weight <= 0`.
    """
    if weight <= 0.0:
        return torch.zeros((), device=gate.device, dtype=gate.dtype)
    return float(weight) * (gate.mean() - float(target)) ** 2


def gate_stats(gate: torch.Tensor, percentiles: tuple[float, ...] = HYBRID_A_GATE_PERCENTILES) -> dict:
    """Summary of one gate map: mean, min, max and `percentiles`.

    Args:
        gate: any-shaped tensor of gate values in [0, 1] (already scaled if the
            caller wants the *effective* gate summarised).
        percentiles: percentile points, in percent.

    Returns:
        `{"mean": float, "min": float, "max": float, "percentiles": {"p50": ...}}`
        -- plain floats, JSON-safe, ready for metrics.json / report.json.
    """
    values = gate.detach().reshape(-1).to(torch.float32).cpu().numpy()
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "percentiles": {}}
    qs = np.percentile(values, list(percentiles))
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "percentiles": {f"p{p:g}": float(q) for p, q in zip(percentiles, np.atleast_1d(qs), strict=True)},
    }


def merge_gate_stats(stats: list[dict]) -> dict:
    """Average a list of `gate_stats` dicts into one (means of means, mins of mins).

    Used for the per-eval and per-report aggregate. The percentile fields are
    averaged across frames rather than recomputed over the pooled pixels: the
    frames all have the same size in every caller here, so the two agree closely,
    and this avoids holding every frame's pixels in memory to report a number.
    """
    if not stats:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "percentiles": {}, "n_frames": 0}
    keys: list[str] = []
    for row in stats:
        for key in row.get("percentiles", {}):
            if key not in keys:
                keys.append(key)
    return {
        "mean": float(np.mean([s["mean"] for s in stats])),
        "min": float(np.min([s["min"] for s in stats])),
        "max": float(np.max([s["max"] for s in stats])),
        "percentiles": {
            k: float(np.mean([s["percentiles"][k] for s in stats if k in s.get("percentiles", {})]))
            for k in keys
        },
        "n_frames": len(stats),
    }


def gate_heatmap(gate: torch.Tensor) -> np.ndarray:
    """A from-scratch uint8 `(H, W, 3)` heatmap of one gate map, safe for an agent to open.

    Built only from the gate values (0 = TRIPS, 1 = splat) through
    `trippy.render.sheets.colorize`'s fixed colour ramp with a fixed [0, 1]
    range, so the colour of a pixel means the same thing in every frame of every
    run and nothing photographed is anywhere in it (AGENTS.md Sec. 6 "Allowed to
    view: abstract heatmaps with no photographic content").
    """
    # Imported here rather than at module scope: `trippy.render.sheets` pulls in PIL,
    # and this module is imported by `trippy.train.config`'s dependency chain (which
    # `trippy eval --help` pays for on every invocation).
    from trippy.render.sheets import colorize

    values = gate.detach().to(torch.float32).cpu().numpy()
    while values.ndim > 2:
        values = values[0]
    return colorize(values, 0.0, 1.0)
