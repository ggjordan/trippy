"""Unit tests for the blend gate's arithmetic (`trippy.hybrid.gate`).

Module: tests.test_hybrid_gate_math
Invariants under test: the two extremes are EXACT, not approximate --
    `gate_scale = 0` returns the TRIPS operand bit for bit, and a scale large
    enough to saturate returns the splat operand bit for bit; the scale is
    clamped at both ends; the gate channel is split off without touching the
    colour channels; a missing Gaussian block propagates as None (never as a
    black splat); and the prior is a mean-only penalty that is exactly zero when
    switched off.
Everything here is pure tensor arithmetic on synthetic values; CPU only, no
scene, no renders, no network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trippy.constants import HYBRID_A_GATE_SCALE_MAX, HYBRID_A_GATE_SCALE_MIN
from trippy.hybrid import gate as gate_mod
from trippy.hybrid.config_a import HybridConfig


def _cfg(**kwargs) -> HybridConfig:
    base = {"enabled": True, "renders_dir": "renders", "gate": {"enabled": True}}
    base.update(kwargs)
    return HybridConfig(**base)


# --- clamp_scale ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [(-1.0, HYBRID_A_GATE_SCALE_MIN), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (9.5, HYBRID_A_GATE_SCALE_MAX)],
)
def test_clamp_scale_bounds_the_knob(given: float, expected: float) -> None:
    assert gate_mod.clamp_scale(given) == expected


# --- split_output ---


def test_split_output_is_a_no_op_when_the_gate_is_off() -> None:
    net_out = torch.randn(1, 3, 4, 5)
    rgb, gate = gate_mod.split_output(net_out, gate_enabled=False)
    assert gate is None
    assert rgb is net_out


def test_split_output_takes_the_fourth_channel_and_leaves_colour_untouched() -> None:
    net_out = torch.randn(1, 4, 4, 5)
    rgb, gate = gate_mod.split_output(net_out, gate_enabled=True)
    assert gate is not None
    assert rgb.shape == (1, 3, 4, 5)
    assert gate.shape == (1, 1, 4, 5)
    # Colour is passed through with no arithmetic at all.
    assert torch.equal(rgb, net_out[:, :3])
    # The gate is the sigmoid of the fourth channel's logit.
    assert torch.allclose(gate, torch.sigmoid(net_out[:, 3:4]))
    assert bool(((gate >= 0.0) & (gate <= 1.0)).all())


def test_split_output_refuses_a_three_channel_output_when_the_gate_is_on() -> None:
    with pytest.raises(ValueError, match="gate logit"):
        gate_mod.split_output(torch.randn(1, 3, 2, 2), gate_enabled=True)


# --- blend, the two exact extremes ---


def _operands(height: int = 3, width: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    splat = torch.rand(1, 3, height, width)
    trips = torch.rand(1, 3, height, width)
    gate = torch.rand(1, 1, height, width)
    return splat, trips, gate


def test_gate_scale_zero_is_exactly_the_trips_path() -> None:
    splat, trips, gate = _operands()
    out = gate_mod.blend(splat, trips, gate, scale=0.0)
    assert torch.equal(out, trips)


def test_a_saturating_scale_is_exactly_the_splat() -> None:
    splat, trips, gate = _operands()
    # Every gate value is >= 0.5 here, so scale 2 clamps all of them to exactly 1.
    gate = gate.clamp(min=0.5)
    out = gate_mod.blend(splat, trips, gate, scale=HYBRID_A_GATE_SCALE_MAX)
    assert torch.equal(out, splat)


def test_scale_one_is_the_plain_convex_combination() -> None:
    splat, trips, gate = _operands()
    out = gate_mod.blend(splat, trips, gate, scale=1.0)
    assert torch.allclose(out, gate * splat + (1.0 - gate) * trips)


def test_blend_stays_between_its_two_operands() -> None:
    splat, trips, gate = _operands()
    out = gate_mod.blend(splat, trips, gate, scale=1.3)
    lo = torch.minimum(splat, trips)
    hi = torch.maximum(splat, trips)
    assert bool(((out >= lo - 1e-6) & (out <= hi + 1e-6)).all())


def test_blend_refuses_mismatched_sizes() -> None:
    with pytest.raises(ValueError, match="matching spatial sizes"):
        gate_mod.blend(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 5, 5), torch.zeros(1, 1, 4, 4))


# --- splat_rgb_from_block ---


def test_a_missing_block_propagates_as_none_never_as_black() -> None:
    assert gate_mod.splat_rgb_from_block(None, _cfg()) is None


def test_masked_by_alpha_blocks_are_taken_verbatim() -> None:
    cfg = _cfg(mask_by_alpha=True)
    block = torch.rand(cfg.num_channels, 3, 4)
    rgb = gate_mod.splat_rgb_from_block(block, cfg)
    assert rgb is not None
    # The block's rgb IS the masked render already: no second multiply.
    assert torch.equal(rgb[0], block[cfg.channel_slice("rgb")])


def test_an_unmasked_block_gets_its_alpha_applied_here() -> None:
    cfg = _cfg(mask_by_alpha=False)
    block = torch.rand(cfg.num_channels, 3, 4)
    rgb = gate_mod.splat_rgb_from_block(block, cfg)
    assert rgb is not None
    expected = block[cfg.channel_slice("rgb")] * block[cfg.channel_slice("alpha")]
    assert torch.allclose(rgb[0], expected)


def test_a_batched_block_keeps_its_batch() -> None:
    cfg = _cfg()
    rgb = gate_mod.splat_rgb_from_block(torch.rand(2, cfg.num_channels, 3, 4), cfg)
    assert rgb is not None and rgb.shape == (2, 3, 3, 4)


# --- the prior ---


def test_the_prior_is_exactly_zero_when_switched_off() -> None:
    loss = gate_mod.gate_prior_loss(torch.rand(1, 1, 4, 4), target=0.5, weight=0.0)
    assert float(loss) == 0.0


def test_the_prior_penalises_the_mean_only() -> None:
    # Two maps with the same mean but wildly different spread must score the same:
    # the prior constrains the average reliance on the splat, not the map.
    flat = torch.full((1, 1, 4, 4), 0.25)
    split = torch.cat([torch.zeros(1, 1, 4, 2), torch.full((1, 1, 4, 2), 0.5)], dim=3)
    a = gate_mod.gate_prior_loss(flat, target=0.5, weight=2.0)
    b = gate_mod.gate_prior_loss(split, target=0.5, weight=2.0)
    assert float(a) == pytest.approx(float(b))
    assert float(a) == pytest.approx(2.0 * (0.25 - 0.5) ** 2)


def test_the_prior_is_differentiable_towards_the_target() -> None:
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    loss = gate_mod.gate_prior_loss(torch.sigmoid(logits), target=1.0, weight=1.0)
    loss.backward()
    # Target above the current mean (0.5) -> the gradient pushes the logits UP.
    assert logits.grad is not None
    assert float(logits.grad.sum()) < 0.0


# --- stats and heatmap ---


def test_gate_stats_reports_mean_and_percentiles() -> None:
    values = torch.linspace(0.0, 1.0, 101).reshape(1, 1, 101)
    stats = gate_mod.gate_stats(values)
    assert stats["mean"] == pytest.approx(0.5, abs=1e-6)
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["percentiles"]["p50"] == pytest.approx(0.5, abs=1e-6)
    assert stats["percentiles"]["p95"] == pytest.approx(0.95, abs=1e-6)


def test_merge_gate_stats_averages_frames() -> None:
    rows = [gate_mod.gate_stats(torch.full((4, 4), v)) for v in (0.2, 0.4)]
    merged = gate_mod.merge_gate_stats(rows)
    assert merged["mean"] == pytest.approx(0.3)
    assert merged["min"] == pytest.approx(0.2)
    assert merged["max"] == pytest.approx(0.4)
    assert merged["n_frames"] == 2


def test_merge_gate_stats_survives_an_empty_list() -> None:
    assert gate_mod.merge_gate_stats([])["n_frames"] == 0


def test_the_heatmap_is_a_uint8_rgb_image_that_varies_with_the_gate() -> None:
    ramp = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
    image = gate_mod.gate_heatmap(ramp)
    assert image.dtype == np.uint8
    assert image.shape == (4, 4, 3)
    # A fixed [0, 1] range means the ends of the ramp always get the ramp's end colours,
    # so one frame's heatmap is comparable with another's.
    assert not np.array_equal(image[0, 0], image[-1, -1])
