"""Metal blend_bwd vs the float64 CPU reference gradients, on MPS. Queue only.

Module: tests.test_raster_bwd_metal
Invariants: every test here is marked `gpu` and must run inside a
    scripts/gpu_submit.sh job (AGENTS.md section 6: never run MPS work
    directly). PYTORCH_ENABLE_MPS_FALLBACK=0 is set by the job wrapper, so an
    unsupported MPS op fails loudly instead of silently landing on the CPU.
    The maths of blend_bwd is already pinned on CPU by
    tests/test_raster_bwd_ref.py; what these tests add is the *Metal
    translation*, the fragment-cap / transmittance-cutoff semantics on real
    kernel output, and the end-to-end optimisability of the MPS path.
Related docs: docs/ARCHITECTURE.md ("Backward pass data flow"),
    docs/LIMITATIONS.md, docs/TRIPS_REFERENCE.md section 4.
"""

from __future__ import annotations

import time

import pytest
import torch
from test_raster_bwd_scenes import make_smooth_scene
from test_raster_scenes import STACK_COUNT, make_scene

from trippy.constants import RASTER_MAX_FRAGS, RASTER_T_CUTOFF
from trippy.raster import render_pyramid

pytestmark = pytest.mark.gpu

# Accepted relative gradient error, float32 Metal vs float64 CPU. "Relative"
# means max|delta| divided by the largest reference component of that same
# gradient tensor -- a scale-invariant measure that does not explode on the
# near-zero entries an elementwise ratio would blow up on.
GRAD_REL_TOL = 1e-3
# Guard for the denominator of that ratio when a whole gradient is ~zero.
GRAD_SCALE_FLOOR = 1e-12

# The learnable inputs whose gradients are compared.
GRAD_INPUTS = ("xyz", "size", "conf", "feat", "pose_delta")

# Mid-size timing case: a quarter-resolution kk crop with a realistic point
# count, at the shipped TRIPS channel count and layer count.
MID_HW = (192, 256)
MID_POINTS = 50_000
MID_CHANNELS = 4
MID_LAYERS = 5
MID_REPEATS = 3

# Fragment-cap / cutoff case: tests.test_raster_scenes stacks 24 fragments on
# one layer-0 pixel, well past RASTER_MAX_FRAGS.
CAP_HW = (32, 32)
CAP_POINTS = 50
CAP_LAYERS = 3
# The fixture's stacked points are deliberately low-confidence so the 16-deep
# cap is what binds. Raising their confidence to this value makes the
# *transmittance cutoff* bind as well, on the corners with the largest
# bilinear weight -- so one scene exercises both stop rules. Verified on CPU
# to leave float32 and float64 with identical `n_used` (the stop point sits
# ~2x away from the cutoff, i.e. ~10^6 float32 ulps).
CAP_STACK_CONF = 0.95

# 2-step-SGD sanity check: features only, so the objective is exactly
# quadratic and plain SGD at a small step size must decrease monotonically.
SGD_STEPS = 20
# Measured on CPU float32 on this exact scene: monotonic up to lr = 0.8, so
# 0.2 sits a factor of 4 inside the stability edge while still cutting the
# loss ~5x in 20 steps (an lr that only nibbles would not prove much).
SGD_LR = 0.2
# Slack on "monotonic": float32 accumulation noise, not a real increase.
SGD_MONOTONIC_SLACK = 1e-6

# A deliberately non-zero pose delta for the timing case: at exactly zero the
# rotation half of xform_b.se3_exp's gradient degenerates to zero (see
# docs/LIMITATIONS.md), which would make a "gradient is non-zero" assertion
# test the wrong thing.
MID_POSE_DELTA = (0.01, -0.02, 0.015, 0.02, -0.01, 0.03)


@pytest.fixture(scope="module", autouse=True)
def require_mps() -> None:
    """Skip the whole module cleanly when MPS is not present."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


def _relative_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """max|got - ref| / max(|ref|), both compared in float64 on CPU."""
    got64 = got.detach().cpu().double()
    ref64 = ref.detach().cpu().double()
    scale = max(float(ref64.abs().max()), GRAD_SCALE_FLOOR)
    return float((got64 - ref64).abs().max()) / scale


def _output_weights(numel: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Deterministic, non-uniform loss weights (so nothing cancels by symmetry)."""
    return torch.linspace(0.3, 1.7, numel, dtype=dtype, device=device)


def _grads_for(scene: dict, device: str, dtype: torch.dtype, **render_kwargs) -> dict:
    """Render `scene` on `device` and return d(weighted sum of layers)/d(input).

    Args:
        scene: a make_smooth_scene()/make_scene() dict (float64, CPU).
        device: "cpu" (float64 reference) or "mps" (float32 Metal path).
        dtype: compute dtype for that device.
        render_kwargs: forwarded to render_pyramid (mode, num_layers, ...).

    Returns:
        {name: gradient tensor} for every entry of GRAD_INPUTS present in the
        scene, plus {"out": the flattened render} for a forward-value check.
    """
    variables = {
        name: scene[name].to(dtype=dtype, device=device).detach().clone().requires_grad_(True)
        for name in GRAD_INPUTS
        if name in scene
    }
    fixed = {
        name: scene[name].to(dtype=dtype, device=device)
        for name in ("K", "R", "t", "bg")
        if name in scene
    }
    layers, aux = render_pyramid(
        variables["xyz"],
        variables["size"],
        variables["feat"],
        variables["conf"],
        fixed["K"],
        fixed["R"],
        fixed["t"],
        scene["image_hw"],
        bg=fixed.get("bg"),
        pose_delta=variables.get("pose_delta"),
        **render_kwargs,
    )
    flat = torch.cat([layer.reshape(-1) for layer in layers])
    weights = _output_weights(flat.numel(), flat.dtype, device)
    (flat * weights).sum().backward()
    result = {name: variable.grad for name, variable in variables.items()}
    result["out"] = flat.detach()
    result["n_used"] = [n.detach().cpu() for n in aux["n_used"]]
    result["t_final_per_layer"] = [t.detach().cpu().double() for t in aux["t_final"]]
    result["n_used_max"] = max(int(n.max()) for n in result["n_used"])
    result["t_final_min"] = min(float(t.detach().min()) for t in aux["t_final"])
    return result


def _compare(label: str, scene: dict, **render_kwargs) -> float:
    """Compare MPS float32 gradients against the CPU float64 reference."""
    ref = _grads_for(scene, "cpu", torch.float64, **render_kwargs)
    gpu = _grads_for(scene, "mps", torch.float32, **render_kwargs)
    out_err = _relative_error(gpu["out"], ref["out"])
    print(f"[{label}] forward rel err {out_err:.3e}")
    worst = 0.0
    for name in GRAD_INPUTS:
        if name not in ref:
            continue
        if ref[name] is None:
            # Expected in mode="broadcast": the layer factor is 1 everywhere,
            # so per-point size feeds nothing and has no gradient on either
            # device (docs/TRIPS_REFERENCE.md section 10.1). Both must agree.
            assert gpu[name] is None, f"{label}: {name} has a gradient on MPS but not on CPU"
            print(f"[{label}] d/d {name:<10} no gradient on either device (expected)")
            continue
        assert gpu[name] is not None, f"{label}: {name} got no gradient on MPS"
        assert torch.isfinite(gpu[name]).all(), f"{label}: {name} gradient is not finite"
        err = _relative_error(gpu[name], ref[name])
        worst = max(worst, err)
        print(
            f"[{label}] d/d {name:<10} rel err {err:.3e}  "
            f"|ref|max {float(ref[name].abs().max()):.4g}"
        )
        assert float(ref[name].abs().max()) > 0.0, f"{label}: {name} reference gradient is zero"
    print(f"[{label}] worst relative gradient error: {worst:.3e}")
    return worst


@pytest.mark.parametrize("mode", ["trilinear", "broadcast", "trips"])
@pytest.mark.parametrize("num_channels", [3, 4])
def test_metal_gradients_match_reference(mode: str, num_channels: int) -> None:
    """All five gradients on MPS must match the float64 reference to 1e-3."""
    scene = make_smooth_scene(num_channels=num_channels)
    worst = _compare(
        f"smooth {mode} C={num_channels}",
        scene,
        num_layers=scene["num_layers"],
        mode=mode,
    )
    assert worst < GRAD_REL_TOL


@pytest.mark.parametrize("pixel_center", ["half", "integer"])
def test_metal_gradients_match_reference_trips_pixel_conventions(pixel_center: str) -> None:
    """Mode "trips" gradients on MPS, in both pixel-centre conventions.

    Mode "trips" is the trainer default, so this is the gradient path every
    training step now takes; `pixel_center="integer"` additionally covers the
    setting `trippy.render.parity`'s native engine renders with.
    """
    scene = make_smooth_scene(num_channels=4)
    worst = _compare(
        f"smooth trips {pixel_center}",
        scene,
        num_layers=scene["num_layers"],
        mode="trips",
        pixel_center=pixel_center,
    )
    assert worst < GRAD_REL_TOL


def test_metal_gradients_with_fragment_cap_and_cutoff() -> None:
    """A pixel past the 16-fragment cap, and one past the cutoff, still work.

    tests.test_raster_scenes stacks 24 fragments on one layer-0 pixel, which
    saturates `n_used` at RASTER_MAX_FRAGS. Raising those points' confidence
    to CAP_STACK_CONF additionally drives transmittance under RASTER_T_CUTOFF
    on the high-weight corners, so *both* forward stop rules fire in one
    scene. Fragments beyond the composited prefix contributed nothing to the
    forward and must receive exactly zero gradient -- which is what comparing
    against the reference (whose `keep` mask implements the same prefix rule)
    verifies, since the reference zeroes them by construction.
    """
    scene = make_scene(
        num_points=CAP_POINTS, height=CAP_HW[0], width=CAP_HW[1], num_channels=3, seed=0
    )
    scene["conf"] = scene["conf"].clone()
    scene["conf"][:STACK_COUNT] = CAP_STACK_CONF

    ref = _grads_for(scene, "cpu", torch.float64, num_layers=CAP_LAYERS)
    gpu = _grads_for(scene, "mps", torch.float32, num_layers=CAP_LAYERS)
    print(
        f"[cap] n_used max ref={ref['n_used_max']} metal={gpu['n_used_max']} "
        f"(cap {RASTER_MAX_FRAGS}); t_final min ref={ref['t_final_min']:.3e} "
        f"metal={gpu['t_final_min']:.3e} (cutoff {RASTER_T_CUTOFF})"
    )
    assert ref["n_used_max"] == RASTER_MAX_FRAGS, "the fixture must reach the cap"
    assert gpu["n_used_max"] == RASTER_MAX_FRAGS

    # The cutoff must be the binding rule somewhere: a pixel that stopped
    # early (n_used < cap) with transmittance already under the threshold.
    cutoff_hit = any(
        bool(((t_layer < RASTER_T_CUTOFF) & (n_layer < RASTER_MAX_FRAGS)).any())
        for t_layer, n_layer in zip(ref["t_final_per_layer"], ref["n_used"], strict=True)
    )
    assert cutoff_hit, "the fixture must trip the transmittance cutoff too"

    # Float32 and float64 must agree on *where* they stopped; if they did
    # not, a gradient mismatch below would be a discrete disagreement, not a
    # kernel bug, and the tolerance check would be meaningless.
    for layer, (ref_n, gpu_n) in enumerate(zip(ref["n_used"], gpu["n_used"], strict=True)):
        assert torch.equal(ref_n, gpu_n.to(ref_n.dtype)), f"n_used differs on layer {layer}"

    worst = 0.0
    for name in GRAD_INPUTS:
        if name not in ref or ref[name] is None:
            continue
        assert torch.isfinite(gpu[name]).all(), name
        err = _relative_error(gpu[name], ref[name])
        worst = max(worst, err)
        print(f"[cap] d/d {name:<10} rel err {err:.3e}")
    print(f"[cap] worst relative gradient error: {worst:.3e}")
    assert worst < GRAD_REL_TOL


def test_metal_backward_midsize_runs_and_is_finite() -> None:
    """256x192, 50k points, C=4, L=5: backward runs, is finite; report ms."""
    scene = make_scene(
        num_points=MID_POINTS,
        height=MID_HW[0],
        width=MID_HW[1],
        num_channels=MID_CHANNELS,
        seed=23,
        dtype=torch.float32,
        device="mps",
    )
    variables = {
        name: scene[name].detach().clone().requires_grad_(True)
        for name in ("xyz", "size", "feat", "conf")
    }
    pose_delta = (
        torch.tensor(MID_POSE_DELTA, dtype=torch.float32, device="mps")
        .clone()
        .requires_grad_(True)
    )

    def render():
        layers, aux = render_pyramid(
            variables["xyz"],
            variables["size"],
            variables["feat"],
            variables["conf"],
            scene["K"],
            scene["R"],
            scene["t"],
            scene["image_hw"],
            num_layers=MID_LAYERS,
            bg=scene["bg"],
            pose_delta=pose_delta,
        )
        return layers, aux

    layers, aux = render()  # warm-up: compiles both kernels
    sum(layer.sum() for layer in layers).backward()
    torch.mps.synchronize()
    for variable in (*variables.values(), pose_delta):
        variable.grad = None

    forward_start = time.perf_counter()
    for _ in range(MID_REPEATS):
        layers, aux = render()
    torch.mps.synchronize()
    forward_ms = 1000.0 * (time.perf_counter() - forward_start) / MID_REPEATS

    both_start = time.perf_counter()
    for _ in range(MID_REPEATS):
        for variable in (*variables.values(), pose_delta):
            variable.grad = None
        layers, aux = render()
        sum(layer.sum() for layer in layers).backward()
    torch.mps.synchronize()
    both_ms = 1000.0 * (time.perf_counter() - both_start) / MID_REPEATS

    print(
        f"[timing] {MID_HW[1]}x{MID_HW[0]} {MID_POINTS} points C={MID_CHANNELS} "
        f"L={MID_LAYERS}: forward {forward_ms:.1f} ms, forward+backward {both_ms:.1f} ms, "
        f"backward {both_ms - forward_ms:.1f} ms, {aux['num_fragments']} fragments"
    )
    for name, variable in (*variables.items(), ("pose_delta", pose_delta)):
        assert variable.grad is not None, name
        assert torch.isfinite(variable.grad).all(), name
        print(f"[timing] d/d {name:<10} |grad|max {float(variable.grad.abs().max()):.4g}")
    assert variables["feat"].grad.abs().max() > 0.0
    assert variables["xyz"].grad.abs().max() > 0.0
    assert pose_delta.grad.abs().max() > 0.0


def test_feature_only_sgd_decreases_the_loss() -> None:
    """20 SGD steps on features alone must reduce a render-matching loss.

    Features enter the composite linearly, so the objective is exactly
    quadratic and a small fixed step size cannot overshoot: any increase here
    is a wrong gradient, not an optimisation artefact.
    """
    scene = make_scene(
        num_points=CAP_POINTS, height=CAP_HW[0], width=CAP_HW[1], num_channels=3, seed=5
    )
    args = [scene[key].to(torch.float32).to("mps") for key in ("xyz", "size", "conf", "K", "R", "t")]
    xyz, size, conf, K, R, t = args
    target_feat = scene["feat"].to(torch.float32).to("mps")
    bg = scene["bg"].to(torch.float32).to("mps")

    def render(feat):
        layers, _ = render_pyramid(
            xyz, size, feat, conf, K, R, t, scene["image_hw"], num_layers=CAP_LAYERS, bg=bg
        )
        return torch.cat([layer.reshape(-1) for layer in layers])

    with torch.no_grad():
        target = render(target_feat)

    # Start from a deliberately wrong, but same-scale, feature set.
    feat = (1.0 - target_feat).detach().clone().requires_grad_(True)
    optimiser = torch.optim.SGD([feat], lr=SGD_LR)

    losses: list[float] = []
    for _ in range(SGD_STEPS):
        optimiser.zero_grad()
        loss = ((render(feat) - target) ** 2).sum()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    print("[sgd] losses: " + ", ".join(f"{value:.5f}" for value in losses))
    print(f"[sgd] first {losses[0]:.5f} -> last {losses[-1]:.5f}")
    for step in range(1, SGD_STEPS):
        assert losses[step] <= losses[step - 1] * (1.0 + SGD_MONOTONIC_SLACK), (
            f"loss increased at step {step}: {losses[step - 1]:.6f} -> {losses[step]:.6f}"
        )
    assert losses[-1] < 0.5 * losses[0], "20 steps should meaningfully reduce the loss"


def test_forward_api_is_unchanged_for_non_grad_callers() -> None:
    """render_pyramid must still return the same numbers, graph or no graph."""
    scene = make_scene(num_points=CAP_POINTS, height=CAP_HW[0], width=CAP_HW[1], seed=9)
    plain = [scene[key].to(torch.float32).to("mps") for key in ("xyz", "size", "feat", "conf")]
    fixed = [scene[key].to(torch.float32).to("mps") for key in ("K", "R", "t")]
    bg = scene["bg"].to(torch.float32).to("mps")

    layers_plain, aux_plain = render_pyramid(
        *plain, *fixed, scene["image_hw"], num_layers=CAP_LAYERS, bg=bg
    )
    assert not layers_plain[0].requires_grad, "no input requires grad -> no graph"

    grad_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in plain]
    layers_grad, aux_grad = render_pyramid(
        *grad_inputs, *fixed, scene["image_hw"], num_layers=CAP_LAYERS, bg=bg
    )
    assert layers_grad[0].requires_grad, "auto-detection must build the graph"

    layers_off, _ = render_pyramid(
        *grad_inputs,
        *fixed,
        scene["image_hw"],
        num_layers=CAP_LAYERS,
        bg=bg,
        differentiable=False,
    )
    assert not layers_off[0].requires_grad, "differentiable=False must not build a graph"
    for a, b, c in zip(layers_plain, layers_grad, layers_off, strict=True):
        assert torch.equal(a, b.detach())
        assert torch.equal(a, c)
    assert aux_plain["num_fragments"] == aux_grad["num_fragments"]
    # depth_sum is documented as non-differentiable on MPS (LIMITATIONS.md).
    assert not aux_grad["depth_sum"][0].requires_grad
