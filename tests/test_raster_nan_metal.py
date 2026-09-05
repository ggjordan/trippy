"""Degenerate-fragment gradient guards on the Metal path. Queue only.

Module: tests.test_raster_nan_metal
Invariants under test: the MPS render must survive the same degenerate
    inputs tests/test_raster_nan_ref.py pins on CPU -- a point whose
    camera-space z is exactly 0.0, depths at/behind the near plane, a
    fragment sitting exactly on a pixel boundary, `size_px` exactly a power
    of two, and fragment alpha exactly 0 or 1 -- with a finite forward and
    finite gradients for every learnable input.

    Two of those guards live in shared torch code and therefore apply to both
    devices: `trippy.raster.emit.safe_depth` (the projection divisions) runs
    before the kernel on MPS too, since `_render_pyramid_mps` builds its
    fragment list with the same `build_sorted_fragments`.

    The third, alpha == 1, is where the two paths differ, and the direction
    of the difference matters: `metal_src/blend_bwd.metal` is division free
    by construction (suffix recurrences `U`/`Q`, never TRIPS's
    `colour_behind / (1 - alpha)`), so the kernel has never needed an
    epsilon and composites alpha == 1 exactly. It was the *torch* twin that
    needed fixing. These tests assert the kernel really does hold that line,
    and that its numbers still agree with the corrected reference.

    Every test here is marked `gpu` and must run inside a
    scripts/gpu_submit.sh job (AGENTS.md section 6: never run MPS work
    directly). PYTORCH_ENABLE_MPS_FALLBACK=0 is set by the job wrapper.
Related docs: docs/LIMITATIONS.md, docs/ARCHITECTURE.md ("Backward pass data
    flow"), trippy.raster.emit.safe_depth.
"""

from __future__ import annotations

import pytest
import torch
from test_raster_nan_ref import (
    ANCHOR_CONF,
    ANCHOR_SIZE,
    ANCHOR_XYZ,
    BOUNDARY_DEPTH,
    BOUNDARY_SIZE,
    BOUNDARY_XY,
    CX,
    CY,
    DEGENERATE_DEPTHS,
    DEGENERATE_SIZE_PX,
    FX,
    IMAGE_HW,
    NUM_LAYERS,
)

from trippy.constants import RASTER_MODES
from trippy.raster import render_pyramid

pytestmark = pytest.mark.gpu

# Accepted relative error, float32 Metal vs float64 CPU, on the forward and
# on each gradient. Same definition and same value as
# tests/test_raster_bwd_metal.py: max|delta| over max|reference|.
GRAD_REL_TOL = 1e-3
# Guard for the denominator when a whole gradient is ~zero.
GRAD_SCALE_FLOOR = 1e-12

GRAD_INPUTS = ("xyz", "size", "conf", "feat", "pose_delta")


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


def _render_grads(
    probe_xyz: tuple[float, float, float],
    probe_size: float,
    probe_conf: float,
    device: str,
    dtype: torch.dtype,
    mode: str,
    alpha_min: float = 0.0,
    anchor_xyz: tuple[float, float, float] = ANCHOR_XYZ,
    anchor_size: float = ANCHOR_SIZE,
    anchor_conf: float = ANCHOR_CONF,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Anchor + probe render on `device`, backpropped with a non-symmetric loss.

    Mirrors tests.test_raster_nan_ref._render_grads exactly, so the CPU
    float64 run of this helper is the reference the MPS run is diffed
    against.

    Returns:
        (flat_render, {name: gradient or None}).
    """
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    R = torch.eye(3, dtype=dtype, device=device)
    t = torch.zeros(3, dtype=dtype, device=device)
    variables = {
        "xyz": torch.tensor([anchor_xyz, probe_xyz], dtype=dtype, device=device),
        "size": torch.tensor([anchor_size, probe_size], dtype=dtype, device=device),
        "conf": torch.tensor([anchor_conf, probe_conf], dtype=dtype, device=device),
        "feat": torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.1, 0.3]], dtype=dtype, device=device),
        "pose_delta": torch.zeros(6, dtype=dtype, device=device),
    }
    for tensor in variables.values():
        tensor.requires_grad_(True)

    layers, _aux = render_pyramid(
        variables["xyz"],
        variables["size"],
        variables["feat"],
        variables["conf"],
        K,
        R,
        t,
        IMAGE_HW,
        num_layers=NUM_LAYERS,
        mode=mode,
        alpha_min=alpha_min,
        pose_delta=variables["pose_delta"],
    )
    flat = torch.cat([layer.reshape(-1) for layer in layers])
    weights = torch.linspace(0.3, 1.7, flat.numel(), dtype=dtype, device=device)
    (flat * weights).sum().backward()
    return flat.detach(), {name: var.grad for name, var in variables.items()}


def _assert_metal_matches_reference(label: str, **case) -> None:
    """Run the case on MPS and on the CPU float64 reference and compare.

    Asserts (a) the MPS forward and every MPS gradient is finite, (b) the
    two devices agree on *which* inputs receive a gradient at all, and (c)
    the values agree to GRAD_REL_TOL.
    """
    ref_flat, ref_grads = _render_grads(device="cpu", dtype=torch.float64, **case)
    gpu_flat, gpu_grads = _render_grads(device="mps", dtype=torch.float32, **case)

    assert torch.isfinite(gpu_flat).all(), f"{label}: non-finite forward on MPS"
    out_err = _relative_error(gpu_flat, ref_flat)
    print(f"[{label}] forward rel err {out_err:.3e}")
    assert out_err < GRAD_REL_TOL, f"{label}: forward rel err {out_err:.3e}"

    for name in GRAD_INPUTS:
        ref, gpu = ref_grads[name], gpu_grads[name]
        if ref is None:
            assert gpu is None, f"{label}: {name} has a gradient on MPS but not on CPU"
            continue
        assert gpu is not None, f"{label}: {name} got no gradient on MPS"
        bad = int((~torch.isfinite(gpu)).sum())
        assert bad == 0, f"{label}: {bad} non-finite entries in d/d {name}: {gpu.reshape(-1).tolist()}"
        err = _relative_error(gpu, ref)
        print(f"[{label}] d/d {name:<10} rel err {err:.3e}")
        assert err < GRAD_REL_TOL, f"{label}: d/d {name} rel err {err:.3e}"


@pytest.mark.parametrize("depth", DEGENERATE_DEPTHS)
@pytest.mark.parametrize("mode", RASTER_MODES)
def test_metal_degenerate_depth_keeps_every_gradient_finite(depth: float, mode: str) -> None:
    """A culled point at a degenerate depth must not poison the MPS gradients.

    `depth == 0.0` is the reproducing input from the kk-coherent CPU run.
    The guard is in shared torch code (`safe_depth`), which is why the same
    case has to hold on MPS.
    """
    _assert_metal_matches_reference(
        f"depth={depth!r} mode={mode}",
        probe_xyz=(1.0, 0.5, depth),
        probe_size=0.1,
        probe_conf=0.9,
        mode=mode,
    )


@pytest.mark.parametrize("size_px", DEGENERATE_SIZE_PX)
@pytest.mark.parametrize("mode", RASTER_MODES)
def test_metal_degenerate_size_px_keeps_every_gradient_finite(size_px: float, mode: str) -> None:
    """size_px == 2**k / == 1 / == 0 are layer-selection branch switches."""
    _assert_metal_matches_reference(
        f"size_px={size_px} mode={mode}",
        probe_xyz=(0.1, -0.2, BOUNDARY_DEPTH),
        probe_size=size_px * BOUNDARY_DEPTH / FX,
        probe_conf=0.9,
        mode=mode,
    )


@pytest.mark.parametrize("mode", RASTER_MODES)
def test_metal_fragment_exactly_on_a_pixel_boundary(mode: str) -> None:
    """frac == 0 exactly: two corners get bilinear weight 0, one gets 1."""
    _assert_metal_matches_reference(
        f"boundary mode={mode}",
        probe_xyz=(BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH),
        probe_size=BOUNDARY_SIZE,
        probe_conf=0.9,
        mode=mode,
    )


@pytest.mark.parametrize("mode", ("trips", "broadcast"))
def test_metal_alpha_exactly_one_renders_finite(mode: str) -> None:
    """alpha == 1 exactly: the kernel is division free, so it needs no guard.

    The Metal forward loops `T *= (1 - a)` and blend_bwd carries the
    division-free suffix recurrences `U`/`Q`, so alpha == 1 is an ordinary
    value there. This pins that, and pins that the corrected torch reference
    (which needed a dtype-aware epsilon on `log1p(-alpha)`) now agrees with
    it rather than producing NaN.
    """
    _assert_metal_matches_reference(
        f"alpha==1 mode={mode}",
        probe_xyz=(BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH),
        probe_size=BOUNDARY_SIZE,
        probe_conf=1.0,
        mode=mode,
    )


def test_metal_survives_every_degenerate_point_at_once() -> None:
    """One render holding a z == 0 point, a boundary point and an alpha == 1 point."""
    device, dtype = "mps", torch.float32
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    xyz = torch.tensor(
        [
            list(ANCHOR_XYZ),
            [BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH],  # boundary + alpha == 1
            [1.0, 0.5, 0.0],                             # camera-space z exactly 0
            [-1.0, 0.25, -2.0],                          # behind the camera
            [0.5, -0.5, 0.5e-3],                         # inside the near plane
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    size = torch.tensor(
        [ANCHOR_SIZE, BOUNDARY_SIZE, 0.1, 0.1, 0.1], dtype=dtype, device=device, requires_grad=True
    )
    conf = torch.tensor(
        [ANCHOR_CONF, 1.0, 0.9, 0.9, 0.9], dtype=dtype, device=device, requires_grad=True
    )
    feat = torch.rand(5, 3, dtype=dtype, device=device, requires_grad=True)
    delta = torch.zeros(6, dtype=dtype, device=device, requires_grad=True)

    layers, aux = render_pyramid(
        xyz,
        size,
        feat,
        conf,
        K,
        torch.eye(3, dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
        IMAGE_HW,
        num_layers=NUM_LAYERS,
        mode="trips",
        alpha_min=0.0,
        pose_delta=delta,
    )
    flat = torch.cat([layer.reshape(-1) for layer in layers])
    assert torch.isfinite(flat).all()
    assert aux["num_fragments"] > 0
    (flat * torch.linspace(0.3, 1.7, flat.numel(), dtype=dtype, device=device)).sum().backward()
    for name, tensor in (("xyz", xyz), ("size", size), ("conf", conf), ("feat", feat), ("delta", delta)):
        assert tensor.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(tensor.grad).all(), f"non-finite d/d {name}: {tensor.grad}"
