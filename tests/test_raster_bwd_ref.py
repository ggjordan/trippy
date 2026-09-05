"""CPU backward gate: float64 gradcheck, plus the blend_bwd formulas in python.

Module: tests.test_raster_bwd_ref
Invariants: never touches MPS. Two independent things are pinned here, both
    on CPU, so a GPU-queue round trip is only ever needed to validate the
    *Metal translation*, never the maths:

    1. `torch.autograd.gradcheck` in float64 on the whole reference render
       (trippy.raster.ref_torch), w.r.t. all five learnable inputs: point
       positions, sizes, confidences, features and the SE(3) pose delta.
    2. A line-for-line python transcription of metal_src/blend_bwd.metal
       (`blend_bwd_python`) checked against autograd on the very compositing
       function it differentiates. If the kernel's suffix recurrences were
       wrong, this fails without a GPU.

Related docs: docs/ARCHITECTURE.md ("Backward pass data flow"),
    docs/TRIPS_REFERENCE.md section 4, docs/LIMITATIONS.md.
"""

from __future__ import annotations

import pytest
import torch
from test_raster_bwd_scenes import make_smooth_scene

from trippy.constants import RASTER_MAX_FRAGS, RASTER_T_CUTOFF
from trippy.raster import blend_fragments, render_pyramid, render_pyramid_ref
from trippy.raster.emit import build_sorted_fragments, layer_grid
from trippy.raster.ref_torch import composite_sorted

# gradcheck settings. eps is the default central-difference step; the scene
# fixture keeps every discrete decision >= 0.05 away from its switch, i.e.
# ~5 * 10^4 steps, so a flip cannot be mistaken for a gradient error.
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-6
GRADCHECK_RTOL = 1e-4

# Tolerance for the python transcription of the kernel vs autograd, float64.
KERNEL_FORMULA_TOL = 1e-11

# The five learnable inputs the backward must reach (docs/SPEC.md v0.2.0).
GRAD_INPUTS = ("xyz", "size", "conf", "feat", "pose_delta")


def _render_flat(scene: dict, **overrides) -> torch.Tensor:
    """Render the fixture and flatten every layer into one vector.

    Args:
        scene: a make_smooth_scene() dict.
        overrides: tensors replacing the scene's own (the gradcheck inputs).

    Returns:
        (sum_l C * h_l * w_l,) float64 tensor -- the full pyramid, so
        gradcheck sees every output the loss could ever see.
    """
    args = dict(scene)
    args.update(overrides)
    layers, _ = render_pyramid_ref(
        args["xyz"],
        args["size"],
        args["feat"],
        args["conf"],
        args["K"],
        args["R"],
        args["t"],
        args["image_hw"],
        num_layers=args["num_layers"],
        bg=args["bg"],
        pose_delta=args["pose_delta"],
    )
    return torch.cat([layer.reshape(-1) for layer in layers])


@pytest.mark.parametrize("name", GRAD_INPUTS)
def test_gradcheck_reference_render(name: str) -> None:
    """float64 gradcheck of the whole render w.r.t. one learnable input."""
    scene = make_smooth_scene()
    variable = scene[name].clone().requires_grad_(True)

    def fn(value: torch.Tensor) -> torch.Tensor:
        return _render_flat(scene, **{name: value})

    assert torch.autograd.gradcheck(
        fn,
        (variable,),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        nondet_tol=0.0,
    )


def test_gradcheck_all_five_inputs_together() -> None:
    """The five gradients must also be right when perturbed jointly."""
    scene = make_smooth_scene()
    variables = tuple(scene[name].clone().requires_grad_(True) for name in GRAD_INPUTS)

    def fn(*values: torch.Tensor) -> torch.Tensor:
        return _render_flat(scene, **dict(zip(GRAD_INPUTS, values, strict=True)))

    assert torch.autograd.gradcheck(
        fn,
        variables,
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        nondet_tol=0.0,
    )


def test_every_learnable_input_gets_a_non_zero_gradient() -> None:
    """A passing gradcheck on an all-zero gradient would prove nothing."""
    scene = make_smooth_scene()
    variables = {name: scene[name].clone().requires_grad_(True) for name in GRAD_INPUTS}
    out = _render_flat(scene, **variables)
    # A non-uniform weighting, so no gradient can cancel by symmetry.
    weights = torch.linspace(0.3, 1.7, out.numel(), dtype=out.dtype)
    (out * weights).sum().backward()
    for name, variable in variables.items():
        assert variable.grad is not None, name
        assert torch.isfinite(variable.grad).all(), name
        assert variable.grad.abs().max() > 0.0, f"{name} received an all-zero gradient"


def test_zero_pose_delta_is_the_identity() -> None:
    """`pose_delta=zeros(6)` must render exactly like passing no delta."""
    scene = make_smooth_scene()
    common = (
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
    )
    layers_a, _ = render_pyramid_ref(*common, num_layers=scene["num_layers"], bg=scene["bg"])
    layers_b, _ = render_pyramid_ref(
        *common,
        num_layers=scene["num_layers"],
        bg=scene["bg"],
        pose_delta=torch.zeros(6, dtype=torch.float64),
    )
    for a, b in zip(layers_a, layers_b, strict=True):
        assert torch.allclose(a, b, atol=1e-12)


def test_pose_delta_reaches_render_pyramid_on_cpu() -> None:
    """The public entry point forwards pose_delta and returns its gradient."""
    scene = make_smooth_scene()
    delta = scene["pose_delta"].clone().requires_grad_(True)
    layers, _ = render_pyramid(
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=scene["num_layers"],
        bg=scene["bg"],
        pose_delta=delta,
    )
    sum(layer.sum() for layer in layers).backward()
    assert delta.grad is not None and delta.grad.abs().max() > 0.0


def test_pose_delta_rotation_gradient_vanishes_at_zero() -> None:
    """KNOWN WART, pinned deliberately: se3_exp is flat in phi at phi == 0.

    trippy.geom.xform_b.se3_exp writes its rotation as
    `a * |phi| * skew(phi / max(|phi|, EPS))`, which is second order in phi at
    the origin, so autograd hands back an exactly zero gradient for
    `delta[3:]` there. The true derivative is the SO(3) generator (magnitude
    1), as the finite difference below shows. Consequence for the trainer: a
    pose delta initialised at exactly zero will never learn rotation, only
    translation. Fixing it means changing xform_b.se3_exp (and re-running the
    xform_a/xform_b agreement test), which is out of scope for the backward
    pass; this test exists so the next person finds it immediately instead of
    debugging a pose refinement that silently does not rotate.

    Away from the origin the gradient is correct -- that is what
    test_gradcheck_reference_render[pose_delta] verifies, at |phi| ~ 0.037.
    """
    from trippy.geom.xform_b import se3_exp

    zero = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    rotation = se3_exp(zero)[:3, :3]
    analytic = torch.autograd.grad(rotation[2, 1], zero)[0]

    step = 1e-5
    plus, minus = torch.zeros(6, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
    plus[3], minus[3] = step, -step
    numeric = float((se3_exp(plus)[2, 1] - se3_exp(minus)[2, 1]) / (2.0 * step))

    assert abs(numeric - 1.0) < 1e-8, "the true derivative is the generator"
    assert float(analytic[3]) == 0.0, "autograd currently returns zero here"
    # Translation, by contrast, is right at the origin.
    translation = torch.autograd.grad(se3_exp(zero)[0, 3], zero)[0]
    assert abs(float(translation[0]) - 1.0) < 1e-12


def test_apply_pose_delta_rejects_bad_shapes() -> None:
    """A (4,) or (1, 6) delta is a caller bug, not something to broadcast."""
    from trippy.raster import apply_pose_delta

    R = torch.eye(3, dtype=torch.float64)
    t = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        apply_pose_delta(R, t, torch.zeros(4, dtype=torch.float64))


# --------------------------------------------------------------------------
# The blend_bwd formulas, transcribed from the kernel and checked on CPU.
# --------------------------------------------------------------------------


def blend_bwd_python(
    offsets: torch.Tensor,
    point_id: torch.Tensor,
    alpha: torch.Tensor,
    feat: torch.Tensor,
    n_used: torch.Tensor,
    grad_out: torch.Tensor,
    grad_t_final: torch.Tensor,
    max_frags: int = RASTER_MAX_FRAGS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Line-for-line python twin of metal_src/blend_bwd.metal.

    Deliberately written as the same two sequential passes with the same
    suffix recurrences (no torch vectorisation), so that a divergence between
    this and autograd points at the *formula*, and a divergence between this
    and the Metal kernel points at the *translation*.

    Args:
        offsets: (P + 1,) int64 segment starts.
        point_id: (F,) int64 index into `feat`.
        alpha: (F,) float per-fragment alpha.
        feat: (N, C) float point features.
        n_used: (P,) int, the forward's composited prefix length.
        grad_out: (P, C) float, dL/d out.
        grad_t_final: (P,) float, dL/d t_final.
        max_frags: per-pixel fragment cap.

    Returns:
        d_alpha: (F,) dL/d alpha_i.
        d_feat: (F, C) per-fragment dL/d feat[point_id_i].
    """
    num_fragments = int(alpha.shape[0])
    num_channels = int(feat.shape[1])
    d_alpha = torch.zeros(num_fragments, dtype=feat.dtype)
    d_feat = torch.zeros((num_fragments, num_channels), dtype=feat.dtype)

    for pixel in range(int(offsets.shape[0]) - 1):
        start = int(offsets[pixel])
        n = min(int(n_used[pixel]), max_frags)
        if n <= 0:
            continue

        # Pass 1: replay T_i.
        t_stack = []
        transmittance = torch.ones((), dtype=feat.dtype)
        for i in range(n):
            t_stack.append(transmittance)
            transmittance = transmittance * (1.0 - alpha[start + i])

        g_out = grad_out[pixel]
        g_t = grad_t_final[pixel]

        # Pass 2: back-to-front suffix recurrences (division free).
        suffix = torch.zeros(num_channels, dtype=feat.dtype)
        suffix_t = torch.ones((), dtype=feat.dtype)
        for i in range(n - 1, -1, -1):
            frag = start + i
            a = alpha[frag]
            t_i = t_stack[i]
            f_in = feat[int(point_id[frag])]
            d_feat[frag] = t_i * a * g_out
            d_alpha[frag] = (g_out * t_i * (f_in - suffix)).sum() - g_t * t_i * suffix_t
            suffix = a * f_in + (1.0 - a) * suffix
            suffix_t = suffix_t * (1.0 - a)
    return d_alpha, d_feat


def _sorted_fragments_for_scene(scene: dict):
    """Emit + sort the fixture's fragments (float64, CPU)."""
    grid = layer_grid(*scene["image_hw"], scene["num_layers"])
    frags = build_sorted_fragments(
        scene["xyz"],
        scene["size"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        grid,
        pose_delta=scene["pose_delta"],
    )
    return grid, frags


def test_kernel_formulas_match_autograd_on_the_reference() -> None:
    """blend_bwd's suffix recurrences == autograd through composite_sorted."""
    scene = make_smooth_scene()
    _, frags = _sorted_fragments_for_scene(scene)
    alpha = frags.alpha.detach().clone().requires_grad_(True)
    feat = scene["feat"].detach().clone().requires_grad_(True)

    out, t_final, n_used, _ = composite_sorted(
        frags.layer_pixel,
        frags.depth,
        frags.point_id,
        alpha,
        frags.offsets,
        feat,
        max_frags=RASTER_MAX_FRAGS,
        t_cutoff=RASTER_T_CUTOFF,
    )
    # Arbitrary, non-uniform upstream gradients, including a non-zero
    # dL/d t_final (what a background term produces).
    generator = torch.Generator().manual_seed(17)
    grad_out = torch.rand(out.shape, generator=generator, dtype=out.dtype) - 0.5
    grad_t_final = torch.rand(t_final.shape, generator=generator, dtype=out.dtype) - 0.5
    torch.autograd.backward((out, t_final), (grad_out, grad_t_final))

    d_alpha, d_feat = blend_bwd_python(
        frags.offsets,
        frags.point_id,
        frags.alpha.detach(),
        scene["feat"],
        n_used,
        grad_out,
        grad_t_final,
    )
    grad_feat = torch.zeros_like(feat).index_add_(0, frags.point_id, d_feat)

    assert alpha.grad is not None and feat.grad is not None
    assert (d_alpha - alpha.grad).abs().max() < KERNEL_FORMULA_TOL
    assert (grad_feat - feat.grad).abs().max() < KERNEL_FORMULA_TOL


def test_kernel_formulas_survive_alpha_close_to_one() -> None:
    """The division-free recurrences need no `1 / (1 - alpha)` guard.

    TRIPS divides by `1 - alpha + 1e-9` (RenderBackward.cu:290). We do not,
    so alpha == 1 -- a fully opaque fragment that hides everything behind it
    -- must still give the exact, finite gradient.
    """
    offsets = torch.tensor([0, 3], dtype=torch.int64)
    point_id = torch.tensor([0, 1, 2], dtype=torch.int64)
    feat = torch.tensor([[0.2, 0.9], [0.7, 0.1], [0.4, 0.5]], dtype=torch.float64)
    n_used = torch.tensor([3], dtype=torch.int64)
    grad_out = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    grad_t_final = torch.tensor([0.25], dtype=torch.float64)

    for leading in (0.5, 1.0 - 1e-9, 1.0):
        alpha = torch.tensor([leading, 0.6, 0.3], dtype=torch.float64)
        d_alpha, d_feat = blend_bwd_python(
            offsets, point_id, alpha, feat, n_used, grad_out, grad_t_final
        )
        assert torch.isfinite(d_alpha).all(), leading
        assert torch.isfinite(d_feat).all(), leading

    # At alpha_0 == 1 the closed form is exact and non-degenerate:
    #   d out / d a_0 = f_0 - (a_1 f_1 + (1 - a_1) a_2 f_2)
    alpha = torch.tensor([1.0, 0.6, 0.3], dtype=torch.float64)
    d_alpha, _ = blend_bwd_python(offsets, point_id, alpha, feat, n_used, grad_out, grad_t_final)
    behind = 0.6 * feat[1] + 0.4 * 0.3 * feat[2]
    expected = float((grad_out[0] * (feat[0] - behind)).sum()) - 0.25 * (0.4 * 0.7)
    assert abs(float(d_alpha[0]) - expected) < KERNEL_FORMULA_TOL


# --------------------------------------------------------------------------
# The device dispatcher.
# --------------------------------------------------------------------------


def test_blend_fragments_on_cpu_is_the_torch_reference() -> None:
    """CPU must take the differentiable pure-torch path, bit for bit."""
    scene = make_smooth_scene()
    _, frags = _sorted_fragments_for_scene(scene)
    feat = scene["feat"].clone().requires_grad_(True)
    out, t_final, n_used, depth_sum = blend_fragments(frags, feat)
    expected = composite_sorted(
        frags.layer_pixel,
        frags.depth,
        frags.point_id,
        frags.alpha,
        frags.offsets,
        scene["feat"],
    )
    for got, want in zip((out, t_final, n_used, depth_sum), expected, strict=True):
        assert torch.equal(got.detach(), want.detach())
    assert out.requires_grad, "the CPU path must stay connected to autograd"


def test_blend_fragments_rejects_mixed_devices() -> None:
    """A features/fragments device mismatch is silent garbage in Metal."""
    scene = make_smooth_scene()
    _, frags = _sorted_fragments_for_scene(scene)
    frags.alpha = frags.alpha.to("meta")
    with pytest.raises(ValueError, match="different device|but fragments are on"):
        blend_fragments(frags, scene["feat"])
