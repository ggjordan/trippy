"""Autograd bridge for the compositing step: Metal on MPS, torch on CPU.

Module: trippy.raster.blend_autograd
Purpose: make the pyramid rasteriser differentiable on MPS. The Metal
    forward kernel (trippy.raster.metal_lib.blend_fwd) knows nothing about
    autograd, so `BlendFunction` wraps the fwd/bwd kernel pair in a
    torch.autograd.Function and reduces the kernel's *per-fragment*
    gradients onto per-point gradients with `index_add_`. `blend_fragments`
    is the device dispatcher: MPS goes through the kernels, everything else
    goes through the already-differentiable torch reference
    (trippy.raster.ref_torch.composite_sorted), so the calling code is
    identical on both devices.

Why per-fragment, then index_add_ (docs/ARCHITECTURE.md "No atomics"):
    TRIPS reduces gradients onto points with `atomicAdd` inside the kernel
    (RenderBackward.cu:384-465). Metal via torch.mps.compile_shader has no
    64-bit atomics, and float atomics on the same address would make the
    result run-order dependent. Instead every fragment is owned by exactly
    one thread, which writes its own slot with no contention; the sum over
    fragments that share a point id is then a single `index_add_` in torch.

What carries a gradient:
    - alpha (F,) -> emission (bilinear weight, confidence, layer factor) ->
      uv/size/depth -> xyz and the camera pose.
    - feat (N, C) -> directly, via the index_add_ reduction.
    - t_final (P,) -> the background term `out += t_final * bg`.
    What does NOT: fragment *ordering* (a discrete function of depth,
    piecewise constant, exactly as in TRIPS), `n_used`, and -- on MPS only --
    `depth_sum`, which is returned detached (docs/LIMITATIONS.md).

Invariants:
    - The backward replays exactly the prefix the forward composited: it is
      handed the forward's `n_used`, never a re-derived cutoff.
    - No `.detach()` / `.item()` on the alpha or feature path; sort
      permutations are applied with `index_select` on detached integer
      indices, which is correct because the ordering is piecewise constant.
    - Double backward is not supported (`once_differentiable`): a second
      backward raises instead of silently returning wrong numbers.
Units / frames: see trippy.raster.emit (alpha dimensionless, depth camera-
    space z in world units).
Related docs: docs/ARCHITECTURE.md ("Backward pass data flow"),
    docs/TRIPS_REFERENCE.md section 4, docs/LIMITATIONS.md.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from trippy.constants import RASTER_MAX_FRAGS, RASTER_T_CUTOFF
from trippy.raster import metal_lib
from trippy.raster.emit import SortedFragments
from trippy.raster.ref_torch import composite_sorted


class BlendFunction(torch.autograd.Function):
    """torch.autograd.Function around the blend_fwd / blend_bwd Metal pair.

    Shapes (F fragments, P layer-pixels, N points, C channels):
        inputs  alpha (F,) float32, feat (N, C) float32, offsets (P + 1,)
                int32, point_id (F,) int32, depth (F,) float32
        outputs out (P, C), t_final (P,), n_used (P,) int32,
                depth_sum (P,)
    Only `alpha` and `feat` receive gradients; `offsets`, `point_id` and
    `depth` are indices/keys and are passed detached (see module docstring).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        alpha: Tensor,
        feat: Tensor,
        offsets: Tensor,
        point_id: Tensor,
        depth: Tensor,
        max_frags: int,
        t_cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Composite one image's sorted fragments with the Metal kernel.

        Args:
            alpha: (F,) float32 MPS, per-fragment alpha in (0, 1).
            feat: (N, C) float32 MPS, per-point features.
            offsets: (P + 1,) int32 MPS, segment starts.
            point_id: (F,) int32 MPS, index into `feat`.
            depth: (F,) float32 MPS, camera-space z, world units.
            max_frags: per-layer-pixel fragment cap.
            t_cutoff: transmittance stop threshold.

        Returns:
            (out, t_final, n_used, depth_sum) as documented on
            trippy.raster.metal_lib.blend_fwd. `n_used` and `depth_sum` are
            marked non-differentiable.
        """
        alpha_c = alpha.contiguous()
        feat_c = feat.contiguous()
        out, t_final, n_used, depth_sum = metal_lib.blend_fwd(
            offsets,
            point_id,
            alpha_c,
            depth,
            feat_c,
            max_frags=max_frags,
            t_cutoff=t_cutoff,
        )
        ctx.save_for_backward(alpha_c, feat_c, offsets, point_id, n_used)
        ctx.max_frags = int(max_frags)
        ctx.num_points = int(feat_c.shape[0])
        ctx.mark_non_differentiable(n_used, depth_sum)
        return out, t_final, n_used, depth_sum

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx,
        grad_out: Tensor | None,
        grad_t_final: Tensor | None,
        grad_n_used: Tensor | None,
        grad_depth_sum: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        """Run blend_bwd, then reduce per-fragment feature grads onto points.

        Args:
            grad_out: (P, C) or None, dL/d out.
            grad_t_final: (P,) or None, dL/d t_final.
            grad_n_used, grad_depth_sum: always None (both outputs are marked
                non-differentiable); accepted to match the output arity.

        Returns:
            (grad_alpha (F,), grad_feat (N, C), None, None, None, None, None).
        """
        del grad_n_used, grad_depth_sum
        alpha, feat, offsets, point_id, n_used = ctx.saved_tensors
        num_pixels = int(offsets.shape[0]) - 1
        num_channels = int(feat.shape[1])
        device = feat.device

        if grad_out is None:
            grad_out = torch.zeros((num_pixels, num_channels), dtype=torch.float32, device=device)
        if grad_t_final is None:
            grad_t_final = torch.zeros(num_pixels, dtype=torch.float32, device=device)

        d_alpha, d_feat = metal_lib.blend_bwd(
            offsets,
            point_id,
            alpha,
            feat,
            n_used,
            grad_out.to(torch.float32).contiguous(),
            grad_t_final.to(torch.float32).contiguous(),
            max_frags=ctx.max_frags,
        )

        grad_alpha = d_alpha if ctx.needs_input_grad[0] else None
        grad_feat = None
        if ctx.needs_input_grad[1]:
            # The only reduction in the whole backward, and it is torch's,
            # not the kernel's -- this is what removes the need for atomics.
            grad_feat = torch.zeros(
                (ctx.num_points, num_channels), dtype=torch.float32, device=device
            )
            grad_feat.index_add_(0, point_id.to(torch.int64), d_feat)
        return grad_alpha, grad_feat, None, None, None, None, None


def blend_fragments(
    frags: SortedFragments,
    feat: Tensor,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Composite sorted fragments on whichever device they live on.

    MPS dispatches to `BlendFunction` (Metal fwd + bwd); every other device
    dispatches to `trippy.raster.ref_torch.composite_sorted`, which is plain
    differentiable torch. Both honour the same fragment cap and transmittance
    cutoff, so the two paths differ only in float32-vs-float64 rounding.

    Args:
        frags: SortedFragments for one image (see trippy.raster.emit). Its
            `alpha` may carry autograd history; its integer index tensors and
            `depth` are treated as constants.
        feat: (N, C) float per-point features, on the same device as `frags`.
            May carry autograd history.
        max_frags: per-layer-pixel fragment cap.
        t_cutoff: transmittance stop threshold.

    Returns:
        out: (P, C) sum of T * alpha * feat, background NOT added.
        t_final: (P,) transmittance left after the composited prefix.
        n_used: (P,) integer fragment count (never differentiable).
        depth_sum: (P,) sum of T * alpha * depth. Differentiable on CPU;
            detached on MPS (docs/LIMITATIONS.md).

    Raises:
        ValueError: if `feat` and the fragments are on different devices.
    """
    if feat.device != frags.alpha.device:
        raise ValueError(
            f"feat is on {feat.device} but fragments are on {frags.alpha.device}"
        )
    if feat.device.type != "mps":
        return composite_sorted(
            frags.layer_pixel,
            frags.depth,
            frags.point_id,
            frags.alpha,
            frags.offsets,
            feat,
            max_frags=max_frags,
            t_cutoff=t_cutoff,
        )
    return BlendFunction.apply(
        frags.alpha.to(torch.float32),
        feat.to(torch.float32),
        frags.offsets.to(torch.int32).contiguous(),
        frags.point_id.to(torch.int32).contiguous(),
        frags.depth.detach().to(torch.float32).contiguous(),
        int(max_frags),
        float(t_cutoff),
    )
