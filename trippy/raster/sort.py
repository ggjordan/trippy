"""Fragment ordering: sort by (layer, pixel, depth) and build segment offsets.

Module: trippy.raster.sort
Purpose: the atomic-free half of the rasteriser design. TRIPS allocates
    per-pixel list slots with `atomicAdd` and bitonic-sorts inside a warp
    (docs/TRIPS_REFERENCE.md section 3); Metal via torch.mps.compile_shader
    has no 64-bit atomics, so we instead sort the whole fragment list once
    and hand each compositing thread a contiguous, read-only segment
    (docs/ARCHITECTURE.md "Core principle: No atomics anywhere" -- our
    redesign, not a port).
Invariants:
    - `layer_pixel` is the flat pyramid index from trippy.raster.emit; sorting
      by it and then by depth *is* sorting by (layer, pixel, depth), because
      the flat index is layer-major.
    - Both sort methods produce the *same* permutation for the same inputs
      (tests/test_raster_sort.py), so the fallback can be switched on at any
      time without changing rendered output.
    - Depth is compared at float32 resolution in both methods (the composite
      key can only hold 32 bits of depth), so a float64 caller gets the same
      order as the float32 GPU path.
Units: depth is camera-space z in world units and must be > 0 (points behind
    the near plane are culled in trippy.raster.emit).
Related docs: docs/ARCHITECTURE.md; docs/LIMITATIONS.md ("int64 argsort at
    50M elements unverified").
"""

from __future__ import annotations

import torch
from torch import Tensor

from trippy.constants import RASTER_SORT_DEPTH_BITS, RASTER_SORT_MAX_LAYER_PIXELS

SORT_METHODS = ("composite", "two_pass")
SEGMENT_METHODS = ("searchsorted", "bincount")

# Smallest positive float32; depth is clamped to it before its bit pattern is
# used as a sort key, because the "IEEE bits sort like the value" trick only
# holds for non-negative floats.
_MIN_SORT_DEPTH = 1e-38


def _depth_key_bits(depth: Tensor) -> Tensor:
    """float32 depth -> int64 whose ordering matches the depth ordering.

    Args:
        depth: (F,) float tensor, > 0 (clamped defensively).

    Returns:
        (F,) int64 in [0, 2**31), the IEEE-754 bit pattern of the float32
        depth. For non-negative floats this pattern is monotonically
        non-decreasing in the value.
    """
    d32 = torch.clamp(depth, min=_MIN_SORT_DEPTH).to(torch.float32).contiguous()
    return d32.view(torch.int32).to(torch.int64)


def sort_fragments(
    layer_pixel: Tensor,
    depth: Tensor,
    method: str = "composite",
    stable: bool = True,
    max_layer_pixel: int | None = None,
) -> Tensor:
    """Permutation that orders fragments by (layer, pixel) then depth ascending.

    Args:
        layer_pixel: (F,) int64, flat pyramid layer-pixel index (see
            trippy.raster.emit.LayerGrid).
        depth: (F,) float, camera-space z, world units, > 0.
        method: "composite" -- one int64 argsort on
            `layer_pixel * 2**32 + float32_bits(depth)`; or "two_pass" -- two
            stable sorts (depth, then layer_pixel), the fallback for backends
            whose 64-bit sort is slow or unstable at 10^7+ elements.
        stable: use a stable sort. Required for the two methods to agree on
            depth ties; pass False only if a backend lacks stable sort.
        max_layer_pixel: caller-supplied upper bound on `layer_pixel.max()`
            for the composite key's range check. `None` reads the real
            maximum off the device, which costs a full device->host
            synchronisation on MPS; emission already guarantees every index
            is < `LayerGrid.total`, so `build_sorted_fragments` passes
            `grid.total - 1` and the sync disappears. A bound that is too
            small is a caller bug and raises exactly as the measured maximum
            would.

    Returns:
        perm: (F,) int64, such that `layer_pixel[perm]` is non-decreasing and
        `depth[perm]` is non-decreasing within each layer_pixel run.
    """
    if method not in SORT_METHODS:
        raise ValueError(f"method must be one of {SORT_METHODS}, got {method!r}")
    if layer_pixel.shape != depth.shape:
        raise ValueError(f"shape mismatch: layer_pixel {layer_pixel.shape} vs depth {depth.shape}")
    if layer_pixel.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=layer_pixel.device)

    if method == "composite":
        # `.max().item()` is a device->host sync; skip it when the caller can
        # already bound the index space (build_sorted_fragments always can).
        max_lp = int(layer_pixel.max().item()) if max_layer_pixel is None else int(max_layer_pixel)
        if max_lp >= RASTER_SORT_MAX_LAYER_PIXELS:
            raise ValueError(
                f"layer_pixel index {max_lp} does not fit the composite sort key "
                f"(limit {RASTER_SORT_MAX_LAYER_PIXELS}); use method='two_pass'"
            )
        # Multiply-add rather than shift-or: the two are equivalent here
        # (the depth bits are < 2**31, so they never collide with the pixel
        # part) and integer multiply is more widely implemented across
        # backends than an int64 bit shift.
        key = layer_pixel * (1 << RASTER_SORT_DEPTH_BITS) + _depth_key_bits(depth)
        return torch.argsort(key, stable=stable)

    perm_depth = torch.argsort(torch.clamp(depth, min=_MIN_SORT_DEPTH).to(torch.float32), stable=True)
    perm_pixel = torch.argsort(layer_pixel.index_select(0, perm_depth), stable=True)
    return perm_depth.index_select(0, perm_pixel)


def segment_offsets(
    layer_pixel_sorted: Tensor,
    total_layer_pixels: int,
    method: str = "searchsorted",
) -> Tensor:
    """Per-(layer, pixel) segment boundaries in the sorted fragment list.

    Args:
        layer_pixel_sorted: (F,) int64, non-decreasing flat layer-pixel
            indices (the output of applying `sort_fragments`' permutation).
        total_layer_pixels: P, LayerGrid.total.
        method: "searchsorted" (works on any backend that has
            torch.searchsorted) or "bincount" (bincount + cumsum).

    Returns:
        offsets: (P + 1,) int64. Fragments of layer-pixel p occupy
        `[offsets[p], offsets[p + 1])`; `offsets[P] == F`.
    """
    if method not in SEGMENT_METHODS:
        raise ValueError(f"method must be one of {SEGMENT_METHODS}, got {method!r}")
    device = layer_pixel_sorted.device
    if method == "searchsorted":
        bins = torch.arange(total_layer_pixels + 1, dtype=torch.int64, device=device)
        return torch.searchsorted(layer_pixel_sorted, bins, right=False).to(torch.int64)

    counts = torch.bincount(layer_pixel_sorted, minlength=total_layer_pixels)
    offsets = torch.zeros(total_layer_pixels + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def fragment_rank(layer_pixel_sorted: Tensor, offsets: Tensor) -> Tensor:
    """0-based position of each sorted fragment inside its own segment.

    Args:
        layer_pixel_sorted: (F,) int64, non-decreasing.
        offsets: (P + 1,) int64 from segment_offsets.

    Returns:
        (F,) int64 rank; rank 0 is the fragment nearest the camera for that
        layer-pixel. Used to apply the per-pixel fragment cap without a loop.
    """
    positions = torch.arange(layer_pixel_sorted.shape[0], dtype=torch.int64, device=offsets.device)
    return positions - offsets.index_select(0, layer_pixel_sorted)
