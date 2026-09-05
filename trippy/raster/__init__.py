"""Trilinear point splatting rasteriser: pyramid emission + sorted compositing.

Module: trippy.raster
Purpose: the forward half of the TRIPS port. A point set plus a camera goes
    in; an L-layer image pyramid of alpha-composited features comes out,
    ready for the U-Net decoder in trippy.net.

Pipeline (docs/ARCHITECTURE.md "Forward pass data flow"):

    project (trippy.geom.xform_b)
      -> emit 2x2 bilinear fragments per (point, layer)   [emit.py]
      -> sort by (layer, pixel, depth) + segment offsets  [sort.py]
      -> front-to-back alpha compositing                  [metal_src/blend_fwd.metal
                                                           on MPS, ref_torch.py on CPU]
      -> add background, split into per-layer images      [pyramid.py]

    backward: metal_src/blend_bwd.metal writes per-fragment d_alpha/d_feat;
    blend_autograd.py reduces d_feat onto points with index_add_ and lets
    autograd carry d_alpha back through emission to xyz / size / conf and
    the SE(3) pose delta.

Invariants:
    - No atomics anywhere. TRIPS uses `atomicAdd` for list-slot allocation and
      for every backward reduction; Metal via torch.mps.compile_shader has no
      64-bit atomics, so we pre-sort instead. This is trippy's redesign, not a
      port (docs/TRIPS_REFERENCE.md section 10.3).
    - Out-of-bounds fragments are dropped, never clamped.
    - Three implementations of the same maths -- numpy (ref_numpy), torch
      (ref_torch), Metal (metal_lib) -- with tests pinning them together.

Example:
    >>> import torch
    >>> from trippy.raster import render_pyramid
    >>> n = 64
    >>> g = torch.Generator().manual_seed(0)
    >>> xyz = torch.rand(n, 3, generator=g) * 2.0 - 1.0
    >>> xyz[:, 2] += 4.0                       # push the cloud in front of the camera
    >>> size = torch.full((n,), 0.02)          # world units (post-softplus)
    >>> feat = torch.rand(n, 3, generator=g)   # linear RGB
    >>> conf = torch.full((n,), 0.9)           # post-sigmoid confidence
    >>> K = torch.tensor([[64.0, 0.0, 16.0], [0.0, 64.0, 16.0], [0.0, 0.0, 1.0]])
    >>> R, t = torch.eye(3), torch.zeros(3)
    >>> layers, aux = render_pyramid(xyz, size, feat, conf, K, R, t, (32, 32), num_layers=3)
    >>> [tuple(layer.shape) for layer in layers]
    [(3, 32, 32), (3, 16, 16), (3, 8, 8)]
    >>> aux["t_final"][0].shape          # coverage / honesty map for layer 0
    torch.Size([32, 32])

Related docs: docs/ARCHITECTURE.md, docs/GEOMETRY.md,
    docs/TRIPS_REFERENCE.md sections 3, 10, 11, docs/LIMITATIONS.md.
"""

from trippy.raster.blend_autograd import BlendFunction, blend_fragments
from trippy.raster.emit import (
    Fragments,
    LayerGrid,
    SortedFragments,
    apply_pose_delta,
    build_sorted_fragments,
    cull_points,
    emit_fragments,
    layer_bounds,
    layer_factor,
    layer_grid,
    project_points,
)
from trippy.raster.pyramid import render_pyramid
from trippy.raster.ref_numpy import render_pyramid_numpy
from trippy.raster.ref_torch import composite_sorted, render_pyramid_ref, split_layers
from trippy.raster.sort import fragment_rank, segment_offsets, sort_fragments

__all__ = [
    "BlendFunction",
    "Fragments",
    "LayerGrid",
    "SortedFragments",
    "apply_pose_delta",
    "blend_fragments",
    "build_sorted_fragments",
    "composite_sorted",
    "cull_points",
    "emit_fragments",
    "fragment_rank",
    "layer_bounds",
    "layer_factor",
    "layer_grid",
    "project_points",
    "render_pyramid",
    "render_pyramid_numpy",
    "render_pyramid_ref",
    "segment_offsets",
    "sort_fragments",
    "split_layers",
]
