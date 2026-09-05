"""numpy reference vs torch float64 reference: the dual-implementation gate.

Module: tests.test_raster_ref
Invariants: the two implementations share no code below the constants module
    -- ref_numpy projects with xform_a and loops per point/layer/corner;
    ref_torch projects with xform_b and composites with vectorised segment
    prefix sums. They must agree to float64 round-off on scenes that include
    a pixel stacked past the 16-fragment cap and points straddling every
    border, in both layer-selection modes.
Related docs: AGENTS.md section 7 ("implement transforms twice
    independently"); docs/TRIPS_REFERENCE.md sections 3 and 10.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_raster_scenes import as_numpy, make_scene

from trippy.constants import RASTER_MAX_FRAGS
from trippy.raster import render_pyramid, render_pyramid_numpy, render_pyramid_ref

NUM_LAYERS = 3
TOL = 1e-6


def _render_both(mode: str, seed: int = 0, num_points: int = 50, bg: bool = True):
    scene = make_scene(num_points=num_points, height=32, width=32, num_channels=3, seed=seed)
    npy = as_numpy(scene)
    bg_t = scene["bg"] if bg else None
    bg_n = npy["bg"] if bg else None
    layers_t, aux_t = render_pyramid_ref(
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=NUM_LAYERS,
        mode=mode,
        bg=bg_t,
    )
    layers_n, aux_n = render_pyramid_numpy(
        npy["xyz"],
        npy["size"],
        npy["feat"],
        npy["conf"],
        npy["K"],
        npy["R"],
        npy["t"],
        npy["image_hw"],
        num_layers=NUM_LAYERS,
        mode=mode,
        bg=bg_n,
    )
    return layers_t, aux_t, layers_n, aux_n


@pytest.mark.parametrize("mode", ["trilinear", "broadcast"])
def test_numpy_and_torch_references_agree(mode: str) -> None:
    """Composited colour, transmittance, depth and fragment counts all match."""
    layers_t, aux_t, layers_n, aux_n = _render_both(mode)
    assert aux_t["num_fragments"] == aux_n["num_fragments"]
    for layer in range(NUM_LAYERS):
        diff = np.abs(layers_t[layer].detach().numpy() - layers_n[layer]).max()
        assert diff < TOL, f"{mode} layer {layer}: max abs diff {diff}"
        assert np.abs(aux_t["t_final"][layer].numpy() - aux_n["t_final"][layer]).max() < TOL
        assert np.abs(aux_t["depth_sum"][layer].numpy() - aux_n["depth_sum"][layer]).max() < TOL
        assert np.array_equal(aux_t["n_used"][layer].numpy(), aux_n["n_used"][layer])


@pytest.mark.parametrize("mode", ["trilinear", "broadcast"])
def test_references_agree_without_background(mode: str) -> None:
    """bg=None must mean a plain zero background, not an implicit one."""
    layers_t, _, layers_n, _ = _render_both(mode, seed=5, bg=False)
    for layer in range(NUM_LAYERS):
        assert np.abs(layers_t[layer].detach().numpy() - layers_n[layer]).max() < TOL


def test_fragment_cap_is_reached_and_respected() -> None:
    """The fixture's stacked pixel really does hit the 16-fragment cap."""
    _, aux_t, _, aux_n = _render_both("trilinear")
    used = aux_t["n_used"][0].numpy()
    assert used.max() == RASTER_MAX_FRAGS
    assert aux_n["n_used"][0].max() == RASTER_MAX_FRAGS
    assert used.max() <= RASTER_MAX_FRAGS


def test_broadcast_writes_more_fragments_than_trilinear() -> None:
    """Mode "broadcast" is TRIPS's shipped default: every point, every layer."""
    _, aux_tri, _, _ = _render_both("trilinear")
    _, aux_bro, _, _ = _render_both("broadcast")
    assert aux_bro["num_fragments"] > aux_tri["num_fragments"]


def test_render_pyramid_cpu_dispatches_to_reference() -> None:
    """render_pyramid on CPU must be bit-identical to ref_torch (it *is* it)."""
    scene = make_scene(seed=1)
    args = (
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
    )
    layers_a, _ = render_pyramid(*args, num_layers=NUM_LAYERS, bg=scene["bg"])
    layers_b, _ = render_pyramid_ref(*args, num_layers=NUM_LAYERS, bg=scene["bg"], compute_dtype=None)
    for a, b in zip(layers_a, layers_b, strict=True):
        assert torch.equal(a, b)


def test_alpha_is_differentiable_through_the_cpu_path() -> None:
    """Gradients must reach conf/size/xyz through emission and compositing."""
    scene = make_scene(seed=2, num_points=40)
    xyz = scene["xyz"].clone().requires_grad_(True)
    conf = scene["conf"].clone().requires_grad_(True)
    size = scene["size"].clone().requires_grad_(True)
    layers, _ = render_pyramid_ref(
        xyz,
        size,
        scene["feat"],
        conf,
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=NUM_LAYERS,
        bg=scene["bg"],
    )
    sum(layer.sum() for layer in layers).backward()
    assert conf.grad is not None and torch.isfinite(conf.grad).all() and conf.grad.abs().sum() > 0
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all() and xyz.grad.abs().sum() > 0
    assert size.grad is not None and torch.isfinite(size.grad).all()


def test_unsupported_device_raises() -> None:
    """No silent fallback (AGENTS.md review checklist)."""
    scene = make_scene(seed=4, num_points=32)
    with pytest.raises(ValueError, match="mode"):
        render_pyramid_numpy(
            *(as_numpy(scene)[k] for k in ("xyz", "size", "feat", "conf", "K", "R", "t")),
            scene["image_hw"],
            mode="nonsense",
        )
