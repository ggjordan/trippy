"""Metal blend_fwd vs the torch float64 reference, on MPS. GPU-queue only.

Module: tests.test_raster_metal
Invariants: every test here is marked `gpu` and must run inside a
    scripts/gpu_submit.sh job (AGENTS.md section 6: never run MPS work
    directly). PYTORCH_ENABLE_MPS_FALLBACK=0 is set by the job wrapper, so an
    unsupported MPS op fails loudly instead of silently landing on the CPU.
    The first test pins the torch primitives the atomic-free design depends
    on (float bitcast, stable sort, searchsorted, bincount) so a failure
    there is diagnosable without another queue round trip.
Related docs: docs/ARCHITECTURE.md ("Validation strategy": Metal kernel vs
    reference on a 32x32 synthetic scene); docs/LIMITATIONS.md.
"""

from __future__ import annotations

import time

import pytest
import torch
from test_raster_scenes import make_scene

from trippy.constants import RASTER_MAX_FRAGS
from trippy.raster import render_pyramid, render_pyramid_ref, segment_offsets, sort_fragments

pytestmark = pytest.mark.gpu

# Tolerance for float32-on-Metal vs float64-in-torch on the 32x32 scenes.
FP32_TOL = 1e-4
# Synthetic full-resolution case (half-res kk frames are 1008 wide per SPEC).
BIG_HW = (756, 1008)
BIG_POINTS = 200_000
BIG_CHANNELS = 4
BIG_LAYERS = 5
BIG_REPEATS = 3
SMALL_LAYERS = 3


@pytest.fixture(scope="module", autouse=True)
def require_mps() -> None:
    """Skip the whole module cleanly when MPS is not present."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


def test_mps_torch_primitives() -> None:
    """The torch ops the sort/segment stage needs must exist on MPS."""
    device = torch.device("mps")
    depth = torch.rand(1024, device=device) + 0.5
    bits = depth.contiguous().view(torch.int32).to(torch.int64)
    assert bits.min().item() >= 0, "float32 bitcast must stay non-negative for positive depths"

    layer_pixel = torch.randint(0, 64, (1024,), device=device, dtype=torch.int64)
    key = layer_pixel * (1 << 32) + bits
    perm_composite = torch.argsort(key, stable=True)
    sorted_key = key.index_select(0, perm_composite)
    assert bool((sorted_key[1:] >= sorted_key[:-1]).all())
    print("mps: int64 composite stable argsort OK")

    sorted_lp = layer_pixel.index_select(0, perm_composite)
    off_search = segment_offsets(sorted_lp, 64, method="searchsorted")
    assert int(off_search[-1]) == 1024
    print("mps: searchsorted segment offsets OK")

    # The two fallbacks are not on the default path; report rather than fail
    # if a backend op is missing, so one queue round trip still yields the
    # result of every other test.
    for label, thunk in (
        ("two_pass sort", lambda: sort_fragments(layer_pixel, depth, method="two_pass")),
        ("bincount offsets", lambda: segment_offsets(sorted_lp, 64, method="bincount")),
    ):
        try:
            result = thunk()
        except NotImplementedError as exc:  # pragma: no cover (backend-dependent)
            print(f"mps: {label} UNSUPPORTED: {exc}")
            continue
        expected = (
            sort_fragments(layer_pixel, depth, method="composite")
            if label == "two_pass sort"
            else off_search
        )
        assert torch.equal(result, expected), f"{label} disagrees with the default path"
        print(f"mps: {label} agrees with the default path")


def _render_mps_and_reference(
    mode: str, seed: int, num_channels: int, num_layers: int, pixel_center: str = "half"
):
    scene = make_scene(
        num_points=50, height=32, width=32, num_channels=num_channels, seed=seed, dtype=torch.float64
    )
    ref_layers, ref_aux = render_pyramid_ref(
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=num_layers,
        mode=mode,
        bg=scene["bg"],
        pixel_center=pixel_center,
    )
    mps_args = [
        scene[key].to(torch.float32).to("mps")
        for key in ("xyz", "size", "feat", "conf", "K", "R", "t")
    ]
    gpu_layers, gpu_aux = render_pyramid(
        *mps_args,
        scene["image_hw"],
        num_layers=num_layers,
        mode=mode,
        bg=scene["bg"].to(torch.float32).to("mps"),
        pixel_center=pixel_center,
    )
    return ref_layers, ref_aux, gpu_layers, gpu_aux


@pytest.mark.parametrize("mode", ["trilinear", "broadcast", "trips"])
@pytest.mark.parametrize("num_channels", [3, 4])
def test_metal_matches_reference(mode: str, num_channels: int) -> None:
    """blend_fwd on MPS must match the float64 CPU reference within 1e-4."""
    ref_layers, ref_aux, gpu_layers, gpu_aux = _render_mps_and_reference(
        mode, seed=0, num_channels=num_channels, num_layers=SMALL_LAYERS
    )
    print(
        f"[{mode} C={num_channels}] fragments ref={ref_aux['num_fragments']} "
        f"metal={gpu_aux['num_fragments']}"
    )
    assert abs(ref_aux["num_fragments"] - gpu_aux["num_fragments"]) <= 2

    worst = 0.0
    for layer in range(SMALL_LAYERS):
        diff = (gpu_layers[layer].cpu().double() - ref_layers[layer]).abs().max().item()
        t_diff = (gpu_aux["t_final"][layer].cpu().double() - ref_aux["t_final"][layer]).abs().max()
        worst = max(worst, diff)
        print(f"[{mode} C={num_channels}] layer {layer}: max|out| {diff:.3e} max|T| {t_diff:.3e}")
        assert diff < FP32_TOL
        assert t_diff.item() < FP32_TOL
    print(f"[{mode} C={num_channels}] worst max abs diff over layers: {worst:.3e}")
    assert int(gpu_aux["n_used"][0].max().item()) <= RASTER_MAX_FRAGS


@pytest.mark.parametrize("pixel_center", ["half", "integer"])
def test_metal_matches_reference_trips_both_pixel_conventions(pixel_center: str) -> None:
    """Mode "trips" on MPS, in both pixel-centre conventions.

    `pixel_center="integer"` is the setting `trippy.render.parity`'s native
    engine renders a TRIPS checkpoint with, so it has to be exercised on the
    device that engine actually runs on.
    """
    ref_layers, ref_aux, gpu_layers, gpu_aux = _render_mps_and_reference(
        "trips", seed=2, num_channels=4, num_layers=SMALL_LAYERS, pixel_center=pixel_center
    )
    print(
        f"[trips {pixel_center}] fragments ref={ref_aux['num_fragments']} "
        f"metal={gpu_aux['num_fragments']} "
        f"per layer ref={ref_aux['fragments_per_layer'].tolist()} "
        f"metal={gpu_aux['fragments_per_layer'].cpu().tolist()}"
    )
    assert abs(ref_aux["num_fragments"] - gpu_aux["num_fragments"]) <= 2
    for layer in range(SMALL_LAYERS):
        diff = (gpu_layers[layer].cpu().double() - ref_layers[layer]).abs().max().item()
        print(f"[trips {pixel_center}] layer {layer}: max|out| {diff:.3e}")
        assert diff < FP32_TOL


def test_metal_sort_fallback_matches_composite() -> None:
    """The two_pass sort fallback renders identically on MPS."""
    scene = make_scene(num_points=50, height=32, width=32, num_channels=3, seed=7)
    args = [
        scene[key].to(torch.float32).to("mps")
        for key in ("xyz", "size", "feat", "conf", "K", "R", "t")
    ]
    layers_a, _ = render_pyramid(*args, scene["image_hw"], num_layers=SMALL_LAYERS)
    try:
        layers_b, _ = render_pyramid(
            *args, scene["image_hw"], num_layers=SMALL_LAYERS, sort_method="two_pass"
        )
    except NotImplementedError as exc:  # pragma: no cover (backend-dependent)
        pytest.skip(f"two_pass sort unsupported on MPS: {exc}")
    for a, b in zip(layers_a, layers_b, strict=True):
        assert torch.equal(a, b)


def test_metal_full_resolution_forward_timing() -> None:
    """A 1008x756, 200k-point forward pass runs; report milliseconds."""
    scene = make_scene(
        num_points=BIG_POINTS,
        height=BIG_HW[0],
        width=BIG_HW[1],
        num_channels=BIG_CHANNELS,
        seed=11,
        dtype=torch.float32,
        device="mps",
    )
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
    layers, aux = render_pyramid(*args, num_layers=BIG_LAYERS, bg=scene["bg"])  # warm-up + compile
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(BIG_REPEATS):
        layers, aux = render_pyramid(*args, num_layers=BIG_LAYERS, bg=scene["bg"])
    torch.mps.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / BIG_REPEATS

    print(
        f"[timing] {BIG_HW[1]}x{BIG_HW[0]} {BIG_POINTS} points C={BIG_CHANNELS} "
        f"L={BIG_LAYERS}: {elapsed_ms:.1f} ms/forward, "
        f"{aux['num_fragments']} fragments"
    )
    assert len(layers) == BIG_LAYERS
    assert layers[0].shape == (BIG_CHANNELS, BIG_HW[0], BIG_HW[1])
    for layer in layers:
        assert torch.isfinite(layer).all()
    assert aux["num_fragments"] > 0
