"""The post-sort fragment cap is bit-identical on MPS. GPU-queue only.

Module: tests.test_raster_cap_metal
Invariants under test: `render_pyramid(..., cap_to_max_frags=True)` -- the MPS
    default since perf/train-step -- must produce **byte-identical** layers,
    `t_final`, `n_used` and `depth_sum` to `cap_to_max_frags=False`, and
    identical gradients, on the real Metal kernels. That is the whole
    correctness argument for the cap: `blend_fwd` checks `used >= MAX_FRAGS`
    *before* consuming a fragment, so a fragment past a layer-pixel's first
    `max_frags` is never read, and neither is it read by `blend_bwd` (which
    replays the forward's own `n_used` prefix). Unlike the CPU twin, whose
    transmittance comes from a global prefix sum, the kernel loops
    sequentially per segment, so truncation cannot move a single bit.

    Every test here is marked `gpu` and must run inside a
    scripts/gpu_submit.sh job (AGENTS.md section 6). Synthetic scenes only.
Related docs: docs/ARCHITECTURE.md ("Where a training step's time goes");
    trippy.raster.emit._cap_segment_indices.
"""

from __future__ import annotations

import pytest
import torch
from test_raster_scenes import make_scene

from trippy.raster import render_pyramid

pytestmark = pytest.mark.gpu

# Wide enough that "broadcast" overflows the 16-deep list on the coarse layers,
# which is exactly the regime the cap exists for.
CAP_HW = (192, 256)
CAP_POINTS = 60_000
CAP_LAYERS = 5
CAP_CHANNELS = 4


@pytest.fixture(scope="module", autouse=True)
def require_mps() -> None:
    """Skip the whole module cleanly when MPS is not present."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


def _mps_scene(seed: int) -> dict:
    scene = make_scene(
        num_points=CAP_POINTS,
        height=CAP_HW[0],
        width=CAP_HW[1],
        num_channels=CAP_CHANNELS,
        seed=seed,
        dtype=torch.float32,
        device="mps",
    )
    return scene


@pytest.mark.parametrize("mode", ["broadcast", "trips", "trilinear"])
def test_cap_is_byte_identical_on_mps(mode: str) -> None:
    scene = _mps_scene(seed=11)

    def render(cap: bool):
        return render_pyramid(
            scene["xyz"],
            scene["size"],
            scene["feat"],
            scene["conf"],
            scene["K"],
            scene["R"],
            scene["t"],
            scene["image_hw"],
            num_layers=CAP_LAYERS,
            mode=mode,
            bg=scene["bg"],
            cap_to_max_frags=cap,
        )

    layers_off, aux_off = render(False)
    layers_on, aux_on = render(True)

    for level, (a, b) in enumerate(zip(layers_off, layers_on, strict=True)):
        assert torch.equal(a, b), f"layer {level} differs with the cap on"
    for key in ("t_final", "n_used", "depth_sum"):
        for level, (a, b) in enumerate(zip(aux_off[key], aux_on[key], strict=True)):
            assert torch.equal(a, b), f"{key} level {level} differs with the cap on"
    # The reported counts describe emission, so they must not move either.
    assert aux_off["num_fragments"] == aux_on["num_fragments"]
    assert torch.equal(aux_off["fragments_per_layer"], aux_on["fragments_per_layer"])
    kept_bound = aux_off["grid"].total * 16
    print(
        f"mode={mode}: {aux_off['num_fragments']} fragments emitted, "
        f"at most {kept_bound} compositable"
    )


def test_cap_gradients_match_within_mps_run_to_run_noise() -> None:
    """Gradients agree with the cap on -- to the precision MPS itself offers.

    The forward is byte-identical (test above). The backward cannot be held to
    that bar, and the reason is not the cap: reducing per-fragment gradients
    onto points is a float `index_add_`, and MPS's is not deterministic, so
    **two identical runs already disagree in the last bits**. This test
    measures that noise floor first -- cap on vs cap on, same inputs -- and
    then requires the cap-on-vs-cap-off difference to be no larger. If the cap
    ever really did change a gradient, it would show up as a difference above
    the machine's own run-to-run spread rather than being hidden by a
    hand-picked tolerance.
    """
    scene = _mps_scene(seed=12)

    def grads(cap: bool):
        xyz = scene["xyz"].detach().clone().requires_grad_(True)
        size = scene["size"].detach().clone().requires_grad_(True)
        conf = scene["conf"].detach().clone().requires_grad_(True)
        feat = scene["feat"].detach().clone().requires_grad_(True)
        layers, aux = render_pyramid(
            xyz,
            size,
            feat,
            conf,
            scene["K"],
            scene["R"],
            scene["t"],
            scene["image_hw"],
            num_layers=CAP_LAYERS,
            mode="trips",
            bg=scene["bg"],
            cap_to_max_frags=cap,
        )
        loss = sum(layer.square().sum() for layer in layers)
        loss = loss + sum(t_final.sum() for t_final in aux["t_final"])
        return torch.autograd.grad(loss, [xyz, size, conf, feat], allow_unused=True)

    names = ("xyz", "size", "conf", "feat")
    capped_a = grads(True)
    capped_b = grads(True)
    uncapped = grads(False)

    def spread(left, right) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, a, b in zip(names, left, right, strict=True):
            if a is None or b is None:
                assert a is None and b is None, f"{name}: only one side has a gradient"
                continue
            scale = max(float(a.abs().max()), 1e-30)
            out[name] = float((a - b).abs().max()) / scale
        return out

    noise = spread(capped_a, capped_b)
    delta = spread(uncapped, capped_a)
    print(f"cap: run-to-run noise floor {noise}")
    print(f"cap: cap-off vs cap-on      {delta}")

    for name in noise:
        # `+ 1e-7` so a parameter whose reduction happens to be deterministic
        # (noise floor exactly 0) is still held to float32 rounding rather than
        # to exact equality.
        assert delta[name] <= noise[name] + 1e-7, (
            f"{name} gradient moved by {delta[name]:.3e} with the cap on, above this "
            f"machine's own run-to-run spread of {noise[name]:.3e}"
        )
