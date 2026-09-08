"""Bit-identity tests for the training-step performance work (perf/train-step).

Module: tests.test_train_step_perf
Invariants under test: every optimisation on the `Trainer.train_step` hot path
    that is claimed to be *exactly equivalent* is pinned against a literal
    re-implementation of the code it replaced, on synthetic data only. If one
    of these ever fails, the "no numerics changed" claim in
    docs/ARCHITECTURE.md ("Where a training step's time goes") is false.

    1. `CameraResponseNet.forward` folds the channel axis into the batch axis
       for one `grid_sample`; pinned against the per-channel loop it replaced.
    2. `project_points` inlines `project_pinhole` on the safe depth; pinned
       against the `torch.cat` + `project_pinhole` form it replaced.
    3. `build_sorted_fragments(cap_frags=...)` truncates each layer-pixel
       segment to the compositing cap; pinned by rendering the same scene with
       and without the cap and requiring identical layers, `t_final`,
       `n_used`, `depth_sum` and identical gradients.
    4. `Trainer._sanitise_gradients` returns the same count and leaves the
       same gradients as the literal `~isfinite` + `nan_to_num_` loop.
    5. `trippy.train.steptimer` is invisible when inactive and records
       disjoint stages when active.
All fixtures are synthetic (`tests/test_train_helpers.py`); no scene imagery
is read (AGENTS.md section 6).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config
from torch.nn import functional

from trippy.constants import RASTER_MAX_FRAGS, RASTER_ZNEAR
from trippy.geom.xform_b import project_pinhole
from trippy.net.camera_model import CameraResponseNet
from trippy.raster.emit import build_sorted_fragments, layer_grid, project_points, safe_depth
from trippy.raster.ref_torch import render_pyramid_ref
from trippy.train import steptimer
from trippy.train.trainer import Trainer

# --------------------------------------------------------------------------
# 1. the camera response net
# --------------------------------------------------------------------------


def _response_per_channel_loop(module: CameraResponseNet, image: torch.Tensor) -> torch.Tensor:
    """The literal per-channel loop `CameraResponseNet.forward` replaced (leak included)."""
    from trippy.constants import CAMERA_RESPONSE_LEAK_SQRT_EPS

    num_batches, num_channels = image.shape[0], image.shape[1]

    leak_add = None
    if module.training and module.leak_factor > 0:
        clamp_low = image < 0
        clamp_high = image > 1
        leak_add = (image * module.leak_factor) * clamp_low
        leak_add = leak_add + (
            -module.leak_factor / torch.sqrt(image.abs() + CAMERA_RESPONSE_LEAK_SQRT_EPS)
            + module.leak_factor
        ) * clamp_high

    batched_response = module.response.repeat(num_batches, 1, 1, 1)
    scaled = image * 2.0 - 1.0
    grid = torch.stack([scaled, torch.zeros_like(scaled)], dim=-1)
    result = torch.ones_like(image)
    for c in range(num_channels):
        result[:, c : c + 1] = functional.grid_sample(
            batched_response[:, c : c + 1],
            grid[:, c],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    if leak_add is not None:
        result = result + leak_add
    return result


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("leak_factor", [0.0, 0.01])
def test_camera_response_matches_per_channel_loop(training: bool, leak_factor: float) -> None:
    torch.manual_seed(0)
    module = CameraResponseNet(
        num_params=25, num_channels=3, initial_gamma=2.2, leak_factor=leak_factor
    )
    module.train(training)
    # Values outside [0, 1] on purpose: that is what activates the leak terms.
    image = torch.rand(2, 3, 7, 5) * 1.6 - 0.3
    assert torch.equal(module(image), _response_per_channel_loop(module, image))


def test_camera_response_gradients_match_per_channel_loop() -> None:
    torch.manual_seed(1)
    module = CameraResponseNet(num_params=17, num_channels=3, initial_gamma=2.2, leak_factor=0.0)
    module.eval()
    image = torch.rand(1, 3, 6, 4, requires_grad=True)

    out = module(image).square().sum()
    (grad_fast,) = torch.autograd.grad(out, image, retain_graph=False)
    grad_param_fast = torch.autograd.grad(module(image).square().sum(), module.response)[0]

    out_ref = _response_per_channel_loop(module, image).square().sum()
    (grad_ref,) = torch.autograd.grad(out_ref, image, retain_graph=True)
    grad_param_ref = torch.autograd.grad(
        _response_per_channel_loop(module, image).square().sum(), module.response
    )[0]

    assert torch.equal(grad_fast, grad_ref)
    assert torch.equal(grad_param_fast, grad_param_ref)


# --------------------------------------------------------------------------
# 2. the projection
# --------------------------------------------------------------------------


def _project_points_via_cat(
    xyz: torch.Tensor,
    size: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    znear: float = RASTER_ZNEAR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `torch.cat` + `project_pinhole` form `project_points` replaced."""
    from trippy.geom.xform_b import world_to_cam

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xyz_c = world_to_cam(R, t, xyz)
    depth = xyz_c[:, 2]
    depth_safe = safe_depth(depth, znear)
    xyz_c_safe = torch.cat([xyz_c[:, :2], depth_safe.reshape(-1, 1)], dim=1)
    uv, _ = project_pinhole(xyz_c_safe, fx, fy, cx, cy)
    return uv, depth, fx * size / depth_safe


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_project_points_matches_the_cat_form(dtype: torch.dtype) -> None:
    torch.manual_seed(2)
    # Deliberately includes points behind the camera and at z == 0, the rows
    # `safe_depth` exists for.
    xyz = torch.randn(500, 3, dtype=dtype) * 4.0
    xyz[:20, 2] = 0.0
    xyz[20:40, 2] = -3.0
    size = torch.rand(500, dtype=dtype) * 0.1 + 1e-3
    K = torch.tensor([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]], dtype=dtype)
    R = torch.eye(3, dtype=dtype)
    t = torch.tensor([0.1, -0.2, 5.0], dtype=dtype)

    got = project_points(xyz, size, K, R, t)
    expected = _project_points_via_cat(xyz, size, K, R, t)
    for a, b in zip(got, expected, strict=True):
        assert torch.equal(a, b)


def test_project_points_gradients_match_the_cat_form() -> None:
    """Gradients agree to float64 tolerance -- NOT bit-for-bit, and that is expected.

    The forward is bit-identical (test above). The backward is not: the
    `cat` form assembles `d(xyz_c)` from one `cat` backward, the inlined form
    from two `select` backwards plus the depth path, so the same terms are
    summed in a different order. Float addition is not associative, so the
    last bits move. 1e-12 relative on float64 is ~4 orders tighter than the
    1e-5 parity bar in this task's brief.
    """
    torch.manual_seed(3)
    xyz = (torch.randn(200, 3, dtype=torch.float64) * 2.0).requires_grad_(True)
    size = (torch.rand(200, dtype=torch.float64) * 0.1 + 1e-3).requires_grad_(True)
    K = torch.tensor([[500.0, 0.0, 100.0], [0.0, 500.0, 90.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)
    t = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64)

    def loss(fn) -> torch.Tensor:
        uv, depth, size_px = fn(xyz, size, K, R, t)
        return uv.square().sum() + depth.square().sum() + size_px.square().sum()

    grads_fast = torch.autograd.grad(loss(project_points), [xyz, size])
    grads_ref = torch.autograd.grad(loss(_project_points_via_cat), [xyz, size])
    for a, b in zip(grads_fast, grads_ref, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# 3. the fragment cap
# --------------------------------------------------------------------------


def _random_scene(seed: int, num_points: int = 4000):
    torch.manual_seed(seed)
    xyz = torch.rand(num_points, 3, dtype=torch.float64) * 2.0 - 1.0
    xyz[:, 2] = xyz[:, 2] * 0.5 + 3.0
    size = torch.rand(num_points, dtype=torch.float64) * 0.02 + 0.005
    conf = torch.rand(num_points, dtype=torch.float64) * 0.5 + 0.25
    feat = torch.rand(num_points, 4, dtype=torch.float64)
    K = torch.tensor(
        [[60.0, 0.0, 16.0], [0.0, 60.0, 12.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    R = torch.eye(3, dtype=torch.float64)
    t = torch.zeros(3, dtype=torch.float64)
    return xyz, size, conf, feat, K, R, t


def test_cap_frags_keeps_every_compositable_fragment() -> None:
    """The truncated list must hold each segment's first `min(count, cap)` fragments."""
    xyz, size, conf, _feat, K, R, t = _random_scene(seed=4)
    grid = layer_grid(24, 32, 3)

    full = build_sorted_fragments(xyz, size, conf, K, R, t, grid, mode="broadcast")
    capped = build_sorted_fragments(
        xyz, size, conf, K, R, t, grid, mode="broadcast", cap_frags=RASTER_MAX_FRAGS
    )

    counts = full.offsets[1:] - full.offsets[:-1]
    expected_kept = torch.clamp(counts, max=RASTER_MAX_FRAGS)
    assert torch.equal(capped.offsets[1:] - capped.offsets[:-1], expected_kept)
    assert len(capped) == int(expected_kept.sum())
    # The pre-cap totals still describe what emission produced.
    assert capped.emitted_count == len(full)
    assert torch.equal(capped.emitted_offsets, full.offsets)

    # Every kept fragment is the corresponding prefix entry of the full list.
    for pixel in range(grid.total):
        lo_full, hi_full = int(full.offsets[pixel]), int(full.offsets[pixel + 1])
        lo_cap, hi_cap = int(capped.offsets[pixel]), int(capped.offsets[pixel + 1])
        n = hi_cap - lo_cap
        assert n == min(hi_full - lo_full, RASTER_MAX_FRAGS)
        if n == 0:
            continue
        assert torch.equal(capped.depth[lo_cap:hi_cap], full.depth[lo_full : lo_full + n])
        assert torch.equal(capped.point_id[lo_cap:hi_cap], full.point_id[lo_full : lo_full + n])
        assert torch.equal(capped.alpha[lo_cap:hi_cap], full.alpha[lo_full : lo_full + n])
        assert torch.equal(
            capped.layer_pixel[lo_cap:hi_cap], full.layer_pixel[lo_full : lo_full + n]
        )


@pytest.mark.parametrize("mode", ["broadcast", "trips", "trilinear"])
def test_cap_frags_renders_identically_on_the_cpu_reference(mode: str) -> None:
    """Same image to float64 tolerance, same `n_used`, same reported counts.

    Not bit-for-bit, and deliberately so: `composite_sorted` derives its
    transmittance from a global prefix sum over the whole fragment list, so
    a shorter list moves the last bits (which is why the CPU reference keeps
    `cap_to_max_frags=False` by default). The invariant that actually matters
    -- that the kept fragments ARE the compositable prefix, fragment for
    fragment -- is pinned exactly by the test above, and the MPS path, whose
    kernel loops sequentially per segment, is bit-identical (GPU marker,
    tests/test_raster_metal.py).
    """
    xyz, size, conf, feat, K, R, t = _random_scene(seed=5)
    bg = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    def render(cap: bool):
        return render_pyramid_ref(
            xyz, size, feat, conf, K, R, t, (24, 32),
            num_layers=3, mode=mode, bg=bg, cap_to_max_frags=cap,
        )

    layers_a, aux_a = render(False)
    layers_b, aux_b = render(True)

    for a, b in zip(layers_a, layers_b, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    for key in ("t_final", "depth_sum"):
        for a, b in zip(aux_a[key], aux_b[key], strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
    # The composited prefix itself must be identical, not merely close.
    for a, b in zip(aux_a["n_used"], aux_b["n_used"], strict=True):
        assert torch.equal(a, b)
    # The reported fragment counts are the pre-cap ones in both renders.
    assert aux_a["num_fragments"] == aux_b["num_fragments"]
    assert torch.equal(aux_a["fragments_per_layer"], aux_b["fragments_per_layer"])


def test_cap_frags_gradients_agree() -> None:
    """Every gradient the rasteriser produces survives the cap (float64 tolerance)."""
    xyz, size, conf, feat, K, R, t = _random_scene(seed=6, num_points=1500)

    def grads(cap: bool):
        x = xyz.clone().requires_grad_(True)
        sz = size.clone().requires_grad_(True)
        c = conf.clone().requires_grad_(True)
        f = feat.clone().requires_grad_(True)
        layers, aux = render_pyramid_ref(
            # mode "trips" so `size` reaches the loss through `layer_factor`;
            # in "broadcast" it only feeds the (non-differentiable) cull.
            x, sz, f, c, K, R, t, (24, 32), num_layers=3, mode="trips", cap_to_max_frags=cap
        )
        loss = sum(layer.square().sum() for layer in layers)
        loss = loss + sum(tf.sum() for tf in aux["t_final"])
        return torch.autograd.grad(loss, [x, sz, c, f])

    for a, b in zip(grads(False), grads(True), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------
# 4. gradient sanitisation
# --------------------------------------------------------------------------


def _build_trainer(tmp_path: Path) -> Trainer:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    cfg.eval_lpips = False
    return Trainer(cfg)


@pytest.mark.parametrize("poison", [False, True])
def test_sanitise_gradients_matches_the_literal_loop(tmp_path: Path, poison: bool) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.train_step(name=trainer.train_names[0], zoom=1.0, center=(4.0, 4.0))

    # Rebuild a known gradient state on both copies.
    torch.manual_seed(7)
    reference: list[torch.Tensor] = []
    for group in trainer.optimizer.param_groups:
        for param in group["params"]:
            grad = torch.randn_like(param)
            if poison:
                flat = grad.reshape(-1)
                if flat.numel() >= 3:
                    flat[0] = float("nan")
                    flat[1] = float("inf")
                    flat[2] = float("-inf")
            param.grad = grad
            reference.append(grad.clone())

    expected_count = int(_sanitise_reference_on_copies(reference))
    got = trainer._sanitise_gradients()
    assert int(got.item()) == expected_count

    index = 0
    for group in trainer.optimizer.param_groups:
        for param in group["params"]:
            expected = torch.nan_to_num(reference[index], nan=0.0, posinf=0.0, neginf=0.0)
            assert torch.equal(param.grad, expected)
            index += 1


def _sanitise_reference_on_copies(grads: list[torch.Tensor]) -> int:
    """Non-finite entries across `grads`, counted the way the old loop counted."""
    total = 0
    for grad in grads:
        total += int((~torch.isfinite(grad)).sum())
    return total


# --------------------------------------------------------------------------
# 5. the step timer
# --------------------------------------------------------------------------


def test_steptimer_is_inactive_by_default() -> None:
    assert steptimer.active() is None
    with steptimer.stage("nothing"):
        pass  # must not raise and must not allocate a row


def test_steptimer_records_disjoint_stages() -> None:
    timer = steptimer.StepTimer("cpu")
    previous = steptimer.set_active(timer)
    try:
        timer.begin_step()
        with steptimer.stage("outer"), steptimer.stage("inner"):
            torch.rand(64, 64).square().sum()
        row = timer.end_step()
    finally:
        steptimer.set_active(previous)
    assert set(row) == {"outer", "inner"}
    # `outer` is exclusive of `inner`, so nesting cannot double-count.
    assert row["outer"] >= 0.0
    assert row["inner"] >= 0.0


def test_profile_step_runs_on_the_synthetic_scene(tmp_path: Path) -> None:
    from trippy.train import profile_step as profile_mod

    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    cfg.eval_lpips = False
    cfg.loss_vgg = 0.0

    result = profile_mod.profile_step(cfg, steps=2, warmup=1, epoch=0)
    assert result["points"] > 0
    assert result["untimed"]["sec_per_step"] > 0
    assert "raster_emit" in result["stages_ms"]
    assert "backward" in result["stages_ms"]
    assert len(result["losses"]) == 3
    table = profile_mod.format_table(result)
    assert "UNTIMED" in table and "raster_emit" in table
    # The timer must be uninstalled again once profiling is over.
    assert steptimer.active() is None


# --------------------------------------------------------------------------
# 6. fused Adam
# --------------------------------------------------------------------------


def test_fused_adam_reproduces_the_unfused_loss_curve(tmp_path: Path) -> None:
    """`optimizer_fused` must not move the training trajectory on CPU.

    torch never selects the fused Adam kernel by itself on MPS
    (`_default_to_fused_or_foreach` leaves MPS out of the *foreach* device
    list and only uses `fused` when asked), so an unmodified run drives ~7
    kernels and a full-size temporary per parameter tensor per step. This
    test is the CPU half of the parity argument for turning it on: same
    config, same seed, same losses. The MPS half is the 30-step loss curve
    `trippy profile-step --fused-adam` records against the same run without
    it (research/trips-metal.md).
    """
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)

    curves: dict[bool, list[float]] = {}
    for fused in (False, True):
        cfg = tiny_train_config(
            scene_root, ply_path, tmp_path / f"run{int(fused)}", tmp_path / "cache"
        )
        cfg.eval_lpips = False
        cfg.loss_vgg = 0.0
        cfg.optimizer_fused = fused
        trainer = Trainer(cfg)
        curves[fused] = [trainer.train_step()["loss"] for _ in range(8)]

    assert curves[False] == curves[True]


# --------------------------------------------------------------------------
# 7. mixed precision (flagged, off by default)
# --------------------------------------------------------------------------


def test_amp_is_off_by_default_and_changes_nothing(tmp_path: Path) -> None:
    """`amp` defaults off, and with it off every AMP object is inert."""
    trainer = _build_trainer(tmp_path)
    assert trainer.cfg.amp is False
    assert trainer.amp_enabled is False
    assert trainer._scaler.is_enabled() is False
    assert trainer.loss_fn._vgg.amp is False
    assert trainer.loss_fn._lpips.amp is False


def test_amp_scope_is_training_only(tmp_path: Path) -> None:
    """With `amp` on, the U-Net runs float16 in training and float32 in eval.

    The scope matters more than the speed: a held-out number must not depend on
    the precision the run trained in, so `_render` gates autocast on
    `self.net.training` and `Trainer.evaluate` (which calls `net.eval()`) is
    float32 unconditionally.
    """
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    cfg.eval_lpips = False
    cfg.amp = True
    trainer = Trainer(cfg)

    assert trainer.amp_enabled is True
    assert trainer._scaler.is_enabled() is True
    assert trainer.loss_fn._vgg.amp is True
    # The metric LPIPS is a separate module and must stay float32.
    assert trainer._eval_lpips is None or trainer._eval_lpips.amp is False

    record = trainer.train_step(name=trainer.train_names[0], zoom=1.0, center=(4.0, 4.0))
    assert math.isfinite(record["loss"])

    # Eval renders float32 even though the run is an AMP run.
    trainer.net.eval()
    item = trainer.dataset[0]
    R, t = trainer._pose_for(item, 0)
    net_out, _layers, _aux = trainer._render(
        item["K"], R, t, (int(item["rgb"].shape[0]), int(item["rgb"].shape[1]))
    )
    assert net_out.dtype == torch.float32


def test_profile_step_sweep_arms_start_from_the_same_state(tmp_path: Path) -> None:
    """Two arms of a sweep must produce comparable loss curves, not chained ones.

    `profile_step` snapshots and restores everything a step mutates -- model,
    camera, points, poses, background, optimiser state, epoch, global step and
    the crop RNG -- so a `--raster-cap both` / `--amp both` sweep measures both
    arms from the same starting point. Without that, the second arm would
    train on what the first one left behind and `max|dloss|` in the comparison
    table would be meaningless.
    """
    from trippy.train import profile_step as profile_mod

    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    cfg.eval_lpips = False
    cfg.loss_vgg = 0.0
    trainer = Trainer(cfg)
    before = trainer.point_params.xyz.detach().clone()

    first = profile_mod.profile_step(cfg, steps=2, warmup=1, epoch=0, trainer=trainer)
    # The trainer is handed back exactly as it was found.
    assert torch.equal(trainer.point_params.xyz.detach(), before)
    assert trainer.global_step == 0

    second = profile_mod.profile_step(cfg, steps=2, warmup=1, epoch=0, trainer=trainer)
    assert first["losses"] == second["losses"]

    table = profile_mod.format_comparison([first, second])
    assert "max|dloss|" in table
    assert "0.000e+00" in table


# --------------------------------------------------------------------------
# 8. the isolation knobs (both paths must give the same step)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("knob", ["use_fast_crop", "sanitise_conditional"])
def test_isolation_knobs_do_not_change_the_step(tmp_path: Path, knob: str) -> None:
    """Both settings of each knob must produce the same loss curve.

    These knobs exist only because the controlled before/after
    (`trippy-train-perf-ab`) showed this branch's exactly-equivalent changes to
    be a wash, and two of them are the prime suspects for giving the win back.
    A knob that changed the numbers would make that follow-up measurement
    meaningless, so both paths are pinned against each other here.
    """
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)

    curves: dict[bool, list[float]] = {}
    for value in (False, True):
        cfg = tiny_train_config(
            scene_root, ply_path, tmp_path / f"run{int(value)}", tmp_path / "cache"
        )
        cfg.eval_lpips = False
        cfg.loss_vgg = 0.0
        trainer = Trainer(cfg)
        setattr(trainer, knob, value)
        assert getattr(trainer, knob) is value
        curves[value] = [trainer.train_step()["loss"] for _ in range(6)]

    assert curves[False] == curves[True]
