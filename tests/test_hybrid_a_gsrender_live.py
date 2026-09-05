"""Tests for `trippy.hybrid.gsrender_live` and the `gaussian_provider` hook in candidate.py.

Module: tests.test_hybrid_a_gsrender_live
Invariants under test: the live renderer builds gsrender's viewmat the same
    way the batch renderer does, always passes an explicit `max_hw`, and loads
    the PLY exactly **once per process** (the real one is 1.7 GB); the provider
    renders at the pose it is *given* and never substitutes the anchor image's
    precomputed render (a dolly/off-path pose is displaced from the photo, so
    that render belongs to a different camera); it degrades to None (an
    all-zero block) when no renderer is configured; and `render_candidate`
    routes every pose through the provider, producing a different image than it
    would with the Gaussian channels zeroed.
Never imports Splats' real `gsrender.py`, never loads a PLY, never touches
MPS: `FakeGsrender` (tests/test_hybrid_a_helpers.py) is injected everywhere.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from test_hybrid_a_helpers import FakeGsrender, hybrid_train_config, write_fake_render
from test_train_helpers import IMG_HEIGHT, IMG_WIDTH

from trippy.hybrid.config_a import HybridConfig
from trippy.hybrid.gaussian_input import GaussianInputs
from trippy.hybrid.gsrender_live import (
    LiveGaussianRenderer,
    clear_ply_cache,
    gaussian_provider_for,
    viewmat_from_rt,
)
from trippy.render.candidate import render_candidate
from trippy.render.dolly import CameraPose, shade_dolly_poses
from trippy.train.trainer import Trainer


@pytest.fixture(autouse=True)
def _clean_ply_cache():
    clear_ply_cache()
    yield
    clear_ply_cache()


def _renderer(fake: FakeGsrender, ply: str = "/nowhere/kkc_15000.ply") -> LiveGaussianRenderer:
    return LiveGaussianRenderer(
        ply, device="cpu", render_fn=fake.render, load_ply_fn=fake.load_ply
    )


def _pose() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = torch.tensor([[48.0, 0.0, 24.0], [0.0, 48.0, 18.0], [0.0, 0.0, 1.0]])
    R = torch.eye(3)
    t = torch.tensor([0.1, -0.2, 0.3])
    return K, R, t


# --- viewmat ---


def test_viewmat_matches_the_batch_renderer_convention() -> None:
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    viewmat = viewmat_from_rt(R, t)
    assert viewmat.shape == (4, 4)
    assert np.allclose(viewmat[:3, :3], R)
    assert np.allclose(viewmat[:3, 3], t)
    assert np.allclose(viewmat[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(viewmat_from_rt(torch.from_numpy(R), torch.from_numpy(t)), viewmat)


# --- live renderer ---


def test_live_render_returns_the_three_channels_at_the_requested_size() -> None:
    fake = FakeGsrender()
    arrays = _renderer(fake).render(*_pose(), (7, 11))
    assert arrays["rgb"].shape == (7, 11, 3)
    assert arrays["depth"].shape == (7, 11)
    assert arrays["alpha"].shape == (7, 11)
    call = fake.render_calls[0]
    assert (call["height"], call["width"]) == (7, 11)
    assert call["max_hw"] == 400, "gsrender's own default of 32 corrupts near-camera footprints"


def test_ply_is_loaded_once_per_process_across_renderers() -> None:
    fake = FakeGsrender()
    first = _renderer(fake)
    first.render(*_pose(), (4, 4))
    first.render(*_pose(), (4, 4))
    second = _renderer(fake)  # e.g. render_candidate called a second time by the report
    second.render(*_pose(), (4, 4))
    assert len(fake.load_calls) == 1, f"PLY loaded {len(fake.load_calls)} times"
    assert len(fake.render_calls) == 3


def test_clear_ply_cache_forces_a_reload() -> None:
    fake = FakeGsrender()
    _renderer(fake).render(*_pose(), (4, 4))
    clear_ply_cache()
    _renderer(fake).render(*_pose(), (4, 4))
    assert len(fake.load_calls) == 2


# --- composed provider ---


def _inputs(tmp_path: Path, **overrides) -> tuple[GaussianInputs, HybridConfig]:
    renders = tmp_path / "renders"
    write_fake_render(renders, "IMG_0", height=IMG_HEIGHT, width=IMG_WIDTH)
    kwargs = {"enabled": True, "renders_dir": str(renders), "depth_scale": 6.0}
    kwargs.update(overrides)
    cfg = HybridConfig(**kwargs)
    return GaussianInputs.build(cfg, ["IMG_0.jpg"]), cfg


def test_provider_never_substitutes_the_anchor_images_render(tmp_path: Path) -> None:
    """A pose anchored to IMG_0.jpg is not IMG_0.jpg's pose -- render live, don't substitute."""
    inputs, cfg = _inputs(tmp_path, ply_path="/nowhere/kkc.ply")
    fake = FakeGsrender()
    provider = gaussian_provider_for(cfg, inputs, live_renderer=_renderer(fake))
    block = provider("IMG_0.jpg", *_pose(), (IMG_HEIGHT, IMG_WIDTH))
    assert block is not None and block.shape == (5, IMG_HEIGHT, IMG_WIDTH)
    assert len(fake.render_calls) == 1, "the anchor image's cached render was reused"
    assert inputs.has("IMG_0.jpg"), "the cached render exists -- it just must not be used here"


def test_provider_renders_live_for_an_unphotographed_pose(tmp_path: Path) -> None:
    inputs, cfg = _inputs(tmp_path, ply_path="/nowhere/kkc.ply", mask_by_alpha=False)
    fake = FakeGsrender()
    provider = gaussian_provider_for(cfg, inputs, live_renderer=_renderer(fake))
    block = provider("dolly_t+0.35", *_pose(), (9, 13))
    assert block is not None and block.shape == (5, 9, 13)
    assert len(fake.render_calls) == 1
    # FakeGsrender's constant render, through the same normalisation the cached path uses.
    assert torch.allclose(block[0:3], torch.full((3, 9, 13), 0.5))
    assert torch.allclose(block[3], torch.full((9, 13), 0.75))
    assert torch.allclose(block[4], torch.full((9, 13), 1.0))  # depth 6.0 / depth_scale 6.0


def test_provider_without_a_ply_returns_none(tmp_path: Path) -> None:
    inputs, cfg = _inputs(tmp_path)  # no ply_path
    provider = gaussian_provider_for(cfg, inputs)
    assert provider("dolly_t+0.35", *_pose(), (9, 13)) is None


# --- render_candidate hook ---


def _with_unknown_name(pose: CameraPose) -> CameraPose:
    """Copy of `pose` anchored to an image the checkpoint has no render for.

    `CameraPose` is frozen (a pose must not mutate between the videos and the
    metrics that describe it), so this is a `dataclasses.replace`, not an
    assignment.
    """
    return dataclasses.replace(pose, image_name="NOT_A_REGISTERED_IMAGE.jpg")


def _hybrid_checkpoint(tmp_path: Path) -> Path:
    cfg, _names = hybrid_train_config(tmp_path, ply_path="/nowhere/kkc.ply")
    return Trainer(cfg).save_checkpoint()


def test_render_candidate_uses_the_injected_gaussian_provider(tmp_path: Path) -> None:
    checkpoint = _hybrid_checkpoint(tmp_path)
    poses = shade_dolly_poses(tmp_path / "scene", pose_name="IMG_0.jpg", n=2, width=IMG_WIDTH)
    calls: list[str | None] = []

    def provider(name, K, R, t, image_hw):
        calls.append(name)
        return torch.full((5, int(image_hw[0]), int(image_hw[1])), 0.7)

    metrics = render_candidate(
        checkpoint,
        poses,
        tmp_path / "with_provider",
        device="cpu",
        write_video_files=False,
        gaussian_provider=provider,
    )
    assert metrics["n_frames"] == 2
    # Every pose goes through the provider, including the ones anchored to IMG_0.jpg --
    # which HAS a precomputed render that must not be substituted for a displaced camera.
    assert calls == ["IMG_0.jpg", "IMG_0.jpg"]

    render_candidate(
        checkpoint,
        [_with_unknown_name(poses[0])],
        tmp_path / "with_provider2",
        device="cpu",
        write_video_files=False,
        gaussian_provider=provider,
    )
    assert calls[-1] == "NOT_A_REGISTERED_IMAGE.jpg"


def test_gaussian_channels_change_the_rendered_frame(tmp_path: Path) -> None:
    """A provider that returns content must not produce the same pixels as one returning None."""
    checkpoint = _hybrid_checkpoint(tmp_path)
    poses = [
        _with_unknown_name(shade_dolly_poses(tmp_path / "scene", pose_name="IMG_0.jpg", n=1, width=IMG_WIDTH)[0])
    ]

    def full(name, K, R, t, image_hw):
        return torch.full((5, int(image_hw[0]), int(image_hw[1])), 0.9)

    def empty(name, K, R, t, image_hw):
        return None

    for tag, provider in (("full", full), ("empty", empty)):
        render_candidate(
            checkpoint,
            poses,
            tmp_path / tag,
            device="cpu",
            write_video_files=False,
            gaussian_provider=provider,
        )
    from PIL import Image

    with Image.open(tmp_path / "full" / "frames" / poses[0].name / "net.png") as a:
        full_np = np.asarray(a.convert("RGB"))
    with Image.open(tmp_path / "empty" / "frames" / poses[0].name / "net.png") as b:
        empty_np = np.asarray(b.convert("RGB"))
    assert not np.array_equal(full_np, empty_np)
