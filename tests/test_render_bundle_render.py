"""Rendering a `trippy-bundle-1` bundle from its own three files, in Python.

Module: tests.test_render_bundle_render
Purpose: `trippy.render.bundle_render` is the numeric half of the viewer
    acceptance check -- it exists so "is the Rust viewer wrong, or is the
    checkpoint?" can be answered without anyone opening a render of Jordan's
    scenes (AGENTS.md Sec. 6). These tests prove it reconstructs the exporting
    network and tone mapper exactly, renders the same picture the Trainer that
    wrote the bundle renders, and encodes 8-bit PNGs with the same rounding
    rule as `brush_pyramid::png::feature_to_rgb8` -- so a PSNR between the two
    paths measures the renderer and nothing else.
    Also pins `trippy.render.bundle.trusted_exposures`, the exposure hygiene
    the exporter applies (see docs/LIMITATIONS.md "Per-image exposure").
Invariants: CPU only, synthetic fixtures only, no GPU, no network, and
    nothing here opens or displays an image.
Related docs: trippy/render/bundle_render.py, trippy/render/bundle.py,
    docs/decisions/ADR-0006-viewer-integration.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy import cli
from trippy.net.export_safetensors import read_safetensors
from trippy.render import bundle_render
from trippy.render.bundle import (
    BUNDLE_EXPOSURE_TRUST_STOPS,
    BUNDLE_JSON_FILENAME,
    BUNDLE_WEIGHTS_FILENAME,
    BundleView,
    default_view_with_trusted_exposure,
    trusted_exposures,
)
from trippy.train.trainer import Trainer

#: The tiny synthetic run these tests export: 3 views of the 48x36 scene, a
#: 3-level pyramid, C = 4. Seconds on CPU.
NUM_IMAGES = 3
LAYERS = 3
CHANNELS = 4

#: How closely the Python bundle render must reproduce the Trainer's own
#: render of the same view. Both run the identical float32 modules on CPU, so
#: the only source of difference is tensor construction order.
RENDER_TOLERANCE = 1e-5

#: dB floor for "the two 8-bit PNGs are the same picture". Two float renders
#: agreeing to `RENDER_TOLERANCE` cannot differ by more than one 8-bit LSB.
PNG_PSNR_FLOOR = 70.0


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Trainer]:
    """A bundle exported from a tiny synthetic checkpoint, plus that Trainer.

    Module-scoped: building the scene and the Trainer is the slow part and
    every test below reads the same bundle.
    """
    tmp_path = tmp_path_factory.mktemp("bundle_render")
    scene_root, point_set = build_synthetic_scene(tmp_path, n_images=NUM_IMAGES)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(
        scene_root,
        ply_path,
        tmp_path / "run",
        tmp_path / "cache",
        layers=LAYERS,
        feature_channels=CHANNELS,
    )
    trainer = Trainer(cfg)
    checkpoint = trainer.save_checkpoint()
    out_dir = tmp_path / "bundle"
    assert (
        cli.main(
            ["export-bundle", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--name", "synthetic"]
        )
        == 0
    )
    return out_dir, trainer


# --- exposure hygiene --------------------------------------------------------


def test_trusted_exposures_replaces_only_the_far_outliers() -> None:
    """A value further than `max_stops` from the median is replaced by the median."""
    # Nine sane exposures around 0.25 and one stuck at the -5.87 the
    # missing-EXIF bug produced on kk-coherent.
    values = np.array([0.2, 0.25, 0.3, 0.1, 0.4, 0.25, -0.9, 0.5, 1.4, -5.87], dtype=np.float32)
    sanitised, replaced, reference = trusted_exposures(values, max_stops=BUNDLE_EXPOSURE_TRUST_STOPS)

    assert replaced == [9]
    assert abs(reference - float(np.median(values))) < 1e-6
    assert abs(float(sanitised[9]) - reference) < 1e-6
    # Everything else is untouched, bit for bit.
    assert np.array_equal(sanitised[:9], values[:9])
    # The input is not mutated.
    assert float(values[9]) == pytest.approx(-5.87, abs=1e-5)
    # And the gain the tone mapper would apply is now ~1, not 58x.
    assert 0.25 < 2.0 ** -float(sanitised[9]) < 4.0


def test_trusted_exposures_leaves_a_well_behaved_scene_alone() -> None:
    """The public horse checkpoint's EV range is +-0.07; nothing may be touched."""
    values = np.linspace(-0.058, 0.066, 151).astype(np.float32)
    sanitised, replaced, _reference = trusted_exposures(values)
    assert replaced == []
    assert np.array_equal(sanitised, values)


def test_trusted_exposures_handles_an_empty_table() -> None:
    sanitised, replaced, reference = trusted_exposures(np.zeros(0, dtype=np.float32))
    assert sanitised.shape == (0,)
    assert replaced == []
    assert reference == 0.0


def test_the_exporter_records_its_exposure_decision(exported: tuple[Path, Trainer]) -> None:
    """Every bundle says in its own metadata whether an exposure was substituted."""
    out_dir, _ = exported
    _tensors, metadata = read_safetensors(out_dir / BUNDLE_WEIGHTS_FILENAME)
    assert metadata["exposure_substituted_count"] == "0"
    assert metadata["exposure_substituted_frames"] == ""
    assert float(metadata["exposure_trust_stops"]) == BUNDLE_EXPOSURE_TRUST_STOPS
    # The synthetic photos all carry EXIF, so the reference EV is the middle
    # of a tight, zero-centred spread.
    assert abs(float(metadata["exposure_reference_ev"])) < 1.0


def test_a_stuck_exposure_is_substituted_on_export(tmp_path: Path) -> None:
    """End to end: a 58x exposure in the checkpoint does not reach the bundle."""
    scene_root, point_set = build_synthetic_scene(tmp_path, n_images=NUM_IMAGES)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(
        scene_root, ply_path, tmp_path / "run", tmp_path / "cache", layers=LAYERS, feature_channels=CHANNELS
    )
    trainer = Trainer(cfg)
    with torch.no_grad():
        trainer.camera.exposures_values[0] = -5.870477
    live = float(trainer.camera.exposures_values[0].detach())
    checkpoint = trainer.save_checkpoint()

    out_dir = tmp_path / "bundle"
    assert cli.main(["export-bundle", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--name", "s"]) == 0
    tensors, metadata = read_safetensors(out_dir / BUNDLE_WEIGHTS_FILENAME)

    assert metadata["exposure_substituted_count"] == "1"
    assert metadata["exposure_substituted_frames"] == "0"
    gain = 2.0 ** -float(tensors["camera.exposure"][0])
    assert 0.25 < gain < 4.0, f"exported gain is still {gain:.1f}x"
    # The Trainer handed to the exporter is not mutated: `train --report`
    # exports a bundle and then keeps evaluating with the same object.
    assert float(trainer.camera.exposures_values[0].detach()) == pytest.approx(live, abs=1e-6)


def _view(index: int) -> BundleView:
    """A minimal view; only `index` matters to the default-view chooser."""
    return BundleView(
        index=index,
        name=f"IMG_{index}.jpg",
        width=8,
        height=6,
        fx=1.0,
        fy=1.0,
        cx=4.0,
        cy=3.0,
        R=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        t=(0.0, 0.0, 0.0),
        distortion=tuple(0.0 for _ in range(8)),
    )


def test_the_default_view_moves_off_an_untrustworthy_exposure() -> None:
    """A bundle must not open on a frame whose exposure was never trained."""
    views = [_view(i) for i in range(5)]
    # View 0 is the kk-coherent case: stuck at the initialisation.
    exposure = np.array([-5.87, 0.9, 0.25, 0.1, 0.3], dtype=np.float32)
    median = float(np.median(exposure))
    chosen = default_view_with_trusted_exposure(views, exposure, fallback=0)
    assert chosen != 0
    # The nearest trustworthy view to the median, not merely the next index.
    assert chosen == min(
        (i for i in range(1, 5)), key=lambda i: (abs(float(exposure[i]) - median), i)
    )


def test_the_default_view_is_untouched_when_it_is_fine() -> None:
    """Well-behaved scenes -- the public horse included -- open where they always did."""
    views = [_view(i) for i in range(5)]
    exposure = np.array([0.0, 0.05, -0.03, 0.02, -0.06], dtype=np.float32)
    assert default_view_with_trusted_exposure(views, exposure, fallback=0) == 0
    assert default_view_with_trusted_exposure(views, exposure, fallback=3) == 3


def test_the_default_view_falls_back_when_nothing_is_trustworthy() -> None:
    """With no exposure table, or no good view, the loader's choice stands."""
    views = [_view(i) for i in range(3)]
    assert default_view_with_trusted_exposure(views, np.zeros(0, dtype=np.float32), fallback=2) == 2
    # A table shorter than the view list: index 2 has no exposure at all.
    assert default_view_with_trusted_exposure(views, np.array([0.0, 0.1], np.float32), fallback=2) == 0


# --- rebuilding the modules from the bundle ---------------------------------


def test_the_bundle_rebuilds_the_exporting_network_exactly(exported: tuple[Path, Trainer]) -> None:
    """Every U-Net parameter comes back bit-for-bit through the safetensors file."""
    out_dir, trainer = exported
    loaded = bundle_render.load_bundle(out_dir)
    original = dict(trainer.net.state_dict())
    rebuilt = dict(loaded.net.state_dict())
    assert set(rebuilt) == set(original)
    for key, value in original.items():
        assert torch.equal(rebuilt[key].cpu(), value.detach().cpu()), key


def test_the_bundle_rebuilds_the_tone_mapper_exactly(exported: tuple[Path, Trainer]) -> None:
    """Exposure, white balance, vignette and the response LUT all round-trip."""
    out_dir, trainer = exported
    loaded = bundle_render.load_bundle(out_dir)
    assert loaded.camera is not None
    assert loaded.camera.config == trainer.camera.config
    assert torch.equal(
        loaded.camera.exposures_values.detach().cpu(), trainer.camera.exposures_values.detach().cpu()
    )
    assert torch.equal(
        loaded.camera.white_balance_values.detach().cpu(),
        trainer.camera.white_balance_values.detach().cpu(),
    )
    assert torch.equal(
        loaded.camera.camera_response.response.detach().cpu(),
        trainer.camera.camera_response.response.detach().cpu(),
    )


def test_the_manifest_and_points_must_agree(exported: tuple[Path, Trainer], tmp_path: Path) -> None:
    """A manifest that lies about N or C is refused rather than rendered."""
    import shutil

    out_dir, _ = exported
    broken = tmp_path / "broken"
    shutil.copytree(out_dir, broken)
    document = json.loads((broken / BUNDLE_JSON_FILENAME).read_text())
    document["num_points"] += 1
    (broken / BUNDLE_JSON_FILENAME).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="points.npz holds"):
        bundle_render.load_bundle(broken)


# --- rendering ---------------------------------------------------------------


def test_the_bundle_render_matches_the_trainer_that_wrote_it(exported: tuple[Path, Trainer]) -> None:
    """The whole point: bundle in, same picture out.

    This is the Python end of the viewer acceptance check. If it holds and the
    Rust `--screenshot` disagrees, the disagreement is in the Rust renderer;
    if it fails, the bundle is not a faithful description of the checkpoint.
    """
    out_dir, trainer = exported
    loaded = bundle_render.load_bundle(out_dir)
    position = loaded.view_position(None)
    rgb, info = bundle_render.render_view(loaded, position)

    frame_index = info["frame_index"]
    item = trainer.dataset[frame_index]
    height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
    trainer.net.eval()
    trainer.camera.eval()
    with torch.no_grad():
        R, t = trainer._pose_for(item, frame_index)
        net_out, _layers, _aux = trainer._render(item["K"], R, t, (height, width))
        reference = trainer._tone_map(net_out, frame_index)[0]

    assert rgb.shape == reference.shape
    difference = float((rgb - reference.cpu()).abs().max())
    assert difference < RENDER_TOLERANCE, f"max |bundle - trainer| = {difference:.2e}"

    # And through the 8-bit encoder both paths' PNGs are the same picture.
    psnr = bundle_render.psnr(bundle_render.to_rgb8(rgb), bundle_render.to_rgb8(reference))
    assert psnr > PNG_PSNR_FLOOR, f"{psnr:.2f} dB"


def test_render_at_half_scale_keeps_the_field_of_view(exported: tuple[Path, Trainer]) -> None:
    """`--scale` changes the resolution, not the camera -- same rule as the viewer."""
    out_dir, _ = exported
    loaded = bundle_render.load_bundle(out_dir)
    view = loaded.manifest["views"][0]
    K_full, _R, _t, width_full, height_full = bundle_render.view_camera(view, 1.0)
    K_half, _R2, _t2, width_half, height_half = bundle_render.view_camera(view, 0.5)

    assert (width_half, height_half) == (round(width_full * 0.5), round(height_full * 0.5))
    # Focal length scales with the width, so the horizontal FOV is unchanged.
    assert float(K_half[0, 0]) == pytest.approx(float(K_full[0, 0]) * width_half / width_full, rel=1e-5)
    assert float(K_half[0, 2]) == pytest.approx(float(K_full[0, 2]) * width_half / width_full, rel=1e-5)


def test_a_distorted_view_is_refused_not_silently_rendered(exported: tuple[Path, Trainer]) -> None:
    """`render_pyramid` is a plain pinhole; the Rust kernel is not. Refuse rather than lie."""
    out_dir, _ = exported
    loaded = bundle_render.load_bundle(out_dir)
    loaded.manifest["views"][0]["distortion"] = [-0.064, 0.044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="lens distortion"):
        bundle_render.render_view(loaded, 0)


def test_the_view_position_lookup_is_by_dataset_index(exported: tuple[Path, Trainer]) -> None:
    out_dir, _ = exported
    loaded = bundle_render.load_bundle(out_dir)
    for position, view in enumerate(loaded.manifest["views"]):
        assert loaded.view_position(int(view["index"])) == position
    assert loaded.view_position(None) == int(loaded.manifest["default_view"])
    with pytest.raises(ValueError, match="no view with dataset index"):
        loaded.view_position(9999)


# --- the numbers a render is judged by --------------------------------------


def test_to_rgb8_uses_the_rust_rounding_rule() -> None:
    """`(clamp(v, 0, 1) * 255 + 0.5) as u8`, byte-identical to brush_pyramid::png."""
    # (C, H, W) = (3, 1, 3): one row of three pixels, channel-major.
    values = torch.tensor([[[-1.0, 0.0, 0.5]], [[1.0, 2.0, 0.002]], [[float("nan"), 0.999, 1.0 / 255.0]]])
    out = bundle_render.to_rgb8(values)
    assert out.shape == (1, 3, 3)
    # negative -> 0, exactly 1 -> 255, NaN -> 0 (Rust's clamp chain does the same).
    assert out[0, 0].tolist() == [0, 255, 0]
    # 0 -> 0, above 1 -> 255, 0.999 * 255 + 0.5 = 255.2 -> 255.
    assert out[0, 1].tolist() == [0, 255, 255]
    # 0.5 * 255 + 0.5 = 128.0 -> 128; 0.002 -> 1; 1/255 -> 1 (0.5 rounds up).
    assert out[0, 2].tolist() == [128, 1, 1]


def test_stats_describe_a_white_frame_as_white() -> None:
    """The exact shape of the "nearly white frame" report, on synthetic pixels."""
    white = np.full((4, 5, 3), 255, dtype=np.uint8)
    numbers = bundle_render.stats(white)
    assert numbers["mean"] == [1.0, 1.0, 1.0]
    assert numbers["saturated"] == 1.0
    assert numbers["black"] == 0.0
    assert numbers["p50"] == [1.0, 1.0, 1.0]
    assert numbers["size"] == [5, 4]

    mid = np.zeros((4, 5, 3), dtype=np.uint8)
    mid[..., 0] = 128
    numbers = bundle_render.stats(mid)
    assert numbers["mean"][0] == pytest.approx(128 / 255)
    assert numbers["mean"][1] == 0.0
    assert numbers["saturated"] == 0.0
    assert numbers["black"] == pytest.approx(2 / 3)


def test_psnr_is_infinite_for_identical_images_and_finite_otherwise() -> None:
    a = np.zeros((3, 3, 3), dtype=np.uint8)
    assert bundle_render.psnr(a, a) == float("inf")
    b = a.copy()
    b[0, 0, 0] = 255
    # One of 27 samples off by 1.0 -> MSE = 1/27.
    assert bundle_render.psnr(a, b) == pytest.approx(-10.0 * np.log10(1.0 / 27.0), abs=1e-6)
    with pytest.raises(ValueError, match="shape mismatch"):
        bundle_render.psnr(a, np.zeros((2, 2, 3), dtype=np.uint8))
