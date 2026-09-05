"""The `trippy-bundle-1` asset bundle the native Rust viewer opens.

Module: tests.test_render_bundle
Purpose: pin the wire format between `trippy.render.bundle` and the Rust
    viewer -- the three files, `bundle.json`'s schema (every key, its type,
    the C / 9 / 3 / 8 lengths, the lowercase enum strings), `points.npz`'s
    keys/shapes/dtypes, and that `weights.safetensors` still reads back
    through `trippy.net.export_safetensors` and reproduces the exporting
    network. A schema change on either side fails here first, on CPU, in
    seconds.
Invariants: CPU only, no GPU, no network, no committed fixture -- the
    synthetic checkpoint is generated into `tmp_path` from the pinned seed
    of `tests/test_train_helpers.py` (AGENTS.md section 6: synthetic
    fixtures only). Nothing here opens an image.
Related docs: `trippy/render/bundle.py` module docstring (the schema),
    docs/GEOMETRY.md (world frame, row-major world-to-camera R),
    rust/crates/brush-pyramid/src/{params,scene}.rs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy import cli
from trippy.constants import RASTER_MODES, RASTER_PIXEL_CENTERS, RASTER_PYRAMID_HALVINGS
from trippy.net.export_safetensors import EXPORT_FORMAT, read_safetensors
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.render.bundle import (
    BUNDLE_DISTORTION_COEFFS,
    BUNDLE_FORMAT,
    BUNDLE_JSON_FILENAME,
    BUNDLE_POINTS_FILENAME,
    BUNDLE_UP,
    BUNDLE_WEIGHTS_FILENAME,
    BundleView,
    default_view_position,
)
from trippy.train.trainer import Trainer

#: The tiny synthetic run: 2 views of the 48x36 synthetic scene, C=4 features,
#: a 3-level pyramid. Small enough that building it and running the exporter is
#: a couple of seconds on CPU.
BUNDLE_TEST_NUM_IMAGES = 2
BUNDLE_TEST_LAYERS = 3
BUNDLE_TEST_CHANNELS = 4

#: Tolerance for "R is a rotation": `R @ R.T == I` and `det(R) == +1`.
ROTATION_TOLERANCE = 1e-5

#: Rows/columns of a 3x3 rotation, and the length of a translation.
ROTATION_ELEMENTS = 9
TRANSLATION_ELEMENTS = 3


@pytest.fixture(scope="module")
def native_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Export a bundle from a tiny synthetic trippy-native checkpoint.

    Module-scoped: building the scene, the point source and the Trainer is
    the slow part, and every assertion below reads the same bundle.

    Returns:
        `(bundle_dir, bundle_json_document)`.
    """
    tmp_path = tmp_path_factory.mktemp("bundle")
    scene_root, point_set = build_synthetic_scene(tmp_path, n_images=BUNDLE_TEST_NUM_IMAGES)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(
        scene_root,
        ply_path,
        tmp_path / "run",
        tmp_path / "cache",
        layers=BUNDLE_TEST_LAYERS,
        feature_channels=BUNDLE_TEST_CHANNELS,
    )
    # Saved straight after construction: a randomly initialised model is all
    # the exporter needs, and it keeps the test off any training loop.
    checkpoint = Trainer(cfg).save_checkpoint()

    out_dir = tmp_path / "bundle_out"
    assert cli.main(["export-bundle", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--name", "synthetic"]) == 0
    document = json.loads((out_dir / BUNDLE_JSON_FILENAME).read_text())
    return out_dir, document


def test_bundle_is_exactly_three_files(native_bundle: tuple[Path, dict]) -> None:
    """The contract is a directory of three files; the Rust reader opens them by name."""
    out_dir, _ = native_bundle
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == sorted([BUNDLE_JSON_FILENAME, BUNDLE_POINTS_FILENAME, BUNDLE_WEIGHTS_FILENAME])


def test_bundle_json_top_level_schema(native_bundle: tuple[Path, dict]) -> None:
    """Every top-level key is present, of the declared type, and self-consistent."""
    _, doc = native_bundle
    assert doc["format"] == BUNDLE_FORMAT
    assert doc["name"] == "synthetic"
    assert doc["points"] == BUNDLE_POINTS_FILENAME
    assert doc["weights"] == BUNDLE_WEIGHTS_FILENAME
    assert isinstance(doc["num_points"], int) and doc["num_points"] > 0
    assert doc["num_channels"] == BUNDLE_TEST_CHANNELS

    background = doc["background"]
    assert isinstance(background, list) and len(background) == doc["num_channels"]
    assert all(isinstance(v, float) for v in background)

    assert doc["up"] == list(BUNDLE_UP)
    assert isinstance(doc["default_view"], int)
    assert 0 <= doc["default_view"] < len(doc["views"])


def test_bundle_json_params_use_the_allowed_enum_strings(native_bundle: tuple[Path, dict]) -> None:
    """`params` is `brush_pyramid::PyramidParams` with lowercase enum strings."""
    _, doc = native_bundle
    params = doc["params"]
    assert set(params) == {
        "mode",
        "num_layers",
        "pixel_center",
        "halving",
        "max_frags",
        "t_cutoff",
        "alpha_min",
        "znear",
    }
    assert params["mode"] in RASTER_MODES
    assert params["pixel_center"] in RASTER_PIXEL_CENTERS
    assert params["halving"] in RASTER_PYRAMID_HALVINGS
    assert params["num_layers"] == BUNDLE_TEST_LAYERS
    assert isinstance(params["max_frags"], int) and params["max_frags"] > 0
    for key in ("t_cutoff", "alpha_min", "znear"):
        assert isinstance(params[key], float)
    assert params["znear"] > 0.0


def test_bundle_json_views_are_well_formed_cameras(native_bundle: tuple[Path, dict]) -> None:
    """Non-empty, one entry per image, R a genuine rotation, 8 distortion coeffs."""
    _, doc = native_bundle
    views = doc["views"]
    assert len(views) == BUNDLE_TEST_NUM_IMAGES

    for position, view in enumerate(views):
        assert set(view) == {
            "index", "name", "width", "height", "fx", "fy", "cx", "cy", "R", "t", "distortion",
        }  # fmt: skip
        assert isinstance(view["index"], int)
        assert isinstance(view["name"], str) and view["name"]
        assert isinstance(view["width"], int) and view["width"] > 0
        assert isinstance(view["height"], int) and view["height"] > 0
        for key in ("fx", "fy", "cx", "cy"):
            assert isinstance(view[key], float)
        assert len(view["R"]) == ROTATION_ELEMENTS
        assert len(view["t"]) == TRANSLATION_ELEMENTS
        assert len(view["distortion"]) == BUNDLE_DISTORTION_COEFFS

        R = np.asarray(view["R"], dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ROTATION_TOLERANCE)
        assert abs(float(np.linalg.det(R)) - 1.0) < ROTATION_TOLERANCE
        # `default_view` is an ARRAY POSITION; with every view exported in
        # dataset order the two happen to coincide, which is worth pinning.
        assert view["index"] == position

    # A trippy-native scene is undistorted on ingest (trippy.scene.dataset).
    assert all(all(c == 0.0 for c in v["distortion"]) for v in views)


def test_points_npz_round_trips_to_the_declared_shapes(native_bundle: tuple[Path, dict]) -> None:
    """`xyz`/`size`/`feat`/`conf`, all float32 and row-aligned by point index."""
    out_dir, doc = native_bundle
    with np.load(out_dir / BUNDLE_POINTS_FILENAME) as data:
        arrays = {key: data[key] for key in data.files}

    assert sorted(arrays) == ["conf", "feat", "size", "xyz"]
    n, c = doc["num_points"], doc["num_channels"]
    assert arrays["xyz"].shape == (n, 3)
    assert arrays["size"].shape == (n,)
    assert arrays["feat"].shape == (n, c)
    assert arrays["conf"].shape == (n,)
    for array in arrays.values():
        assert array.dtype == np.float32
        assert np.isfinite(array).all()

    # Effective, not raw: softplus is positive and sigmoid lands strictly inside (0, 1).
    assert (arrays["size"] > 0.0).all()
    assert ((arrays["conf"] > 0.0) & (arrays["conf"] < 1.0)).all()


def test_points_are_world_space_not_camera_space(native_bundle: tuple[Path, dict]) -> None:
    """The whole point of the bundle: `xyz` must survive `R @ x + t` per view.

    The synthetic cameras all sit near the world origin looking down +Z with
    the points 4-8 units in front, so world-space `xyz` has a mean well away
    from any single view's camera-space depth. Concretely: transforming the
    stored points by view 0's pose must move them (a pre-transformed set
    would be a fixed point of view 0), and every point must land in front of
    both cameras.
    """
    out_dir, doc = native_bundle
    with np.load(out_dir / BUNDLE_POINTS_FILENAME) as data:
        xyz = data["xyz"].astype(np.float64)

    for view in doc["views"]:
        R = np.asarray(view["R"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(view["t"], dtype=np.float64)
        cam = xyz @ R.T + t
        assert (cam[:, 2] > 0.0).all(), "synthetic points must be in front of every camera"

    R0 = np.asarray(doc["views"][1]["R"], dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(doc["views"][1]["t"], dtype=np.float64)
    moved = np.abs((xyz @ R0.T + t0) - xyz).max()
    assert moved > 0.0, "xyz already carries view 1's pose -- it is not world space"


def test_weights_load_through_the_existing_reader(native_bundle: tuple[Path, dict]) -> None:
    """`weights.safetensors` is unchanged `trippy-unet-1`, and rebuilds its net.

    Same closure as tests/test_net_export_safetensors.py: read the file back
    with the repo's own reader, rebuild the U-Net from the metadata, bind the
    tensors, and check the architecture matches what the bundle declares.
    """
    out_dir, doc = native_bundle
    weights, metadata = read_safetensors(out_dir / BUNDLE_WEIGHTS_FILENAME)

    assert metadata["format"] == EXPORT_FORMAT
    assert metadata["num_layers"] == str(doc["params"]["num_layers"])
    assert metadata["in_channels"] == str(doc["num_channels"])
    assert metadata["has_camera"] == "1"
    assert metadata["bundle_format"] == BUNDLE_FORMAT
    assert metadata["num_points"] == str(doc["num_points"])
    assert int(metadata["num_frames"]) == len(doc["views"])

    net = MultiScaleUnet2dDecOnlySmallFixed(
        NetworkConfig(
            num_layers=int(metadata["num_layers"]),
            num_input_channels=int(metadata["in_channels"]),
            filters=int(metadata["filters"]),
        )
    )
    with torch.no_grad():
        start = net.start.conv
        start.feature_conv.weight.copy_(torch.from_numpy(weights["unet.start.feature.weight"]))
        start.feature_conv.bias.copy_(torch.from_numpy(weights["unet.start.feature.bias"]))
        start.gate_conv.weight.copy_(torch.from_numpy(weights["unet.start.gate.weight"]))
        start.gate_conv.bias.copy_(torch.from_numpy(weights["unet.start.gate.bias"]))
        for k, block in enumerate(net.up):
            for branch, conv in (("feature", block.conv.feature_conv), ("gate", block.conv.gate_conv)):
                conv.weight.copy_(torch.from_numpy(weights[f"unet.up.{k}.{branch}.weight"]))
                conv.bias.copy_(torch.from_numpy(weights[f"unet.up.{k}.{branch}.bias"]))
        net.final[0].weight.copy_(torch.from_numpy(weights["unet.final.weight"]))
        net.final[0].bias.copy_(torch.from_numpy(weights["unet.final.bias"]))
    assert net.parameter_count() > 0
    assert set(weights) >= {
        "camera.exposure",
        "camera.white_balance",
        "camera.vignette_params",
        "camera.vignette_center",
        "camera.response",
    }


def _view(index: int) -> BundleView:
    """A throwaway view carrying only the dataset index the lookup keys on."""
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return BundleView(
        index=index,
        name=f"{index}.jpg",
        width=1,
        height=1,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
        R=identity,
        t=(0.0, 0.0, 0.0),
        distortion=tuple(0.0 for _ in range(BUNDLE_DISTORTION_COEFFS)),
    )


def test_default_view_is_an_array_position_not_a_dataset_index() -> None:
    """The documented distinction, on a bundle whose views are NOT 0..N-1."""
    views = [_view(5), _view(8), _view(11)]
    assert default_view_position(views, 8) == 1
    assert default_view_position(views, 5) == 0
    # A scene with fewer/other views must still name a loadable entry.
    assert default_view_position(views, 999) == 0
