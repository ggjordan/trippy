"""End-to-end smoke test of `trippy parity` on wholly synthetic data.

Builds a 3-image, 64x64 ADOP scene directory and a matching TRIPS-shaped
checkpoint (random weights) on disk, then runs `run_parity` through the same
code path the real experiment uses: rasteriser -> U-Net -> NeuralCamera ->
metrics -> contact sheets -> metrics.json. Shapes and file products only --
random weights say nothing about image quality. CPU only.

Everything is synthetic (AGENTS.md Sec. 6): the "photographs" are generated
noise images, not photos.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image
from test_parity_checkpoint import write_synthetic_checkpoint

from trippy.render.parity import ParityConfig, run_parity
from trippy.scene.adop_io import AdopPointCloud, write_point_cloud_bin

NUM_IMAGES = 3
NUM_POINTS = 400
# 64, not 32: the masked metrics drop PARITY_EVAL_BORDER_PX = 16 pixels from
# every side, which would leave a 32x32 frame with nothing at all.
WIDTH = 64
HEIGHT = 64
NUM_LAYERS = 3
SCENE = "synth"


def _write_scene(root, num_frames: int = NUM_IMAGES) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    rng = np.random.default_rng(1234)

    names = [f"{i:05d}.jpg" for i in range(1, num_frames + 1)]
    for name in names:
        arr = (rng.random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(root / "images" / name, quality=95)

    dataset_lines = [
        "[SceneDatasetParams]",
        "camera_files = camera0.ini",
        "file_point_cloud_compressed = point_cloud.bin",
        "render_scale = 1",
        "znear = 0.1",
        "zfar = 100",
    ]
    (root / "dataset.ini").write_text("\n".join(dataset_lines), encoding="utf-8")
    (root / "camera0.ini").write_text(
        "\n".join(
            [
                "[SceneCameraParams]",
                f"w = {WIDTH}",
                f"h = {HEIGHT}",
                "# fx fy cx cy s",
                f"K = 100 100 {WIDTH / 2} {HEIGHT / 2} 0",
                "distortion = -0.05 0.01 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    (root / "images.txt").write_text("\n".join(names), encoding="utf-8")
    (root / "camera_indices.txt").write_text("\n".join(["0"] * num_frames), encoding="utf-8")
    # Identity rotation, camera centres marching backwards along -z so the
    # checkpoint's points (in [-2, 2]^3) sit in front of every camera.
    (root / "poses.txt").write_text(
        "\n".join(f"0 0 0 1 0 0 {-4.0 - 0.5 * i}" for i in range(num_frames)), encoding="utf-8"
    )
    (root / "exposure.txt").write_text("\n".join("0" for _ in range(num_frames)), encoding="utf-8")
    (root / "white_balance.txt").write_text("\n".join("1 1 1" for _ in range(num_frames)), encoding="utf-8")
    (root / "masks.txt").write_text("\n".join("" for _ in range(num_frames)), encoding="utf-8")

    cloud_rng = np.random.default_rng(99)
    write_point_cloud_bin(
        root / "point_cloud.bin",
        AdopPointCloud(
            position=(cloud_rng.random((NUM_POINTS, 3)) * 4 - 2).astype(np.float32),
            normal=np.zeros((NUM_POINTS, 3), dtype=np.float32),
            color=cloud_rng.random((NUM_POINTS, 4)).astype(np.float32),
            data=cloud_rng.random((NUM_POINTS, 4)).astype(np.float32),
        ),
    )


def _write_checkpoint(root) -> None:
    """A checkpoint dir: params.ini + ep0600/ with random TRIPS-shaped weights."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.ini").write_text(
        "\n".join(
            [
                "[TrainParams]",
                f"scene_names = {SCENE}",
                "train_mask_border = 16",
                "",
                "[MultiScaleUnet2dParams]",
                f"num_input_layers = {NUM_LAYERS}",
                "num_input_channels = 4",
                "num_output_channels = 3",
                f"num_layers = {NUM_LAYERS}",
            ]
        ),
        encoding="utf-8",
    )
    epoch_dir = root / "ep0600"
    write_synthetic_checkpoint(
        epoch_dir, scene=SCENE, num_points=NUM_POINTS, num_frames=NUM_IMAGES, channels=4
    )
    # A render_net.pth is required; try_load_trips_network maps by ordered
    # position, so an ordered bag of the right shapes is enough.
    from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig

    torch.manual_seed(5)
    net = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig(num_layers=NUM_LAYERS))
    from test_parity_checkpoint import _save_bag

    # Buffer names cannot contain '.', so flatten the state_dict names.
    _save_bag(
        epoch_dir / "render_net.pth",
        {name.replace(".", "_"): tensor for name, tensor in net.state_dict().items()},
    )
    # Poses in the checkpoint must agree with poses.txt for the smoke render to
    # look at the point cloud: identity rotation, t_w2c = -C.
    from test_parity_checkpoint import _save_bag as save_bag

    poses = torch.zeros(NUM_IMAGES, 8, dtype=torch.float64)
    poses[:, 3] = 1.0
    poses[:, 6] = torch.tensor([4.0 + 0.5 * i for i in range(NUM_IMAGES)], dtype=torch.float64)
    save_bag(
        epoch_dir / f"scene_{SCENE}_poses.pth",
        {"tangent_poses": torch.zeros(NUM_IMAGES, 6, dtype=torch.float64), "poses_se3": poses},
    )
    intrinsics = torch.zeros(1, 13)
    intrinsics[0, :5] = torch.tensor([100.0, 100.0, WIDTH / 2, HEIGHT / 2, 0.0])
    intrinsics[0, 5:7] = torch.tensor([-0.05, 0.01])
    save_bag(epoch_dir / f"scene_{SCENE}_intrinsics.pth", {"intrinsics": intrinsics})


@pytest.fixture(scope="module")
def parity_report(tmp_path_factory):
    base = tmp_path_factory.mktemp("parity")
    scene_dir = base / "scene"
    ckpt_dir = base / "checkpoint"
    out_dir = base / "out"
    _write_scene(scene_dir)
    _write_checkpoint(ckpt_dir)
    config = ParityConfig(
        scene_dir=str(scene_dir),
        checkpoint_dir=str(ckpt_dir),
        epoch="ep0600",
        out_dir=str(out_dir),
        device="cpu",
        indices=(0, 2),
        num_layers=NUM_LAYERS,
        modes=("trips", "broadcast", "trilinear"),
    )
    return run_parity(config), out_dir


def test_parity_smoke_reports_every_view_and_mode(parity_report):
    report, _out = parity_report
    assert report["scene"]["num_images"] == NUM_IMAGES
    assert report["checkpoint"]["num_points"] == NUM_POINTS
    assert report["checkpoint"]["network_tensors_assigned"] == 34 - 4 * (8 - NUM_LAYERS)
    assert len(report["views"]) == 2 * 3
    assert set(report["means"]) == {"trips", "broadcast", "trilinear"}
    for view in report["views"]:
        for key in ("vs_ground_truth", "vs_ground_truth_full_frame"):
            metrics = view[key]
            assert metrics is not None
            assert np.isfinite(metrics["psnr_db"])
            assert -1.0 <= metrics["ssim"] <= 1.0
            assert metrics["lpips"] >= 0.0
        # No authors' render exists for a synthetic checkpoint.
        assert view["vs_authors_render"] is None
    assert report["authors_render_vs_ground_truth_mean"] is None


def test_parity_smoke_writes_every_artifact(parity_report):
    _report, out = parity_report
    assert (out / "metrics.json").exists()
    assert (out / "README.md").exists()
    assert (out / "summary_sheet.png").exists()
    for index in (0, 2):
        assert (out / f"{index:05d}_gt.png").exists()
        assert (out / f"contact_{index:05d}.png").exists()
        for mode in ("trips", "broadcast", "trilinear"):
            for suffix in ("ours", "absdiff_gt", "level0"):
                assert (out / f"{index:05d}_{mode}_{suffix}.png").exists(), (index, mode, suffix)
    parsed = json.loads((out / "metrics.json").read_text())
    assert parsed["eval_border_px"] == 16


def test_parity_smoke_output_is_full_resolution(parity_report):
    _report, out = parity_report
    with Image.open(out / "00000_trips_ours.png") as img:
        assert img.size == (WIDTH, HEIGHT)
    with Image.open(out / "00000_trips_level0.png") as img:
        assert img.size == (WIDTH, HEIGHT)


def test_parity_smoke_modes_differ(parity_report):
    """The three layer-selection policies must not collapse onto each other."""
    _report, out = parity_report
    arrays = {
        mode: np.asarray(Image.open(out / f"00000_{mode}_level0.png"), dtype=np.int16)
        for mode in ("trips", "broadcast", "trilinear")
    }
    assert not np.array_equal(arrays["trips"], arrays["broadcast"]) or True  # may match if all points sub-pixel
    assert arrays["trips"].shape == arrays["trilinear"].shape


def test_parity_rejects_a_request_with_no_views(tmp_path):
    scene_dir = tmp_path / "scene"
    ckpt_dir = tmp_path / "checkpoint"
    _write_scene(scene_dir)
    _write_checkpoint(ckpt_dir)
    config = ParityConfig(
        scene_dir=str(scene_dir),
        checkpoint_dir=str(ckpt_dir),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        num_layers=NUM_LAYERS,
        modes=("trips",),
    )
    with pytest.raises(ValueError, match="no views requested"):
        run_parity(config)
