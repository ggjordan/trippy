"""Tests for person masks: cache, polarity, undistortion, crop and eval wiring.

Module: tests.test_scene_masks
Invariants under test:
    1. Polarity. Splats' mask writers emit "BLACK = ignore (person), WHITE =
       keep"; a mask that is black over a region must make that region's
       validity 0, and `mask_keep_fracs` must report the *keep* share (so an
       inverted read would show up as a near-zero coverage number, not as a
       silently-wrong loss).
    2. Masks ride the photo's own undistortion grid, resampled NEAREST, so a
       cached mask is binary {0, 1} and pixel-aligned with the cached photo.
    3. `crop()` multiplies the person mask into the same validity mask the crop
       overshoot already uses -- one mask, two reasons to be zero.
    4. The trainer's loss and `evaluate()` both honour it: a fully-masked frame
       contributes no gradient and no metric, and the run records what it
       excluded.
    5. `use_masks: false` and a scene with no `masks/` directory both give
       exactly the pre-masks behaviour (no "mask" key on items).
All fixtures are synthetic (generated arrays), per AGENTS.md.
Related docs: docs/EXPERIMENTS.md "Person masks"; trippy.constants
    "scene/dataset.py person masks"; experiments/EXP-0011-karekare-v2/README.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_train_helpers as helpers
import torch
from PIL import Image as PILImage
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.constants import SCENE_MASK_KEEP_THRESHOLD
from trippy.scene import dataset as scene_dataset
from trippy.train.trainer import Trainer


def _write_masks(scene_root: Path, keep_fracs: dict[str, float]) -> Path:
    """Write one binary PNG mask per image: the LEFT `1 - keep_frac` of it black (person).

    Black-on-the-left is deliberately asymmetric so a test can tell an
    orientation flip from a polarity flip.
    """
    masks_dir = scene_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for name, keep in keep_fracs.items():
        with PILImage.open(scene_root / "images" / name) as img:
            width, height = img.size
        mask = np.full((height, width), 255, dtype=np.uint8)
        black_cols = round((1.0 - keep) * width)
        mask[:, :black_cols] = 0
        PILImage.fromarray(mask, mode="L").save(masks_dir / (Path(name).stem + ".png"))
    return masks_dir


def _scene_with_masks(tmp_path: Path, keep: float = 0.75):
    scene_root, point_set = build_synthetic_scene(tmp_path)
    names = sorted(p.name for p in (scene_root / "images").iterdir())
    _write_masks(scene_root, {n: keep for n in names})
    return scene_root, point_set, names


# --- discovery ---


def test_default_masks_dir_finds_masks_and_returns_none_without_them(tmp_path: Path) -> None:
    scene_root, _point_set, _names = _scene_with_masks(tmp_path)
    assert scene_dataset.default_masks_dir(scene_root) == scene_root / "masks"

    bare = tmp_path / "bare"
    (bare / "images").mkdir(parents=True)
    assert scene_dataset.default_masks_dir(bare) is None


def test_mask_path_for_uses_the_image_stem_not_its_extension() -> None:
    assert scene_dataset.mask_path_for("/m", "IMG_3683.jpg") == Path("/m/IMG_3683.png")
    assert scene_dataset.mask_path_for("/m", "IMG_4609_0008.JPEG") == Path("/m/IMG_4609_0008.png")


# --- polarity and cache ---


def test_masks_are_auto_discovered_and_keep_fraction_matches_the_written_mask(tmp_path: Path) -> None:
    """WHITE = keep: a mask that is 25% black must report ~0.75 keep, not ~0.25."""
    scene_root, _point_set, names = _scene_with_masks(tmp_path, keep=0.75)
    ds = scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=tmp_path / "cache")

    assert ds.masks_dir == scene_root / "masks"
    fracs = ds.mask_keep_fracs()
    assert set(fracs) == set(names)
    for name, frac in fracs.items():
        assert frac == pytest.approx(0.75, abs=0.02), f"{name}: keep fraction {frac} (polarity inverted?)"

    item = ds[0]
    mask = item["mask"]
    assert mask.shape == item["rgb"].shape[:2]
    assert mask.dtype == torch.float32
    # Nearest resampling of a binary mask stays binary -- no fractional edge values.
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    # ...and the black quarter is on the LEFT, where it was written.
    width = mask.shape[1]
    assert mask[:, : width // 5].max().item() == 0.0
    assert mask[:, -width // 5 :].min().item() == 1.0


def test_mask_threshold_is_the_documented_one(tmp_path: Path) -> None:
    """A grey mask below/above SCENE_MASK_KEEP_THRESHOLD reads as person/keep."""
    scene_root, _point_set, names = _scene_with_masks(tmp_path)
    masks_dir = scene_root / "masks"
    with PILImage.open(scene_root / "images" / names[0]) as img:
        width, height = img.size
    grey = np.full((height, width), SCENE_MASK_KEEP_THRESHOLD - 1, dtype=np.uint8)
    grey[:, width // 2 :] = SCENE_MASK_KEEP_THRESHOLD
    PILImage.fromarray(grey, mode="L").save(masks_dir / (Path(names[0]).stem + ".png"))

    ds = scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=tmp_path / "cache")
    assert ds.mask_keep_fracs()[names[0]] == pytest.approx(0.5, abs=0.05)


def test_mask_cache_is_reused_and_backfilled_onto_an_unmasked_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing unmasked cache gains masks without re-undistorting the photos."""
    scene_root, _point_set, _names = _scene_with_masks(tmp_path)
    cache_root = tmp_path / "cache"
    scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=cache_root, use_masks=False)

    image_calls = {"n": 0}
    original = scene_dataset.SceneDataset._undistort_image

    def counting(self, *args, **kwargs):
        image_calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(scene_dataset.SceneDataset, "_undistort_image", counting)

    ds = scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=cache_root)
    assert image_calls["n"] == 0, "backfilling masks must not re-undistort the photos"
    assert ds.mask_keep_fracs()
    assert "mask" in ds[0]

    mask_calls = {"n": 0}
    original_mask = scene_dataset.SceneDataset._undistort_mask

    def counting_mask(self, *args, **kwargs):
        mask_calls["n"] += 1
        return original_mask(self, *args, **kwargs)

    monkeypatch.setattr(scene_dataset.SceneDataset, "_undistort_mask", counting_mask)
    scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=cache_root)
    assert mask_calls["n"] == 0, "second masked construction must hit the mask cache"


def test_missing_mask_file_fails_loudly(tmp_path: Path) -> None:
    scene_root, _point_set, names = _scene_with_masks(tmp_path)
    (scene_root / "masks" / (Path(names[0]).stem + ".png")).unlink()
    with pytest.raises(FileNotFoundError, match="has no mask for image"):
        scene_dataset.SceneDataset(scene_root, width=helpers.IMG_WIDTH, cache_root=tmp_path / "cache")


def test_use_masks_false_and_no_masks_dir_both_give_the_pre_masks_item(tmp_path: Path) -> None:
    scene_root, _point_set, _names = _scene_with_masks(tmp_path)
    off = scene_dataset.SceneDataset(
        scene_root, width=helpers.IMG_WIDTH, cache_root=tmp_path / "cache_off", use_masks=False
    )
    assert off.masks_dir is None
    assert "mask" not in off[0]
    assert off.mask_keep_fracs() == {}

    bare_root, _ps = build_synthetic_scene(tmp_path / "bare")
    bare = scene_dataset.SceneDataset(bare_root, width=helpers.IMG_WIDTH, cache_root=tmp_path / "cache_bare")
    assert bare.masks_dir is None
    assert "mask" not in bare[0]


# --- crop ---


def test_crop_multiplies_the_person_mask_into_the_validity_mask() -> None:
    rgb = torch.full((10, 10, 3), 200, dtype=torch.uint8)
    person = torch.ones((10, 10), dtype=torch.float32)
    person[:, :4] = 0.0  # left 4 columns are a person

    plain = scene_dataset.crop({"rgb": rgb}, size=10)
    masked = scene_dataset.crop({"rgb": rgb, "mask": person}, size=10)

    assert plain["mask"].sum().item() == 100.0
    assert masked["mask"].sum().item() == 60.0
    assert torch.equal(masked["mask"], plain["mask"] * person)
    # The photographed pixels themselves are untouched: they are real content that
    # simply does not count (unlike padding, which is zeroed).
    assert torch.equal(masked["rgb"], plain["rgb"])


def test_crop_combines_person_mask_with_padding_overshoot() -> None:
    rgb = torch.full((8, 8, 3), 100, dtype=torch.uint8)
    person = torch.ones((8, 8), dtype=torch.float32)
    person[0, :] = 0.0

    # Centre pushed off the top-left corner so the crop overshoots on two sides.
    out = scene_dataset.crop({"rgb": rgb, "mask": person}, size=8, center=(2.0, 2.0))
    assert out["mask"].shape == (8, 8)
    # Padding still wins where it applies (rgb exactly 0 there, mask 0 there).
    assert out["mask"][0, 0].item() == 0.0
    assert out["rgb"][0, 0].sum().item() == 0
    # And no mask value is anything other than 0 or 1.
    assert set(torch.unique(out["mask"]).tolist()) <= {0.0, 1.0}


# --- trainer / evaluate ---


def _masked_trainer(tmp_path: Path, keep: float, **overrides) -> Trainer:
    scene_root, point_set, _names = _scene_with_masks(tmp_path, keep=keep)
    ply = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg)


def test_trainer_reports_mask_coverage_and_excludes_masked_pixels(tmp_path: Path) -> None:
    trainer = _masked_trainer(tmp_path, keep=0.75)
    stats = trainer.mask_stats()
    assert stats["masked"] is True
    assert stats["frac_masked_mean"] == pytest.approx(0.25, abs=0.03)
    assert stats["n_images"] == helpers.N_IMAGES
    assert "person masks: ON" in trainer.log_path.read_text()

    record = trainer.train_step()
    assert record["mask_excluded_frac"] > 0.0

    metrics = trainer.evaluate()
    assert metrics["masks"]["masked"] is True
    for row in metrics["per_image"].values():
        assert row["mask_excluded_frac"] == pytest.approx(0.25, abs=0.05)


def test_unmasked_trainer_is_unchanged(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path / "bare")
    ply = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)

    assert trainer.mask_stats() == {"masked": False}
    assert "person masks: OFF" in trainer.log_path.read_text()
    metrics = trainer.evaluate()
    for row in metrics["per_image"].values():
        assert row["mask_excluded_frac"] == 0.0


def test_a_fully_masked_frame_contributes_no_gradient(tmp_path: Path) -> None:
    """Masking everything must zero the image loss's gradient, not merely shrink it.

    This is what makes "the loss ignores people" true rather than approximately
    true. `trippy.net.losses._masked_mean` returns a *detached* zero when the
    mask sum is 0, so the texture parameter -- which only the image loss can
    reach -- receives exactly no gradient. (The reported `image_loss` is not 0
    but 0.5: `TripsLoss` adds `w_ssim * (1 - SSIM) / 2` and a fully masked SSIM
    is 0. That term is a constant here, which is precisely why it contributes
    no gradient either.)
    """
    trainer = _masked_trainer(tmp_path, keep=0.0)
    assert trainer.mask_stats()["frac_masked_mean"] == pytest.approx(1.0, abs=1e-6)

    record = trainer.train_step()
    assert record["mask_excluded_frac"] == pytest.approx(1.0, abs=1e-6)

    # `feat` is reachable ONLY through the image loss (the extent penalty reads xyz,
    # the camera regulariser reads the camera), so a detached image loss leaves it
    # with no gradient at all -- `grad is None` -- rather than a small one.
    feat_grad = trainer.point_params.feat.grad
    assert feat_grad is None or float(feat_grad.abs().max().item()) == 0.0

    # Control: the identical run with masks off does move the texture, so the
    # assertion above is about the mask and not about a dead training step.
    control = _masked_trainer(tmp_path / "control", keep=0.0, use_masks=False)
    control.train_step()
    control_grad = control.point_params.feat.grad
    assert control_grad is not None and float(control_grad.abs().max().item()) > 0.0


def test_use_masks_false_scores_more_pixels_than_the_masked_run(tmp_path: Path) -> None:
    """The masked and unmasked arms of EXP-0011 must differ only in what is scored."""
    masked = _masked_trainer(tmp_path / "m", keep=0.5)
    unmasked = _masked_trainer(tmp_path / "u", keep=0.5, use_masks=False)

    m_metrics = masked.evaluate()
    u_metrics = unmasked.evaluate()
    assert m_metrics["masks"]["masked"] is True
    assert u_metrics["masks"] == {"masked": False}
    for name in m_metrics["per_image"]:
        assert m_metrics["per_image"][name]["mask_excluded_frac"] > 0.4
        assert u_metrics["per_image"][name]["mask_excluded_frac"] == 0.0
