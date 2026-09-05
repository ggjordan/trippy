"""Tests for trippy.train.config: TrainConfig/PointSourceConfig YAML round-trip and defaults.

Module: tests.test_train_config
Invariants under test: `TrainConfig.load_yaml`/`save_yaml` round-trip
    exactly; a YAML file only needs to state what differs from the
    scaled-for-trippy defaults; nested `point_source` (including a
    "union"'s child `sources`) survives the round trip as `PointSourceConfig`
    objects, not plain dicts; the derived epoch properties
    (`lock_cameras_epochs`, `lock_structure_epochs`, `vgg_start_epoch`)
    scale with `epochs`; invalid fields raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trippy.train.config import PointSourceConfig, TrainConfig, steps_per_epoch


def test_defaults_construct_without_arguments() -> None:
    cfg = TrainConfig()
    assert cfg.width > 0
    assert cfg.crop > 0
    assert isinstance(cfg.point_source, PointSourceConfig)
    assert cfg.point_source.type == "gaussian"


def test_yaml_round_trip_only_overrides_stated_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("scene_root: /some/scene\nepochs: 42\ncrop: 256\n")
    cfg = TrainConfig.load_yaml(path)

    assert cfg.scene_root == "/some/scene"
    assert cfg.epochs == 42
    assert cfg.crop == 256
    # Everything else keeps its scaled-for-trippy default.
    default = TrainConfig()
    assert cfg.width == default.width
    assert cfg.lr_network == default.lr_network


def test_save_yaml_then_load_yaml_is_lossless(tmp_path: Path) -> None:
    cfg = TrainConfig(
        scene_root="/scene",
        run_dir=str(tmp_path / "run"),
        point_source=PointSourceConfig(type="union", sources=[PointSourceConfig(type="gaussian", path="a.ply")]),
        forced_heldout=["IMG_1.jpg", "IMG_2.jpg"],
    )
    path = cfg.save_yaml(tmp_path / "roundtrip.yaml")
    loaded = TrainConfig.load_yaml(path)

    assert loaded.to_dict() == cfg.to_dict()
    assert isinstance(loaded.point_source, PointSourceConfig)
    assert isinstance(loaded.point_source.sources[0], PointSourceConfig)
    assert loaded.point_source.sources[0].path == "a.ply"


def test_point_source_config_from_plain_dict() -> None:
    cfg = TrainConfig(point_source={"type": "colmap", "path": "sparse_txt"})
    assert isinstance(cfg.point_source, PointSourceConfig)
    assert cfg.point_source.type == "colmap"
    assert cfg.point_source.path == "sparse_txt"


def test_unknown_key_raises() -> None:
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"not_a_real_field": 1})


@pytest.mark.parametrize("field,value", [("crop", 0), ("width", -1), ("epochs", 0), ("feature_channels", 2)])
def test_invalid_fields_raise(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**{field: value})


def test_lock_and_vgg_epoch_fractions_scale_with_epochs() -> None:
    cfg = TrainConfig(epochs=600, lock_cameras_frac=100 / 600, lock_structure_frac=10 / 600, vgg_start_frac=100 / 600)
    assert cfg.lock_cameras_epochs == 100
    assert cfg.lock_structure_epochs == 10
    assert cfg.vgg_start_epoch == 100

    scaled = TrainConfig(epochs=150, lock_cameras_frac=100 / 600, lock_structure_frac=10 / 600, vgg_start_frac=100 / 600)
    assert scaled.lock_cameras_epochs == 25
    assert scaled.lock_structure_epochs == round(150 * 10 / 600)
    assert scaled.vgg_start_epoch == 25


def test_steps_per_epoch_is_ceil_and_floored_at_one() -> None:
    assert steps_per_epoch(0.125, 8) == 1
    assert steps_per_epoch(0.125, 9) == 2  # ceil(1.125) = 2
    assert steps_per_epoch(0.125, 1) == 1  # floored at 1, never 0


def test_train_config_steps_per_epoch_method_matches_module_function() -> None:
    cfg = TrainConfig(train_factor=0.25)
    assert cfg.steps_per_epoch(10) == steps_per_epoch(0.25, 10)
