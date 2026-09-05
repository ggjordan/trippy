"""Tests for `trippy.hybrid.config_a.HybridConfig` and its `TrainConfig` integration.

Module: tests.test_hybrid_a_config
Invariants under test: channel bookkeeping (`num_channels`, `channel_slice`,
    canonical order), validation of every enum-ish field, `TrainConfig.
    net_input_channels` adaptation, YAML round-trip of a hybrid config
    (including the two shipped EXP-0009 config files), and -- the one that
    matters most -- that `enabled: false` leaves `net_input_channels` exactly
    at `feature_channels`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trippy.constants import HYBRID_A_CHANNEL_ORDER
from trippy.hybrid.config_a import HybridConfig, gaussian_channel_count
from trippy.net.unet import NetworkConfig
from trippy.train.config import TrainConfig

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "EXP-0009-hybrid-a"


def test_default_hybrid_is_disabled_and_costs_no_channels() -> None:
    cfg = TrainConfig()
    assert cfg.hybrid.enabled is False
    assert cfg.net_input_channels == cfg.feature_channels


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        (["rgb", "alpha", "depth"], 5),
        (["rgb", "alpha"], 4),
        (["rgb"], 3),
        (["alpha", "depth"], 2),
        (["depth"], 1),
    ],
)
def test_gaussian_channel_count(channels: list[str], expected: int) -> None:
    assert gaussian_channel_count(channels) == expected
    assert HybridConfig(enabled=True, renders_dir="r", channels=channels).num_channels == expected


def test_channels_are_canonicalised_and_deduplicated() -> None:
    cfg = HybridConfig(enabled=True, renders_dir="r", channels=["depth", "rgb", "rgb", "alpha"])
    assert cfg.channels == list(HYBRID_A_CHANNEL_ORDER)


def test_channel_slice_partitions_the_block() -> None:
    cfg = HybridConfig(enabled=True, renders_dir="r", channels=["rgb", "alpha", "depth"])
    assert cfg.channel_slice("rgb") == slice(0, 3)
    assert cfg.channel_slice("alpha") == slice(3, 4)
    assert cfg.channel_slice("depth") == slice(4, 5)

    without_alpha = HybridConfig(enabled=True, renders_dir="r", channels=["rgb", "depth"])
    assert without_alpha.channel_slice("depth") == slice(3, 4)
    with pytest.raises(ValueError, match="not in hybrid.channels"):
        without_alpha.channel_slice("alpha")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": ["rgb", "normals"]}, "hybrid.channels entries"),
        ({"channels": []}, "must not be empty"),
        ({"mode": "level7"}, "hybrid.mode"),
        ({"dropout_gaussian_p": 1.5}, "dropout_gaussian_p"),
        ({"dropout_gaussian_p": -0.1}, "dropout_gaussian_p"),
        ({"missing": "shrug"}, "hybrid.missing"),
        ({"depth_scale": 0.0}, "depth_scale must be positive"),
    ],
)
def test_invalid_fields_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HybridConfig(enabled=True, renders_dir="r", **kwargs)


def test_enabled_without_renders_dir_raises() -> None:
    with pytest.raises(ValueError, match="renders_dir is empty"):
        HybridConfig(enabled=True)


def test_train_config_net_input_channels_adapts_and_fits_the_unet() -> None:
    cfg = TrainConfig(
        feature_channels=4,
        hybrid=HybridConfig(enabled=True, renders_dir="r", channels=["rgb", "alpha", "depth"]),
    )
    assert cfg.net_input_channels == 9
    # The U-Net's own constraint (filters > 2 * num_input_channels) must still hold, and
    # the channel arithmetic of every block must close -- constructing it proves both.
    net = NetworkConfig(num_input_channels=cfg.net_input_channels, num_layers=cfg.layers)
    assert net.num_input_channels == 9


def test_train_config_accepts_a_plain_dict_hybrid_block() -> None:
    cfg = TrainConfig.from_dict(
        {"hybrid": {"enabled": True, "renders_dir": "somewhere", "channels": ["rgb", "alpha"]}}
    )
    assert isinstance(cfg.hybrid, HybridConfig)
    assert cfg.net_input_channels == cfg.feature_channels + 4


def test_yaml_round_trip_preserves_the_hybrid_block(tmp_path: Path) -> None:
    cfg = TrainConfig(
        run_dir=str(tmp_path / "run"),
        hybrid=HybridConfig(
            enabled=True,
            renders_dir="output/hybrid-c/renders/w1008",
            channels=["rgb", "alpha", "depth"],
            mode="concat_level0",
            dropout_gaussian_p=0.35,
            mask_by_alpha=False,
            depth_scale=12.5,
            ply_path="/nowhere/kkc.ply",
        ),
    )
    path = cfg.save_yaml(tmp_path / "cfg.yaml")
    reloaded = TrainConfig.load_yaml(path)
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.hybrid.depth_scale == 12.5
    assert reloaded.net_input_channels == cfg.net_input_channels


@pytest.mark.parametrize("filename", ["config.yaml", "config_smoke.yaml"])
def test_shipped_experiment_configs_load(filename: str) -> None:
    path = _EXPERIMENT_DIR / filename
    cfg = TrainConfig.load_yaml(path)
    assert cfg.hybrid.enabled is True
    assert cfg.hybrid.mode == "all_levels"
    assert cfg.hybrid.channels == list(HYBRID_A_CHANNEL_ORDER)
    assert cfg.net_input_channels == cfg.feature_channels + 5
    # queue_training.sh reads `run_dir:` straight out of the YAML and uses its basename
    # as the queue job name -- keep that contract testable here too.
    raw = yaml.safe_load(path.read_text())
    assert raw["run_dir"]
    assert Path(raw["run_dir"]).name.replace("_", "").replace("-", "").replace(".", "").isalnum()
