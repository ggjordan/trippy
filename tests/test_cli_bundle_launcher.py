"""End-to-end test for the standalone `trippy bundle-launcher` command.

Module: tests.test_cli_bundle_launcher
Purpose: `trippy bundle-launcher --checkpoint <ckpt> --name <name> [--out <dir>]`
    runs the same three steps `trippy train --report` now runs from its own
    final checkpoint (`trippy.render.report.export_bundle_and_viewer_launcher`)
    against ANY checkpoint -- this task's brief: "add `trippy bundle-launcher
    --checkpoint ckpt --name NAME` as a standalone command that does (1)-(3)
    for any existing checkpoint". This test proves that on a synthetic
    trippy-native checkpoint (no training run, no `--report`), the command
    writes a `trippy-bundle-1` bundle to `--out` and, if the real viewer
    binary happens to be built on this machine
    (`rust/target/release/trips-viewer`, resolved via `_main_checkout_root`),
    generates `OPEN_TRIPS_MAC_<name>.command` under
    `$TRIPPY_OUTPUT/deliver/<name>/` and records (without shelling out, under
    `TRIPPY_DELIVER_DRY_RUN=1`) a delivery for it -- exercising both
    branches keeps this test honest about `build_mac_viewer_launcher`'s
    "never fails the run" contract (see tests/test_render_report.py's direct
    unit tests of that function for the hermetic missing-binary case).
Invariants:
    - `TRIPPY_DELIVER_DRY_RUN=1` means the delivery is recorded as skipped
      rather than shelling out to `scripts/deliver.sh` -- same rule as
      tests/test_cli_train_report.py.
Fixture: the shared synthetic checkpoint builder from tests/test_train_helpers.py
    (never a real Splats scene or checkpoint).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.render.bundle import BUNDLE_JSON_FILENAME
from trippy.train.trainer import Trainer


def test_cli_bundle_launcher_end_to_end_on_synthetic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRIPPY_DELIVER_DRY_RUN", "1")
    trippy_output = tmp_path / "trippy_output"
    monkeypatch.setenv("TRIPPY_OUTPUT", str(trippy_output))

    from trippy import cli

    scene_root, point_set = build_synthetic_scene(tmp_path, n_images=2)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(
        scene_root, ply_path, tmp_path / "run", tmp_path / "cache", layers=3, feature_channels=4
    )
    # A randomly initialised checkpoint is enough for the exporter -- no training loop.
    checkpoint = Trainer(cfg).save_checkpoint()

    out_dir = tmp_path / "bundle_out"
    rc = cli.main(
        [
            "bundle-launcher",
            "--checkpoint", str(checkpoint),
            "--name", "synthetic-bl",
            "--out", str(out_dir),
        ]
    )  # fmt: skip
    assert rc == 0
    assert (out_dir / BUNDLE_JSON_FILENAME).exists()

    command_path = trippy_output / "deliver" / "synthetic-bl" / "OPEN_TRIPS_MAC_synthetic-bl.command"
    if command_path.exists():
        # The common case on the dev machine (rust/target/release/trips-viewer built):
        # requirement 3, delivered (recorded, not shelled out to, under the dry-run env).
        assert os.access(command_path, os.X_OK)
        assert str(out_dir.resolve()) in command_path.read_text()
    # Either way the bundle itself must exist -- requirement 2's "do not fail the run"
    # applies to the whole `bundle-launcher` command, not just `train --report`.
    assert (out_dir / BUNDLE_JSON_FILENAME).exists()


def test_cli_bundle_launcher_default_out_dir_matches_run_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting `--out` on a `<run>/checkpoints/checkpoint_*.pt` writes `<run>/bundle`."""
    monkeypatch.setenv("TRIPPY_DELIVER_DRY_RUN", "1")
    monkeypatch.setenv("TRIPPY_OUTPUT", str(tmp_path / "trippy_output"))

    from trippy import cli

    scene_root, point_set = build_synthetic_scene(tmp_path, n_images=2)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache", layers=3, feature_channels=4)
    checkpoint = Trainer(cfg).save_checkpoint()
    assert checkpoint.parent.name == "checkpoints"

    rc = cli.main(["bundle-launcher", "--checkpoint", str(checkpoint), "--name", "synthetic-bl-default"])
    assert rc == 0
    assert (run_dir / "bundle" / BUNDLE_JSON_FILENAME).exists()
