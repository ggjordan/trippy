"""Stage sequencing, partial outputs and memory release for the post-training wrap-up.

Module: tests.test_report_stages
Invariants under test (all on synthetic data, no scene, no checkpoint, no MPS):
    - `trippy.render.report.run_report_stages` runs `REPORT_STAGES` in order
      and writes `<run_dir>/report/stages/<stage>.json` after each one, so a
      kill costs at most the stage that was running. This is the fix for the
      2026-09-13 bug: two full karekare-v2 runs (kkv2-8-full-masked-cont
      epoch 259, kkv2-5c-hybrid-cont epoch 244) trained to budget, saved
      `checkpoint_latest`, then died `Killed: 9` about an hour into the
      wrap-up, losing every artifact that had been produced up to that point.
    - A re-run after such a kill reuses the completed stages (their stage
      function is NOT called again) and re-runs only what was lost.
    - `finalize` is never reused -- it appends to README.md and re-delivers,
      which must happen exactly once per report run.
    - A stage file recorded at a DIFFERENT epoch is ignored, never reused:
      reporting an older epoch's dolly numbers under a newer epoch's heading
      would be a silently wrong number (AGENTS.md honesty rule).
    - **Peak memory does not scale with the number of stages** (this task's
      acceptance criterion, the CPU stand-in for "does not scale with
      training length"): each stage's working set is unreachable once the
      stage returns, verified with `weakref` rather than with an RSS
      threshold, which would be flaky.
    - `trippy.train.memlog` never raises, whatever it is handed.
Fixture: fake stage functions and plain dicts only -- nothing here loads a
    Trainer, a PLY or a photograph.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest

from trippy.render import report as report_mod
from trippy.render.report import REPORT_STAGES, ReportContext, run_report_stages
from trippy.train import memlog as memlog_mod


def _context(tmp_path: Path, epoch: int = 7) -> ReportContext:
    """A ReportContext pointing at an empty tmp run dir (no checkpoint needed by fakes)."""
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return ReportContext(
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoints" / "checkpoint_latest.pt",
        export_path=run_dir / "export.ply",
        cfg=None,
        device="cpu",
        epoch=epoch,
        dolly_pose_name="IMG_1.jpg",
        offpath_names=["IMG_1.jpg"],
    )


def _install_fake_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str], boom: str | None = None) -> None:
    """Replace every real stage with a recorder; `boom` names a stage that raises."""

    def make(stage: str):
        def _fake(ctx: ReportContext, state: dict) -> dict:
            calls.append(stage)
            if stage == boom:
                raise RuntimeError(f"{stage} was killed")
            return {"stage": stage, "summary_line": f"{stage} line"}

        return _fake

    monkeypatch.setattr(report_mod, "_STAGE_FUNCTIONS", {s: make(s) for s in REPORT_STAGES})


def test_stages_run_in_order_and_each_writes_its_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)

    report = run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    assert calls == list(REPORT_STAGES)
    assert report["summary_line"] == "finalize line"
    for stage in REPORT_STAGES:
        payload = json.loads((ctx.stages_dir / f"{stage}.json").read_text())
        assert payload["stage"] == stage
        assert payload["epoch"] == ctx.epoch
        assert payload["result"]["stage"] == stage


def test_a_killed_stage_keeps_every_earlier_stages_output_and_the_rerun_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual failure mode: die mid-wrap-up, then finish it with a second invocation."""
    ctx = _context(tmp_path)
    first_calls: list[str] = []
    _install_fake_stages(monkeypatch, first_calls, boom="offpath")

    with pytest.raises(RuntimeError, match="offpath was killed"):
        run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    # Everything before the kill survived on disk; the killed stage left nothing.
    assert (ctx.stages_dir / "eval.json").exists()
    assert (ctx.stages_dir / "dolly.json").exists()
    assert not (ctx.stages_dir / "offpath.json").exists()

    second_calls: list[str] = []
    _install_fake_stages(monkeypatch, second_calls)
    report = run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    # The two expensive stages that already finished are NOT recomputed.
    assert second_calls == ["offpath", "audits", "bundle", "finalize"]
    assert report["summary_line"] == "finalize line"


def test_finalize_is_never_reused_even_when_a_stored_result_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    calls.clear()
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))
    assert calls == ["finalize"]


def test_force_recomputes_every_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    calls.clear()
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None), force=True)
    assert calls == list(REPORT_STAGES)


def test_a_stage_result_from_a_different_epoch_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path, epoch=7)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    later = _context(tmp_path, epoch=8)
    calls.clear()
    run_report_stages(later, memlog=memlog_mod.MemoryLog(None))
    assert calls == list(REPORT_STAGES)


def test_stage_subset_reuses_stored_results_for_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    calls.clear()
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None), stages=["bundle", "finalize"])
    assert calls == ["bundle", "finalize"]


def test_a_stage_subset_missing_a_dependency_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    calls: list[str] = []
    _install_fake_stages(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="needs stage"):
        run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None), stages=["bundle"])


def test_unknown_stage_name_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    _install_fake_stages(monkeypatch, [])
    with pytest.raises(ValueError, match="unknown report stage"):
        run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None), stages=["dolly", "nope"])


def test_no_stages_working_set_survives_its_own_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Peak memory is one stage's peak, not the sum: proven with weakrefs, not RSS.

    Every stage allocates a "working set" object and returns only a small
    summary. If the runner held any of them -- in `state`, in a closure, in a
    stage file kept in memory -- the weakrefs would still be alive at the end
    and the wrap-up's peak would grow with the number of stages, which is
    exactly how the karekare-v2 reports died.
    """

    class WorkingSet:
        """Stand-in for a stage's big tensors (a plain object so it can be weakref'd)."""

        def __init__(self, stage: str) -> None:
            self.stage = stage
            self.payload = bytearray(1024)

    refs: dict[str, weakref.ref] = {}

    def make(stage: str):
        def _fake(ctx: ReportContext, state: dict) -> dict:
            working = WorkingSet(stage)
            refs[stage] = weakref.ref(working)
            return {"stage": stage, "summary_line": f"{stage} line", "n_bytes": len(working.payload)}

        return _fake

    monkeypatch.setattr(report_mod, "_STAGE_FUNCTIONS", {s: make(s) for s in REPORT_STAGES})

    ctx = _context(tmp_path)
    run_report_stages(ctx, memlog=memlog_mod.MemoryLog(None))

    gc.collect()
    alive = sorted(stage for stage, ref in refs.items() if ref() is not None)
    assert refs.keys() == set(REPORT_STAGES)
    assert alive == []


def test_memory_log_records_a_begin_and_end_sample_per_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _context(tmp_path)
    _install_fake_stages(monkeypatch, [])
    log_path = tmp_path / "memory.jsonl"
    memlog = memlog_mod.MemoryLog(log_path)

    run_report_stages(ctx, memlog=memlog)

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    labels = [record["label"] for record in lines]
    assert labels[0] == "wrapup:begin"
    assert labels[-1] == "wrapup:end"
    for stage in REPORT_STAGES:
        assert f"{stage}:begin" in labels
        assert f"{stage}:end" in labels
    # Every sample carries the four counters, even where a platform cannot supply them.
    for record in lines:
        assert set(record) >= {"label", "t", "rss", "peak_rss", "mps_allocated", "mps_driver"}


# --- trippy.train.memlog's own "never raises" contract ---


def test_memlog_sample_has_the_documented_shape() -> None:
    record = memlog_mod.sample("probe")
    assert record["label"] == "probe"
    assert record["rss"] is None or record["rss"] > 0
    assert record["peak_rss"] is None or record["peak_rss"] > 0


def test_memlog_never_raises_on_an_unwritable_path(tmp_path: Path) -> None:
    # A path whose parent is a FILE cannot be created -- the log must degrade, not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    log = memlog_mod.MemoryLog(blocker / "sub" / "memory.jsonl")
    log.sample("probe")
    with log.stage("stage"):
        pass
    assert [r["label"] for r in log.records] == ["probe", "stage:begin", "stage:end"]


def test_memlog_stage_samples_even_when_the_body_raises() -> None:
    log = memlog_mod.MemoryLog(None)
    with pytest.raises(RuntimeError), log.stage("dolly"):
        raise RuntimeError("killed")
    assert [r["label"] for r in log.records] == ["dolly:begin", "dolly:end"]


def test_memlog_format_sample_is_one_readable_line() -> None:
    text = memlog_mod.format_sample({"label": "x", "rss": 2 * memlog_mod.BYTES_PER_GIB, "peak_rss": None})
    assert "mem x:" in text
    assert "rss=2.00G" in text
    assert "peak=n/a" in text


def test_heldout_metrics_on_disk_only_accepts_the_matching_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "eval_ep0244").mkdir(parents=True)
    (run_dir / "eval_ep0244" / "metrics.json").write_text(json.dumps({"epoch": 244, "psnr_mean": 14.1}))

    assert report_mod.heldout_metrics_on_disk(run_dir, 244)["psnr_mean"] == pytest.approx(14.1)
    # An older/newer epoch must NOT silently borrow epoch 244's numbers.
    assert report_mod.heldout_metrics_on_disk(run_dir, 240) == {}


# --- Trainer.release_for_report: the other half of the fix ---


def test_release_for_report_drops_the_training_objects_and_reports_memory(tmp_path: Path) -> None:
    """The finished Trainer must hand back optimizer state, point cloud and dataset.

    Holding all of that while the wrap-up rebuilds an equivalent Trainer from
    the checkpoint is what pushed the two karekare-v2 reports over the kill
    threshold. A weakref on the optimizer proves the object is genuinely
    collectable afterwards, not merely detached from one attribute.
    """
    from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

    from trippy.train.trainer import Trainer

    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    optimizer_ref = weakref.ref(trainer.optimizer)

    memlog = memlog_mod.MemoryLog(None)
    samples = trainer.release_for_report(memlog)

    assert trainer.optimizer is None
    assert trainer.point_params is None
    assert trainer.dataset is None
    assert trainer.net is None
    assert trainer._released_for_report is True
    gc.collect()
    assert optimizer_ref() is None
    assert samples["before"]["label"] == "release_for_report:begin"
    assert samples["after"]["label"] == "release_for_report:end"


def test_report_context_from_trainer_survives_release_for_report(tmp_path: Path) -> None:
    """The context must carry everything the wrap-up needs, so the Trainer can go.

    `trippy.cli._train_and_report_context` relies on exactly this ordering:
    snapshot first, release second, wrap-up third.
    """
    from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

    from trippy.train.trainer import Trainer

    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache")
    trainer = Trainer(cfg)

    ctx = report_mod.report_context_from_trainer(trainer, {"epoch": 3})
    trainer.release_for_report(memlog_mod.MemoryLog(None))

    assert ctx.run_dir == run_dir
    assert ctx.epoch == 3
    assert ctx.checkpoint_path == run_dir / "checkpoints" / "checkpoint_latest.pt"
    assert ctx.export_path == run_dir / "export.ply"
    assert ctx.device == "cpu"
    assert ctx.dolly_pose_name  # a real image name, read before the dataset was dropped
    assert ctx.offpath_names


def test_run_train_report_releases_the_trainer_before_any_stage_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compatibility wrapper must release first and run stages second, in that order.

    `trippy.cli` uses `run_report_stages` directly (so the Trainer is also out
    of scope), but `run_train_report(trainer, metrics)` is still the documented
    entry point, and its whole value is that a caller holding a finished
    Trainer gets the memory back before the wrap-up starts.
    """
    from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

    from trippy.train.trainer import Trainer

    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache")
    trainer = Trainer(cfg)

    seen: dict = {}

    def _fake_stages(ctx, held_out_metrics=None, memlog=None, **kwargs):
        # By the time any stage could run, the Trainer must already be empty.
        seen["released"] = trainer.point_params is None and trainer.optimizer is None
        seen["epoch"] = ctx.epoch
        return {"summary_line": "ok"}

    monkeypatch.setattr(report_mod, "run_report_stages", _fake_stages)
    report = report_mod.run_train_report(trainer, {"epoch": 5})

    assert report == {"summary_line": "ok"}
    assert seen["released"] is True
    assert seen["epoch"] == 5
