# trips-metal — running log

This is the chronological experiment and decision log for the trippy project. Each entry is **appended** (never rewritten) and records a decision or experiment outcome with supporting numbers and artifacts.

Entries follow this format:

```
## YYYY-MM-DD HH:MM — [Experiment / Decision / Milestone]

[One sentence summary]

**Question**: What was tested?
**Job name**: reference to `output/jobs/trippy-<name>.sh` (if applicable)
**Numbers**: key metrics (FPS, PSNR, shade audit, etc.)
**Verdict**: PASS / FAIL / INCONCLUSIVE / DECISION_MADE
**Artifact**: path to rendered output, PLY, logs, or decision doc
```

Entries describe observed facts, not intentions. Once written, an entry is not edited.

---

## 2026-09-05 22:30 — Plan approved and skeleton initialized

Decisions D1–D12 locked. Repository skeleton created: AGENTS.md, CLAUDE.md, README, STATE, VERSION, scripts, and docs/decisions/ with four ADRs. All phase 1 infrastructure in place.

**Question**: Is the repo skeleton complete and ready for research work?
**Verdict**: PASS
**Artifact**: this commit; `git ls-files` confirms no images/plys/checkpoints.
- 2026-09-05T10:49:54Z submitted job trippy-smoke prio 15: trippy smoke --device mps
- 2026-09-05T11:11:03Z smoke job trippy-smoke rc=0: torch 2.14.0 on MPS inside the Splats GPU queue, inline Metal kernel ran (add_one -> 1.0 x8). Queue round-trip proven. Log: output/logs/trippy-smoke.log
- 2026-09-05T11:24:04Z submitted job trippy-raster-gpu-tests prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/raster && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m pytest -q -m gpu -s tests/test_raster_metal.py
- 2026-09-05T11:35:26Z raster GPU tests rc=0: Metal blend_fwd vs float64 refs max|out| 1.5e-6 (trilinear+broadcast, C=3/4); 1008x756, 200k pts, L=5: 41.6 ms/forward, 1.52M fragments. int64 argsort/searchsorted/bincount all OK on MPS (no fallback needed). Log: output/logs/trippy-raster-gpu-tests.log
