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
