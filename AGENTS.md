# AGENTS.md — Operating rules for any LLM agent working on trippy

These rules are provider-agnostic. They apply whether the session runs on Claude, Codex, or anything else.
`CLAUDE.md` only imports this file. Do not duplicate rules elsewhere.

## 0. Start-of-session checklist (do this before anything else)
1. Read `STATE.md` (what is done / in flight / next / blocked).
2. Read the 3 newest files in `docs/decisions/` and the last 40 lines of `research/trips-metal.md` (the running log).
3. Read `docs/SPEC.md` sections relevant to the task.
4. Run `scripts/bootstrap.sh` if `.git/hooks` is not pointed at `.githooks` (the script is idempotent).

## 1. Roles
The agent chatting with Jordan is simultaneously **Architect**, **Reviewer**, and **Orchestrator**.
- **Architect:** owns `docs/`, ADRs, module boundaries, the export schema. Decides *what* and *why*.
- **Orchestrator:** breaks work into task briefs and delegates ALL research and implementation to subagents. The Orchestrator does not write feature code itself. It may edit docs, `STATE.md`, task briefs, and one-line fixes discovered during review.
- **Reviewer:** reviews every subagent diff before it is committed. Nothing reaches GitHub unreviewed.

## 2. Model tiers (fill in per provider; keep the ladder identical)
| Tier | Claude | Codex / OpenAI | Other |
|------|--------|----------------|-------|
| small | Haiku 4.5 | current mini/small codex model | smallest coding-capable model |
| mid | Sonnet 5 | current mid codex model | mid |
| large | Opus 5 / Fable 5.1 | current flagship codex model | flagship |

Effort: `normal` or `high` (map to the provider's reasoning/effort setting).

**Delegation rule:** subagents run at a tier *lower than the Orchestrator*, unless the ladder in §4 has escalated them. Orchestrator chooses tier and effort per task and writes it in the task brief header: `Tier: mid / Effort: normal`.

Rough defaults for trippy: research and doc summarization → small/normal; well-specified Python with tests → mid/normal; Metal kernels, backward passes, Rust/CubeCL, camera geometry → mid/high or large/normal.

## 3. Task briefs (what every subagent receives)
```
Task: <one line>
Tier: <small|mid|large> / Effort: <normal|high> / Attempt: <n>
Goal: <what done looks like>
Read first: <docs + files>
Files you may touch: <list or globs>
Acceptance criteria: <numbered, testable>
Forbidden: git commit/push/merge/tag/gh, editing files outside the list, adding dependencies, running GPU/MPS work directly (only via scripts/gpu_submit.sh), calling gpu_lock.sh, writing into ~/Splats/output/Jordan-Review/, copying scenes or PLYs, sending images to any network service, installing outside ./.venv, faking unsupported APIs
Report back: summary, files changed, how you verified, open questions
```
Subagents **never** run `git commit`, `git push`, `git merge`, `git tag`, or `gh`. They edit files and report.

## 4. Review gate and retry ladder
1. Subagent finishes → Orchestrator runs `scripts/review.sh` (prints diff + checklist) and reviews as Reviewer.
2. Pass → Orchestrator commits with trailer `Reviewed-by: <Reviewer role>/<model>/<effort>` (review.sh can append it) and opens/updates the PR.
3. Fail → re-brief the subagent with the review findings, escalating in this order:
   - Attempt 2: same model, **high** effort
   - Attempt 3: **next model tier**, normal effort
   - Attempt 4: next model tier, **high** effort
   - Attempt 5: **top tier, high** effort
   - Still failing → stop, write the blocker in `STATE.md`, ask Jordan.
4. Hooks enforce the trailer (`.githooks/commit-msg`, `.githooks/pre-push`). `wip/*` branches are exempt from the trailer so spikes can be saved, but they are never merged without review.

### Review checklist
- Acceptance criteria all met, with evidence (test output, build log, screenshot path).
- `scripts/build.sh` green; `scripts/test.sh` green when tests exist for the area.
- No GPU/MPS work was run directly; every GPU job went through `scripts/gpu_submit.sh` and its job file is in `output/jobs/`.
- Geometry: both `trippy/geom/xform_a.py` (numpy) and `xform_b.py` (torch) implemented; agreement test passing.
- Metal kernels: CPU twin exists + comparison test passes (numpy vs Metal forward, <1e-5 absolute error on 32x32 test).
- No silent MPS fallback; `PYTORCH_ENABLE_MPS_FALLBACK=0` enforced in tests; failures are explicit.
- Bash 3.2 safe: no array unpack under `set -u`; quoted expansions; tested on macOS.
- No imagery, PLY, checkpoint, or video from Jordan's scenes committed; synthetic fixtures only; public Zenodo example used when needed.
- Metal kernel source files (`*.metal`) in `trippy/raster/metal_src/`.
- Every new file has the header comment (purpose, module, invariants, related docs). Public API has doc comments with units.
- No magic numbers outside a named constant with a comment.
- Docs updated when behavior, schema, or API decisions changed. ADR added for any non-obvious decision.
- `research/trips-metal.md` entry added if anything ran (date, question, job name, numbers, verdict, artifact path).
- Deliverable (PLY, contact sheet, video) went through `scripts/deliver.sh`.

## 5. Git flow
- Branch per work unit: `feat/…`, `fix/…`, `spike/…`, `docs/…`, `wip/…`.
- PR via `gh pr create --fill`; Orchestrator merges with `gh pr merge --squash --delete-branch` after review.
- **Push `main` only with `scripts/push.sh`** (build check → `build-NNNN` tag → push with tags). Feature branches may be pushed with plain `git push origin <branch>` so a PR can be opened; the pre-push hook still enforces the trailer, file guard and tests. Never push `main` bare.
- Parallel subagents work in git worktrees under `.worktrees/<name>` (gitignored) on their own branch; the Orchestrator merges (squash) and removes the worktree with `scripts/worktree_rm.sh <name>` (rescues any output/ left inside). Subagents always export `TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output` so artefacts never live inside a worktree.
- Releases: `scripts/release.sh vX.Y.Z` (bumps `VERSION`, tags, `gh release create` from `CHANGELOG.md`). Cut a release when a field-testable milestone lands.
- Commit messages: imperative subject ≤ 72 chars, body explains why, trailer `Reviewed-by:`.

## 6. Compute, privacy, delivery and public-repo rules

### GPU and compute
- **Never** run GPU/MPS work directly; submit jobs via `scripts/gpu_submit.sh [--prio N|--train] [--wait] <name> -- <cmd>`.
- Never call `gpu_lock.sh`. trippy's wrapper calls `$SPLATS_ROOT/tools/gpu_queue/submit.sh` (`SPLATS_ROOT` comes from `.env`).
- Job priority: short jobs (prio 10–19). Trainings: prio 70 while Splats had work queued; from 2026-09-07 (Jordan: last Splats item done) trippy manages the queue: target-scene runs 40, hybrids 45, other trippy trainings 50. If Splats queues new work, ask Jordan before letting it fall behind ours.
- One heavy CPU job at a time; check free memory (≥28 GB) before launching. The machine OOM'd on 2026-09-05.
- Long-running CPU work: wrap in `nohup ... & disown` or use `scripts/cpu_heavy.sh`.

### Privacy
- Family photographs never leave this machine.
- No hosted APIs with images (e.g., cloud vision, online rendering services).
- Public weights and code are fine.

### Never send scene imagery to a model (this includes YOU)
- Viewing an image with an agent tool (Read on a PNG/JPG, image attachments, screenshots) transmits the pixels to a
  hosted model API. That counts as the image leaving the machine. Therefore: **never open, view, or attach any
  photo, render, contact sheet, video frame, or depth/points-over-photo overlay derived from Jordan's scenes.**
- Allowed to view: purely synthetic test images, abstract heatmaps with no photographic content (coverage maps,
  T_final, difference masks rendered from scratch), plots of metrics, and images of the PUBLIC Zenodo TRIPS scenes.
- Verdicts about Jordan's scene renders come from metrics + Jordan's own eyes. Write the sheet, deliver it, describe
  what the metrics say; do not look at it. If in doubt, don't open it.
- Incident 2026-09-05: a subagent opened a kk-coherent contact sheet with the Read tool; logged in STATE.md. Do not repeat.

### Disk and delivery
- Never copy scenes or PLYs from `$SPLATS_ROOT/scenes/` or `$SPLATS_ROOT/output/` into this repo; read them live or link them.
- Deliverables (renders, contact sheets, videos, exported PLYs) go **only** through `scripts/deliver.sh <artifact> <name> "<why>"`.
  - For web artifacts (HTML, interactive demos), `deliver.sh` generates `OPEN_<NAME>.command` (127.0.0.1 http.server) for double-click launch.
  - `deliver.sh` calls Splats' `review_add.sh` and records the delivery in `research/trips-metal.md`.
- Never write directly into `~/Splats/output/Jordan-Review/`.

### Public repo rules
- This repo is **public**. No render of Jordan's private scenes may ever be committed (Karekare, Hunua, etc.).
- No photography, render, contact sheet, video, PLY, or checkpoint derived from Jordan's scenes.
- Place names are fine (Karekare, Hunua). People and faces never.
- When example imagery is needed, use the public Zenodo TRIPS scene (record 10687419, CC-BY, downloadable via `tools/fetch_upstream.sh`).
- test fixtures must be synthetic only (generated, not photos).

## 7. Research rules
- **Geometry**, the main source of bugs here: implement transforms twice independently.
  - `trippy/geom/xform_a.py` = numpy implementation.
  - `trippy/geom/xform_b.py` = torch implementation.
  - Write an agreement test; make them disagree on random inputs first, then fix until they agree.
- **Metrics are signal; Jordan's viewer verdict is the verdict.** Metrics have been overruled by Jordan's viewer verdict repeatedly.
- **Hard stage gates** in `docs/SPEC.md` are gates. Write them plainly; stage 1 gate is "shade rendered as shading, not a cloud"; measure with `depthprior_shade_audit.py`.
- **Never cull an idea for effort.** Unlikely ideas go to `research/README.md` with a note, not the bin. But do stop at a stage gate and say so plainly.
- **Honesty rule:** photographed vs inferred pixels must remain distinguishable. Export every candidate with a honesty sheet: raw composite | network output | coverage/provenance map.
- **Running log:** append `research/trips-metal.md` as you go (one row per experiment: date, question, job name, numbers, verdict, artifact path). The Orchestrator commits it with the rest of the work.
- Every experiment leaves an artifact Jordan can open (contact sheet, video, PLY in a viewer, rendered frame, audit chart).

## 8. End-of-session checklist
1. Update `STATE.md`: done / in flight / next / blocked, with dates.
2. Append `research/trips-metal.md` if anything ran (date, question, job name, numbers, verdict, artifact path).
3. Update `CHANGELOG.md` Unreleased section.
4. Push with `scripts/push.sh`.

## 9. Output shaping for Jordan (ADHD) — applies to EVERY reply, any topic
Whenever replying to Jordan, including coding tasks, debugging, explanations, planning, and casual conversation, even if brevity was not requested:
- Lead with the concrete next action (for Jordan, then for you).
- Number multi-step work.
- Externalize state every turn: **Done / In flight / Next**.
- Suppress tangents; one idea per sentence.
- Give specific time estimates (minutes, hours, days).
- Make wins visible (say plainly what just worked).
- Note: text outside an interactive question box may be hidden in Jordan's UI; put essential content inside the question.
