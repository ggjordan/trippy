# EXP-0010: point removal — TRIPS's own rule, and one aimed at the audit

## Question

TRIPS-from-Gaussians keeps **36.9%** of the opacity mass in the walkable shade volume dark
(Rec.709 luminance < 0.25), against **19.9%** for the Gaussians it started from, and 300
epochs did not move it (EXP-0003 `full2-broadcast`). One plain explanation is that nothing
in trippy's trainer has ever removed a point: TRIPS *has* a point removal/adding module,
and trippy never ported it. So:

1. Does TRIPS's own confidence-based removal, ported faithfully, move the dark-mass
   fraction at all on kk-coherent?
2. If it does not, does a rule aimed directly at the audit's own region move it — and what
   does that cost in held-out shade PSNR?

Question 2 is deliberately loaded and is reported as such (see "Honesty" below).

## What TRIPS actually does (read from source, third_party/TRIPS @ a59a65b6)

**Removal.** One rule, one threshold, one schedule, no gradient/visibility/error term:

```cpp
// src/apps/train.cpp:846-851
auto indices_to_remove =
    torch::where(tex->confidence_value_of_point.squeeze() <
                     params->points_adding_params.removal_confidence_cutoff, 1, 0).nonzero();
if (indices_to_remove.size(0) > 0) {
    train_scenes->data[i].scene->RemovePoints(indices_to_remove);
    train_scenes->data[i].scene->OptimizerClear(epoch_id, false);
}
```

- The quantity thresholded is the *effective* confidence,
  `confidence_value_of_point = sigmoid((10 + narrowing) * confidence_raw)`
  (`src/lib/models/NeuralTexture.h:42`), with `sigmoid_narrowing_factor = 0` in the shipped
  config (`configs/train_normalnet.ini:139`), i.e. plain `sigmoid(10 * confidence_raw)` —
  the same parametrisation `trippy.train.params.PointParams.conf()` already used
  (docs/TRIPS_REFERENCE.md §2).
- Cutoff: **0.3** code default (`src/lib/data/Settings.h:427`), **0.500000119** in the
  shipped ini (`configs/train_normalnet.ini:134`).
- Schedule: epochs `start_removing_points_epoch + i * point_removal_epoch_interval`, built
  at startup in `src/apps/train.cpp:533-538`; code defaults **200 / 50**
  (`Settings.h:403-406`). Called once per epoch, at the top, before that epoch's training
  steps, and only for `epoch_id > 0` (`train.cpp:670-674`).
- **Disabled in every shipped config**: `start_removing_points_epoch = 2000` and
  `start_adding_points_epoch = 2020` with `num_epochs = 600`
  (`train_normalnet.ini:8,130-133`), so neither ever fires in a stock TRIPS run.
- Surgery: `NeuralScene::RemovePoints` (`NeuralScene.cpp:1375-1470`) rebuilds the point
  cloud, index-selects `texture_raw` / `confidence_raw` / `confidence_value_of_point`
  (`NeuralTexture.h:88-95`), then `ShrinkTextureOptimizer` (`NeuralScene.cpp:362-370`)
  → `MyAdam::shrinkInternalState` (`src/lib/models/MyAdam.cu:346-374`), which
  `index_select`s the first moments, second moments and per-element step counters onto the
  survivors, and finally zeroes every gradient.

**Adding — NOT ported.** Three code paths exist and none is portable in a day:

| path | why not |
|---|---|
| NeAT CT reconstruction (the default, `neat_use_as_subprocess_ct_reco = true`) | shells out to an external `NeAT/bin/reconstruct` binary on per-epoch L1-loss images, behind `#ifdef COMPILE_WITH_VET`; the density volume it returns is sampled by `NeuralScene::AddNewRandomPointsFromCTHdr` (`NeuralScene.cpp:859-1000`). That is a separate research codebase, not a rule. |
| grid loss (`AddNewRandomPointsInValuefilledBB`, `NeuralScene.cpp:1330-1373`) | **dead code.** It adds points per cell in proportion to `t_cell_value`, and *nothing in the shipped renderer ever writes that buffer* — `SetValueForCell` / `GetPointerForValueForCell` (`NeuralPointCloudCuda.h:201-203`) have zero callers, so `t_cell_value` stays at its `zeros` init (`NeuralPointCloudCuda.cpp:182`), `num_max_points_to_add` computes to 0, and it adds nothing. It also has a placement bug: it multiplies the random offsets by `cell_bb_min` instead of adding it (the correct line is commented out two lines below). |
| point growing (`AddPointsViaPointGrowing`, `NeuralScene.cpp:1295-1327`) | not a "high error region" rule at all — it duplicates every existing point `factor` times with a random offset. Portable, but it is densification-everywhere, which is the opposite of what this experiment is about. |

Parked, with the finding written down, per AGENTS.md §7 ("never cull an idea for effort"):
a *trippy-native* adder driven by rendered depth in high-error regions is a real idea, but
it would be trippy's design, not a port, and belongs in its own experiment.

## The one deviation in arm A, and why

TRIPS initialises **every** confidence at `sigmoid(10 * 0.5) = 0.9933`
(`NeuralTexture.cpp:42`), so a 0.3 or 0.5 cutoff means "training pushed this point down".
trippy initialises confidence from the **source PLY's own opacity** instead
(`PointParams.raw_conf = logit(conf0)/10`). Measured on `kkc_15000` with `min_opacity =
0.05` (400k-point sample):

| conf quantile | p1 | p5 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|---|
| value | 0.052 | 0.060 | 0.105 | 0.179 | 0.311 | 0.499 |

So **74% of points are already below TRIPS's 0.3 code default and 90% below its shipped
0.5 at epoch 0**. Running TRIPS's own number here would delete the scene on the first pass
and measure nothing about training. Arm A therefore keeps the rule's exact shape (a fixed
cutoff on `sigmoid(10*raw_conf)`, on TRIPS's own schedule ratios) with
`conf_threshold: 0.1`, which cuts only into the low-opacity tail, plus a `min_points`
floor trippy adds and TRIPS does not have.

## Arm A': the relative analogue (`point_removal.mode: relative`)

Arm A's `conf_threshold: 0.1` is a workaround, not a port: it exists only because an
absolute cutoff under trippy's own confidence init mostly measures where a point
*started*, not what training did to it. `trippy.train.prune_config.PointRemovalConfig`
now has a `mode` field (`absolute`, the default and exactly arm A's rule above; or
`relative`) plus `rel_factor` (default 0.3). In `mode: relative`, a point is removed once
`sigmoid(10*raw_conf) < rel_factor * init_conf`, where `init_conf` is that SAME point's own
confidence at construction (`trippy.train.params.PointParams.init_conf`, a buffer
snapshotted once and carried — index-selected, never optimizer-shrunk — through every
later removal pass and checkpoint round trip). This is what an absolute cutoff means for
free under TRIPS's own uniform init, and is the faithful analogue for trippy's: it rewards
or punishes *movement*, not starting position. `conf_threshold` doubles as an optional
absolute floor in this mode (an independent OR trigger on top of the relative test, so a
point that starts, and stays, negligible is still caught even though it can never fall by
`rel_factor` from an already-tiny start) — see `trippy.train.prune.confidence_drop_mask`
and `trippy.constants.POINT_REMOVAL_MODE_ABSOLUTE` for the exact rule and the full
rationale. `shade_prune` gets the same `mode`/`rel_factor` fields, independently, for its
own confidence leg.

`config_removal_rel.yaml` (arm A') is `config_removal.yaml` (arm A) with exactly two
fields changed: `point_removal.mode: relative` and `point_removal.rel_factor: 0.3`.
Everything else — recipe, schedule, `conf_threshold: 0.1` (kept as the optional floor) —
is identical, so arm A vs. arm A' isolates the one question this analogue exists to
answer: does thresholding on a point's own decline, instead of its absolute value, change
which points get removed or how the dark-mass fraction moves?

## Configs

Both long arms are EXP-0003's fast recipe (`config_full2_broadcast.yaml`: broadcast, kNN
sizes, width 1008, crop 384, 300 epochs, `train_factor 1.0`, the six shade frames forced
held out) and differ from it only in the blocks below.

- **`config_removal.yaml` (arm A)** — TRIPS's rule only.
  `point_removal: {enabled, start_epoch 100, every_epochs 25, conf_threshold 0.1,
  min_points 1000000}`. The schedule is TRIPS's own ratios scaled to 300 epochs
  (200/600 → 100, 50/600 → 25), so removal fires eight times: 100, 125, …, 275.
  `shade_prune.enabled: false` — but `log_dark_mass` is on, so this arm still reports the
  audit statistic every eval.
- **`config_shade_prune.yaml` (arm B)** — arm A **plus**
  `shade_prune: {enabled, frames = SHADE_FRAMES_KK, znear_frac 0.05, zfar_frac 0.5,
  lum_threshold 0.25, conf_threshold 0.5, start_epoch 100, every_epochs 25}`.
- **`config_removal_rel.yaml` (arm A')** — arm A with `point_removal.mode: relative` and
  `point_removal.rel_factor: 0.3`, nothing else changed (see "Arm A': the relative
  analogue" above). Its `run_dir` is an absolute path
  (`/Users/nzbirdranch/trippy/output/runs/EXP-0010-point-removal/removal-rel`), not a
  relative one, because it was queued from a git worktree (see "Artefact location
  warning" below for why that matters).
- **`config_smoke.yaml`** — MPS path proof: `max_points 200000`, width 504, 4 epochs,
  removal every epoch from epoch 1, shade prune at epoch 2. Minutes, not hours.

## Honesty — read before quoting arm B

`shade_prune` **removes exactly the points the shade audit counts**. It is a heuristic
aimed at a metric, not a claim that those points are wrong. Two consequences:

1. Arm B's dark-mass fraction is not evidence that the shade renders as shading. It is
   evidence that a rule which deletes dark in-region mass deletes dark in-region mass.
2. Arm B is only interesting **next to its held-out shade PSNR**. If shade PSNR falls
   relative to arm A, the removed points were carrying real signal, and the experiment has
   found a way to game the audit rather than a way to fix the geometry. Report both
   numbers in the same sentence, always.

The verdict on either arm is Jordan's in the viewer (AGENTS.md §7), not the audit.

## What is measured, and where

Every `Trainer.evaluate` now writes a `points` block into `metrics.json` and
`metrics.jsonl`:

```json
"points": {
  "n_points": 5738xxx,
  "n_removed_total": 0,
  "shade_region": {
    "n": 5738xxx, "n_in_region": 1633974, "mass_in_region": 336873.5,
    "dark_mass_lum0.25": 67068.8, "dark_n_lum0.25": 286884,
    "dark_mass_fraction": 0.1991
  }
}
```

computed in-process from the live parameters by `trippy.train.prune.dark_mass_stats`.
That port is verified against the real tool: on `kkc_15000` it reproduces
`depthprior_shade_audit.py`'s cached numbers **exactly** — `n_in_region` 1,633,974,
`mass_in_region` 336873.52631, `dark_mass_lum0.25` 67068.80576, fraction **0.199092**
(the "19.9% baseline") — and reproduces all six views' `d` / `nobs` / `znear` / `zfar`,
while reading the binary `sparse/0` model where the tool reads `sparse_txt`. Cost: under a
second for 7.36M points, so it runs at every eval.

## Commands

```bash
# Arms A, A' and B (prio 70, behind Splats' own jobs; they join a long queue).
bash scripts/queue_training.sh experiments/EXP-0010-point-removal/config_removal.yaml --max-minutes 300
bash scripts/queue_training.sh experiments/EXP-0010-point-removal/config_removal_rel.yaml --max-minutes 300
bash scripts/queue_training.sh experiments/EXP-0010-point-removal/config_shade_prune.yaml --max-minutes 300

# MPS smoke (prio 16, jumps the training queue).
bash scripts/gpu_submit.sh --prio 16 exp0010-removal-smoke -- \
  trippy train --config experiments/EXP-0010-point-removal/config_smoke.yaml --report
bash scripts/gpu_wait.sh exp0010-removal-smoke

# Point count and dark-mass fraction over training, from either run's metrics.jsonl:
python - <<'PY'
import json
for line in open("output/runs/EXP-0010-point-removal/exp0010-removal/metrics.jsonl"):
    row = json.loads(line)
    if not row.get("eval"):
        continue
    pts = row["points"]
    reg = pts.get("shade_region", {})
    print(row["epoch"], pts["n_points"], pts["n_removed_total"],
          round(reg.get("dark_mass_fraction", float("nan")), 4),
          round(row["shade_eval"]["psnr"], 2))
PY
```

## Results

### `trippy-exp0010-removal-smoke` — rc 0 (prio 16, MPS, 4 epochs, 200k points, width 504)

Both rules run on MPS. Optimiser-state surgery holds: training continues across every pass
with no shape error, and the run exports and self-reports normally.

| epoch | points | removed (cum.) | n in region | mass in region | dark mass (lum<0.25) | **dark fraction** | held-out PSNR | **held-out shade PSNR** |
|---|---|---|---|---|---|---|---|---|
| 0 | 200,000 | 0 | 48,265 | 11,496.1 | 2,207.6 | **0.1920** | 12.52 | **11.24** |
| 1 | 154,660 | 45,340 | 37,099 | 10,681.5 | 2,160.3 | **0.2023** | 12.48 | **10.74** |
| 2 | 147,488 | 52,512 | 31,003 | 9,314.1 | 970.5 | **0.1042** | 12.47 | **10.75** |
| 3 | 146,528 | 53,472 | 30,802 | 9,326.6 | 1,116.1 | **0.1197** | 12.50 | **10.68** |

Passes, from the run log:
`epoch 1 point_removal 200,000 → 154,660` (45,340 = 22.7%, the conf < 0.1 tail),
`epoch 2 point_removal 154,660 → 153,158` (1,502) then `shade_prune 153,158 → 147,488` (5,670),
`epoch 3 point_removal 147,488 → 146,528` (960).

Three things to read out of it, none of them a verdict (4 epochs on a 200k subsample):

1. **TRIPS's rule alone did not lower the dark fraction — it raised it.** Epoch 1 removed 22.7%
   of the points and the fraction went **0.1920 → 0.2023**: the confidence tail it deletes is
   not preferentially dark. That is the honest early signal on question 1, and exactly what the
   300-epoch arm A exists to confirm or refute.
2. **`shade_prune` moves it, and not permanently.** The single epoch-2 pass took 5,670 points
   and the fraction fell **0.2023 → 0.1042** — then drifted back to **0.1197** by epoch 3 with no
   further prune, because the surviving points' colours keep training. Deleting the measured
   mass does not stop it re-forming.
3. **The PSNR cost sits with TRIPS's rule, not with the shade prune.** Held-out shade PSNR fell
   **11.24 → 10.74 dB** across epoch 1's confidence removal, and then did **not** fall further
   across the shade prune (10.740 → 10.746). On this evidence the audit-targeting prune is the
   cheaper of the two — but four epochs is nowhere near enough to claim that, and the run has
   not re-converged after either deletion.

### Long runs

| job | rc | epochs | points start → end | dark-mass fraction start → end | held-out shade PSNR |
|---|---|---|---|---|---|
| `trippy-exp0010-removal` (arm A, prio 70) | _running_ | 300 | | | |
| `trippy-removal-rel` (arm A', prio 70) | _queued_ | 300 | | | |
| `trippy-exp0010-shade-prune` (arm B, prio 70) | _queued_ | 300 | | | |

Arm A' was submitted 2026-09-06 via `scripts/queue_training.sh
experiments/EXP-0010-point-removal/config_removal_rel.yaml --max-minutes 300`, run_dir
`/Users/nzbirdranch/trippy/output/runs/EXP-0010-point-removal/removal-rel` (absolute, per
the artefact-location warning below — queued from `.worktrees/relative-removal`); see
`research/trips-metal.md` for the submit line.

Arm A's own epoch-0 reading (full cloud, logged by the run itself): **5,736,619 points,
1,387,211 in region, mass 334,395.6, dark mass 80,859.2, fraction 0.2418**, held-out shade
12.55 dB. Note the starting fraction is **24.2%**, not the raw PLY's 19.9%: `min_opacity: 0.05`
drops 1.6M near-transparent points and removes proportionally more bright mass than dark. So
the trajectory this experiment is watching is **24.2% at epoch 0 → 36.9% at epoch 300**
(EXP-0003's number for the same recipe without removal), and the question is whether removal
bends it.

**Artefact location warning.** Both long jobs were submitted from the
`.worktrees/point-removal` git worktree, and their configs use a *relative* `run_dir`, so they
write to `.worktrees/point-removal/output/runs/EXP-0010-point-removal/…`, not the main
checkout's `output/`. Do not delete that worktree by hand while they run; use
`scripts/worktree_rm.sh point-removal`, which rescues `output/` out of it. (Their `--report`
deliverables do land in the main `$TRIPPY_OUTPUT/deliver/`, since `gpu_submit.sh` exports it.)
Arm A' avoids this trap on purpose: it was submitted from `.worktrees/relative-removal` with
`run_dir` set to an *absolute* path
(`/Users/nzbirdranch/trippy/output/runs/EXP-0010-point-removal/removal-rel`), so it writes
straight to the main checkout's `output/` regardless of what happens to the worktree it was
queued from.

Baselines to beat, both on `kk-coherent` (EXP-0003, docs/EXPERIMENTS.md):
plain Gaussians `kkc_15000` dark mass **19.9%**; TRIPS `full2-broadcast` after 300 epochs
**36.9%**, held-out shade 15.27 dB (neighbour exposure) vs the Gaussians' 14.94 dB.
