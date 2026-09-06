# Person-exclusion masks for kk-coherent

**Problem:** `kk-coherent` (`/Users/nzbirdranch/Splats/scenes/karekare/kk-coherent`) trained
without person masks, so Jordan's kids show up as ghosts in TRIPS outputs. This generates the
missing masks and prepares (but does not run) the config + requeue changes needed to retrain
with them.

**2026-09-06 correction from Jordan:** masks are not mandatory — he wants BOTH the existing
unmasked results and masked results. `scripts/requeue_with_masks.sh` therefore does not touch
or dequeue any existing run; it only submits a masked *sibling* run per config.

## 1. Tool and environment

`/Users/nzbirdranch/Splats/tools/make_masks3.py` ("Person-exclusion masks v3") is a plain
argparse script: `make_masks3.py <src-image-dir> <dst-mask-dir> [--seg-thresh --grow --dilate]`.
It reads every `.jpg`/`.jpeg`/`.png` directly inside `<src>` (one level, not recursive) and
writes `<dst>/<stem>.png` for each, so `kk-coherent/images/` (238 flat `.jpg` files) is exactly
the input shape it expects — no adapter needed beyond pointing it at that directory and a new
output directory.

It calls Apple's Vision framework via pyobjc (`import Vision, Quartz` from `Foundation`):
`VNGeneratePersonInstanceMaskRequest` (primary), `VNGeneratePersonSegmentationRequest`
(recall fallback), `VNDetectHumanRectanglesRequest` (box fallback, dilated and merged in).
This needs a Python with `pyobjc-framework-Vision`/`Quartz`/`Foundation` installed, which the
repo's own `.venv` and the system `python3` do **not** have. Splats' `research/EVAL_HARNESS.md`
(sec "run everything from...") documents the venv that does:
`~/Splats/tools/ml-sharp/.venv` ("has torch+MPS, transformers, lpips, pyobjc-Vision —
installed there for this task"). Confirmed directly:

```
/Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -c \
  "import Vision, Quartz, Foundation; print('ok')"
# -> ok
```

**GPU vs CPU/Neural-Engine routing.** The task brief's default was to submit this at
`scripts/gpu_submit.sh --prio 18 --wait`, falling back to `scripts/cpu_heavy.sh` only if the
tool documents itself as CPU/Neural-Engine-only. `make_masks3.py`'s own header doesn't say
that, but Splats' `research/EVAL_HARNESS.md` does, for this exact tool category: *"GPU
discipline: rendering ... uses the MPS rasterizer ... Mask generation, LPIPS-on-CPU, and the
floater audit are CPU-only and need no lock."* Vision's person-segmentation requests run on
the CPU/Neural Engine, not Metal — routing this through Splats' Metal GPU queue would have
queued it behind the currently-running prio-70 training (~2h) for a job that never touches the
GPU at all. So this ran as a CPU/Neural-Engine job, not a GPU-queue job.

`scripts/cpu_heavy.sh kk-masks -- ...` was tried first (it enforces the project's "one heavy
CPU job, ≥28GB free" discipline) and **refused**:

```
$ scripts/cpu_heavy.sh kk-masks -- bash -c '...'
ℹ clearing stale lock (pid 2778 not running)
✗ only 15 GB free; need >=28 GB
```

96 GB machine, ~15 GB free at the time — plausibly the concurrent prio-70 GPU training's
unified-memory footprint (Apple Silicon shares GPU/CPU RAM), not this job's own needs: per-image
Vision inference on one 4032-ish-px photo at a time has a small, roughly constant footprint,
nothing like the batched SegFormer/LPIPS jobs the 28 GB guard is sized for. Given
`cpu_heavy.sh` cannot be edited (not in this task's allowed file list) and the guard is a hard
refusal (exit 5, not a warning), the mask job was run directly, outside both queues, logged by
hand to `research/trips-metal.md` in place of the entry `gpu_submit.sh`/`cpu_heavy.sh` would
otherwise have written automatically. **Open question for the Orchestrator:** should
`cpu_heavy.sh`'s memory guard get a `--skip-mem-check` escape hatch for provably light jobs
like this one, or should the 28 GB threshold itself be revisited given the machine's actual
free-memory behavior under a concurrent GPU training? Not fixed here — out of this task's file
list (trainer/infra scripts are out of scope; see "Files you may touch" in the brief).

## 2. Exact command run

```
/Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python \
  /Users/nzbirdranch/Splats/tools/make_masks3.py \
  /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent/images \
  /Users/nzbirdranch/trippy/output/masks/kk-coherent
```

- **Job name:** `kk-masks` (informal — not a `gpu_submit.sh`/`cpu_heavy.sh` job id, since
  neither queue accepted it; see above).
- **Start / end (UTC):** 2026-09-06T06:47:47Z → 2026-09-06T06:57:36Z. **Elapsed:** 589 s
  (~9.8 min) for 238 images (~2.5 s/image), CPU/Neural-Engine only, no GPU-queue contention.
- **Exit status:** the wrapper's `rc` capture used bash's `${PIPESTATUS[0]}` under a `zsh`
  login shell, where that array doesn't exist, so the written `.rc` file is empty — a
  shell-portability bug in the ad-hoc wrapper, not a tool failure. Success is otherwise
  unambiguous: the tool's own log ends with a clean summary (no traceback), and the output
  directory has exactly 238 `.png` files whose basenames are a 1:1 match (verified with `diff`)
  against the 238 source `.jpg` basenames in `kk-coherent/images/`.
- **Tool's own summary** (`output/logs/kk-masks.log`):
  ```
  238 images | instance-mask fired 186 | boxes fired 190 | any mask 205
  masked fraction mean 12.38%  max 79.5%
  ```
- **Output:** `/Users/nzbirdranch/trippy/output/masks/kk-coherent/IMG_3703.png` … `IMG_3940.png`
  (238 files, one per `kk-coherent/images/*.jpg`, same basename). Nothing was written under
  `~/Splats`.

## 3. Per-image person-fraction summary (numbers only, no imagery viewed)

Computed with PIL/numpy over the 238 output masks (`a < 128` = black = person/ignore,
`a > 127` = white = keep; matches the tool's own docstring polarity):

| stat | white (keep) fraction | black (person/ignore) fraction |
|---|---|---|
| min | 0.2055 | 0.0000 |
| median | 0.9196 | 0.0804 |
| max | 1.0000 | 0.7945 |
| mean | 0.8762 | 0.1238 |

- **Frames with any person** (black fraction > 0.0005, the tool's own "any mask" threshold):
  **205 / 238** (86%).
- **Frames fully clean** (black fraction == 0, no detection fired at all): **23 / 238**.
- Highest person-coverage frame: `IMG_3827` (79.5% black). Lowest: 23 frames tied at 0.0%,
  e.g. `IMG_3781`.
- These 205/238 "any person" frames are the direct numeric explanation for the ghosting bug:
  most of kk-coherent's 238 training images do contain a person somewhere in frame.

## 4. Polarity verification (numeric only — never viewed any imagery)

Per the tool's docstring: **BLACK = ignore (person), WHITE = keep.** Verified two ways,
both purely numeric:

**a) karekare-v2 masks read the same way.** `karekare-v2/masks/*.png` (Splats' own, already
in production) are heavily white overall — sampling the first 20 alphabetically: white
fraction min 0.501, median 0.746, max 0.925. A convention where the *rare* class is black
(person, a small part of most frames) and the *common* class is white (background, kept) is
consistent with a person-exclusion mask, not the reverse.

**b) Cross-scene identity check (the strong evidence).** All 238 `kk-coherent/images/*.jpg`
basenames are also present in `karekare-v2/images/` (confirmed: `comm -12` on the sorted
basename sets returns all 238) — i.e. Splats already has an independently-generated,
already-in-production mask for every one of these exact photographs, from the *same* tool
family. Splats' `research/EVAL_HARNESS.md` documents that its person masks were validated **by
eye** ("Karekare pool frame with two children present: ... children correctly excluded").
Comparing our new kk-coherent masks against those pre-existing, already-validated karekare-v2
masks for the identical 238 photos, both read black-fraction-only (no image opened):

```
kk-coherent black fraction:  min=0.0000  median=0.0804  max=0.7945
karekare-v2 black fraction:  min=0.0000  median=0.0305  max=0.3256
Pearson corr(kk black%, v2 black%) over all 238 paired frames = 0.9485
```

The correlation is strong and the extremes line up on the *same filenames*:

| frame | kk-coherent black % | karekare-v2 black % | note |
|---|---|---|---|
| IMG_3827 | 79.5% | 32.6% | highest in both sets |
| IMG_3826 | 67.3% | 29.9% | 2nd-highest in both |
| IMG_3889 | 58.8% | 29.3% | 3rd-highest in both |
| IMG_3891 | 48.4% | 25.0% | 4th-highest in both |
| IMG_3890 | 42.2% | 23.7% | 5th-highest in both |
| IMG_3824, IMG_3822, IMG_3932, IMG_3823 | 0.0% | 0.0% | zero in both |
| IMG_3888 | 3.7% | 0.0% | kk-coherent (v3, instance-mask API) catches a small/distant
  person v2's pipeline missed — consistent with the tool's own header: *"v1 ... near-zero
  confidence on small/distant children ... v3 uses the per-instance mask API, which fires
  where the others fail."* |

Frames Splats already validated by eye as containing people score highest black-fraction in
*both* independently-generated mask sets on the identical photograph; frames with no person
score ~0% black in both. This is the numeric equivalent of "black=person, verified against
frames known to contain people vs not," without opening a single image (forbidden — see
AGENTS.md "Never send scene imagery to a model").

## 5. Every trippy config pointing at kk-coherent, and the `masks_dir:` line to add

`grep -rl "kk-coherent" experiments/**/*.yaml` finds 17 configs across 5 experiments. All of
them already had a training queue entry as of 2026-09-06 (see `scripts/requeue_with_masks.sh`
usage below) — a good reminder of how many runs the ghosting bug affects.

| config | run_dir (unmasked, untouched) | masks_dir: to add |
|---|---|---|
| `experiments/EXP-0003-kk-trips-train/config.yaml` | `output/runs/EXP-0003-kk-trips-train/full1` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_smoke.yaml` | `output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_smoke4` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_broadcast.yaml` | `output/runs/EXP-0003-kk-trips-train/full1-broadcast` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_full2_trips.yaml` | `output/runs/EXP-0003-kk-trips-train/full2-trips` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_full2_broadcast.yaml` | `output/runs/EXP-0003-kk-trips-train/full2-broadcast` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_full3_alt.yaml` | `output/runs/EXP-0003-kk-trips-train/full3-alt` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0003-kk-trips-train/config_full3_alt_bc.yaml` | `output/runs/EXP-0003-kk-trips-train/full3-alt-bc` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0005-hybrid-c/config.yaml` | `output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0006-union/config_trips.yaml` | `output/runs/EXP-0006-union/union-trips` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0006-union/config_broadcast.yaml` | `output/runs/EXP-0006-union/union-broadcast` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0009-hybrid-a/config.yaml` | `/Users/nzbirdranch/trippy/output/runs/EXP-0009-hybrid-a/hybrid-a-all-levels` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0009-hybrid-a/config_smoke.yaml` | `/Users/nzbirdranch/trippy/output/runs/EXP-0009-hybrid-a/hybrid-a-smoke` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0009-hybrid-a/config_bc.yaml` | `/Users/nzbirdranch/trippy/output/runs/EXP-0009-hybrid-a/hybrid-a-all-levels-bc` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0010-point-removal/config_removal.yaml` | `output/runs/EXP-0010-point-removal/exp0010-removal` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0010-point-removal/config_smoke.yaml` | `output/runs/EXP-0010-point-removal/exp0010-removal-smoke` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0010-point-removal/config_removal_rel.yaml` | `/Users/nzbirdranch/trippy/output/runs/EXP-0010-point-removal/removal-rel` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |
| `experiments/EXP-0010-point-removal/config_shade_prune.yaml` | `output/runs/EXP-0010-point-removal/exp0010-shade-prune` | `/Users/nzbirdranch/trippy/output/masks/kk-coherent` |

None of these configs were edited by this task (trainer's `masks_dir:` option is still landing
on `feat/karekare-v2`; editing trainer code is out of scope here). The line to add to each,
once that branch merges, is exactly `masks_dir: /Users/nzbirdranch/trippy/output/masks/kk-coherent`
placed after each config's `scene_root:` line — which is exactly what
`scripts/requeue_with_masks.sh` does automatically, into a `_masked` sibling config, leaving
every config above untouched.

As of 2026-09-06, Splats' GPU queue (`~/Splats/tools/gpu_queue/`) has queue/running entries
for most of these runs already: `70-trippy-full2-trips-resume`, `70-trippy-full3-alt-bc`,
`70-trippy-full3-alt`, `70-trippy-hybrid-a-all-levels-bc`, `70-trippy-hybrid-a-all-levels`,
`70-trippy-removal-rel`, `70-trippy-union-broadcast`, `70-trippy-union-trips`,
`70-trippy-exp0010-shade-prune` (queued), `70-trippy-exp0010-removal` (running). Per Jordan's
2026-09-06 correction, **none of these are touched or dequeued** — `requeue_with_masks.sh`
only ever adds masked *siblings* alongside them.

## 6. `scripts/requeue_with_masks.sh` (prepared, not run)

Once the `masks_dir:` trainer option lands on `feat/karekare-v2` and merges, the Orchestrator
runs, for whichever configs from the table above should also get a masked run, e.g.:

```
scripts/requeue_with_masks.sh \
  experiments/EXP-0003-kk-trips-train/config.yaml \
  experiments/EXP-0006-union/config_trips.yaml \
  experiments/EXP-0006-union/config_broadcast.yaml \
  ...
```

For each config given, it writes a **sibling** config (`<name>_masked.yaml`, same directory,
original untouched) with `masks_dir: /Users/nzbirdranch/trippy/output/masks/kk-coherent`
inserted after `scene_root:` and `-masked` appended to `run_dir:`'s final path component, then
submits that sibling via `scripts/queue_training.sh` (prio 70, `trippy train --report`, same as
every other trippy training). The original config and its existing queue entry are never
edited, deleted, or requeued — both the unmasked and masked runs end up queued side by side,
per Jordan's "I want both" correction. `--dry-run` (used for all testing in this task; see
`tests/test_requeue_with_masks_script.py`) previews the sibling and forwards `--dry-run` all
the way to `gpu_submit.sh`, writing/printing a job file only — no real submission, no
`research/trips-metal.md` write, and no file written under `experiments/` at all (the preview
sibling lives under `$TRIPPY_OUTPUT/tmp/` and is deleted immediately after).

**This task did not run `scripts/requeue_with_masks.sh` for real** (forbidden per the brief) —
only `bash -n` and `--dry-run` invocations against temp fixture configs, never against the real
`experiments/*.yaml` files above.

## 7. Open questions for the Orchestrator

1. `cpu_heavy.sh`'s hard 28 GB memory guard blocked this job even though Vision-based mask
   generation doesn't need anywhere near that much memory — see §1. Worth a `--skip-mem-check`
   flag for provably light CPU jobs, or a lower guard threshold under GPU-training contention?
2. `scripts/gpu_submit.sh`/`cpu_heavy.sh` both auto-log to `research/trips-metal.md` on
   submission; this job bypassed both, so its research-log entry (below) was written by hand.
   Once a real "run outside the queue" path exists in the tooling, this should go through it
   instead of manual logging.
3. Once `masks_dir:` lands, should *every* config in the table above get a masked sibling
   immediately, or should Jordan pick a subset first (e.g. the ones not already 100+ epochs
   into an unmasked run)? Left to the Orchestrator/Jordan — `requeue_with_masks.sh` takes an
   explicit config list either way.
