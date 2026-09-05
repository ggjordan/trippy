# trippy

TRIPS on Apple Silicon: Trilinear Point Splatting ported to Mac Studio M3 Ultra, trained on Jordan's outdoor scenes, hybridised with Gaussian splatting, with live viewers (Mac + web).

The shade under trees is rendered as shading, not as a dark cloud you walk through.

## For agents

Agents: read `AGENTS.md` first, then `STATE.md`.

Quick start (5 lines):
1. `scripts/bootstrap.sh` — installs hooks, configures `.env`, verifies `SPLATS_ROOT`, reports GPU queue status.
2. `scripts/build.sh` — compile check (Python + Rust when present). `scripts/test.sh` — unit tests (CPU only).
3. Work on a branch. `scripts/review.sh` shows the diff + review checklist. Commit with `Reviewed-by:` trailer.
4. `scripts/push.sh` tags `build-NNNN` and pushes.
5. Read `research/trips-metal.md` (last 40 lines) before each session; append to it as you go.

## For humans

**Status: early skeleton (v0.1.0 not yet reached).** See `STATE.md` for what exists today and `docs/SPEC.md` for the plan.

The plan: a PyTorch-on-MPS port of TRIPS with the trilinear rasteriser as inline Metal kernels
(`torch.mps.compile_shader`), trained on real outdoor scenes, then hybridised with Gaussian splats and
ported into a fork of [Brush](https://github.com/ArthurBrussee/brush) for a native Mac viewer and a WebGPU web viewer.
Nobody human edits this code: LLM agents run the repo under the rules in `AGENTS.md`.

### What will be here
- The Python package `trippy/` (geometry, point sources, Metal rasteriser, U-Net, trainer, renderers).
- CPU-only tests under `tests/` (under 60 s); MPS tests are marked `gpu` and run only inside GPU-queue jobs.
- Docs and decision records under `docs/`; the running experiment log in `research/`.
- Later: the Brush fork under `rust/`.

### What is NOT here, ever
- The owner's private scenes, photos, renders, splat files, or checkpoints. This repo is public; those stay on the owner's machine.
- Redistributed TRIPS data. The public scenes and checkpoints live on Zenodo (record 10687419, CC-BY 4.0);
  `tools/fetch_upstream.sh` clones the TRIPS code and prints the download commands.

### Licenses
- trippy: MIT, copyright 2026 Jordan.
- TRIPS model code: MIT (Franke et al. 2024). Zenodo TRIPS scenes: CC-BY.
- Brush fork (`rust/brush-trips`): Apache-2.0 with NOTICE, building on the original Brush splat renderer.

See `docs/UPSTREAM.md` for full attribution and commit hashes.
