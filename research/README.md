# Research index

This is the experiment and milestone tracking index for the trippy project. Each row corresponds to one experiment or major milestone. Update this table as experiments run and gates are evaluated.

| Experiment | Stage | Status | Verdict | Artifact |
|---|---|---|---|---|
| EXP-0001-forward-pyramid | v0.1.0 | done | PASS on speed (135.7 ms worst-case/frame, trilinear, 1008 wide); shade-region coverage measurably lower than non-shade (numeric, from T_final: ~0.17-0.23 vs ~0.21-0.34) -- holes are honest, not a bug, and are the U-Net's job to fill | `output/runs/EXP-0001/{trilinear,broadcast}/summary_sheet.png` + `metrics.json` (delivered via `scripts/deliver.sh`) |
| EXP-0001-forward-pyramid | v0.1.0 | planned | — | contact sheet: photo / ref / Metal / 5 pyramid levels |
| EXP-0004-monodepth-points | v0.2.0 | done | inconclusive (coverage metric saturates) / signal found (scale-alignment) | `output/points/kk-coherent-monodepth-12.npz`; sheet delivered as `EXP-0004-monodepth-shade-coverage` |
| EXP-0002-horse-parity | v0.1.0 | done 2026-09-06 | **PASS** — 22.27 dB vs the authors' own 22.34 dB (0.07 dB gap); 37.0 dB against their render | `output/EXP-0002-horse-parity/summary_sheet.png`, delivered as `Jordan-Review/4-other/EXP-0002-horse-parity.png` |
| EXP-0005-hybrid-c | v0.3.0 | done | learned renderer improves non-shade PSNR/SSIM/LPIPS (+0.45 dB / +0.051 / -0.018) but *worsens* shade PSNR by ~2 dB (14.94 -> 12.97 dB) despite small SSIM/LPIPS gains there; aggregate PSNR flat | `output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/`; sheet delivered as `EXP-0005-hybrid-c-refine` |
| EXP-0006-union | v0.2.0 | points built, training queued | DepthPro rc=0 on all 219 images (293.6s); MonoDepth-219 3,786,345 pts (median nn 0.2806); Union(Gaussian, MonoDepth-219, voxel=0.03) 5,887,647 pts (median nn 0.2984; 38.2% of raw points collapsed by the voxel dedupe, mostly Gaussian-vs-Gaussian); `train-union-broadcast`/`train-union-trips` submitted prio 70, 300 epochs, train_factor 1.0, --max-minutes 330 | `output/points/kk-coherent-{monodepth-219,union-full}.npz`; `experiments/EXP-0006-union/README.md`; training verdict pending in `output/runs/EXP-0006-union/{broadcast,trips}/` |
| EXP-0009-hybrid-a | v0.3.0 | built, smoke rc=0, run queued | Design A ("Splats combined with TRIPS", Jordan 2026-09-06): the Gaussian render (rgb+alpha+normalised depth) concatenated onto every level of the TRIPS pyramid inside the existing point-based `Trainer`, trained end to end; ablations `dropout_gaussian_p=0.2` and `mask_by_alpha=true`; live gsrender for dolly/off-path poses. Must beat BOTH plain Gaussians (15.53 dB all / 14.94 dB shade, EXP-0005) and plain TRIPS (14.42 dB @40 ep, EXP-0003 -- compare against `full2-trips` at equal epochs when it lands). Smoke `trippy-hybrid-a-smoke` rc=0 (18.5 min, 2 ep @ w504/200k pts): held-out 7.40 -> 8.88 dB, 219/219 renders found, 48 dolly + 12 off-path frames rendered by LIVE gsrender on MPS. Verdict pending on `trippy-hybrid-a-all-levels` | `experiments/EXP-0009-hybrid-a/README.md`; `output/runs/EXP-0009-hybrid-a/hybrid-a-all-levels/` |
| EXP-0008-distill | design B (docs/SPEC.md D2) | pipeline built, render+brush-train queued | Pipeline proof against the weak `EXP-0003-kk-trips-train/full1-broadcast` checkpoint (40 epochs, 14.42 dB): `trippy distill` (`trippy/distill/{cameras,colmap_writer,render_set,brush_runner,compare}.py`) built and CPU-tested; Brush `brush-cli` release binary built (`scripts/cpu_heavy.sh`, rc=0, 2m44s); render stage submitted prio 15 (`distill-render-full1-broadcast`); Brush training (prio 70, 6000 iters) to follow, behind the existing prio-70 queue | `experiments/EXP-0008-distill/README.md`; `output/runs/EXP-0008-distill/full1-broadcast/` once the render job returns |

## Parked ideas (never culled for effort)

Ideas that are out of scope for the current milestones but valuable for future work. These are not forgotten; they remain in the backlog unless explicitly closed with a note.

- **Design A variants not run yet**: `mode: concat_level0` (Gaussian block on level 0 only) is implemented and configurable but unrun -- `all_levels` was chosen as the default on the CombineBridge argument, not on measurement. Also unrun: `dropout_gaussian_p` sweeps, `mask_by_alpha: false`, and channel subsets (`[rgb, alpha]`, `[alpha, depth]` -- the latter would test whether the *geometry* of the Gaussians helps without their colour).
- **A2: EWA footprints per level**: Extend trilinear splatting with elliptical weighted average (EWA) footprints. Improves sharpness on high-curvature surfaces.
- **Low-coverage point spawning**: Detect regions with <0.3 coverage (inferred vs. photographed) and spawn new points from monocular depth to fill holes. Reduces U-Net hallucination.
- **Distortion refinement**: COLMAP distortion parameters are applied once during dataset loading. Refining them during training (joint pose + distortion optimization) could improve geometry.
- **Training the U-Net inside Brush (Burn backward)**: v0.4.0 ports the U-Net to Burn (inference only). A future milestone could add full backward passes to Burn, enabling training directly in the Brush viewer app (called "Burn backward" in the backlog).

## Running log

See `trips-metal.md` for the chronological experiment log (appended as you work, never rewritten).
