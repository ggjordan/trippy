# Research index

This is the experiment and milestone tracking index for the trippy project. Each row corresponds to one experiment or major milestone. Update this table as experiments run and gates are evaluated.

| Experiment | Stage | Status | Verdict | Artifact |
|---|---|---|---|---|
| EXP-0001-forward-pyramid | v0.1.0 | done | PASS on speed (135.7 ms worst-case/frame, trilinear, 1008 wide); shade-region coverage measurably lower than non-shade (numeric, from T_final: ~0.17-0.23 vs ~0.21-0.34) -- holes are honest, not a bug, and are the U-Net's job to fill | `output/runs/EXP-0001/{trilinear,broadcast}/summary_sheet.png` + `metrics.json` (delivered via `scripts/deliver.sh`) |
| EXP-0001-forward-pyramid | v0.1.0 | planned | — | contact sheet: photo / ref / Metal / 5 pyramid levels |
| EXP-0004-monodepth-points | v0.2.0 | done | inconclusive (coverage metric saturates) / signal found (scale-alignment) | `output/points/kk-coherent-monodepth-12.npz`; sheet delivered as `EXP-0004-monodepth-shade-coverage` |
| EXP-0002-horse-parity | v0.1.0 | done 2026-09-06 | **PASS** — 22.27 dB vs the authors' own 22.34 dB (0.07 dB gap); 37.0 dB against their render | `output/EXP-0002-horse-parity/summary_sheet.png`, delivered as `Jordan-Review/4-other/EXP-0002-horse-parity.png` |

## Parked ideas (never culled for effort)

Ideas that are out of scope for the current milestones but valuable for future work. These are not forgotten; they remain in the backlog unless explicitly closed with a note.

- **A2: EWA footprints per level**: Extend trilinear splatting with elliptical weighted average (EWA) footprints. Improves sharpness on high-curvature surfaces.
- **Low-coverage point spawning**: Detect regions with <0.3 coverage (inferred vs. photographed) and spawn new points from monocular depth to fill holes. Reduces U-Net hallucination.
- **Distortion refinement**: COLMAP distortion parameters are applied once during dataset loading. Refining them during training (joint pose + distortion optimization) could improve geometry.
- **Training the U-Net inside Brush (Burn backward)**: v0.4.0 ports the U-Net to Burn (inference only). A future milestone could add full backward passes to Burn, enabling training directly in the Brush viewer app (called "Burn backward" in the backlog).

## Running log

See `trips-metal.md` for the chronological experiment log (appended as you work, never rewritten).
