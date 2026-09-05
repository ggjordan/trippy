# Experiments: structure, metrics, and verdicts

## Experiment folder layout

Each experiment lives under `experiments/EXP-NNNN-<slug>/`:

```
experiments/
└── EXP-0001-forward-pyramid/
    ├── README.md          (question, point source, config, job name(s), gate, verdict)
    ├── config.toml        (training parameters, optional, cited in README)
    └── (no artifacts here; see "Run location" below)
```

The README contains:
- **Question**: What does this experiment answer? E.g., "Does our Metal forward equal the numpy reference and render 3 kk-coherent frames at 1008 wide in <100 ms?"
- **Point source**: Which source is being tested? (1 = Gaussians, 2 = monocular depth, 3 = union, etc.)
- **Config**: Reference to config.toml or inline parameters.
- **Planned commands**: shell commands to run the experiment (placeholder until executed).
- **Gate**: What passes this experiment? E.g., "v0.1.0 acceptance".
- **Verdict**: Outcome after run (filled in post-run).

## Naming convention

- **EXP-NNNN**: zero-padded 4-digit experiment counter.
- **slug**: hyphenated lowercase description, ≤40 chars. E.g., `forward-pyramid`, `gaussian-density-on-hunua`.

## Run location

Experiments do not write artifacts to `experiments/EXP-NNNN-*/`. Instead:

```
output/
├── runs/
│   └── <exp>/
│       ├── EXP-0001-forward-pyramid_1/
│       │   ├── config.yaml       (actual run parameters)
│       │   ├── log.txt           (stdout/stderr)
│       │   ├── checkpoints/
│       │   ├── export.ply        (final model as 3DGS PLY)
│       │   ├── shade_audit.txt   (metrics)
│       │   └── honesty_sheet.png (raw | network | coverage/provenance)
│       └── EXP-0001-forward-pyramid_2/   (if re-run)
├── Training-Data/
│   └── karekare/kk-coherent/
│       └── candidates/
│           ├── EXP-0001-gaussian-source/
│           ├── EXP-0001-depth-source/
│           └── EXP-0001-union-source/
└── deliver/
    └── (artifacts for Jordan review, via scripts/deliver.sh)
```

Run directories are **gitignored**. Only the `experiments/` README lives in version control.

## Ranking candidates: metrics and gates

### Shade audit

```bash
python ~/Splats/tools/depthprior_shade_audit.py \
  --scene ~/Splats/scenes/karekare/kk-coherent/sparse_txt \
  output/Training-Data/karekare/kk-coherent/candidates/EXP-NNNN-<source>/export.ply
```

Output: opacity mass in shade region (lower is better; 0 = shade is transparent).

### Held-out PSNR and LPIPS

After exporting the model to PLY, render with:

```bash
python ~/Splats/tools/gsrender.py <export.ply> --outdir output/renders
```

Use a modulo-8 split (hold out every 8th training view as test). Report:
- **PSNR** on non-shade frames (frames outside the shade region).
- **LPIPS** on the same split.

Target: v0.2.0 acceptance requires PSNR within 1.5 dB of the best plain Gaussian on non-shade frames.

### Extent gate

Prevents scene sprawl. Compute the bounding box of the trained point set:

```bash
python ~/Splats/tools/tmp/extent-audit/extent_gate.py output/Training-Data/karekare/kk-coherent/candidates/EXP-NNNN-<source>/export.ply
```

Output: radius p99, p99.9, max. These should not exceed the extent of the original sparse COLMAP points by >20%. Scene sprawl makes rendering unusable.

## Point sources

`trippy/points/` (D4) turns a scene into a `PointSet` (xyz, size0, rgb0, conf0, provenance):

- `GaussianPlySource`: trained 3DGS Gaussian centres from a binary PLY. `min_opacity` filters by `sigmoid(opacity)`; `size_mode="scale"` uses the trained `exp(log_scale)` extent, `size_mode="knn"` ignores it and uses local point spacing instead. Reads the ~7M-row author PLYs in ~1-2 s (structured-dtype `np.fromfile`, no plyfile, no per-row loop).
- `ColmapSparseSource`: the sparse triangulated points from `points3D.txt`, fixed `conf0=0.5`, size from kNN spacing.
- `UnionSource`: concatenates any sources; with `voxel` set, dedupes colliding points keeping the highest-`conf0` survivor per cell.
- `MonoDepthSource` / `LidarSource`: not implemented yet (v0.2.0 and "later" respectively); constructors document planned inputs.

Inspect any source without training via the CLI:

```bash
trippy density --source gaussian --path <ply> --min-opacity 0.05 --size-mode scale --max-points 200000
trippy density --source colmap --path <sparse_txt_dir>
```

`density` prints `PointSet.summary()` (count, bbox, median nearest-neighbour distance on a subsample, provenance histogram) as both a human table and a `JSON:`-prefixed line, and can also write it to `--out summary.json`. `size_mode="knn"` on the full multi-million-point Gaussian PLY is expensive by design (kNN over the whole cloud) -- pass `--max-points` first when exploring interactively.

One-shot numbers on `kkc_15000.ply` (7.36M Gaussians) and its `sparse_txt` COLMAP model, `min_opacity=0.05`, `size_mode=scale`:

| Source | Count (post-filter) | bbox (world units) | median nn-distance |
|---|---|---|---|
| gaussian (full) | 5,736,619 | [-79.3,-85.6,-61.4] to [83.1,68.3,94.1] | 0.0795 |
| gaussian (`--max-points 200000`) | 200,000 | [-32.5,-83.8,-55.5] to [81.2,24.1,74.0] | 0.0803 |
| colmap | 153,515 | [-130.6,-117.2,-207.3] to [213.5,168.7,188.1] | 0.0865 |

The COLMAP sparse cloud's bbox is much larger than the Gaussians' -- expected, since sparse triangulated points include noisy far-field/sky points that training prunes away.

## Mandatory honesty sheet

Every candidate must include a three-panel image:

1. **Raw composite** (level-0 blend_fwd output, no U-Net).
2. **Network output** (after U-Net tone mapping).
3. **Coverage/provenance map**: colourised by coverage count and point source (Gaussians = red, depth = blue, union = purple). Pixels with coverage <0.3 are outlined in white.

The honesty sheet makes clear which image regions are photographed (covered by point sources) and which are inferred (U-Net hallucination). Jordan reviews these alongside the dolly video.

## Dolly camera paths

The dolly renders a fixed path through the shade region:

```bash
python ~/Splats/tools/depthprior_shade_dolly.py <export.ply> --outdir output/renders
```

**Pose source**: use the same pose as `IMG_3830.jpg` (the centre of the shade region), then translate the camera along the `+X` axis (horizontal, perpendicular to viewing direction) from `-0.35` m to `+1.20` m. This walks the observer through the shade volume and shows whether TRIPS renders it as a lighting effect or as a cloud of points.

Output: MP4 video (typically 2–5 seconds at 24 fps).

## Jordan's viewer verdict is final

All metrics are rankings. **Jordan's visual assessment in the viewer overrides any metric.** If PSNR is high but the shade looks wrong, the metric is wrong. If LPIPS is high but the scene looks good, fine. The only verdict that matters: can you step into the scene and see shading, not a cloud?

## Experiment tracking: research/trips-metal.md

Each completed experiment adds an entry to the running log `research/trips-metal.md`. Entries are appended chronologically (never rewritten) and record:

- **Date** (YYYY-MM-DD HH:MM).
- **Question**: What was tested?
- **Job name**: reference to `output/jobs/trippy-<name>.sh`.
- **Numbers**: PSNR, shade audit, extent radius, FPS (if applicable).
- **Verdict**: Pass/fail/inconclusive.
- **Artifact path**: where to find the export PLY, video, honesty sheet.

Example:

```
## 2026-09-06 10:30 — EXP-0001 forward pass validation
Question: does our Metal forward equal the numpy reference?
Job: trippy-forward-check
Numbers: agreement <1e-5, 3 frames at 1008 wide rendered in 87 ms (11.5 fps)
Verdict: PASS
Artifact: output/runs/EXP-0001-forward-pyramid_1/
```

This log serves as the experiment audit trail.
