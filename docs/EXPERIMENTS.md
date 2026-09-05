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

## Exporting TRIPS point sets as PLY

`trippy.train.export` (`write_gaussian_ply` / `export_pointset_ply`) writes
any `PointSet` (or raw `xyz`/`rgb`/`conf`/`size` arrays) as a
3DGS-compatible binary PLY, so Splats' existing audit tools (extent gate,
`ply_extract.py`, `depthprior_shade_audit.py`) and the Brush viewer can
open a TRIPS point set unchanged -- exactly the mapping documented in
`docs/GEOMETRY.md` "3DGS PLY export mapping":

```
f_dc_{0,1,2}  = (rgb - 0.5) / SH_C0
opacity       = logit(clamp(conf, 1e-4, 1 - 1e-4))
scale_{0,1,2} = log(size)                    (isotropic)
rot_{0,1,2,3} = (1, 0, 0, 0)                 (wxyz identity; TRIPS has no rotation)
nx, ny, nz    = 0
```

The writer is the exact mathematical inverse of `GaussianPlySource`'s read
side (`trippy/points/gaussian_ply.py`): a 200-point synthetic round trip
(`write_gaussian_ply` -> `GaussianPlySource`) reproduces `xyz`/`rgb0`/
`conf0`/`size0` to within float32 precision, and the point count matches
exactly (`tests/test_export_ply.py`). Higher-order SH is zero (`sh_degree`
defaults to 0, no `f_rest_*` properties); passing `sh_degree=3` also
writes 45 zero-filled `f_rest_0..44` properties for viewers that expect a
full SH basis. A per-point `provenance` array, if supplied, is written
alongside the `.ply` as a `<path>.provenance.npy` sidecar
(`write_provenance_sidecar`) for post hoc per-source diagnostics -- not
read by any 3DGS tool.

Verified against Splats' own (unmodified) `extent_gate.py` on a synthetic
200-point PLY:

```
$ ~/Splats/tools/ml-sharp/.venv/bin/python \
    ~/Splats/tools/tmp/extent-audit/extent_gate.py synthetic.ply

synthetic.ply  (200 gaussians)
  median centre           [ 0.1767 -0.3056  0.2197]
  radius p50/p99/p99.9/max  4.85 / 7.30 / 7.70 / 7.77
  scene diagonal (min/max box)  17.11
  non-finite means         0
  non-finite scales        0
```
exit code 0 -- the gate accepts the synthetic PLY with no changes to
Splats' code. `tests/test_export_ply.py::test_splats_extent_gate_accepts_synthetic_ply`
runs this same check via subprocess and skips cleanly on a machine
without the Splats `ml-sharp` venv.

## `trippy render` output layout

`trippy render` (`trippy/render/pyramid_render.py`, `render_frames`) is the
CLI entry point for the no-U-Net forward pass in this document's "Mandatory
honesty sheet" spirit, applied per pyramid level. Given `--out <dir>` and
`--frames a.jpg,b.jpg`, it writes:

```
<dir>/
├── a/
│   ├── photo.png        (undistorted source image)
│   ├── level_0.png .. level_{L-1}.png   (native per-level resolution)
│   ├── coverage.png     (1 - T_final at level 0, colorized)
│   ├── depth.png        (expected depth at level 0, colorized; uncovered
│   │                      pixels are exactly black, never a fabricated value)
│   └── sheet.png         (photo | L0 .. L{L-1} | coverage | depth, one row;
│                           levels are nearest-upsampled to level-0 size only
│                           for this sheet, so coarse blockiness stays visible)
├── b/ (same layout)
├── summary_sheet.png     (all frames, one row each: photo | L0 | coverage)
├── metrics.json          (per frame: image_hw, timing_ms {emit, sort, blend,
│                           total}, num_fragments, points_visible)
└── README.md             (the command that produced the run + the timing table)
```

`points_visible` is the count of points that survived the conservative
view-frustum cull (`trippy.raster.cull_points`) -- candidates handed to
fragment emission, not the (more expensive) count of points that actually
survived the per-pixel fragment cap/transmittance cutoff inside compositing.
## Monocular depth points

`trippy.points.monodepth.MonoDepthSource` is D4 point source 2: per-image
Apple DepthPro metric depth (via Splats' `tools/ldi/depth_batch.py`, run
only through `scripts/gpu_submit.sh`) -> median-ratio scale alignment
against reprojected COLMAP sparse depth -> unprojection to a world-frame
`PointSet`, voxel-deduped, `provenance=MONODEPTH`. `trippy.points.depth_io`
builds depth_batch.py's manifest and parses its `<id>_depth.npy` /
`_mask.npy` / `_meta.json` outputs; it also reimplements
`trippy.scene.dataset.SceneDataset`'s undistortion+cache step for an
arbitrary curated image subset (SceneDataset's own constructor only
supports a `limit`-first-N-sorted-images slice, which would force
undistorting most of a scene just to reach frames near the end of it).

Resolution/EXIF choice: DepthPro is always run on the same undistorted
pinhole image `SceneDataset`'s cache would produce (never the original
distorted capture), so DepthPro's pixel grid and the scale-alignment /
unprojection math share one `K` -- no separate "undistort these keypoints"
step is needed. EXIF orientation is left untouched, matching both
`SceneDataset` and `depth_batch.py`'s own documented convention for these
photo folders.

```bash
# Print the exact GPU job to run (writes the manifest + undistorted PNG
# inputs as a side effect; exits 3 while depth outputs are missing):
trippy depth-points --scene ~/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,IMG_3829.jpg,... --width 1008 \
  --depth-dir output/depth/kk-coherent --run-depth

# Then, after the printed scripts/gpu_submit.sh command has completed:
trippy depth-points --scene ~/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,IMG_3829.jpg,... --width 1008 \
  --depth-dir output/depth/kk-coherent --out output/points/kk-coherent-monodepth-12.npz
```

One-shot numbers on kk-coherent, 12 frames (6 shade + 6 spread across the
219-image sequence), `width=1008`, `stride=6`, `voxel=0.03` -- see
`research/trips-metal.md`'s 2026-09-05 13:20 entry and
`experiments/EXP-0004-monodepth-points/README.md` for the full breakdown:
234,712 points after dedupe (from 254,016 raw), median nn-distance 0.166.
Shade frames (IMG_3828-3833) average 1,914 usable sparse-COLMAP matches for
scale alignment vs 4,373 for the spread frames -- fewer keypoints in the
darker region, as expected -- but a *lower* mean MAD (0.188 vs 0.249),
i.e. the few matches shade frames get agree well with each other. An
8px-radius point-presence coverage check (projecting the full 12-frame
union into each shade camera) comes back ~100% for every shade frame both
over the whole image and a central 50% box -- this mostly reflects
DepthPro's `valid_fraction=1.0` and stride-6 density (a point lands near
almost every pixel by construction) rather than confirming the depth
values themselves are *correct* there; that needs the shade audit /
Jordan's viewer verdict once this source feeds a training run.

`trippy.render.sheets` (`contact_sheet`, `side_by_side`, `colorize`,
`save_png`) and `trippy.render.video` (`write_video`, `frames_from_dir`)
give every export an inspectable artifact: a labelled contact sheet
(PIL only, no matplotlib) and an ffmpeg-piped MP4 (`h264_videotoolbox`
hardware encoder when ffmpeg reports it available, `libx264` otherwise;
raises a clear `RuntimeError` if ffmpeg isn't on `PATH` rather than
failing silently). `colorize` maps depth/coverage scalars to RGB with a
hand-picked 5-stop viridis-like ramp implemented in numpy, so
`docs/SPEC.md`'s honesty sheet (raw composite | network output |
coverage/provenance map) needs no new plotting dependency.
