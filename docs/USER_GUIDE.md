# User guide: for Jordan

This guide explains how to review trippy work without using the terminal.

## Where deliverables appear

All finished artifacts (viewer demos, videos, exported point clouds) appear in:

```
~/Splats/output/Jordan-Review/README.md
```

This file is the index. It has sections for each delivery:
- **2-open-in-brush**: Interactive 3D viewer on the Mac (double-click `OPEN_TRIPS_MAC.command`).
- **4-other**: MP4 videos, exported PLY files for Brush inspection, metrics reports.

Artifacts **never** appear in the `trippy` repo itself; they live in the `Splats` review folder where you already review other experiments.

## How to open a .command file

A `.command` file is a macOS double-click shortcut:

1. Find it in Finder (usually in `~/Splats/output/Jordan-Review/`).
2. **Double-click** the file.
3. It opens a local web server (127.0.0.1, no internet) in your browser.
4. Use your mouse and keyboard to navigate the viewer (standard 3D controls).
5. Close the browser tab when done; the server shuts down automatically.

**Security**: `.command` files only serve files from the local machine on loopback (127.0.0.1). No data leaves your Mac.

## How to open a .ply file in Brush

To inspect a raw point cloud (before U-Net refinement) in the Brush viewer:

1. Launch Brush: `~/Splats/tools/brush-final/target/release/brush`
2. Menu → **Open Scene**
3. Select the `.ply` file from `~/Splats/output/Jordan-Review/`
4. Use the viewer controls to inspect the points.

Brush will render the points as Gaussians (standard 3D splatting). This shows the raw output before any U-Net refinement and is useful for comparing point sources (Gaussians vs. monocular depth vs. union).

## How to open the native TRIPS viewer (Mac)

The native viewer renders a TRIPS scene **live** — the point cloud is rasterised into an
image pyramid and decoded by the U-Net on every frame, at whatever size the window is.
It is a separate app from Brush; Brush still opens `.ply` files exactly as before.

**Double-click** `OPEN_TRIPS_MAC_<name>.command` in your review folder. That's it.

Controls:

| | |
|---|---|
| `W` `A` `S` `D` | move forward / left / back / right |
| `Q` / `E` | move down / up |
| drag the mouse | look around |
| scroll | change how fast you fly |
| `V` | cycle the view: **network** -> **raw level-0** -> **coverage** |
| `-` / `=` | render at a smaller / larger fraction of the window |
| `TAB` | hide the panel |

The three views are the honesty sheet, live:

- **network** — the finished frame, what the model wants you to see.
- **raw level-0** — the rasteriser's own finest layer, before the network touched it.
  Sparse and speckled is normal; this is the evidence.
- **coverage** — bright where the rasteriser actually drew something, **dark where the
  network invented every pixel**. If a region you care about is dark, the model is
  making it up.

The top line of the panel is the frame time and frames per second. If it is too slow,
press `-` once or twice (rendering at 0.75 or 0.5 of the window and upscaling) — that is
usually enough, and it costs sharpness rather than correctness.

You will notice **raw level-0 and coverage are about ten times faster than network**.
That is not a trick: those two views stop before the neural network, which is where
almost the whole frame time goes. So if you want to fly around fast and find the spot
you want to judge, fly in `coverage`, then press `V` twice to look at it.

The panel's checkboxes are speed/accuracy trades. They all start off except the one the
launcher turned on. "exact pipeline" in the panel means nothing has been traded away.

To open a different scene, run the viewer with a bundle folder:

```
rust/target/release/trips-viewer <bundle folder>
```

or with no argument at all, and it opens a folder picker.

## How to make a scene openable in the native viewer (`export-bundle`)

`trippy export-bundle` packages a trained scene into one folder the native (Rust) viewer can open:

```
trippy export-bundle --checkpoint <checkpoint> --out <folder> [--name NAME]
```

The folder holds exactly three files — `bundle.json` (the cameras and render settings), `points.npz` (the point cloud) and `weights.safetensors` (the network). The points are stored in **world space** with every real camera of the scene listed alongside them, so the viewer can fly anywhere rather than replay one fixed frame; it opens on the scene's reference view. It accepts either a published TRIPS checkpoint (add `--scene <scene folder>`) or one of trippy's own training checkpoints, and prints which kind it found. It runs on the CPU, so it does not need to queue for the GPU.

## How to change GPU priority

If a job is running too slowly (bogging down other work) or too fast (starving training for GPU time), you can re-prioritise it by asking:

> "Re-prioritise the current trippy job to priority X."

The agent will adjust the queue position. Priority ranges:
- **10–19**: short jobs (smoke tests, small renders).
- **60**: Splats trainings (current baseline).
- **70**: trippy trainings (lower priority, longer wall-clock time expected).

## How to ask for a release

When a milestone is ready to ship (e.g., v0.1.0 complete), ask:

> "Release trippy v0.1.0."

The agent will:
1. Run `scripts/test.sh` to verify all tests pass.
2. Tag the repo with `v0.1.0`.
3. Create a GitHub Release with release notes.
4. Deliver the acceptance artifact (e.g., contact sheet) to the review folder.

You don't need to do anything; just ask and wait.

## Privacy and what's not in the repo

This repo is public on GitHub. It contains:
- Training code, tests, documentation.
- Synthetic test fixtures.
- References to public Zenodo data (not the data itself).

It does **not** contain:
- Photos from Karekare, Hunua, or any family scene.
- Trained model checkpoints.
- Rendered images or videos from your scenes.
- Point clouds (`.ply` files).

All your deliverables live in `~/Splats/output/Jordan-Review/`, which is private and never committed to the public repo.
