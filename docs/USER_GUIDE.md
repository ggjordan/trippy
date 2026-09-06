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

## How to read the leaderboard

`~/Splats/output/Jordan-Review/4-other/trips-leaderboard.png` (also `.md` next to it under
`$TRIPPY_OUTPUT/leaderboard/`) is one table comparing every TRIPS training run that has finished
a self-report against the plain-Gaussian baseline and Design C. It regenerates and re-delivers
itself automatically at the end of every `trippy train --report` run, so it's always the
newest run's own README plus every earlier run in one place -- open it first before digging into
an individual run's own README/dolly video.

Rows are sorted by shade dark-mass fraction ascending (lower/closer to the Gaussian baseline's
19.9% first), then held-out PSNR descending -- the rows nearest the top are the closest to "no
worse than plain Gaussians in the shade, and as sharp everywhere else". A run named with
`(smoke)` was a queue-rehearsal run (a few epochs, proving the pipeline works), not a real result
-- read it for "did the run complete", not for its numbers.

The "held-out shade" PSNR/SSIM/LPIPS column now fills in for real trippy runs too: the trainer
records a separate shade-vs-other breakdown every time it evaluates (mid-training, `--report`, or a
standalone re-eval). A row still reads `n/a` when the run finished training before this existed and
hasn't been re-evaluated since -- ask for `trippy eval --checkpoint <run>/checkpoints/checkpoint_latest.pt`
to be run against it (no retraining needed) and the next leaderboard rebuild will pick up the number.

**"Held-out all"/"Held-out shade" now headline the neighbour-exposure number, not the strict one.**
A held-out photo's own exposure/white-balance is never trained, so it can start miles off (the
kk-coherent no-EXIF frames were 58x too bright) and that alone used to cost some rows many dB that
had nothing to do with how good the reconstruction actually is. These two columns now copy each
held-out frame's exposure from its nearest TRAINING neighbours instead (never reading that frame's
own photo) before scoring it, which is the fix and the new headline number. The strict, unmodified
number -- the one the leaderboard always showed -- is still right there: the compact "Strict
own-exposure PSNR (all/shade)" column, and it's what `trippy eval`'s own printout calls "own". A run
whose report predates this fix shows `" (own)"` on the two headline columns instead of a number that
doesn't exist yet -- that just means there was nothing to switch to yet, not that anything is wrong
with the run. Raw Gaussians and Design C (the two fixed baseline rows) always show `" (own)"` here:
neither has a per-image exposure model this fix could do anything to.

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

It opens **on a real camera of the capture**, looking at what that camera saw, and it
starts in **orbit** mode: left-drag turns you around the scene rather than spinning you
on the spot, and you cannot leave the area the real cameras covered. That is the mode
for judging a scene. Press `F` when you want to fly through it instead.

Controls:

| | |
|---|---|
| **left-drag** | **orbit** the scene — or look around, in free mode |
| **right-drag** or **middle-drag** | pan sideways / up and down |
| **scroll** | orbit: move closer or further · free: change how fast you fly |
| `W` `A` `S` `D` | move forward / left / back / right |
| `Q` / `E` | move down / up |
| `R` | back to the view it opened at (use this whenever you are lost) |
| `N` / `P` | jump to the next / previous real camera of the capture |
| `F` | switch between **orbit** and **free fly** |
| `V` | cycle the view: **network** -> **raw level-0** -> **coverage** |
| `-` / `=` | render at a smaller / larger fraction of the window |
| `TAB` | hide the panel |

**Speed is set from the scene, not guessed.** The viewer measures how far apart the real
cameras are and flies at half that distance per second, so one tap of `W` is a step, not
a teleport, whether the scene is a horse on a plinth or a whole beach. The panel shows
both numbers: world units per second, and the same speed as a fraction of the captured
area per second. Scroll changes it by 25% a notch, between 1/100 and 10 times the
default.

If you do fly out of the captured area in free mode, the panel says so and tells you to
press `R`. In orbit mode you cannot: the point you are turning around is pinned inside
the box the real cameras occupy, and the camera moves with it.

The "jump to view" dropdown lists every real camera in the capture; picking one puts you
exactly where that photograph was taken, which is the fair place to compare the render
with the photograph.

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

## How to open the TRIPS viewer in a web browser (Mac, v0.5.0)

The same scene, the same renderer, in a browser tab instead of an app window.
**Double-click** `OPEN_TRIPS_WEB_<name>.command`. A tiny web server starts on
your own machine (127.0.0.1 — nothing leaves this Mac) and the page opens.

**Use Chrome.** Safari will open the page and draw something, but the picture is
wrong — one of the renderer's shaders will not compile in Safari, and what you
get is stripe noise rather than the scene. The page prints the error on screen
in red and says the image is not trustworthy, so you will know; but the answer
is to use Chrome.

The controls are the native viewer's — `W A S D` to move, `Q`/`E` down and up,
drag to look, scroll for speed, `V` to cycle views, `-`/`=` for render scale,
`R` to jump back to the scene's own camera. The frame rate is in the top-left.

**What you will and will not see.** The browser build renders the **rasteriser**:
`raw level-0` (the photographed-ish evidence) and `coverage` (bright where the
rasteriser drew, dark where the network would have invented the pixels). The
**network view is not available in the browser** — a bug in a library trippy
depends on makes it impossible to get the network's frame onto the screen there,
and the page says so on-screen instead of pretending. For the finished,
network-decoded image, use the native Mac viewer
(`OPEN_TRIPS_MAC_<name>.command`); it is also about fourteen times faster.

Expect about 3 frames per second on the horse scene (2.9 fps measured, while a
training was running on the same GPU). That is slow, and honest: the browser
re-uploads the whole two-million-point cloud on every frame.

## How to make a scene openable in the native viewer (`export-bundle`)

`trippy export-bundle` packages a trained scene into one folder the native (Rust) viewer can open:

```
trippy export-bundle --checkpoint <checkpoint> --out <folder> [--name NAME]
```

The folder holds exactly three files — `bundle.json` (the cameras and render settings), `points.npz` (the point cloud) and `weights.safetensors` (the network). The points are stored in **world space** with every real camera of the scene listed alongside them, so the viewer can fly anywhere rather than replay one fixed frame; it opens on the scene's reference view. It accepts either a published TRIPS checkpoint (add `--scene <scene folder>`) or one of trippy's own training checkpoints, and prints which kind it found. It runs on the CPU, so it does not need to queue for the GPU.

**Every self-reporting training run does this for you automatically.** `trippy train --report`
exports this same bundle from the run's own final checkpoint and delivers a
`OPEN_TRIPS_MAC_<run>.command` launcher — it's the *first* thing listed for a finished run, ahead of
the dolly video, because a fixed dolly path is hard to judge; the bundle lets you fly through the
scene yourself instead. If you have a checkpoint from *before* this existed (or just want a fresh
launcher without re-training), run:

```
trippy bundle-launcher --checkpoint <checkpoint> --name <name>
```

That's the same export + launcher + delivery, for any checkpoint on demand.

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
