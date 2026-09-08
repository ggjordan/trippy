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

It opens **on a real camera of the capture**, looking at what that camera saw. A card
called **Controls** is up on the first frame; press **`?`** at any time to bring it back.

**The mouse works like Brush.** (Changed 2026-09-08, at Jordan's request. If you had
learned the old scheme: the right button now looks around instead of panning, and the
wheel now always moves you closer or further.)

| | |
|---|---|
| **left-drag** | **turn around** what you are looking at |
| **right-drag** | **look around** from where you are — and `W` `A` `S` `D` `Q` `E` fly while you hold it |
| **middle-drag**, or **shift + left-drag** | slide sideways and up/down |
| **scroll** | move **closer / further** |
| **shift + scroll** (or scroll while the right button is down) | fly **faster / slower** |
| **double-click** something | **look at THAT**: the point you turn around jumps onto what you clicked |

And the keys:

| | |
|---|---|
| `W` `A` `S` `D` | move forward / left / back / right |
| `Q` / `E` | move down / up |
| `R` | back to the view it opened at (use this whenever you are lost) |
| `?` or `F1` | the Controls card |
| `N` / `P` | jump to the next / previous real camera of the capture |
| `F` | fence on/off: **stay in the photographed area** or **fly anywhere** |
| `V` | cycle the view: **network** -> **raw level-0** -> **coverage** |
| `X` | cycle the **exposure**: auto -> this view's -> scene median -> manual |
| `-` / `=` | render at a smaller / larger fraction of the window |
| `M` | the editor |
| `TAB` | hide the panel |

**Too slow? Hold shift and scroll up.** Each notch is 25% faster and the wheel goes to
**50x** the default, which crosses a whole capture in under a second.

`F` is now only a **fence**, not a different set of controls. It starts on: the point
you turn around is pinned inside the box the real cameras occupy, so you cannot get
lost. Press `F` and you can fly out of it; the panel says so when you have, and `R`
brings you home.

**Speed is set from the scene, not guessed.** The viewer measures how far apart the real
cameras are and flies **two of those gaps per second**, so one second of held `W` takes
you past two capture positions whether the scene is a horse on a plinth or a whole
beach. The panel shows both numbers: world units per second, and the same speed as a
fraction of the captured area per second. Scroll changes it by 25% a notch, between
1/100 and **50** times the default.

(That default was 0.5 gaps per second up to 2026-09-06 — 4x slower. On Karekare the
photographs are ~0.2 world units apart, so the old default needed about two and a half
minutes of held `W` to cross the capture. It now takes about forty seconds, and one
scroll-up ladder takes it to under a second.)

The "jump to view" dropdown lists every real camera in the capture; picking one puts you
exactly where that photograph was taken, which is the fair place to compare the render
with the photograph.

**Why the brightness can change when you move.** Each photograph was taken at its own
exposure, and the model learned that per photograph — it is the only part of the colour
handling that differs between two views of one scene. A viewpoint that is *not* a
photograph has no exposure of its own, so the viewer has to choose one. The default,
`auto`, uses the exposure of the camera you are sitting on and the scene's **median**
once you move off it; the panel always says which EV it is applying. Press `X` to force
`view` (always the nearest capture camera's — right when you are comparing a render with
its photograph), `median` (one grade for the whole scene — right when you are flying
around judging it), or `manual` and set the EV yourself.

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

## The Blend panel: splat vs TRIPS, with a slider

Appears on any scene whose bundle names a Gaussian `.ply` — every **hybrid** run, and every
plain run seeded from a splat (which on Karekare is all of them: they all start from
`kklid_20000.ply`). On a scene built from COLMAP points or monodepth there is no splat to
mix in and the viewer looks exactly as it always has.

A hybrid run draws the scene from two things at once: your Gaussian splat, and TRIPS's point
cloud. Until now, how much of any pixel came from which was buried in the network's weights.
The Blend panel puts it on screen and puts a slider on it.

**Press `B`** to cycle the five modes, or click them in the panel:

| Mode | What you get |
|---|---|
| **TRIPS only** | the network's own frame, no splat mixed in. The default, and the frame every number in the run's report was measured on. |
| **splat only** | the Gaussian render at this pose, on its own. |
| **gated blend** | what the run itself chose, per pixel. |
| **manual mix** | you choose one blend for the whole frame. |
| **split screen** | splat on the left, TRIPS on the right, same pose, one slider to move the seam. |

Two sliders:

- **gate scale** (gated blend only), **0 to 2**. 0 is pure TRIPS, 1 is what training chose, 2
  pushes every pixel the network already leaned towards the splat all the way to the splat.
  This is the knob for "the network was too shy / too eager about the Gaussians".
- **mix** (manual mix only), **0 = splat, 1 = TRIPS**. This one ignores what the network
  learned and just crossfades.

### The splat is rendered live, wherever you fly

When the viewer opens a bundle it also opens the Gaussian `.ply` that bundle names
(`blend.splat_ply`) and keeps it on the graphics card. Every frame, Brush's own Gaussian
renderer draws it **from wherever your camera is**, and the Blend panel mixes that with the
TRIPS frame. So every mode works everywhere: fly off the capture path, look backwards, go
somewhere no photograph was ever taken — the splat follows.

You will see two lines about it:

```
loading splat /Users/.../kklid_20000.ply ...
splat loaded: 8910382 Gaussians, SH degree 3, 2731 ms
```

Those are the real numbers for `kklid_20000.ply` (2.1 GB, 8.9 million Gaussians): **about
2.7 seconds** to open, once, when the scene opens. After that it is **27 ms a frame at
1080p** — but the TRIPS half of the same frame is ~185 ms, so in practice turning the Blend
panel on costs you roughly 15% and you will not notice it. The panel's bottom line says which
you are looking at — `rendered LIVE at this pose` or `a precomputed render of this capture
view`.

**If the `.ply` cannot be opened** (it was moved, or the bundle came from another machine) the
viewer says so on stderr, keeps working, and falls back to what it did before: **precomputed
renders of the capture views**, carried inside the bundle as a `splat.npz` (a dozen views,
downscaled to 512 px on the long edge). In that state:

- on a capture view that has one — press `N`/`P` to step between them — every mode works;
- the moment you fly off one, there is no splat for *that pose*. The panel greys out the modes
  that need one and says so. **It will not fade to black and it will not reuse the view you
  just left**, because a stored render is a picture of the Gaussians taken from that view's
  camera — showing it at your new pose would be showing a photograph of somewhere else.

`--splat-ply <path>` points the viewer at a different `.ply` than the bundle names.
`--no-live-splat` turns the live path off and gives you exactly the older behaviour.
`--splat-subsample <n>` keeps every n-th Gaussian, which is the lever to reach for if a very
large `.ply` will not fit — on Karekare, `--splat-subsample 4` takes the splat render from
27 ms to 10 ms and the memory with it.

### What the two halves have in common, and what they do not

- **Exposure.** Both halves are in *display* space when they are mixed. The TRIPS frame has
  already been through the tone mapper (exposure, white balance, vignette, response curve); the
  splat's colours were fitted against the photographs directly. Neither is converted, so what
  you see is a straight crossfade between two pictures graded the same way. If a scene's TRIPS
  half looks brighter or flatter than its splat half, that is the tone mapper's grade, and `X`
  cycles it.
- **Lens distortion.** The TRIPS half applies the capture's lens distortion; the splat half is
  rendered as a plain pinhole, because the two renderers parameterise distortion differently.
  On a scene with strong distortion the two can disagree by a few pixels at the very corners.
- **The gate is still a capture-view opinion, on a hybrid scene.** A hybrid network is *fed* the
  Gaussian render as input, and that input still comes from the bundle's precomputed views — the
  live splat feeds the blend, not the network. So off a capture view the TRIPS half (and the
  gate `g` it produces) is the network's "no Gaussian information here" answer, which is what it
  was trained to fall back to. `splat only`, `manual mix` and `split screen` are unaffected, and
  so is every non-hybrid scene. `docs/LIMITATIONS.md` has the reason (there is no per-pixel depth
  out of either renderer to fill the block's depth channel with).

### From the command line

Useful for making two frames to compare side by side:

```
rust/target/release/trips-viewer <bundle> --blend-mode mix --mix 0 --screenshot splat.png
rust/target/release/trips-viewer <bundle> --blend-mode mix --mix 1 --screenshot trips.png
# ... and the same two from a pose no camera stood at, which is the whole point:
rust/target/release/trips-viewer <bundle> --camera-yaw-deg 20 --blend-mode mix --mix 0 \
    --screenshot splat-off-path.png
rust/target/release/trips-viewer <bundle> --blend-mode gated --gate-scale 2 --screenshot pushed.png
rust/target/release/trips-viewer <bundle> --blend-mode split --split 0.5 --screenshot split.png
```

### The gate map, as a picture

Every eval of a gate run writes a **heatmap of the mix** per held-out frame, at
`<run>/eval_ep*/gate/<name>.gate.png`, and the candidate report writes one per dolly frame at
`<report>/dolly/frames/<pose>/gate.png`. Dark is TRIPS, bright is splat. These are drawn from
the gate values alone and contain no photograph, so they are safe for anyone (and any agent) to
open. `report.json` carries the same thing as numbers: mean and percentiles.

To re-score a finished run at a different mix without retraining:

```
trippy eval --checkpoint <run>/checkpoints/checkpoint_best.pt --gate-scale 0
trippy eval --checkpoint <run>/checkpoints/checkpoint_best.pt --gate-scale 2
```

## The Editor (press `M`)

Press **`M`** in the viewer (or launch it with `--edit` to start there). Press `M` again
and it is gone; nothing about the scene changes while it is hidden, and a bundle with no
edits renders exactly the frame it rendered before the editor existed.

The editor opens in **Simple Mode**. There is a **Simple / Advanced** switch at the top
of the panel; everything the old editor had is under **Advanced**, unchanged.

Everything you do is saved in a file called `edits.json`, next to `bundle.json`. Delete
that file and the scene is back to untouched. **`Cmd-Z` undoes anything.**

### Simple Mode: four steps

Do them in any order. Nothing here is permanent until you press `Cmd-S`.

1. **Find shade clouds.** Press **Find them**. The dark blobs of nothing that hang in
   the shade under trees light up magenta, and the panel says how many points that is
   and what percentage of the scene. Then **Soften them** (keep the points, fade them
   towards the splat) or **Remove them** (take them out; the model fills the hole in).

2. **Select an object.** Press **Turn on**, then **click** the thing you want — a plain
   click, no modifier. It lights up magenta and the panel says how many points it took.
   Use **bigger** and **smaller** until it is the thing you meant, then **Keep this as
   an object**. One click will never take the whole scene: how far the selection can
   spread is set from how far away the thing you clicked is, and it stops where the
   points thin out.

3. **Paint an area.** Press **Turn on**, then **drag** on the scene. A magenta circle
   follows the cursor showing exactly how big the brush is where you are pointing, and
   what you paint lights up magenta as soon as you let go. **Hold `ALT` to rub it out.**
   `[` and `]` make the brush smaller and bigger. Under the buttons the panel says in
   one sentence what will happen where you paint — **soften**, **remove** or **choose**.

4. **What to show here.** Every object you have made gets a row: its name, how many
   points it holds, a **Splat &lt;—&gt; TRIPS** slider, and the three buttons **remove /
   soften / choose**. Uncheck the box to switch an edit off without losing it; press
   **forget it** to delete it.

   If the slider is greyed out and says *"This bundle has no splat to mix. Open a
   combined bundle."*, that scene has no Gaussian half — the slider genuinely cannot do
   anything, and it now says so instead of moving and having no effect.

**Put a shape somewhere.** Press **+ box**, **+ ball** or **+ pool lid**, then
**click the spot in the scene**. The shape is created *on the point you clicked*, sized
to how far away it is (about a tenth of the frame across), never at the origin and never
at the camera. Press the button again to cancel.

**Move arrows.** A selected shape has three coloured arrows on it — red, green, blue.
Drag one to move the shape along it; **shift-drag** to resize; **ctrl-drag** to spin a
box. A drag anywhere else still moves the camera. (These used to be called "gizmos".)

### The three things a region can do

| op | what happens inside the region |
|---|---|
| **blend** | the mix slider decides: 0 = show the Gaussian splat there, 1 = show TRIPS there |
| **fade** | the same, but it *multiplies* whatever an earlier region already decided — the gentle version |
| **delete** | the points AND the Gaussians inside are removed before anything is drawn. Nothing can bring them back at any mix, from any angle |

Regions are a **stack**. The one lowest in the list wins where two overlap, and a
`delete` always beats a `blend` above it — so a slider on an unrelated region can
never un-delete something.

### Making one

1. Press `M`, then click **+ box**, **+ ball** or **+ pool lid**.
2. **Click the spot in the scene.** The region is created *there*, on the point under
   your click, sized to 6% of how far away that point is — so it is about a tenth of the
   frame across wherever you put it. Press the button again to cancel instead.
3. Drag the **mix** slider, or click **delete** for the hard version.
4. **Drag one of the three coloured move arrows** (red = world X, green = Y, blue = Z)
   to move it along that axis; **Shift-drag** an arrow to resize; **Ctrl-drag**
   one to rotate a box. A **lid** gets a **fourth, yellow arrow** on its own
   plane normal — drag it to tilt the plane, instead of typing `up` by hand.
   A drag anywhere *else* still moves the camera, so navigation is never taken away.
   The arrow keys, `PageUp`/`PageDown` and `[` / `]` still nudge and resize by
   exact steps, and the Inspector still takes typed numbers (for a lid, `up`
   can be either dragged or typed — both write the same field).
5. **Cmd-S** writes `edits.json`. Closing and reopening the bundle restores the
   regions, the mixes AND the undo history.

**+ pool lid** keeps the Karekare pool *plane* that was already fitted and A/B-checked
— only where the lid sits, and how wide it is, come from your click.

### Keys

| key | action |
|---|---|
| `M` | show/hide the editor |
| `T` | cycle the Advanced Tools panel: Regions → shade-cloud finder → click-to-cluster → SAM 3 lift → brush |
| `H` | preview highlight on/off (for whichever tool has focus) |
| **click** on the render | place the shape you armed with **+ box** / **+ ball** / **+ pool lid**; in Simple Mode's step 2, select the object under the pointer |
| **Shift-click** on the render | select the object under the pointer (click-to-cluster), in every mode |
| **drag** on the render | draw a box for the SAM 3 lift, or paint — *only* while that tool has focus. Right-drag still looks around and shift+left-drag still slides |
| **Alt-click / Alt-drag** | point prompt for the SAM 3 lift; erase, with the brush |
| **drag a move arrow** | move the selected region along that axis (Shift: resize, Ctrl: rotate a box); on a lid, the 4th yellow arrow tilts its plane normal |
| arrows, `PageUp`/`PageDown` | nudge the selected region along world X/Z and Y |
| `[` / `]` | shrink / grow the selected region — or the brush radius, while the brush has focus |
| `Delete` / `Backspace` | remove the selected region |
| `Cmd-Z` / `Cmd-Shift-Z` | undo / redo |
| `Cmd-S` | save `edits.json` |

Every key the viewer already used (`V X B Tab - = F R N P W A S D Q E`) still does
what it did.

### Paint an edit by hand (the brush)

Press `T` until the Tools panel says **brush**, then **drag on the render**. Each
dab is a sphere of the radius in the panel, placed at the depth of the nearest
point **inside the magenta ring** under your cursor — so you paint *on* the scene, not
on the glass. **Alt-drag erases**. `[` and `]` change the radius while the tool has
focus, and the magenta ring shows exactly how big the brush is at the depth it is
painting.

Three things changed on 2026-09-08, after "brushing seemed to blur the foreground and
the background":

- The default op is **soften** (`fade`), not **remove** (`delete`). Painting no longer
  takes geometry out unless you ask it to.
- What you painted is **highlighted magenta** as soon as you let go, whatever the op is,
  so you can see what you did. `H` turns the highlight off.
- The stroke **stays on one surface**: the depth it paints at may move at most 1.5 brush
  radii between two samples of one drag, and the point it anchors on has to be inside
  the ring you can see. Before, one drag could anchor on a twig in front and then on the
  hillside behind, and paint a sphere spanning both.
- The default radius is **3% of how far away you are looking**, not a fixed fraction of
  the whole scene. On Karekare the old default was metres across.

- The stroke goes into a **brush region**: a sparse set of voxel cells, painted
  where you dragged. It behaves like every other region — `delete` removes the
  points inside it, `blend`/`fade` mix them towards the splat, and the Named
  Objects panel lists it like any other.
- **One stroke is one `Cmd-Z`**, however long you dragged.
- Later strokes go into the **same** region until you press **start a new region**,
  so an object can be painted in several passes and still be one named thing.
- The **weight** slider decides how strongly a painted cell claims a point — the
  brush's own soft edge, independent of the region's `mix`.
- The region's voxel grid is fixed when it is created (from the radius at that
  moment), so changing the radius later changes the *stroke*, never the cells you
  already painted.

Headless, for a script or a check:

```
rust/target/release/trips-viewer <bundle> --brush 240 180 --brush-radius 0.5 \
    --brush-op delete --screenshot painted.png
rust/target/release/trips-viewer <bundle> --brush 240 180 --brush-to 300 210 \
    --brush-radius 0.5 --brush-op delete --brush-undo --screenshot undone.png
```

The second run paints the same stroke and undoes it; its frame is byte-identical to
one with no `--brush` at all, which is how the undo is checked on every test run.

### Name, group and solo your objects

The **Named Objects** panel is the list of everything you have made, grouped by the
tool that made it (`brush`, `click`, `shade-clouds`, `sam-box`, and
`hand-authored` for regions you placed by hand). Each row shows the name, kind,
op, and how many points it is actually claiming right now, and carries:

- a **checkbox** to switch it off without deleting it,
- a **mix** slider (the whole drag is one undo step),
- **rename** — the names are the same auto-numbered ones the command line gives
  (`brush-1`, `click-2`, `sam-box-IMG_3703-3`), counted across every tool so the
  list reads as one numbered sequence,
- **remove**,
- **solo**: render *only* that region's effect, so you can see what one object is
  doing without turning the others off one at a time. Solo is a way of looking, not
  an edit — it is never saved into `edits.json`.

Headlessly, `--solo <name-or-id>` does the same thing, and
`--move-region <name-or-id> <dx> <dy> <dz>` moves a region exactly as dragging its
translate handle does.

### Click an object to select it

In Simple Mode: step 2, **Turn on**, then plain-click. In Advanced: press `T` until the
Tools panel says **click-to-cluster**, then **Shift-click** whatever you want in the
render. A plain drag still turns the camera; only a *click* selects, so navigation is not
taken away from you.

**One click can no longer take the whole scene** (2026-09-08). How far a selection may
spread is now the smaller of **15% of how far away the thing you clicked is** and **2% of
the photographed area**, never less than the little circle you clicked in, times whatever
**bigger** / **smaller** you have pressed. On top of that, growth **stops where the points
thin out** — it will not walk from a dense object out into sparse background — and the
seed is clipped to the surface under the cursor rather than a cone reaching through the
whole cloud.

Measured on a synthetic block of **5,000,000** points with no depth gap anywhere (the
shape that caused the problem): the old rule took 200,000 points and was still going when
it hit the hard cap; the new one takes **608 points, 0.012% of the cloud**, in 27 ms
instead of 4.7 s. On a 200,000-point version of the same block the old rule took
**82.7%** of it.

What happens: every point is projected into the camera you are looking through, the
ones landing within a few pixels of your click are collected, the group **nearest
the camera** among those becomes the seed (so clicking a tree does not also grab the
same-coloured hillside glimpsed through a gap behind it), and the selection grows
outward from there by nearest-neighbour hops in 3D — taking in points of a similar
colour, and stopping at a hard distance from the seed.

Four sliders steer it, and moving any of them re-runs the *same* click:

| slider | what it does |
|---|---|
| **radius (px)** | how far from your click a point may land and still be caught |
| **colour tol** | how different a colour may be and still join the object |
| **max radius** | how far, in world units, the selection may reach from the seed. Starts at the scene's own median camera spacing |
| **max points** | a hard cap, so a mis-click on a wall cannot select millions of points |

The panel shows how many points were caught, how many seeded the nearest depth
mode, and how long the clustering took. `H` (or the checkbox) tints the selection
magenta and dims the rest, exactly as the shade finder's preview does.

- **add as region** commits the selection as a `pointset` region with the op
  (`blend` / `fade` / `delete`) and mix you picked next to the button — undoable,
  savable and publishable like any other region.
- **clear** drops the selection, the tint and the neighbour index it built, and the
  frame goes back to exactly what it was.

Headless, for a script or a check:

```
rust/target/release/trips-viewer <bundle> --click 240 180 --dump-click ids.json
rust/target/release/trips-viewer <bundle> --click 240 180 --screenshot selected.png
```

`--dump-click` writes the selected point ids; it must agree exactly with
`trippy edits click --bundle <bundle> --view <IMG> --px 240 180`, and that agreement
is checked on every test run.

### Find shade clouds

Press `T` to switch the Tools panel to the **shade-cloud finder**. Four sliders —
darkness, confidence, and the near/far ends of the depth slab — pick out the dark,
low-confidence points floating in front of what the shade photographs actually saw.
The panel counts them live and shows the audit's own **dark mass fraction** next to
them.

- **`H`** tints the current selection magenta and dims everything else. Press `V`
  to switch to the *raw level-0* view, where the tint is the rasteriser's own
  pixels rather than the network's guess about them — that is the honest picture of
  what is selected.
- **add as region (fade)** or **(delete)** turns the live preview into a real
  region you can then undo, re-mix or save like any other.

The finder needs to know where the shade photographs were taken. That comes from a
small file, `shade_views.json`, written once per bundle:

```
PYTHONPATH=. python -c "
from trippy.edit.golden import write_shade_views
from trippy.train import prune
views = prune.build_shade_region('<scene>/sparse/0', ['IMG_3828.jpg', 'IMG_3830.jpg'], 0.05, 0.5)
write_shade_views('<bundle>', views, 0.05, 0.5)
"
```

Without it the panel says so and the finder is unavailable, rather than quietly
guessing.

### Cut an object out with SAM 3

This is the tool for "get rid of that", "make just this bit splat", or "select
that whole thing" when a colour-based click cannot tell the object from its
background. It segments a **photograph** with the SAM 3 model already on this
machine and lifts the mask onto the scene's points.

1. Press **`N`** / **`P`** until you are looking through the capture view that
   shows the object best. (You must be *on* a view — the tool needs the
   photograph. If you have flown off one, it snaps you to the nearest view and
   says so; then drag again.)
2. Press **`T`** until the Tools panel says **SAM 3 lift**.
3. **Drag a box** round the object on the render. Or **Alt-click** it. A pink
   rectangle follows the drag.
4. Choose **op** (fade / blend / delete) and **mix**, then press **run SAM
   lift**.
5. Watch the log. It prints a line per stage; **cancel** kills it at any point.
   About 9 seconds per view on the CPU.
6. When it lands, the selected points light up magenta and the region is in the
   Regions list. **`Cmd-Z`** takes it straight back out if it grabbed the wrong
   thing; **`Cmd-S`** keeps it.

Two settings worth knowing:

- **views around** — how many neighbouring photographs also get segmented, so a
  point is only kept if a majority of the views that can see it agree. Use **0**
  (trust the one view) or **2 or more**. One neighbour is not a vote, it is an
  intersection, and it will throw away most of the selection.
- **device** — `cpu` by default and that is usually the right answer. `mps` is
  your own interactive use of your own GPU, which is fine; a batch of lifts is
  not, and should go through the queue.

Nothing here sends a photograph anywhere. The image is opened by one local
child process, the mask is an array, and the viewer itself never decodes a
pixel of it.

If the panel says the bundle **records no `scene_root`**, it was exported before
this feature existed and cannot find its own photographs — re-run
`trippy export-bundle` for that run and the tool lights up.

### Checking an edit without opening a window

```
# what weight did every point end up with?  (no GPU, no window)
rust/target/release/trips-viewer <bundle> --dump-weights weights.json

# what would the finder select at these thresholds?
rust/target/release/trips-viewer <bundle> --dump-shade shade.json \
    --shade-lum 0.25 --shade-conf 0.5

# what would a click at pixel (240, 180) of the default view select?
rust/target/release/trips-viewer <bundle> --click 240 180 --dump-click ids.json \
    --click-radius-px 12 --click-colour-tol 0.15

# run the SAM lift without a window (--sam-fake needs no SAM 3 and no GPU;
# the box is in the chosen view's own pixels, which is what a drag maps to)
rust/target/release/trips-viewer <bundle> --sam-box 12 9 36 27 --sam-op delete \
    --screenshot lifted.png
rust/target/release/trips-viewer <bundle> --sam-box 12 9 36 27 --sam-fake \
    --sam-undo --screenshot back-to-normal.png

# a picture of the edited scene, same code path as the window
rust/target/release/trips-viewer <bundle> --screenshot edited.png
rust/target/release/trips-viewer <bundle> --edits nowhere.json --screenshot clean.png
```

`--dump-weights` is the file `trippy apply-edits` has to agree with, to six decimal
places. That agreement is checked automatically on every test run.

### Publishing an edit

`edits.json` is the input to the Python side:

```
trippy apply-edits --bundle <bundle> --out <edited-bundle>
```

which writes a new bundle with the deleted points gone, the per-point weights
alongside, and (if the bundle names a Gaussian `.ply`) a filtered copy of it.

### What is not built yet

- The SAM lift starts a fresh process per view, so `views around 4` loads the
  3.4 GB model five times. Use 0 unless a vote is really wanted.
- No per-point tidy-up of what SAM returned: if the selection is slightly wrong
  at the edges, the answer today is a different box, not a brush.
- A region with `mix < 1` needs a Gaussian splat to mix *with*. On a bundle with no
  `blend.splat_ply` the Splat/TRIPS slider is greyed out and says
  *"This bundle has no splat to mix. Open a combined bundle."*
- The brush's highlight is computed when you let go of the button, not while you drag.
  During the stroke you have the ring; after it you have the magenta.
- Simple Mode has no SAM step yet: cutting an object out with SAM 3 is still under
  **Advanced**.

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

If any of the scene's photographs had a per-image exposure the training never converged
on (a photo with no EXIF, or one that was held out), `export-bundle` says so and
substitutes the scene's median exposure for those views, so the viewer cannot open on a
blown-white frame. It prints a line like `10 of 219 per-view exposures were more than 2.0
stops from the scene median`; that line is normal and the substitution is recorded inside
the bundle. See docs/LIMITATIONS.md "Per-image exposure".

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

## Cleaning a splat with TRIPS (`trippy splat-clean`)

**What it is for.** Your Gaussian splat "felt like being in the scene", but it has
fog in it — the canopy shade cloud, and other floaters. The full-scene TRIPS runs
make that fog disappear, but TRIPS does not look like a photograph elsewhere. So
this tool keeps the splat and uses TRIPS only as a *judge*: it asks the trained
TRIPS model which of your Gaussians it stopped believing in, and deletes exactly
those. Nothing else about the splat changes — every surviving Gaussian keeps its
own colour, opacity, shape and rotation, copied through byte for byte.

**Why this is not the prune that failed before.** The earlier Splats-side prune
keyed on "dark, and in the shade volume", which is also true of the *ground* under
the tree, so it took the ground with it. TRIPS renders that ground, so its
confidence separates the two: a point in free space is contradicted by every view
that sees past it and its confidence falls; a point on a surface is confirmed and
its confidence rises. The tool measures this before it deletes anything and puts
the numbers in `summary.json`.

**What you get.** Three PLYs at increasing aggressiveness, so you pick with your
eyes rather than from a metric:

| variant | what it deletes |
|---|---|
| `kklid-tripsclean-shade` | the least-believed points, inside the measured shade volume only — the rest of the scene is untouched |
| `kklid-tripsclean-005` | the same threshold, everywhere in the scene |
| `kklid-tripsclean-015` | a wider threshold, everywhere — the most aggressive |

Open them in Brush the usual way (see "How to open a .ply file in Brush"). Start
with `shade`; if the fog is still there, try `005`, then `015`. If `015` starts
eating things you wanted, say so and the threshold moves.

**The honesty artifact.** Each variant also writes
`kklid-tripsclean-<variant>-deleted-density.png`: a top-down map of the scene with
no photographic content in it at all. The left panel is how many Gaussians were
deleted per cell, the right panel is what *fraction* of each cell went. The right
panel is the one to read — if the deletion were a blind shave off the whole scene
it would be flat, and if it is finding the fog it is concentrated.

**Running it.**

```
PYTHONPATH=. .venv/bin/python -m trippy.cli splat-clean \
  --checkpoint <run>/checkpoints/checkpoint_latest.pt \
  --ply <the splat that run was seeded from>.ply \
  --out $TRIPPY_OUTPUT/clean/<name> \
  --scene <scene>/sparse_txt \
  --frames-json $TRIPPY_OUTPUT/scratch/shade_frames.json --frames-key big_tree
```

It is CPU-only and takes a few minutes per variant on a 9M-Gaussian splat. Add
`--variant 005` (repeatable) to build only some of them, `--no-audit` to skip the
Splats shade/extent audits, and `--no-freespace` to skip the in-front-of-surface
check.

**What `summary.json` tells you.**

- `mapping` — how the tool matched Gaussians to TRIPS points, and the evidence.
  `fingerprint_max_abs_err: 0.0` means every point was matched to the exact
  Gaussian that seeded it, with no guessing.
- `confidence` — the distribution of what TRIPS believes, so you can see where the
  thresholds fall in it.
- `variants.<name>.freespace` — of the points being deleted, how many sit in front
  of a surface TRIPS believes in (`front`), on it (`on`), or behind it (`behind`),
  and above all `pixels_emptied`: how many pixels held *something* before the
  deletion and hold *nothing* after. That last number is the hole count. It should
  be zero or near it.
- `audits` — Splats' own shade audit and extent gate, run on the original splat and
  on every variant, so the numbers sit in the same column as every training run's.

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
