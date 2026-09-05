# ADR-0006: Viewer integration — a separate binary in trippy's workspace, not crates moved into the fork

Date: 2026-09-06 · Status: Accepted

## Context

v0.4.0's last piece is a **native Mac viewer** that renders a TRIPS scene live:
per frame, `brush-pyramid` rasterises the point set into an image pyramid,
`brush-unet` decodes it and tone-maps it, and the result reaches the screen.

ADR-0005 deliberately left this open. It settled that `brush-pyramid` and
`brush-unet` can live in trippy's thin workspace and reach *into* the Brush
submodule by path (`brush-cube`, `brush-sort`, `brush-prefix-sum`), and it
recorded the one thing that still argued for moving them:

> `apps/brush-app` lives in the submodule and cannot path-depend outward into
> trippy without the fork's manifest referring to a directory that only exists
> in a trippy checkout. When `splat_backbuffer.rs` is wired up, either the
> crates move into the fork or the viewer integration lives on trippy's side as
> a separate binary. That decision belongs to the task that does the wiring.

The two options, concretely:

**(a) Move the crates into the fork.** `git mv` `brush-pyramid` and
`brush-unet` into `rust/brush-trips/crates/` on the `trippy-fork` branch, push
to `ggjordan/brush`, bump the submodule pin, delete trippy's thin workspace's
members, and add a TRIPS panel to `apps/brush-app`. Then `brush-app` can
`brush-pyramid.path = "../../crates/brush-pyramid"` like every other crate.

**(b) A separate viewer binary in trippy's own workspace.** A new crate
`rust/crates/trips-viewer` that depends on `brush-pyramid`/`brush-unet` by path
(already proven to work) and pulls `eframe`/`egui`/`wgpu` from crates.io, using
the same `[patch]` tables the thin workspace already carries. It reproduces the
~40 lines of Brush's wgpu/egui scaffolding it actually needs — the device
descriptor in `ui/mod.rs::create_egui_options` and the paint-callback shape of
`ui/splat_backbuffer.rs` — rather than importing the whole app.

## Decision

**Option (b): a separate `trips-viewer` binary in trippy's thin workspace.**

The brief asked for whichever gets a working window fastest. (b) does, and by a
wide margin, for four reasons that turned out to be more than convenience:

1. **Nothing has to be pushed anywhere.** (a) needs a fork push, a submodule
   pin bump, a `.gitmodules` change and a re-clone story before a single line of
   viewer code compiles. (b) needed one `members = [...]` entry.
2. **The regression requirement is satisfied by construction.** The acceptance
   criteria include "Brush's own splat viewing still opens a `.ply`". Under (b)
   `apps/brush-app` is not touched at all — not one line — so its `.ply` path
   *cannot* regress and no test is needed to prove it didn't. Under (a) the
   TRIPS panel shares `App`, the tile tree and the wgpu setup with the splat
   path, and every one of those is a place to break it.
3. **The build stays fast where it matters.** `scripts/build.sh` and
   `scripts/test.sh` still only compile `brush-pyramid`/`brush-unet` without the
   `gpu` feature — seconds. Under (a) those crates would live in the Brush
   workspace, and checking them would mean resolving Brush's whole graph.
4. **The viewer is not a training app.** `brush-app` is built around
   `brush-process`: a dataset loader, a trainer, a stats panel, a tile layout,
   an actor pipeline. A TRIPS bundle has no training loop and no dataset stream.
   Nearly all of that scaffolding would have been bypassed rather than reused.

### What (b) costs, honestly

- **Duplicated scaffolding.** `create_egui_options` (the device descriptor that
  excludes `MAPPABLE_PRIMARY_BUFFERS` and enables passthrough shaders) and the
  `CallbackTrait` blit are copied, with attribution, into
  `crates/trips-viewer/src/main.rs` and `src/blit.rs`. If Brush changes the
  feature mask its kernels need, we will not find out from a compile error. Both
  copies carry a comment saying where they came from.
- **A second set of window deps** in `rust/Cargo.toml`'s
  `[workspace.dependencies]` — `egui`, `eframe`, `wgpu`, `rfd`, `env_logger` —
  which must keep matching the submodule's specs for exactly the reason the burn
  specs already do: the viewer binds Burn's buffers into egui's render pass, and
  that is only sound if both halves link **one** wgpu.
- **Two binaries to ship** (`brush` and `trips-viewer`) instead of one app with
  two modes.

None of those is worse than option (a)'s standing cost: every future upstream
rebase of `ggjordan/brush` would have to carry trippy's own crates through it.

## How it works

```
eframe creates the wgpu Adapter/Device/Queue (create_egui_options' descriptor)
        │
        ├── burn_wgpu::init_device(setup, ExclusivePages)   → Burn shares it
        │
        ├── brush_pyramid::gpu::render_pyramid  → CubeTensor (P, C) f32
        ├── brush_unet Unet + NeuralCamera      → Tensor<4> [1, 3, H, W]
        │        └── burn_bridge::resolve_to_cube_float → CubeTensor
        │
        └── egui paint callback binds that buffer as a storage buffer
            and a fullscreen triangle samples it (src/shaders/blit.wgsl)
```

The frame **never leaves the GPU**. This is the same trick `brush-app` plays for
its splat backbuffer, and it only works because Burn was handed eframe's device
rather than making its own.

Two pieces of new glue were needed and both live in `brush-pyramid`, next to the
existing forward bridge, so the viewer does not have to depend on the splat
rasteriser for them:

- `gpu::burn_bridge::resolve_to_cube_float` — the reverse of the existing
  `float_tensor`, for getting the *network's* output back to a bindable buffer.
- `gpu::sync` — drain the device queue, the honest end-of-frame barrier for the
  benchmark (a readback would charge the frame for 24 MB of transfer the window
  never pays).

## Consequences

### `brush-app` is untouched

`git diff` against the pinned submodule commit is empty. `rust/brush-trips` is
still at `b2f2c3ea27e39c28509fc470b528cfee4cf6f6f6` and `.gitmodules` is
unchanged. Nothing was pushed to `ggjordan/brush` for this work. Brush's `.ply`
viewing is byte-identical to what ADR-0005 shipped.

### The bundle format is the interface

The viewer does not know about checkpoints, ADOP scenes or trippy's Python at
all. It reads a **bundle directory** — `bundle.json` + `points.npz` +
`weights.safetensors`, format `trippy-bundle-1` — written by
`trippy export-bundle`. That is what makes "open a Karekare checkpoint the same
way" a one-command job the day a Karekare checkpoint is worth opening, with no
Rust change.

One schema decision inside that is load-bearing: the bundle stores points in
**world space** with each view's lens distortion as data, where the older
`tools/export_unet_safetensors.py horse-e2e` export baked one view's pose *and*
its distortion into the point positions. A free-flying camera cannot use
pre-baked points. So `brush_pyramid::scene::Camera` gained a `distortion: [f32;
8]` field (Saiga order, defaulting to all-zeros = the identity = the previous
behaviour) and both the CPU reference and the CubeCL kernel now apply it. On the
horse this is not cosmetic: `k1 = -0.064`, `k2 = 0.044` move a corner pixel by
about 21 px, which is the difference between a 40 dB screenshot check and a
meaningless one.

### What the levers turned out to be worth (and the premise that was wrong)

The task this ADR belongs to was briefed on the assumption, recorded in
`research/trips-metal.md` after the first Mac timing, that the 193 ms frame was
"sort-dominated over 10.4 M fragments". Five levers were specified against that
assumption. Measuring them says it is false: with the U-Net removed and the
*identical* rasteriser rendering the `raw level-0` view, the frame is **21.5 ms**
instead of 204 ms. The rasteriser is ~11 % of the frame and the network is ~89 %.

That is not a criticism of the earlier measurement — it timed cumulative prefixes
and reported that "the pyramid alone" was ~205 ms, i.e. indistinguishable from the
whole frame, which is exactly what you would see if the *last* stage dominated and
the barrier semantics put its cost in every prefix. The cheap thing that settles it
is the one the viewer made available for free: a view mode that renders the pyramid
and stops.

The consequence for this ADR is that the levers stay (they are correct, and they are
the right levers for a scene with far more fragments) but the *shipped* setting is
not among them: it is resolution and network precision. See `docs/LIMITATIONS.md`
for the table and `research/trips-metal.md` for the numbers.

### Performance levers are render parameters, not viewer state

Every speed/quality trade lives in `brush_pyramid::params::PyramidParams`
(`frustum_cull`, `layer_floor`, `sort`, `feature_store`, `depth_range`) with a
serde default that is the **exact** pipeline, so:

- an old `params.json` or bundle still loads as exact;
- the parity tests and `render_frame_full` are unaffected;
- the CPU reference implements the one lever that changes *which fragments
  exist* (`layer_floor`) and **refuses** the two that are pure GPU storage and
  ordering strategies, rather than silently ignoring them.

`render_scale` is the exception and is viewer-side only: it is a choice of
resolution, not of arithmetic.

### Verification without a window

The viewer cannot be checked by looking at it — no agent may open a render, and
the person who can is not in the loop during the build. So the same binary has a
`--screenshot` path that runs the identical render code headlessly and writes a
PNG, and `--bench`, which times frames with a device sync. The acceptance check
is PSNR between `--screenshot` and `render_frame_full`'s PNG.

## Related

- ADR-0005 — the fork layout this refines, and the open question it left.
- ADR-0004 — MIT for trippy's crates, Apache-2.0 for the fork; the copied
  scaffolding is attributed in-file.
- `rust/README.md` — build and run instructions.
- `docs/USER_GUIDE.md` — how Jordan opens a scene.
- `docs/LIMITATIONS.md` — what the levers cost and what the viewer still cannot do.
