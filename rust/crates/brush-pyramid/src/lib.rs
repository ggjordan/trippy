//! `brush-pyramid`: TRIPS's image-pyramid rasteriser, forward pass, in Rust.
//!
//! Module: `brush_pyramid`
//! Purpose: the v0.4.0 port of trippy's Python/Metal pyramid rasteriser
//!     (`trippy/raster/`) to Rust, so the Mac viewer (`apps/brush-app`, wgpu
//!     on Metal) and later the web viewer (WebGPU) can render a TRIPS point
//!     set natively. A point set plus a camera goes in; `L` alpha-composited
//!     feature images, finest first, come out, ready for the U-Net decoder.
//!
//! # Layout
//!
//! - [`params`] — every render knob, mirroring `trippy.constants`.
//! - [`grid`] — pyramid shapes and the layer-major flat index space.
//! - [`factor`] — TRIPS's `compute_point_size_fac` and `layer_bounds`.
//! - [`scene`] — [`PointSet`] and [`Camera`], with npz/JSON loaders.
//! - [`npz`] — a dependency-light reader for numpy archives.
//! - [`cpu`] — the reference forward pass, explicit loops, no GPU.
//! - [`output`] — the host-side result type both paths produce.
//! - [`fixture`] — loader for the synthetic Python parity fixtures.
//! - `gpu` — the CubeCL kernels and the Burn entry point
//!   (`gpu::render_pyramid`). **Behind the `gpu` feature**, which is off by
//!   default so that `cargo check`/`cargo test -p brush-pyramid` (run on
//!   every push by `scripts/build.sh` / `scripts/test.sh`) never has to
//!   compile Burn, CubeCL and wgpu.
//! - [`png`] — a tiny PNG writer for the `render_frame` example.
//!
//! # The pipeline
//!
//! Identical in both implementations, and atomic-free by design
//! (`docs/ARCHITECTURE.md`, "Core principle: No atomics anywhere"):
//!
//! 1. **project & count** — per point: `uv`, depth, `size_px`, the near-plane
//!    and visibility cull, and how many `(layer, corner)` slots it reserves.
//! 2. **prefix sum** over those counts, giving each point its write offset.
//! 3. **emit** — one fragment per slot: `(layer, pixel)` key, float32 depth
//!    bits, alpha, point id.
//! 4. **sort** — by depth, then by the `(layer, pixel)` key. Both passes are
//!    stable LSB radix sorts, so the result is ordered by
//!    `(layer, pixel, depth)` with ties broken by point id.
//! 5. **segment offsets** — the `[start, end)` run of each layer-pixel.
//! 6. **blend** — one thread per layer-pixel walks its run front-to-back:
//!    `out += T * alpha * feature`, `T *= 1 - alpha`, stopping at
//!    `max_frags` or when `T` drops below `t_cutoff`.
//!
//! Background is added afterwards as `out += t_final * bg`, never inside the
//! blend loop, exactly as TRIPS does (`RenderForward.cu:3610-3620`).
//!
//! # Invariants
//!
//! - Numerics are float32 end to end, on both paths and on the Python side
//!   the fixtures come from, so parity is a real comparison rather than a
//!   dtype artefact.
//! - Out-of-bounds fragments are **dropped, never clamped**
//!   (`docs/GEOMETRY.md` bug class 3).
//! - Every quirk of TRIPS's blend factor is reproduced deliberately; see
//!   [`factor::layer_factor`] before "simplifying" any of it.
//!
//! # Example
//!
//! ```no_run
//! use brush_pyramid::{cpu::render_pyramid_cpu, params::PyramidParams, scene::{Camera, PointSet}};
//! use std::path::Path;
//!
//! let points = PointSet::from_npz(Path::new("points.npz")).expect("points");
//! let camera = Camera::from_json(Path::new("camera.json")).expect("camera");
//! let images = render_pyramid_cpu(&points, &camera, &PyramidParams::default(), None)
//!     .expect("render");
//! assert_eq!(images.layers.len(), PyramidParams::default().num_layers);
//! ```
//!
//! Related docs: `docs/ARCHITECTURE.md`, `docs/GEOMETRY.md`,
//! `docs/TRIPS_REFERENCE.md` sections 3/3a/3b,
//! `docs/decisions/ADR-0005-brush-fork-layout.md`, `rust/README.md`.

#![forbid(unsafe_code)]

pub mod cpu;
pub mod factor;
pub mod fixture;
pub mod grid;
pub mod npz;
pub mod output;
pub mod params;
pub mod png;
pub mod scene;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use grid::LayerGrid;
pub use output::{LayerImage, PyramidImages};
pub use params::{
    DepthRange, FeatureStore, LayerFloor, Mode, PixelCenter, PyramidHalving, PyramidParams, SortMode,
};
pub use scene::{Camera, PointSet};
