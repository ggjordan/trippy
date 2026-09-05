//! `trips-viewer` as a **library**: the platform-neutral half of the viewer.
//!
//! Module: `trips_viewer` (library crate)
//! Purpose: bundle loading, the fly camera and the per-frame TRIPS pipeline
//!     (`brush-pyramid` -> `brush-unet` -> a device buffer) with **no window
//!     toolkit attached**, so the same code drives two front ends:
//!     - `src/main.rs`, the native eframe/egui binary (this package's `[[bin]]`);
//!     - `rust/crates/trips-web`, the wasm32/WebGPU browser viewer.
//! Invariants:
//!     - Nothing in this library may reference `eframe`, `egui`, `rfd` or
//!       `winit`. Those are `[target.'cfg(not(target_family = "wasm"))'`
//!       dependencies of the binary only; pulling one in here would break the
//!       wasm build (see `docs/WEB_VIEWER.md`).
//!     - Nothing here may call `std::time::Instant::now()` or
//!       `brush_pyramid::gpu::block_on` unconditionally: the first panics on
//!       `wasm32-unknown-unknown` ("time not implemented on this platform")
//!       and the second parks a thread the browser has no way to unpark. Both
//!       are `cfg`-gated at their definition sites.
//!     - `std::fs` is still used by [`bundle::Bundle::load`], which is the
//!       *native* entry point; the web front end calls
//!       [`bundle::Bundle::from_parts`] with bytes it fetched instead. `fs`
//!       compiles for wasm32 (it just fails at runtime), so no `cfg` is
//!       needed — only the discipline of not calling it.
//! Related docs: `docs/WEB_VIEWER.md`; `docs/decisions/ADR-0006-viewer-integration.md`.

pub mod bundle;
pub mod camera;
pub mod renderer;

/// The blit shader, shared verbatim by both front ends.
///
/// The native binary hands it to `egui_wgpu`'s paint callback
/// (`src/blit.rs`); the web viewer creates its own render pipeline from the
/// same string (`trips-web/src/blit.rs`). One source means the two cannot
/// drift into rendering the same buffer two different ways — which is exactly
/// what the screenshot PSNR check would otherwise be measuring.
pub const BLIT_WGSL: &str = include_str!("shaders/blit.wgsl");
