//! `brush-unet`: TRIPS's decoder-only gated U-Net and tone mapper, in Burn.
//!
//! Module: `brush_unet`
//! Purpose: the second half of the v0.4.0 forward pass. `brush-pyramid`
//!     rasterises a point set into `L` alpha-composited feature images;
//!     this crate fuses them into RGB (`net::Unet`) and applies the
//!     per-image tone mapper (`camera::NeuralCamera`), both on wgpu through
//!     Burn. Weights come from `trippy.net.export_safetensors`, so the Mac
//!     and web viewers load exactly what the Python trainer produced.
//!
//! # Layout
//!
//! - [`config`] — architecture description + the safetensors key schema.
//!   **No dependencies**, so `scripts/build.sh` / `scripts/test.sh` compile
//!   and test it on every push.
//! - [`weights`] — the safetensors reader. Host-side only (`Vec<f32>`), so
//!   the schema is testable without a GPU.
//! - `net` — the Burn graph (`GatedBlock`, `UpBlock`, `Unet`).
//!   **Behind the `gpu` feature.**
//! - `camera` — the Burn tone mapper. Behind the `gpu` feature.
//!
//! # The forward pass
//!
//! ```text
//! points + camera --brush-pyramid--> L x Tensor<4> [1,C,h,w]  (finest first)
//!                 --Unet::forward--> Tensor<4> [1,3,H,W]      (linear-ish)
//!    --NeuralCamera::forward(frame)--> Tensor<4> [1,3,H,W]    (display RGB)
//! ```
//!
//! # Parity
//!
//! `tests/parity_gpu.rs` (feature `gpu`) checks three things against PyTorch:
//! a random-weight 5-layer U-Net on a 32x24 pyramid, the tone mapper on the
//! same fixture, and — when the exports exist under `$TRIPPY_OUTPUT/brush` —
//! the whole pipeline on the public Zenodo horse scene at 1920x1080 against
//! `trippy.render.parity`'s own frame.
//!
//! Related docs: `rust/README.md` ("brush-unet weight schema"),
//! `docs/TRIPS_REFERENCE.md` Sec. 5/5a and 6/6a, `docs/ARCHITECTURE.md`,
//! `docs/LIMITATIONS.md`.

#![forbid(unsafe_code)]

pub mod config;
pub mod weights;

#[cfg(feature = "gpu")]
pub mod camera;
#[cfg(feature = "gpu")]
pub mod net;

pub use config::{CameraConfig, UnetConfig};
pub use weights::{HostTensor, Weights};

#[cfg(feature = "gpu")]
pub use camera::NeuralCamera;
#[cfg(feature = "gpu")]
pub use net::{combine_bridge, GatedBlock, UpBlock, Unet};

/// Repository root, derived from this crate's manifest directory.
///
/// `CARGO_MANIFEST_DIR` is `<repo>/rust/crates/brush-unet`, so the root is
/// three levels up. Used by the tests and the example's default paths, the
/// same way `brush_pyramid::fixture::repo_root` is.
#[must_use]
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate is nested at <repo>/rust/crates/brush-unet")
        .to_path_buf()
}

/// Directory holding the committed random-weight parity fixture.
#[must_use]
pub fn fixture_dir() -> std::path::PathBuf {
    repo_root().join("tests/fixtures/synthetic/unet_fixture_small")
}
