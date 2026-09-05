//! brush-unet: TRIPS U-Net decoder inference (skeleton).
//!
//! Module: brush_unet
//! Purpose: v0.4.0 placeholder for the Rust/Burn port of the TRIPS U-Net
//!     decoder (5 levels, 32 base filters, ~130k params; see
//!     docs/UPSTREAM.md). This crate currently exposes only the
//!     architecture's shape as a config struct so downstream crates
//!     (`apps/brush-app`) and CI have something concrete to build against.
//!     The real Burn conv2d graph and the safetensors weight loader (mirrors
//!     `crates/lpips-convert` in the Brush fork) are future work tracked in
//!     docs/SPEC.md's v0.4.0 row.
//! Invariants:
//!     - `UnetConfig::default()` must describe the same architecture as the
//!       trained PyTorch checkpoints this crate will eventually load
//!       (5 levels, 32 filters) so parity tests have a fixed target.
//! Related docs: docs/UPSTREAM.md ("TRIPS: original paper and code");
//!     docs/SPEC.md v0.4.0 row; docs/decisions/ADR-0005-brush-fork-layout.md.

/// Shape of the U-Net decoder. Placeholder: no weights, no forward pass yet.
///
/// Mirrors the TRIPS reference architecture (arXiv 2401.06003): an input
/// packing colour + coverage/confidence channels, an RGB output, `levels`
/// encoder/decoder stages, and `base_filters` channels at the finest level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnetConfig {
    /// Input channel count (TRIPS: RGB + per-pixel coverage/confidence).
    pub in_channels: u32,
    /// Output channel count (RGB).
    pub out_channels: u32,
    /// Channel count at the finest encoder/decoder level; doubles per level
    /// going down, per the standard U-Net convention.
    pub base_filters: u32,
    /// Number of encoder/decoder levels.
    pub levels: u32,
}

impl Default for UnetConfig {
    /// The TRIPS paper's reference configuration: 5 levels, 32 base filters,
    /// ~130k parameters total (docs/UPSTREAM.md).
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 3,
            base_filters: 32,
            levels: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_trips_reference_architecture() {
        let cfg = UnetConfig::default();
        assert_eq!(cfg.levels, 5, "TRIPS U-Net has 5 encoder/decoder levels");
        assert_eq!(cfg.base_filters, 32, "TRIPS U-Net has 32 base filters");
        assert_eq!(cfg.out_channels, 3, "decoder output is RGB");
    }

    #[test]
    fn config_is_plain_data() {
        // Placeholder skeleton: no weights, no forward pass. This test only
        // pins that the config type is cheap, comparable data so it can be
        // threaded through the app without ceremony once the real network
        // lands.
        let a = UnetConfig::default();
        let b = a;
        assert_eq!(a, b);
    }
}
