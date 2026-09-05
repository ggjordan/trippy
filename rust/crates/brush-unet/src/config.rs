//! Architecture description and the safetensors key schema.
//!
//! Module: `brush_unet::config`
//! Purpose: everything about the network's *shape* that can be known without
//!     a GPU, a device or Burn: channel bookkeeping, the per-block output
//!     widths, and the exact tensor names
//!     `trippy.net.export_safetensors` writes. Keeping this dependency-free
//!     is what lets `scripts/build.sh` / `scripts/test.sh` compile and test
//!     the schema on every push while the Burn graph stays behind the `gpu`
//!     feature.
//! Invariants:
//!     - `filters > 2 * in_channels`, so the "-2C" block width stays
//!       positive (mirrors `NetworkConfig.__post_init__`'s check).
//!     - `up` blocks are numbered in **application order**: block 0 consumes
//!       the coarsest pyramid input (`inputs[num_layers - 2]`) and block
//!       `num_layers - 2` is the `last = true` block that consumes
//!       `inputs[0]`. This matches the Python `up` ModuleList index, which is
//!       what the exporter uses; it is NOT the pyramid level index.
//! Units: all counts are channels or pixels.
//! Related docs: `rust/README.md` ("brush-unet weight schema");
//!     `docs/TRIPS_REFERENCE.md` Sec. 5/5a; `trippy/net/unet.py`.

/// The gated block's convolution kernel; Saiga asserts it is always 3.
pub const GATED_KERNEL: usize = 3;

/// `padding = dilation * (kernel - 1) / 2` (PartialConvUnet2d.h:114).
pub const GATED_PADDING: usize = (GATED_KERNEL - 1) / 2;

/// Upsampling factor between two pyramid levels.
pub const UPSAMPLE_SCALE: usize = 2;

/// `format` metadata value this crate can read.
pub const EXPORT_FORMAT: &str = "trippy-unet-1";

/// Shape of `MultiScaleUnet2dDecOnlySmallFixed`.
///
/// Defaults are TRIPS's shipped `train_normalnet.ini` values, i.e. what
/// `trippy.net.unet.NetworkConfig` defaults to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnetConfig {
    /// `C`: channels per raw pyramid input (the rasteriser's feature width).
    pub in_channels: usize,
    /// `O`: channels of the final image (RGB).
    pub out_channels: usize,
    /// `F`: the constant channel budget at every level.
    pub filters: usize,
    /// `L`: pyramid levels consumed, and therefore `len(inputs)`.
    pub num_layers: usize,
}

impl Default for UnetConfig {
    /// TRIPS's reference configuration: 5 levels, 32 filters, C = 4, RGB out.
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 3,
            filters: 32,
            num_layers: 5,
        }
    }
}

impl UnetConfig {
    /// Validate the channel bookkeeping.
    ///
    /// # Errors
    /// Returns `Err` if `num_layers < 2` or `filters <= 2 * in_channels`
    /// (either makes a block's output width zero or negative).
    pub fn validate(&self) -> Result<(), String> {
        if self.num_layers < 2 {
            return Err(format!(
                "num_layers must be >= 2 (a decoder-only U-Net needs a start block and at \
                 least one upsample block); got {}",
                self.num_layers
            ));
        }
        if self.filters <= 2 * self.in_channels {
            return Err(format!(
                "filters must be > 2*in_channels so the start block's output width \
                 filters-2*in_channels stays positive; got filters={}, in_channels={}",
                self.filters, self.in_channels
            ));
        }
        Ok(())
    }

    /// Output width of the start block's gated convolution: `F - 2C`.
    #[must_use]
    pub const fn start_out_channels(&self) -> usize {
        self.filters - 2 * self.in_channels
    }

    /// Number of `up` blocks: `L - 1`.
    #[must_use]
    pub const fn num_up_blocks(&self) -> usize {
        self.num_layers - 1
    }

    /// Whether `up` block `k` is the `last = true` one (Networks.h:1034).
    #[must_use]
    pub const fn is_last_up(&self, k: usize) -> bool {
        k + 2 == self.num_layers
    }

    /// Pyramid level `up` block `k` consumes as its raw skip input.
    ///
    /// Application order, so `k = 0` reads level `L - 2` and
    /// `k = L - 2` reads level 0.
    #[must_use]
    pub const fn up_level(&self, k: usize) -> usize {
        self.num_layers - 2 - k
    }

    /// Output width of `up` block `k`: `F - C` for the last block,
    /// `F - 2C` otherwise (Networks.h:1033-1034, the "-2C" trick).
    #[must_use]
    pub const fn up_out_channels(&self, k: usize) -> usize {
        if self.is_last_up(k) {
            self.filters - self.in_channels
        } else {
            self.filters - 2 * self.in_channels
        }
    }

    /// Pyramid layer shapes with TRIPS's `ceil` halving, finest first.
    #[must_use]
    pub fn layer_shapes(&self, height: usize, width: usize) -> Vec<(usize, usize)> {
        let mut shapes = Vec::with_capacity(self.num_layers);
        let (mut h, mut w) = (height, width);
        for _ in 0..self.num_layers {
            shapes.push((h, w));
            h = h.div_ceil(UPSAMPLE_SCALE);
            w = w.div_ceil(UPSAMPLE_SCALE);
        }
        shapes
    }

    /// Total learnable scalars, for cross-checking against the Python
    /// `parameter_count()`.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let k2 = GATED_KERNEL * GATED_KERNEL;
        // Start: two convs C -> F-2C, each with a bias.
        let start_out = self.start_out_channels();
        let mut total = 2 * (start_out * self.in_channels * k2 + start_out);
        for k in 0..self.num_up_blocks() {
            let out = self.up_out_channels(k);
            total += 2 * (out * self.filters * k2 + out);
        }
        total += self.out_channels * self.filters + self.out_channels;
        total
    }

    /// Every U-Net tensor name, in the order the exporter writes them.
    #[must_use]
    pub fn weight_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for branch in ["feature", "gate"] {
            keys.push(format!("unet.start.{branch}.weight"));
            keys.push(format!("unet.start.{branch}.bias"));
        }
        for k in 0..self.num_up_blocks() {
            for branch in ["feature", "gate"] {
                keys.push(format!("unet.up.{k}.{branch}.weight"));
                keys.push(format!("unet.up.{k}.{branch}.bias"));
            }
        }
        keys.push("unet.final.weight".into());
        keys.push("unet.final.bias".into());
        keys
    }

    /// Expected shape of each key in [`Self::weight_keys`], same order.
    #[must_use]
    pub fn weight_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let k = GATED_KERNEL;
        let mut out = Vec::new();
        let start_out = self.start_out_channels();
        for branch in ["feature", "gate"] {
            out.push((
                format!("unet.start.{branch}.weight"),
                vec![start_out, self.in_channels, k, k],
            ));
            out.push((format!("unet.start.{branch}.bias"), vec![start_out]));
        }
        for block in 0..self.num_up_blocks() {
            let block_out = self.up_out_channels(block);
            for branch in ["feature", "gate"] {
                out.push((
                    format!("unet.up.{block}.{branch}.weight"),
                    vec![block_out, self.filters, k, k],
                ));
                out.push((format!("unet.up.{block}.{branch}.bias"), vec![block_out]));
            }
        }
        out.push((
            "unet.final.weight".into(),
            vec![self.out_channels, self.filters, 1, 1],
        ));
        out.push(("unet.final.bias".into(), vec![self.out_channels]));
        out
    }
}

/// The tone mapper's shape and which of its stages are enabled.
///
/// Mirrors `trippy.net.camera_model.NeuralCameraConfig` plus the two sizes
/// (`num_frames`, `image_height/width`) the Python module takes as
/// constructor arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraConfig {
    /// `M`: number of images with their own exposure / white balance.
    pub num_frames: usize,
    /// `P`: response-LUT control points per channel.
    pub response_params: usize,
    /// Render height the vignette's aspect correction was fitted at.
    pub image_height: usize,
    /// Render width, likewise.
    pub image_width: usize,
    /// Per-image exposure, applied as `x * 2**-ev`.
    pub enable_exposure: bool,
    /// Per-image white balance gains.
    pub enable_white_balance: bool,
    /// Radial vignette polynomial.
    pub enable_vignette: bool,
    /// Response LUT; when false the stage is `clamp(x, 0, 1)`.
    pub enable_response: bool,
}

impl CameraConfig {
    /// Every camera tensor name that must be present for this configuration.
    #[must_use]
    pub fn weight_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        if self.enable_exposure {
            keys.push("camera.exposure".into());
        }
        if self.enable_white_balance {
            keys.push("camera.white_balance".into());
        }
        if self.enable_vignette {
            keys.push("camera.vignette_params".into());
            keys.push("camera.vignette_center".into());
        }
        if self.enable_response {
            keys.push("camera.response".into());
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_the_trips_reference_architecture() {
        let cfg = UnetConfig::default();
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.filters, 32);
        assert_eq!(cfg.in_channels, 4);
        assert_eq!(cfg.out_channels, 3);
        cfg.validate().expect("default config is valid");
    }

    #[test]
    fn only_the_last_up_block_widens_to_filters_minus_c() {
        let cfg = UnetConfig::default();
        assert_eq!(cfg.num_up_blocks(), 4);
        // Blocks 0..2 are "not last": F - 2C = 32 - 8 = 24.
        for k in 0..3 {
            assert!(!cfg.is_last_up(k));
            assert_eq!(cfg.up_out_channels(k), 24);
        }
        // Block 3 is the last: F - C = 32 - 4 = 28, so the final bridge
        // concat lands on exactly F = 32 channels.
        assert!(cfg.is_last_up(3));
        assert_eq!(cfg.up_out_channels(3), 28);
        assert_eq!(cfg.up_out_channels(3) + cfg.in_channels, cfg.filters);
    }

    #[test]
    fn up_blocks_are_indexed_in_application_order() {
        let cfg = UnetConfig::default();
        // Block 0 reads the coarsest input that is not the start block's.
        assert_eq!(cfg.up_level(0), 3);
        // The last block reads the full-resolution input.
        assert_eq!(cfg.up_level(cfg.num_up_blocks() - 1), 0);
    }

    #[test]
    fn parameter_count_matches_the_python_port() {
        // trippy's MultiScaleUnet2dDecOnlySmallFixed reports 59,675 for the
        // 5-layer fixture and 101,291 for the 8-layer horse checkpoint
        // (experiments/EXP-0002-horse-parity/README.md).
        assert_eq!(UnetConfig::default().parameter_count(), 59_675);
        let horse = UnetConfig {
            num_layers: 8,
            ..UnetConfig::default()
        };
        assert_eq!(horse.parameter_count(), 101_291);
    }

    #[test]
    fn layer_shapes_use_ceil_halving() {
        let cfg = UnetConfig::default();
        assert_eq!(
            cfg.layer_shapes(24, 32),
            vec![(24, 32), (12, 16), (6, 8), (3, 4), (2, 2)]
        );
        // 1080p with 8 levels, the horse configuration.
        let horse = UnetConfig {
            num_layers: 8,
            ..cfg
        };
        assert_eq!(
            horse.layer_shapes(1080, 1920),
            vec![
                (1080, 1920),
                (540, 960),
                (270, 480),
                (135, 240),
                (68, 120),
                (34, 60),
                (17, 30),
                (9, 15)
            ]
        );
    }

    #[test]
    fn weight_keys_and_shapes_agree() {
        let cfg = UnetConfig::default();
        let keys = cfg.weight_keys();
        let shapes = cfg.weight_shapes();
        assert_eq!(keys.len(), shapes.len());
        for (key, (name, _)) in keys.iter().zip(&shapes) {
            assert_eq!(key, name);
        }
        // 4 start tensors + 4 per up block + 2 final.
        assert_eq!(keys.len(), 4 + 4 * cfg.num_up_blocks() + 2);
    }

    #[test]
    fn validate_rejects_impossible_channel_budgets() {
        let too_narrow = UnetConfig {
            filters: 8,
            in_channels: 4,
            ..UnetConfig::default()
        };
        assert!(too_narrow.validate().is_err());
        let too_shallow = UnetConfig {
            num_layers: 1,
            ..UnetConfig::default()
        };
        assert!(too_shallow.validate().is_err());
    }
}
