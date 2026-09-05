//! The decoder-only gated U-Net, as Burn modules.
//!
//! Module: `brush_unet::net`
//! Purpose: a 1:1 Burn port of
//!     `trippy.net.unet.MultiScaleUnet2dDecOnlySmallFixed` (itself a port of
//!     TRIPS's `MultiScaleUnet2dDecOnlySmallFixedImpl`, Networks.h:1100-1208)
//!     and of the Saiga gated block it is built from. Runs on wgpu; the
//!     `tests/parity_gpu.rs` fixture pins it to PyTorch at 1e-4.
//! Invariants:
//!     - Inputs are ordered **finest first**: `inputs[0]` is full render
//!       resolution, `inputs[L-1]` the coarsest — the same convention
//!       `brush_pyramid::gpu::PyramidRender::layer_tensors` produces.
//!     - The gated block's two convolutions read the SAME input `x`
//!       (PartialConvUnet2d.h:139-145), never each other's output.
//!     - `combine_bridge` centre-crops **whichever** side is larger down to
//!       the shared minimum, exactly like `trippy.net.unet.combine_bridge`;
//!       see that module's docstring for why TRIPS's own crop-`skip`-only
//!       version cannot handle a `ceil`-halved pyramid with an odd
//!       intermediate dimension.
//!     - Upsampling is bilinear with `align_corners = false`, matching
//!       `nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)`.
//! Units: activations are dimensionless features; the output is
//!     linear-ish RGB, before the tone mapper.
//! Related docs: `docs/TRIPS_REFERENCE.md` Sec. 5/5a; `docs/LIMITATIONS.md`;
//!     `rust/README.md`.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::{elu, sigmoid};
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use burn::tensor::{Device, Tensor, TensorData};

use crate::config::{UnetConfig, GATED_KERNEL, GATED_PADDING, UPSAMPLE_SCALE};
use crate::weights::{HostTensor, Weights};

/// ELU's `alpha`. PyTorch's `nn.ELU()` and libtorch's both default to 1.
const ELU_ALPHA: f64 = 1.0;

/// Upload a host tensor as a rank-`D` Burn tensor.
///
/// # Errors
/// Returns `Err` if the host tensor's rank is not `D`.
pub fn upload<const D: usize>(host: &HostTensor, device: &Device) -> Result<Tensor<D>, String> {
    if host.shape.len() != D {
        return Err(format!(
            "expected a rank-{D} tensor, got shape {:?}",
            host.shape
        ));
    }
    let data = TensorData::new(host.data.clone(), host.shape.clone());
    Ok(Tensor::from_data(data, device))
}

fn load_conv(conv: Conv2d, weights: &Weights, prefix: &str, device: &Device) -> Result<Conv2d, String> {
    let weight: Tensor<4> = upload(weights.get(&format!("{prefix}.weight"))?, device)?;
    let bias: Tensor<1> = upload(weights.get(&format!("{prefix}.bias"))?, device)?;
    Ok(Conv2d {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
        ..conv
    })
}

/// Saiga's `GatedBlockImpl`: two independent 3x3 convolutions over the same
/// input, `elu` on one, `sigmoid` on the other, multiplied.
///
/// The post-gate norm is `Identity` in every shipped TRIPS config
/// (`norm_layer_up = id`) and [`Weights`] refuses to load anything else, so
/// it is not represented here.
#[derive(Module, Debug)]
pub struct GatedBlock {
    /// The `feature_transform` branch (Conv2d -> ELU).
    pub feature: Conv2d,
    /// The `mask_transform` branch (Conv2d -> Sigmoid).
    pub gate: Conv2d,
}

impl GatedBlock {
    /// Build with Burn's default initialisation (weights are overwritten by
    /// [`Self::load`] before any real use).
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, device: &Device) -> Self {
        let config = Conv2dConfig::new([in_channels, out_channels], [GATED_KERNEL, GATED_KERNEL])
            .with_stride([1, 1])
            .with_dilation([1, 1])
            .with_padding(PaddingConfig2d::Explicit(
                GATED_PADDING,
                GATED_PADDING,
                GATED_PADDING,
                GATED_PADDING,
            ))
            .with_bias(true);
        Self {
            feature: config.init(device),
            gate: config.init(device),
        }
    }

    /// Bind this block's four tensors from `weights` under `prefix`
    /// (e.g. `"unet.start"`).
    ///
    /// # Errors
    /// Returns `Err` if any of the four keys is missing or mis-shaped.
    pub fn load(self, weights: &Weights, prefix: &str, device: &Device) -> Result<Self, String> {
        Ok(Self {
            feature: load_conv(self.feature, weights, &format!("{prefix}.feature"), device)?,
            gate: load_conv(self.gate, weights, &format!("{prefix}.gate"), device)?,
        })
    }

    /// `norm(elu(feature_conv(x)) * sigmoid(gate_conv(x)))`, spatial size
    /// preserved.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let feature = elu(self.feature.forward(x.clone()), ELU_ALPHA);
        let gate = sigmoid(self.gate.forward(x));
        feature * gate
    }
}

/// Concatenate on the channel axis, centre-cropping both sides to their
/// shared minimum spatial size first.
///
/// Port of `trippy.net.unet.combine_bridge`, which generalises TRIPS's
/// `CombineBridge` (Networks.h:766-773, 1060-1067): TRIPS crops only `skip`,
/// assuming `skip >= below`, which is false whenever a `ceil`-halved pyramid
/// level had an odd dimension. Cropping whichever side is larger is
/// bit-identical to TRIPS in the case TRIPS can handle, and defined in the
/// case it cannot.
#[must_use]
pub fn combine_bridge(below: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
    let [_, _, bh, bw] = below.dims();
    let [_, _, sh, sw] = skip.dims();
    if bh == sh && bw == sw {
        return Tensor::cat(vec![below, skip], 1);
    }
    let th = bh.min(sh);
    let tw = bw.min(sw);
    Tensor::cat(vec![centre_crop(below, th, tw), centre_crop(skip, th, tw)], 1)
}

/// Symmetric centre crop to `(target_h, target_w)`; a no-op on an axis that
/// already matches. The offset uses truncating division, matching the C++
/// `int(diff)/2` and Python's `//`.
#[must_use]
pub fn centre_crop(x: Tensor<4>, target_h: usize, target_w: usize) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    debug_assert!(h >= target_h && w >= target_w, "centre_crop requires x >= target");
    let off_h = (h - target_h) / 2;
    let off_w = (w - target_w) / 2;
    x.slice_dim(2, off_h..off_h + target_h)
        .slice_dim(3, off_w..off_w + target_w)
}

/// One `UpsampleDecOnlySmallBlockFixedImpl` (Networks.h:999-1097).
#[derive(Module, Debug)]
pub struct UpBlock {
    /// The gated convolution, `F -> F-2C` (or `F -> F-C` when last).
    pub conv: GatedBlock,
}

impl UpBlock {
    /// Build block `k` of `config` with default initialisation.
    #[must_use]
    pub fn new(config: &UnetConfig, k: usize, device: &Device) -> Self {
        Self {
            conv: GatedBlock::new(config.filters, config.up_out_channels(k), device),
        }
    }

    /// Bind from `weights` under `unet.up.{k}`.
    ///
    /// # Errors
    /// As [`GatedBlock::load`].
    pub fn load(self, weights: &Weights, k: usize, device: &Device) -> Result<Self, String> {
        Ok(Self {
            conv: self.conv.load(weights, &format!("unet.up.{k}"), device)?,
        })
    }

    /// `below` is the running decoder state, `raw` this level's pyramid
    /// input. Returns the new bridge, `raw` concatenated with the block's
    /// convolution output.
    #[must_use]
    pub fn forward(&self, below: Tensor<4>, raw: Tensor<4>) -> Tensor<4> {
        let [b, c, h, w] = below.dims();
        let upsampled = interpolate(
            below,
            [h * UPSAMPLE_SCALE, w * UPSAMPLE_SCALE],
            InterpolateOptions {
                mode: InterpolateMode::Bilinear,
                align_corners: false,
            },
        );
        debug_assert_eq!(upsampled.dims()[0..2], [b, c]);
        // Networks.h:1082: below = the raw pyramid input, skip = the
        // upsampled path.
        let combined = combine_bridge(raw.clone(), upsampled);
        let conv_out = self.conv.forward(combined);
        // Networks.h:1088.
        combine_bridge(raw, conv_out)
    }
}

/// `MultiScaleUnet2dDecOnlySmallFixed`.
#[derive(Module, Debug)]
pub struct Unet {
    /// `SmallDecStartBlockImpl`'s gated conv, `C -> F-2C`.
    pub start: GatedBlock,
    /// `L-1` upsample blocks, in application order (coarsest first).
    pub up: Vec<UpBlock>,
    /// The 1x1 output convolution, `F -> O`. `last_act` is `id`, so no
    /// activation follows it.
    pub final_conv: Conv2d,
    /// The architecture. `#[module(skip)]` because it holds no tensors and
    /// must not appear in a record (the same attribute `Conv2d` uses for its
    /// `padding` field).
    #[module(skip)]
    config: UnetConfig,
}

impl Unet {
    /// Build the graph with Burn's default initialisation.
    ///
    /// # Panics
    /// Panics if `config` does not [`UnetConfig::validate`].
    #[must_use]
    pub fn new(config: UnetConfig, device: &Device) -> Self {
        config.validate().expect("valid UnetConfig");
        let start = GatedBlock::new(config.in_channels, config.start_out_channels(), device);
        let up = (0..config.num_up_blocks())
            .map(|k| UpBlock::new(&config, k, device))
            .collect();
        let final_conv = Conv2dConfig::new([config.filters, config.out_channels], [1, 1])
            .with_bias(true)
            .init(device);
        Self {
            start,
            up,
            final_conv,
            config,
        }
    }

    /// Build and immediately bind every tensor from `weights`.
    ///
    /// # Errors
    /// Returns `Err` if any expected key is missing or mis-shaped.
    pub fn load(weights: &Weights, device: &Device) -> Result<Self, String> {
        let config = weights.unet;
        let net = Self::new(config, device);
        let start = net.start.load(weights, "unet.start", device)?;
        let mut up = Vec::with_capacity(net.up.len());
        for (k, block) in net.up.into_iter().enumerate() {
            up.push(block.load(weights, k, device)?);
        }
        let final_conv = load_conv(net.final_conv, weights, "unet.final", device)?;
        Ok(Self {
            start,
            up,
            final_conv,
            config,
        })
    }

    /// The architecture this graph was built for.
    #[must_use]
    pub fn config(&self) -> UnetConfig {
        self.config
    }

    /// Run the decoder.
    ///
    /// # Arguments
    /// - `inputs`: exactly `num_layers` tensors, **finest first**, each
    ///   `[1, C, h_l, w_l]`.
    ///
    /// # Errors
    /// Returns `Err` if the number of levels or a channel count is wrong.
    pub fn forward(&self, inputs: &[Tensor<4>]) -> Result<Tensor<4>, String> {
        let config = self.config();
        if inputs.len() != config.num_layers {
            return Err(format!(
                "expected {} pyramid levels, got {}",
                config.num_layers,
                inputs.len()
            ));
        }
        for (level, x) in inputs.iter().enumerate() {
            let channels = x.dims()[1];
            if channels != config.in_channels {
                return Err(format!(
                    "inputs[{level}] has {channels} channels, expected {}",
                    config.in_channels
                ));
            }
        }

        // Start block on the coarsest level; Networks.h:780 concatenates the
        // raw input back on, which is always the equal-size branch.
        let coarsest = inputs[config.num_layers - 1].clone();
        let conv_out = self.start.forward(coarsest.clone());
        let mut state = combine_bridge(coarsest, conv_out);

        for (k, block) in self.up.iter().enumerate() {
            state = block.forward(state, inputs[config.up_level(k)].clone());
        }
        Ok(self.final_conv.forward(state))
    }
}
