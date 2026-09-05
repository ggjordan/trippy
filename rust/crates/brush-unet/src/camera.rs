//! `NeuralCamera`: per-image exposure / white balance / vignette / response.
//!
//! Module: `brush_unet::camera`
//! Purpose: a Burn port of `trippy.net.camera_model.NeuralCamera`, itself a
//!     port of TRIPS's `NeuralCameraImpl` (NeuralCamera.cpp:258-390). Takes
//!     the U-Net's raw `[1, 3, H, W]` output and produces the displayed
//!     frame.
//! Invariants:
//!     - Stage order is fixed and is the source of the whole result:
//!       exposure, then white balance, then vignette, then response LUT
//!       (`clamp(x, 0, 1)` when the LUT is disabled).
//!     - **Eval-mode semantics only.** TRIPS's "leaky" extrapolation outside
//!       `[0, 1]` is training-only (`self.training` in the Python port) and
//!       is deliberately not implemented; the parity target runs `eval()`.
//!     - The uv grid puts the image centre at `(0, 0)` and the corner *pixel
//!       centres* at `+-1` (`texel = pixel / (size - 1)`,
//!       Dataset.cpp:11-30), i.e. `linspace(-1, 1, n)` per axis. Only the
//!       u/x channel gets the aspect correction (NeuralCamera.cpp:33).
//!     - Rolling shutter is not ported (off by default in TRIPS; see
//!       `docs/LIMITATIONS.md`).
//! Units: exposure is EV (applied as `x * 2**-ev`); white balance, vignette
//!     and LUT values are dimensionless.
//! Related docs: `docs/TRIPS_REFERENCE.md` Sec. 6/6a;
//!     `trippy/net/camera_model.py`.

use burn::tensor::{Device, Int, Tensor};

use crate::config::CameraConfig;
use crate::net::upload;
use crate::weights::Weights;

/// Number of colour channels the tone mapper operates on.
const RGB: usize = 3;

/// Terms in the vignette's radial polynomial: `1 + p0*r2 + p1*r4 + p2*r6`.
const VIGNETTE_TERMS: usize = 3;

/// The tone mapper, with every learned tensor resident on the device.
///
/// Not a `#[derive(Module)]` type on purpose: nothing here is ever trained
/// in the viewer, the per-image parameters are indexed rather than convolved,
/// and keeping it a plain struct means the response LUT can stay a host-side
/// `Vec<Tensor<2>>` slice-per-channel without fighting the record machinery.
#[derive(Debug, Clone)]
pub struct NeuralCamera {
    config: CameraConfig,
    /// `[M]` per-image EV, or `None`.
    exposure: Option<Vec<f32>>,
    /// `[M][3]` per-image white-balance gains, or `None`.
    white_balance: Option<Vec<[f32; RGB]>>,
    /// `[p0, p1, p2]` radial polynomial coefficients, or `None`.
    vignette_params: Option<[f32; VIGNETTE_TERMS]>,
    /// `(u, v)` centre of the vignette in uv space.
    vignette_center: [f32; 2],
    /// `[1, O * P]` response LUT control points, channel-major, or `None`.
    response: Option<Tensor<2>>,
}

impl NeuralCamera {
    /// Build from a parsed weight file.
    ///
    /// # Errors
    /// Returns `Err` if the file has no camera block or a tensor is missing.
    pub fn load(weights: &Weights, device: &Device) -> Result<Self, String> {
        let config = weights
            .camera
            .ok_or_else(|| "weight file has no camera block (has_camera = 0)".to_string())?;

        let exposure = if config.enable_exposure {
            Some(weights.get("camera.exposure")?.data.clone())
        } else {
            None
        };
        let white_balance = if config.enable_white_balance {
            let raw = &weights.get("camera.white_balance")?.data;
            Some(
                raw.chunks_exact(RGB)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let (vignette_params, vignette_center) = if config.enable_vignette {
            let p = &weights.get("camera.vignette_params")?.data;
            let c = &weights.get("camera.vignette_center")?.data;
            (Some([p[0], p[1], p[2]]), [c[0], c[1]])
        } else {
            (None, [0.0, 0.0])
        };
        let response = if config.enable_response {
            let host = weights.get("camera.response")?;
            // `[O, P]` -> `[1, O*P]`: one flat LUT the per-channel gather
            // indexes with `channel * P + i`.
            let flat: Tensor<2> = upload::<2>(host, device)?
                .reshape([1, config.response_params * weights.unet.out_channels]);
            Some(flat)
        } else {
            None
        };

        Ok(Self {
            config,
            exposure,
            white_balance,
            vignette_params,
            vignette_center,
            response,
        })
    }

    /// The tone mapper's shape and enable flags.
    #[must_use]
    pub fn config(&self) -> CameraConfig {
        self.config
    }

    /// Apply the tone mapper to `x`, `[1, 3, H, W]`, for image `frame`.
    ///
    /// # Arguments
    /// - `x`: the U-Net's raw output.
    /// - `frame`: index into the per-image exposure / white-balance tables.
    ///
    /// # Errors
    /// Returns `Err` on a wrong channel count or an out-of-range `frame`.
    pub fn forward(&self, x: Tensor<4>, frame: usize) -> Result<Tensor<4>, String> {
        let [batch, channels, height, width] = x.dims();
        if channels != RGB {
            return Err(format!("expected {RGB} channels, got {channels}"));
        }
        if batch != 1 {
            return Err(format!("batch must be 1 (one frame index), got {batch}"));
        }
        let device = x.device();
        let mut out = x;

        if let Some(exposure) = &self.exposure {
            let ev = *exposure
                .get(frame)
                .ok_or_else(|| format!("frame {frame} out of range ({} images)", exposure.len()))?;
            out = out.mul_scalar((-ev).exp2());
        }

        if let Some(white_balance) = &self.white_balance {
            let gains = *white_balance
                .get(frame)
                .ok_or_else(|| format!("frame {frame} out of range ({} images)", white_balance.len()))?;
            let wb = Tensor::<1>::from_floats(gains, &device).reshape([1, RGB, 1, 1]);
            out = out * wb;
        }

        if let Some(params) = self.vignette_params {
            out = out * self.vignette(height, width, params, &device);
        }

        out = match &self.response {
            Some(lut) => self.apply_response(out, lut),
            // NeuralCamera.cpp:388 -- the disabled branch is a hard clamp.
            None => out.clamp(0.0, 1.0),
        };
        Ok(out)
    }

    /// The `[1, 1, H, W]` multiplicative vignette factor
    /// `1 + p0*r2 + p1*r4 + p2*r6`.
    ///
    /// `r` is measured from `vignette_center` in the uv frame, with the u
    /// axis scaled by the image aspect ratio (NeuralCamera.cpp:27-40).
    fn vignette(
        &self,
        height: usize,
        width: usize,
        params: [f32; VIGNETTE_TERMS],
        device: &Device,
    ) -> Tensor<4> {
        // The aspect ratio is the one the parameters were FITTED at, not the
        // one being rendered: `NeuralCameraImpl` builds `VignetteNet` from the
        // dataset's image size once. Rendering at another size and using the
        // render's own aspect would silently change the falloff.
        let aspect = self.config.image_width as f32 / self.config.image_height as f32;
        let u = linspace_centered(width, device).reshape([1, 1, 1, width]);
        let v = linspace_centered(height, device).reshape([1, 1, height, 1]);
        let du = (u - self.vignette_center[0]).mul_scalar(aspect);
        let dv = v - self.vignette_center[1];
        let r2 = du.clone() * du + dv.clone() * dv;
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();
        r2.mul_scalar(params[0])
            .add(r4.mul_scalar(params[1]))
            .add(r6.mul_scalar(params[2]))
            .add_scalar(1.0)
    }

    /// The per-channel response LUT, evaluated by linear interpolation.
    ///
    /// Equivalent to TRIPS's `grid_sample(response, grid, bilinear,
    /// padding_mode = border, align_corners = true)` on a `(1, C, 1, P)`
    /// texture: with `align_corners = true` the sample coordinate is
    /// `s = x * (P - 1)`, and `padding_mode = border` clips the *coordinate*
    /// before the two taps are read, which is exactly `clamp(x, 0, 1)`
    /// followed by an unclamped lerp. The LUT is flattened to `[1, C*P]` and
    /// each channel's two taps are read with `gather`, offsetting the index by
    /// `c * P`, so no per-channel sub-tensor has to be materialised.
    fn apply_response(&self, x: Tensor<4>, lut: &Tensor<2>) -> Tensor<4> {
        let [_, _, height, width] = x.dims();
        let points = self.config.response_params;
        let last = (points - 1) as f32;
        let pixels = height * width;

        let mut channels = Vec::with_capacity(RGB);
        for c in 0..RGB {
            let base = (c * points) as i32;
            let value = x
                .clone()
                .slice_dim(1, c..c + 1)
                .reshape([1, pixels])
                .clamp(0.0, 1.0)
                .mul_scalar(last);
            let low = value.clone().floor();
            let frac = value - low.clone();
            let index_low: Tensor<2, Int> = low.int();
            let index_high = index_low
                .clone()
                .add_scalar(1)
                .clamp(0, (points - 1) as i32)
                .add_scalar(base);
            let index_low = index_low.add_scalar(base);
            let tap_low = lut.clone().gather(1, index_low);
            let tap_high = lut.clone().gather(1, index_high);
            let interpolated = tap_low.clone() + (tap_high - tap_low) * frac;
            channels.push(interpolated.reshape([1, 1, height, width]));
        }
        Tensor::cat(channels, 1)
    }
}

/// `linspace(-1, 1, n)` as a `[n]` tensor — TRIPS's `InitialUVImage` spacing,
/// which puts the first and last pixel centres exactly on `-1` and `+1`.
fn linspace_centered(n: usize, device: &Device) -> Tensor<1> {
    if n == 1 {
        return Tensor::<1>::from_floats([-1.0], device);
    }
    let step = 2.0 / (n - 1) as f32;
    let values: Vec<f32> = (0..n).map(|i| -1.0 + step * i as f32).collect();
    Tensor::<1>::from_data(burn::tensor::TensorData::new(values, [n]), device)
}
