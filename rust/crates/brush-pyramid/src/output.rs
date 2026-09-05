//! What a forward pass returns, on the host.
//!
//! Module: `brush_pyramid::output`
//! Purpose: one shared result type for the CPU reference ([`crate::cpu`]) and
//!     for the GPU pass (`brush_pyramid::gpu`) read back to host memory, so the
//!     parity test compares both against the same Python `.npy` fixtures
//!     through the same code path.
//! Invariants:
//!     - `feature` is **channel-first** `(C, h_l, w_l)`, matching what
//!       `trippy.raster.ref_torch.split_layers` produces and what
//!       `torch.nn.Conv2d` (and therefore the U-Net that consumes this)
//!       expects. `t_final` and `n_used` are `(h_l, w_l)`.
//!     - Layers are ordered **finest first**: `layers[0]` is full resolution.
//!     - `t_final == 1.0` means nothing was drawn at that pixel — it is the
//!       coverage/honesty map, not a leftover.
//! Units: `feature` is in whatever the point features hold (linear RGB or
//!     learned channels); `t_final` is a dimensionless transmittance in
//!     `[0, 1]`; `n_used` is a fragment count.
//! Related docs: `docs/ARCHITECTURE.md` (forward data flow).

/// One pyramid layer's composited output.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerImage {
    /// Layer height in pixels.
    pub height: usize,
    /// Layer width in pixels.
    pub width: usize,
    /// Feature width `C`.
    pub channels: usize,
    /// `C * height * width` values, channel-first.
    pub feature: Vec<f32>,
    /// `height * width` transmittances left after compositing.
    pub t_final: Vec<f32>,
    /// `height * width` fragment counts; equals `max_frags` where the
    /// per-pixel list overflowed.
    pub n_used: Vec<u32>,
}

impl LayerImage {
    /// Value of channel `c` at pixel `(y, x)`.
    #[must_use]
    pub fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.feature[(c * self.height + y) * self.width + x]
    }
}

/// A whole rendered pyramid plus the fragment statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidImages {
    /// `L` layers, finest first.
    pub layers: Vec<LayerImage>,
    /// Total fragments that survived emission (matches the Python
    /// `aux["num_fragments"]`).
    pub num_fragments: u32,
    /// Fragments per layer (matches `aux["fragments_per_layer"]`).
    pub fragments_per_layer: Vec<u32>,
}

/// Worst-case differences between two renders of the same scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Discrepancy {
    /// Largest absolute difference in any feature channel.
    pub max_feature: f32,
    /// Largest absolute difference in `t_final`.
    pub max_t_final: f32,
    /// Number of pixels whose `n_used` differs.
    pub n_used_mismatches: usize,
}

impl PyramidImages {
    /// Compare against a reference render, e.g. the Python `.npy` fixtures.
    ///
    /// Feature values and `t_final` are compared with an absolute tolerance;
    /// `n_used` and the fragment counts are integers and must match exactly —
    /// an off-by-one there means the emission or the sort diverged, which a
    /// float tolerance would happily hide.
    ///
    /// # Arguments
    /// - `expected`: the reference render.
    /// - `tol`: absolute tolerance for `feature` and `t_final`.
    ///
    /// # Errors
    /// Returns `Err` with a message naming the first structural mismatch
    /// (layer count, shape, fragment counts) or the worst out-of-tolerance
    /// sample, including its layer, channel and pixel.
    pub fn compare(&self, expected: &Self, tol: f32) -> Result<Discrepancy, String> {
        if self.layers.len() != expected.layers.len() {
            return Err(format!(
                "layer count: got {}, expected {}",
                self.layers.len(),
                expected.layers.len()
            ));
        }
        if self.num_fragments != expected.num_fragments
            || self.fragments_per_layer != expected.fragments_per_layer
        {
            // Report both together: the per-layer split is what says whether a
            // point lost a whole pyramid layer (a layer-selection or gate bug)
            // or scattered corners (a bounds or alpha_min bug).
            return Err(format!(
                "fragment counts: got {} {:?}, expected {} {:?}",
                self.num_fragments,
                self.fragments_per_layer,
                expected.num_fragments,
                expected.fragments_per_layer
            ));
        }

        let mut worst = Discrepancy {
            max_feature: 0.0,
            max_t_final: 0.0,
            n_used_mismatches: 0,
        };
        let mut worst_note = String::new();

        for (l, (got, want)) in self.layers.iter().zip(&expected.layers).enumerate() {
            if got.height != want.height || got.width != want.width || got.channels != want.channels {
                return Err(format!(
                    "layer {l} shape: got {}x{}x{}, expected {}x{}x{}",
                    got.channels, got.height, got.width, want.channels, want.height, want.width
                ));
            }
            for (i, (&a, &b)) in got.feature.iter().zip(&want.feature).enumerate() {
                let diff = (a - b).abs();
                if diff > worst.max_feature {
                    worst.max_feature = diff;
                    let c = i / (got.height * got.width);
                    let rest = i % (got.height * got.width);
                    worst_note = format!(
                        "layer {l} channel {c} pixel ({}, {}): got {a}, expected {b}",
                        rest / got.width,
                        rest % got.width
                    );
                }
            }
            for (&a, &b) in got.t_final.iter().zip(&want.t_final) {
                worst.max_t_final = worst.max_t_final.max((a - b).abs());
            }
            worst.n_used_mismatches += got
                .n_used
                .iter()
                .zip(&want.n_used)
                .filter(|(a, b)| a != b)
                .count();
        }

        if worst.n_used_mismatches > 0 {
            return Err(format!(
                "n_used differs at {} pixels (max |feature| diff {:.3e})",
                worst.n_used_mismatches, worst.max_feature
            ));
        }
        if worst.max_feature > tol || worst.max_t_final > tol {
            return Err(format!(
                "max |feature| diff {:.3e}, max |t_final| diff {:.3e} exceed tol {tol:.1e}; worst at {worst_note}",
                worst.max_feature, worst.max_t_final
            ));
        }
        Ok(worst)
    }
}
