//! The Blend panel's state: how much of the frame is splat and how much is TRIPS.
//!
//! Module: `trips_viewer::blend`
//! Purpose: hybrid design A renders a scene from two things at once -- a
//!     Gaussian splat and the TRIPS point pyramid -- and a run trained with the
//!     blend gate (`trippy.hybrid.gate`) carries a per-pixel weight `g` saying
//!     which one it trusted where. This module is the viewer's half of that:
//!     the modes, the two sliders, and the arithmetic that turns them into one
//!     image. Nothing here touches Burn or a device; the composition itself
//!     lives in [`crate::renderer`] so this stays testable on the CPU and
//!     compilable for wasm.
//! Invariants:
//!     - Every mode except [`BlendMode::Trips`] needs a splat image for the
//!       frame being drawn. When there is none the renderer falls back to
//!       `Trips` and reports it ([`crate::renderer::FrameStats`]); it never
//!       invents a splat, and it never blends against black.
//!     - [`BlendMode::Gated`] additionally needs the gate head. A bundle whose
//!       weights are three-channel has no gate, so `Gated` is unavailable and
//!       the panel says so.
//!     - `mix` is stated as **0 = splat, 1 = TRIPS** (the task brief's own
//!       convention), which is the opposite sense to `g`. [`Blend::uniform_gate`]
//!       is the single place that inversion happens, so it cannot be done twice.
//! Units: `gate_scale` is a dimensionless multiplier in
//!     `[GATE_SCALE_MIN, GATE_SCALE_MAX]`; `mix` and `split` are fractions in
//!     [0, 1]; `split` is a fraction of the frame's **width**.
//! Related docs: `docs/USER_GUIDE.md` "Blend panel"; `trippy/hybrid/gate.py`;
//!     `docs/EXPERIMENTS.md` "The blend gate".

/// Smallest `gate_scale` the panel offers: the pure TRIPS path.
pub const GATE_SCALE_MIN: f32 = 0.0;

/// Largest `gate_scale` the panel offers. Matches
/// `trippy.constants.HYBRID_A_GATE_SCALE_MAX`, and 2 is enough that every pixel
/// whose trained gate is at least 0.5 saturates to pure splat.
pub const GATE_SCALE_MAX: f32 = 2.0;

/// What the Blend panel draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// The TRIPS network's own frame, with no splat mixed in at all. The
    /// default, and exactly what the viewer drew before the Blend panel
    /// existed, so a bundle with no `blend` block is unaffected.
    #[default]
    Trips,
    /// The Gaussian splat alone, at this view's own pose.
    Splat,
    /// `g * splat + (1 - g) * trips` with `g` the trained gate, scaled by
    /// [`Blend::gate_scale`]. What the run itself chose.
    Gated,
    /// A uniform mix, ignoring the trained gate: [`Blend::mix`] of TRIPS.
    Mix,
    /// Splat on the left of [`Blend::split`], TRIPS on the right, same pose.
    Split,
}

impl BlendMode {
    /// Cycle order for the panel's hotkey.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Trips => Self::Splat,
            Self::Splat => Self::Gated,
            Self::Gated => Self::Mix,
            Self::Mix => Self::Split,
            Self::Split => Self::Trips,
        }
    }

    /// Short label for the panel and the readout.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Trips => "TRIPS only",
            Self::Splat => "splat only",
            Self::Gated => "gated blend",
            Self::Mix => "manual mix",
            Self::Split => "split screen",
        }
    }

    /// Whether this mode needs a splat image to draw anything different.
    #[must_use]
    pub const fn needs_splat(self) -> bool {
        !matches!(self, Self::Trips)
    }

    /// Whether this mode needs the network's gate head.
    #[must_use]
    pub const fn needs_gate(self) -> bool {
        matches!(self, Self::Gated)
    }

    /// Parse a `--blend-mode` argument.
    ///
    /// # Errors
    /// Returns `Err` with the accepted spellings when `text` is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "trips" => Ok(Self::Trips),
            "splat" => Ok(Self::Splat),
            "gated" => Ok(Self::Gated),
            "mix" => Ok(Self::Mix),
            "split" => Ok(Self::Split),
            other => Err(format!(
                "unknown blend mode {other:?}; expected trips|splat|gated|mix|split"
            )),
        }
    }
}

/// The Blend panel's full state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blend {
    /// Which composition to draw.
    pub mode: BlendMode,
    /// Multiplier on the trained gate, clamped into
    /// `[GATE_SCALE_MIN, GATE_SCALE_MAX]` on use. Only read by
    /// [`BlendMode::Gated`].
    pub gate_scale: f32,
    /// **0 = splat, 1 = TRIPS.** Only read by [`BlendMode::Mix`].
    pub mix: f32,
    /// Fraction of the width where [`BlendMode::Split`] changes over.
    pub split: f32,
}

impl Default for Blend {
    /// TRIPS only, at the gate scale a bundle with no opinion implies.
    fn default() -> Self {
        Self {
            mode: BlendMode::Trips,
            gate_scale: 1.0,
            mix: 0.5,
            split: 0.5,
        }
    }
}

impl Blend {
    /// The panel's starting state for a bundle, honouring its own `gate_scale`.
    ///
    /// Deliberately starts in [`BlendMode::Trips`] even on a gate bundle: the
    /// first frame a viewer shows should be the one every metric in the run's
    /// report was measured on, and moving to the gated blend should be the
    /// user's decision, taken while looking at it.
    #[must_use]
    pub fn for_bundle(gate_scale: f32) -> Self {
        Self {
            gate_scale: gate_scale.clamp(GATE_SCALE_MIN, GATE_SCALE_MAX),
            ..Self::default()
        }
    }

    /// `gate_scale`, clamped to the range the panel offers.
    #[must_use]
    pub fn clamped_gate_scale(&self) -> f32 {
        self.gate_scale.clamp(GATE_SCALE_MIN, GATE_SCALE_MAX)
    }

    /// The constant gate [`BlendMode::Mix`] applies: `1 - mix`.
    ///
    /// The one place the "0 = splat, 1 = TRIPS" slider is inverted into the
    /// "1 = splat" weight the blend arithmetic uses.
    #[must_use]
    pub fn uniform_gate(&self) -> f32 {
        1.0 - self.mix.clamp(0.0, 1.0)
    }

    /// Column index where [`BlendMode::Split`] changes over, in `[0, width]`.
    #[must_use]
    pub fn split_column(&self, width: usize) -> usize {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let column = (self.split.clamp(0.0, 1.0) * width as f32).round() as usize;
        column.min(width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hotkey_cycle_visits_every_mode_and_returns() {
        let mut seen = vec![BlendMode::Trips];
        let mut mode = BlendMode::Trips;
        for _ in 0..5 {
            mode = mode.next();
            seen.push(mode);
        }
        assert_eq!(seen.len(), 6);
        assert_eq!(seen[5], BlendMode::Trips, "the cycle must return home");
        for candidate in [
            BlendMode::Trips,
            BlendMode::Splat,
            BlendMode::Gated,
            BlendMode::Mix,
            BlendMode::Split,
        ] {
            assert!(seen.contains(&candidate), "{candidate:?} is unreachable");
        }
    }

    #[test]
    fn only_trips_needs_neither_a_splat_nor_a_gate() {
        assert!(!BlendMode::Trips.needs_splat());
        assert!(!BlendMode::Trips.needs_gate());
        for mode in [BlendMode::Splat, BlendMode::Gated, BlendMode::Mix, BlendMode::Split] {
            assert!(mode.needs_splat(), "{mode:?}");
        }
        assert!(BlendMode::Gated.needs_gate());
        assert!(!BlendMode::Mix.needs_gate(), "a manual mix ignores the trained gate");
    }

    #[test]
    fn the_mix_slider_reads_zero_as_splat_and_one_as_trips() {
        let mut blend = Blend::default();
        blend.mix = 0.0;
        assert_eq!(blend.uniform_gate(), 1.0, "mix 0 is all splat, i.e. gate 1");
        blend.mix = 1.0;
        assert_eq!(blend.uniform_gate(), 0.0, "mix 1 is all TRIPS, i.e. gate 0");
        blend.mix = 0.25;
        assert_eq!(blend.uniform_gate(), 0.75);
    }

    #[test]
    fn the_mix_slider_is_clamped() {
        let mut blend = Blend::default();
        blend.mix = -3.0;
        assert_eq!(blend.uniform_gate(), 1.0);
        blend.mix = 7.0;
        assert_eq!(blend.uniform_gate(), 0.0);
    }

    #[test]
    fn the_gate_scale_is_clamped_to_the_documented_range() {
        let mut blend = Blend::default();
        blend.gate_scale = -1.0;
        assert_eq!(blend.clamped_gate_scale(), GATE_SCALE_MIN);
        blend.gate_scale = 99.0;
        assert_eq!(blend.clamped_gate_scale(), GATE_SCALE_MAX);
        assert_eq!(Blend::for_bundle(1.5).gate_scale, 1.5);
        assert_eq!(Blend::for_bundle(9.0).gate_scale, GATE_SCALE_MAX);
    }

    #[test]
    fn a_bundle_opens_on_the_frame_its_metrics_describe() {
        assert_eq!(Blend::for_bundle(1.0).mode, BlendMode::Trips);
    }

    #[test]
    fn the_split_column_spans_the_whole_width_and_never_overruns() {
        let mut blend = Blend::default();
        blend.split = 0.0;
        assert_eq!(blend.split_column(100), 0);
        blend.split = 1.0;
        assert_eq!(blend.split_column(100), 100);
        blend.split = 0.5;
        assert_eq!(blend.split_column(100), 50);
        blend.split = 2.0;
        assert_eq!(blend.split_column(100), 100, "clamped, never past the frame");
    }

    #[test]
    fn blend_modes_round_trip_through_their_argument_spelling() {
        for mode in [
            BlendMode::Trips,
            BlendMode::Splat,
            BlendMode::Gated,
            BlendMode::Mix,
            BlendMode::Split,
        ] {
            let text = match mode {
                BlendMode::Trips => "trips",
                BlendMode::Splat => "splat",
                BlendMode::Gated => "gated",
                BlendMode::Mix => "mix",
                BlendMode::Split => "split",
            };
            assert_eq!(BlendMode::parse(text).expect("parses"), mode);
        }
        assert!(BlendMode::parse("sideways").is_err());
    }
}
