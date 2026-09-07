//! The shade-cloud finder: the audit's own rule, live, as a `pointset` selection.
//!
//! Module: `trips_viewer::edit::shade`
//! Purpose: `docs/EDITOR.md` §3 "2. Shade-cloud finder (E2)" — find the dark,
//!     low-confidence points sitting in front of the geometry the shade frames
//!     actually see, and hand them to the Regions panel as a `pointset` region
//!     Jordan can fade or delete. This is a **port of the arithmetic** in
//!     `trippy/train/prune.py` (`in_region`, `luminance`, `dark_mass_stats`,
//!     `confidence_drop_mask`'s absolute branch), not a call into it: the four
//!     thresholds are sliders, so the test has to re-run at interaction speed
//!     with no subprocess in the loop.
//! Invariants:
//!     - Everything a slider moves is computed here; everything a slider cannot
//!       move comes from the sidecar. The one number that needs the COLMAP
//!       sparse model — each frame's median observed depth `d` — is not a
//!       function of any threshold, so it is precomputed once into
//!       [`SHADE_VIEWS_FILENAME`] (`trippy.edit.golden.write_shade_views`) and
//!       read here. `znear = znear_frac * d` and `zfar = zfar_frac * d` are then
//!       re-derived per slider move, exactly as `prune.build_shade_region` bakes
//!       them.
//!     - Without that sidecar the finder still runs, with `d` estimated from the
//!       bundle's OWN points ([`ShadeView::estimate_median_depth`]). That is a
//!       different number from COLMAP's sparse median, so
//!       [`ShadeViews::depth_is_estimated`] is true and the panel says so —
//!       an estimate that announces itself, never a silent substitution.
//!     - `in_region`'s half-open pixel test (`0 <= u < W`) and strict depth test
//!       (`znear < z < zfar`) are the audit's, kept literally.
//!     - The selection is `inside AND dark AND (conf < conf_threshold)` — the
//!       same AND `prune.shade_prune_keep_mask` negates, without its
//!       `min_points` floor (a finder has no reason to guarantee a minimum
//!       selection size). `mode="relative"` is deliberately NOT ported: it needs
//!       each point's `init_conf`, which no bundle carries.
//! Units: world units for positions/depths; pixels for intrinsics; luminance and
//!     confidence are dimensionless in `[0, 1]`.
//! Related docs: `docs/EDITOR.md` §3; `docs/EXPERIMENTS.md` "Shade audit";
//!     `trippy/train/prune.py`; `trippy/edit/shade_finder.py`.

use serde_json::Value;

/// The sidecar the shade finder reads, next to `bundle.json`.
pub const SHADE_VIEWS_FILENAME: &str = "shade_views.json";

/// The sidecar's `"format"` field.
pub const SHADE_VIEWS_FORMAT: &str = "trippy-shade-views-1";

/// Rec.709 luma weights — `trippy.constants.REC709_LUMA_WEIGHTS`.
pub const REC709_LUMA_WEIGHTS: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// `trippy.constants.SHADE_PRUNE_DEFAULT_ZNEAR_FRAC`.
pub const DEFAULT_ZNEAR_FRAC: f64 = 0.05;
/// `trippy.constants.SHADE_PRUNE_DEFAULT_ZFAR_FRAC`.
pub const DEFAULT_ZFAR_FRAC: f64 = 0.50;
/// `trippy.constants.SHADE_PRUNE_DEFAULT_LUM_THRESHOLD`.
pub const DEFAULT_LUM_THRESHOLD: f64 = 0.25;
/// `trippy.constants.SHADE_PRUNE_DEFAULT_CONF_THRESHOLD`.
pub const DEFAULT_CONF_THRESHOLD: f64 = 0.5;

/// One shade frame's contribution to the audit region — the twin of
/// `trippy.train.prune.ShadeView`, minus the two fields the sliders derive.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadeView {
    /// Image filename, e.g. `"IMG_3830.jpg"`.
    pub name: String,
    /// Row-major `3x3` world->camera rotation. Row 2 dotted with `p - c` is depth.
    pub r: [f64; 9],
    /// Camera centre in world coordinates, `-R^T t`.
    pub c: [f64; 3],
    /// Median camera-space depth of this frame's own observed points, world units.
    pub d: f64,
    /// Pinhole intrinsics in pixels of the as-captured image.
    pub fx: f64,
    /// See [`Self::fx`].
    pub fy: f64,
    /// See [`Self::fx`].
    pub cx: f64,
    /// See [`Self::fx`].
    pub cy: f64,
    /// As-captured image width, pixels.
    pub width: f64,
    /// As-captured image height, pixels.
    pub height: f64,
}

impl ShadeView {
    /// Row `i` of `R` dotted with `p - c` (`i = 0` right, `1` down, `2` forward).
    fn axis(&self, i: usize, rel: [f64; 3]) -> f64 {
        rel[0].mul_add(
            self.r[3 * i],
            rel[1].mul_add(self.r[3 * i + 1], rel[2] * self.r[3 * i + 2]),
        )
    }

    /// Build a view from a bundle's own camera (`crate::bundle::BundleView`) with
    /// `d` estimated from `xyz` — the no-sidecar fallback.
    ///
    /// `d` is the median camera-space depth of the points this camera sees in
    /// front of it, which stands in for `prune.build_shade_region`'s median over
    /// the frame's COLMAP *observations*. The two agree in spirit and not in
    /// value; see the module's second invariant.
    ///
    /// # Arguments
    /// - `xyz`: flat `(N, 3)` world positions to estimate from.
    #[must_use]
    pub fn estimate_median_depth(&self, xyz: &[f64]) -> f64 {
        let mut depths: Vec<f64> = xyz
            .chunks_exact(3)
            .filter_map(|p| {
                let rel = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
                let z = self.axis(2, rel);
                (z > 0.0).then_some(z)
            })
            .collect();
        if depths.is_empty() {
            return 1.0;
        }
        depths.sort_by(f64::total_cmp);
        let mid = depths.len() / 2;
        if depths.len() % 2 == 0 {
            f64::midpoint(depths[mid - 1], depths[mid])
        } else {
            depths[mid]
        }
    }

    /// Parse one `"views"` entry of [`SHADE_VIEWS_FILENAME`].
    ///
    /// # Errors
    /// Returns `Err` naming the first field that is missing or the wrong shape.
    pub fn from_json(doc: &Value) -> Result<Self, String> {
        let num = |key: &str| -> Result<f64, String> {
            doc.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("shade view: {key:?} must be a number"))
        };
        let vec = |key: &str, n: usize| -> Result<Vec<f64>, String> {
            let array = doc
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("shade view: {key:?} must be an array of {n} numbers"))?;
            if array.len() != n {
                return Err(format!("shade view: {key:?} must have {n} numbers"));
            }
            array
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("shade view: {key:?} must be numbers"))
                })
                .collect()
        };
        let r = vec("r", 9)?;
        let c = vec("c", 3)?;
        Ok(Self {
            name: doc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            r: [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]],
            c: [c[0], c[1], c[2]],
            d: num("d")?,
            fx: num("fx")?,
            fy: num("fy")?,
            cx: num("cx")?,
            cy: num("cy")?,
            width: num("width")?,
            height: num("height")?,
        })
    }
}

/// The shade frames the finder tests against, and where they came from.
#[derive(Debug, Clone)]
pub struct ShadeViews {
    /// One per shade frame, in the order the frame list named them.
    pub views: Vec<ShadeView>,
    /// True when `d` was estimated from the bundle's own points rather than read
    /// from the COLMAP-derived sidecar. The panel must say so.
    pub depth_is_estimated: bool,
    /// Where the frame list came from, for the panel's provenance line.
    pub source: String,
    /// The `znear_frac` the sidecar recorded — the panel's opening slider
    /// position, not a baked-in plane.
    pub znear_frac: f64,
    /// The `zfar_frac` the sidecar recorded.
    pub zfar_frac: f64,
}

impl Default for ShadeViews {
    fn default() -> Self {
        Self {
            views: Vec::new(),
            depth_is_estimated: false,
            source: String::new(),
            znear_frac: DEFAULT_ZNEAR_FRAC,
            zfar_frac: DEFAULT_ZFAR_FRAC,
        }
    }
}

impl ShadeViews {
    /// Read `<bundle>/shade_views.json`, or `Ok(None)` when there is none.
    ///
    /// The sidecar is written by `trippy.edit.golden.write_shade_views` (see
    /// `docs/USER_GUIDE.md` "Shade finder"); its absence is not an error, it is
    /// the ordinary state of a bundle nobody has run the precompute on.
    ///
    /// # Errors
    /// Returns `Err` when the file exists but does not parse, or declares a
    /// format this build does not read.
    pub fn load(dir: &std::path::Path) -> Result<Option<Self>, String> {
        let path = dir.join(SHADE_VIEWS_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let doc: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let format = doc
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or(SHADE_VIEWS_FORMAT);
        if format != SHADE_VIEWS_FORMAT {
            return Err(format!(
                "{}: format {format:?}, expected {SHADE_VIEWS_FORMAT:?}",
                path.display()
            ));
        }
        let views = doc
            .get("views")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{}: no \"views\" array", path.display()))?
            .iter()
            .map(ShadeView::from_json)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Some(Self {
            views,
            depth_is_estimated: false,
            source: path.display().to_string(),
            znear_frac: doc
                .get("znear_frac")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_ZNEAR_FRAC),
            zfar_frac: doc
                .get("zfar_frac")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_ZFAR_FRAC),
        }))
    }
}

/// The four numbers the panel's sliders hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Near plane as a fraction of each frame's own median observed depth.
    pub znear_frac: f64,
    /// Far plane, same units.
    pub zfar_frac: f64,
    /// "Dark" Rec.709 luminance cutoff.
    pub lum_threshold: f64,
    /// Absolute effective-confidence cutoff (`prune.confidence_drop_mask`,
    /// `mode="absolute"`).
    pub conf_threshold: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            znear_frac: DEFAULT_ZNEAR_FRAC,
            zfar_frac: DEFAULT_ZFAR_FRAC,
            lum_threshold: DEFAULT_LUM_THRESHOLD,
            conf_threshold: DEFAULT_CONF_THRESHOLD,
        }
    }
}

/// Rec.709 luminance of one linear RGB triple — `prune.luminance`'s weights.
#[must_use]
pub fn luminance(rgb: [f64; 3]) -> f64 {
    rgb[0].mul_add(
        REC709_LUMA_WEIGHTS[0],
        rgb[1].mul_add(REC709_LUMA_WEIGHTS[1], rgb[2] * REC709_LUMA_WEIGHTS[2]),
    )
}

/// Which points sit in the union of the shade frames' near depth slabs.
///
/// A port of `trippy.train.prune.in_region`, with `znear`/`zfar` re-derived from
/// `znear_frac`/`zfar_frac` and each view's own `d` so the two sliders are live.
///
/// # Arguments
/// - `views`: the shade frames.
/// - `xyz`: flat `(N, 3)` world positions.
/// - `znear_frac`, `zfar_frac`: the depth slab, as fractions of `view.d`.
///
/// # Returns
/// `(N,)` booleans: in the union of the slabs, projected inside the image, and
/// finite.
///
/// # Panics
/// Panics if `xyz.len()` is not a multiple of 3.
#[must_use]
pub fn in_region(views: &[ShadeView], xyz: &[f64], znear_frac: f64, zfar_frac: f64) -> Vec<bool> {
    assert!(xyz.len() % 3 == 0, "xyz must be a flat (N, 3) array");
    let n = xyz.len() / 3;
    let mut inside = vec![false; n];
    for v in views {
        let znear = znear_frac * v.d;
        let zfar = zfar_frac * v.d;
        for (i, hit) in inside.iter_mut().enumerate() {
            if *hit {
                continue;
            }
            let p = [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
            if !p.iter().all(|c| c.is_finite()) {
                continue;
            }
            let rel = [p[0] - v.c[0], p[1] - v.c[1], p[2] - v.c[2]];
            let z = v.axis(2, rel);
            // The audit's own strict depth test.
            if !(z > znear && z < zfar) {
                continue;
            }
            let u = v.fx.mul_add(v.axis(0, rel) / z, v.cx);
            let w = v.fy.mul_add(v.axis(1, rel) / z, v.cy);
            // The audit's own half-open pixel test.
            if u >= 0.0 && u < v.width && w >= 0.0 && w < v.height {
                *hit = true;
            }
        }
    }
    inside
}

/// What the finder found: the selection, and the audit numbers next to it.
#[derive(Debug, Clone, Default)]
pub struct ShadeSelection {
    /// Indices into the TRIPS cloud's own row order — a `pointset` region's `point_ids`.
    pub point_ids: Vec<u32>,
    /// How many points fell in the audit region at all.
    pub n_in_region: usize,
    /// `sum(conf)` over the region — `prune.dark_mass_stats`'s `mass_in_region`.
    pub mass_in_region: f64,
    /// `sum(conf)` over the dark points of the region.
    pub dark_mass: f64,
    /// `dark_mass / max(mass_in_region, 1e-9)` — the audit's headline number.
    pub dark_mass_fraction: f64,
}

/// Run the finder over one point cloud.
///
/// # Arguments
/// - `views`: the shade frames.
/// - `xyz`: flat `(N, 3)` world positions.
/// - `rgb`: flat `(N, 3)` base colour, already clipped to `[0, 1]` (the
///   exporter's `clip(feat[:, :3], 0, 1)`, which is what the audit reads back).
/// - `conf`: `(N,)` effective confidence in `(0, 1)`.
/// - `t`: the slider values.
///
/// # Panics
/// Panics if `xyz`/`rgb` are not flat `(N, 3)` arrays of the same `N` as `conf`.
#[must_use]
pub fn find(
    views: &[ShadeView],
    xyz: &[f64],
    rgb: &[f64],
    conf: &[f64],
    t: Thresholds,
) -> ShadeSelection {
    let n = conf.len();
    assert!(xyz.len() == 3 * n, "xyz must be (N, 3) with N = conf.len()");
    assert!(rgb.len() == 3 * n, "rgb must be (N, 3) with N = conf.len()");
    let inside = in_region(views, xyz, t.znear_frac, t.zfar_frac);

    let mut selection = ShadeSelection::default();
    for i in 0..n {
        if !inside[i] {
            continue;
        }
        selection.n_in_region += 1;
        selection.mass_in_region += conf[i];
        let lum = luminance([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]]);
        if lum < t.lum_threshold {
            selection.dark_mass += conf[i];
            // `mode="absolute"`: TRIPS's own literal rule (`train.cpp:846-851`).
            if conf[i] < t.conf_threshold {
                if let Ok(id) = u32::try_from(i) {
                    selection.point_ids.push(id);
                }
            }
        }
    }
    selection.dark_mass_fraction = selection.dark_mass / selection.mass_in_region.max(1e-9);
    selection
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at the origin looking down +Z, 100 px focal, 64x64 image, d = 4.
    fn view() -> ShadeView {
        ShadeView {
            name: "synthetic.jpg".to_owned(),
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            c: [0.0, 0.0, 0.0],
            d: 4.0,
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 32.0,
            width: 64.0,
            height: 64.0,
        }
    }

    #[test]
    fn the_depth_slab_is_a_fraction_of_the_frames_own_median_depth() {
        let v = view();
        // znear = 0.05 * 4 = 0.2, zfar = 0.5 * 4 = 2.0.
        let xyz = [
            0.0, 0.0, 0.1, // in front of znear: out
            0.0, 0.0, 1.0, // in the slab: in
            0.0, 0.0, 3.0, // past zfar: out
            0.0, 0.0, -1.0, // behind the camera: out
        ];
        let inside = in_region(std::slice::from_ref(&v), &xyz, 0.05, 0.5);
        assert_eq!(inside, vec![false, true, false, false]);

        // Widening zfar_frac brings the far point in — this is what the slider does.
        let inside = in_region(&[v], &xyz, 0.05, 1.0);
        assert_eq!(inside, vec![false, true, true, false]);
    }

    #[test]
    fn a_point_that_projects_outside_the_image_is_not_in_the_region() {
        // At z = 1, x = 0.32 lands on u = 64.0, which the half-open test excludes.
        let xyz = [0.32, 0.0, 1.0, 0.31, 0.0, 1.0];
        let inside = in_region(&[view()], &xyz, 0.05, 0.5);
        assert_eq!(inside, vec![false, true]);
    }

    #[test]
    fn the_selection_is_in_region_and_dark_and_low_confidence() {
        let v = [view()];
        // Four points in the slab: bright/high, bright/low, dark/high, dark/low.
        let xyz = [
            0.0, 0.0, 1.0, //
            0.05, 0.0, 1.0, //
            -0.05, 0.0, 1.0, //
            0.0, 0.05, 1.0,
        ];
        let rgb = [
            0.9, 0.9, 0.9, //
            0.9, 0.9, 0.9, //
            0.02, 0.02, 0.02, //
            0.02, 0.02, 0.02,
        ];
        let conf = [0.9, 0.1, 0.9, 0.1];
        let found = find(&v, &xyz, &rgb, &conf, Thresholds::default());
        assert_eq!(found.point_ids, vec![3]);
        assert_eq!(found.n_in_region, 4);
        // dark mass = 0.9 + 0.1 over a total of 2.0.
        assert!((found.dark_mass - 1.0).abs() < 1e-12);
        assert!((found.dark_mass_fraction - 0.5).abs() < 1e-12);
    }

    #[test]
    fn luminance_is_rec709() {
        assert!((luminance([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
        assert!((luminance([0.0, 1.0, 0.0]) - 0.7152).abs() < 1e-12);
    }

    #[test]
    fn an_estimated_median_depth_ignores_points_behind_the_camera() {
        let v = view();
        let xyz = [0.0, 0.0, -5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 6.0];
        assert!((v.estimate_median_depth(&xyz) - 4.0).abs() < 1e-12);
    }
}
