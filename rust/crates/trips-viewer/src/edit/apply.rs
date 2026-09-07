//! Turning composed weights into the two point sets the rasteriser renders.
//!
//! Module: `trips_viewer::edit::apply`
//! Purpose: `docs/EDITOR.md` §2's "Deletions" and "The weight, precisely",
//!     made concrete. Given a bundle's [`PointSet`] and the
//!     [`ComposedWeights`] the regions produced, build:
//!     1. the **kept** point set — deleted rows removed before rasterisation,
//!        the same `index_select`-and-rebuild
//!        `trippy.train.trainer.Trainer._apply_keep_mask` performs at an epoch
//!        boundary, done here at edit time;
//!     2. the **probe** point set — the same surviving rows with a
//!        three-channel feature vector `[w_edit, touched, 1]`, rasterised as
//!        its own tiny pyramid pass so that
//!        `w_pix = w_sum / (coverage + eps)` at every pixel is exactly the
//!        alpha-weighted average of `w_edit` over the points that covered it.
//! Invariants:
//!     - The two sets share `xyz`, `size` and `conf` row for row, so the two
//!       pyramid passes place identical fragments with identical alphas and the
//!       probe's third channel really is the main pass's own coverage
//!       `sum(T_i a_i) = 1 - t_final`.
//!     - The probe's `background` must be all zeros (see
//!         [`PROBE_BACKGROUND`]): a pixel no point covers has to come back with
//!       coverage 0, which is what makes `touched / (coverage + eps)` fall to 0
//!       and hand the pixel back to whatever the Blend panel and the gate had
//!       already decided. A non-zero background here would silently turn every
//!       empty pixel into an edited one.
//!     - `C = 3` is deliberate: `brush_pyramid::params::SUPPORTED_CHANNELS` is
//!       `[3, 4, 8]`, so the probe compiles the rasteriser's *existing* 3-channel
//!       pipeline and needs no kernel change, no new fixture and no widening of
//!       the U-Net's input. `docs/EDITOR.md` §2's "a sibling buffer rendered as
//!       its own tiny pyramid pass" is this.
//!     - `delete` removes rows; `blend`/`fade` do **not** touch `conf`. Scaling
//!       alpha by `w_edit` as well would dim the TRIPS half *and* then blend the
//!       splat over it, counting the same edit twice; see the "why not alpha"
//!       note on [`edited_points`].
//! Units: world units; weights dimensionless in `[0, 1]`.
//! Related docs: `docs/EDITOR.md` §2; `docs/decisions/ADR-0007-viewer-editing.md`.

use brush_pyramid::scene::PointSet;

use super::weights::ComposedWeights;

/// Feature width of the probe cloud: `[w_edit, touched, coverage]`.
pub const PROBE_CHANNELS: usize = 3;

/// The probe pass's background. Must stay all zeros — see the module invariants.
pub const PROBE_BACKGROUND: [f32; PROBE_CHANNELS] = [0.0, 0.0, 0.0];

/// Guard against dividing the probe's channels by a coverage of zero.
///
/// Coverage is `sum(T_i a_i) <= 1`, and both numerators are bounded above by it,
/// so `x / (coverage + EPS)` tends to 0 as coverage does — which is precisely
/// the "no point covered this pixel, so no region has an opinion about it"
/// fallback. No comparison or mask is needed.
pub const COVERAGE_EPS: f32 = 1e-6;

/// What [`edited_points`] produced, and the numbers the HUD reports.
pub struct EditedPoints {
    /// The bundle's points with every `delete`-op row removed.
    pub points: PointSet,
    /// The three-channel probe cloud, or `None` when no enabled `blend`/`fade`
    /// region claims anything (in which case the second pyramid pass is skipped
    /// entirely and the frame costs exactly what it did before).
    pub probe: Option<PointSet>,
    /// Rows a `delete` region removed.
    pub num_deleted: usize,
    /// Surviving rows an enabled `blend`/`fade` region claims.
    pub num_touched: usize,
}

/// Build the kept and probe point sets from composed weights.
///
/// # Why the weight does not multiply `conf`
///
/// The obvious reading of "the weight multiplies the point's alpha" makes a
/// `mix = 0` region invisible in the TRIPS pass — and then the per-pixel blend,
/// seeing no coverage there, has nothing to mix the splat *into*. The two
/// mechanisms would cancel. So `delete` removes rows (a weight of 0 that really
/// does mean "not rendered"), and `blend`/`fade` leave alpha alone and let the
/// probe's per-pixel `w_edit` decide how much splat replaces the TRIPS pixel.
/// `docs/EDITOR.md` §2's "the composited value at a pixel is `w_edit`'s
/// alpha-weighted average" only means anything if the alphas are the untouched
/// ones.
///
/// # Arguments
/// - `base`: the bundle's own point set.
/// - `composed`: [`super::weights::compose_trips_weights`] over `base`'s `xyz`.
///
/// # Returns
/// `Err` when `composed` does not describe `base` (a caller bug), or when the
/// probe's shape is rejected by [`PointSet::new`].
///
/// # Errors
/// See above.
pub fn edited_points(base: &PointSet, composed: &ComposedWeights) -> Result<EditedPoints, String> {
    let n = base.len();
    if composed.weight.len() != n || composed.delete_mask.len() != n {
        return Err(format!(
            "composed weights describe {} points but the cloud has {n}",
            composed.weight.len()
        ));
    }
    let num_deleted = composed.num_deleted();
    let channels = base.num_channels;

    // 1. The kept set. `index_select` in spirit; a straight row copy in fact,
    //    because `PointSet` is host-side `Vec<f32>`s.
    let (points, kept_rows) = if num_deleted == 0 {
        (base.clone(), None)
    } else {
        let keep: Vec<usize> = (0..n).filter(|i| !composed.delete_mask[*i]).collect();
        let mut xyz = Vec::with_capacity(keep.len() * 3);
        let mut size = Vec::with_capacity(keep.len());
        let mut feat = Vec::with_capacity(keep.len() * channels);
        let mut conf = Vec::with_capacity(keep.len());
        for &i in &keep {
            xyz.extend_from_slice(&base.xyz[3 * i..3 * i + 3]);
            size.push(base.size[i]);
            feat.extend_from_slice(&base.feat[i * channels..(i + 1) * channels]);
            conf.push(base.conf[i]);
        }
        (PointSet::new(xyz, size, feat, conf, channels)?, Some(keep))
    };

    // 2. The probe set, over the SAME surviving rows.
    let touched_rows: Vec<usize> = match &kept_rows {
        Some(keep) => keep.clone(),
        None => (0..n).collect(),
    };
    let num_touched = touched_rows
        .iter()
        .filter(|i| composed.touched[**i] > 0.0)
        .count();
    let probe = if num_touched == 0 {
        None
    } else {
        let mut feat = Vec::with_capacity(touched_rows.len() * PROBE_CHANNELS);
        for &i in &touched_rows {
            #[allow(clippy::cast_possible_truncation)]
            {
                feat.push(composed.weight[i] as f32);
                feat.push(composed.touched[i] as f32);
                feat.push(1.0_f32);
            }
        }
        Some(PointSet::new(
            points.xyz.clone(),
            points.size.clone(),
            feat,
            points.conf.clone(),
            PROBE_CHANNELS,
        )?)
    };

    Ok(EditedPoints {
        points,
        probe,
        num_deleted,
        num_touched,
    })
}

/// Highlight colour the shade finder's preview paints selected points.
///
/// Saturated magenta: no natural surface in a photograph sits near it, so a
/// tinted point cannot be mistaken for the scene's own colour.
pub const PREVIEW_TINT: [f32; 3] = [1.0, 0.0, 1.0];

/// How far the preview dims everything it did NOT select.
pub const PREVIEW_DIM: f32 = 0.25;

/// A copy of `base` with `ids` painted [`PREVIEW_TINT`] and everything else
/// dimmed — `docs/EDITOR.md` §3's "the current threshold's matched points at
/// high saturation against a dimmed scene".
///
/// Only the first three feature channels are touched, because those are the
/// base colour every trippy-native bundle seeds
/// (`trippy.train.params.PointParams.__init__`: `feat[:, :3] = rgb0`). The
/// remaining channels are the network's learned descriptor and are left alone;
/// the preview is a change of what is being *shown*, not of what the network is
/// asked to invent.
///
/// Ids outside the cloud are ignored, exactly as
/// `trippy.edit.model.pointset_membership` ignores them.
#[must_use]
pub fn tinted_points(base: &PointSet, ids: &[u32]) -> PointSet {
    let mut out = base.clone();
    let channels = out.num_channels;
    let colour = channels.min(3);
    for row in 0..out.len() {
        for c in 0..colour {
            out.feat[row * channels + c] *= PREVIEW_DIM;
        }
    }
    for id in ids {
        let row = *id as usize;
        if row >= out.len() {
            continue;
        }
        for c in 0..colour {
            out.feat[row * channels + c] = PREVIEW_TINT[c];
        }
    }
    out
}

/// The Gaussian half: a per-Gaussian opacity multiplier, `0.0` where a
/// `delete`-op region removed it and `1.0` everywhere else.
///
/// `blend`/`fade` regions deliberately do NOT scale a Gaussian's opacity. Their
/// mix is a per-pixel choice between the two renders, made in
/// `Renderer::compose`; dimming the Gaussians as well would apply it twice, and
/// in the wrong direction (a `mix = 0` region asks for MORE splat, not less).
/// This is exactly what `trippy.edit.apply.apply_edits` does on the publish
/// side, where the Gaussian PLY is filtered by
/// `compose_gaussian_weights(...).delete_mask` and nothing else.
///
/// Returns `None` when nothing is deleted, so the caller can leave the loaded
/// splat untouched.
#[must_use]
pub fn gaussian_opacity_scale(composed: &ComposedWeights) -> Option<Vec<f32>> {
    if composed.delete_mask.iter().all(|d| !d) {
        return None;
    }
    Some(
        composed
            .delete_mask
            .iter()
            .map(|deleted| if *deleted { 0.0 } else { 1.0 })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::model::{EditDocument, Op, Params, Region};
    use crate::edit::weights::compose_trips_weights;

    fn cloud() -> PointSet {
        // Four points on the x axis, C = 4.
        let xyz = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0,
        ];
        let size = vec![0.1, 0.2, 0.3, 0.4];
        let feat: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let conf = vec![0.9, 0.8, 0.7, 0.6];
        PointSet::new(xyz, size, feat, conf, 4).expect("valid")
    }

    fn sphere(id: &str, center: [f64; 3], radius: f64, mix: f64, op: Op) -> Region {
        Region::new(
            id.to_owned(),
            id.to_owned(),
            Params::Sphere { center, radius },
            mix,
            op,
        )
    }

    #[test]
    fn no_edits_leaves_the_cloud_alone_and_skips_the_probe_pass() {
        let base = cloud();
        let doc = EditDocument::default();
        let composed = compose_trips_weights(&doc, &crate::edit::weights::widen(&base.xyz));
        let edited = edited_points(&base, &composed).unwrap();
        assert_eq!(edited.points, base);
        assert!(edited.probe.is_none(), "no second pyramid pass to pay for");
        assert_eq!(edited.num_deleted, 0);
        assert!(gaussian_opacity_scale(&composed).is_none());
    }

    #[test]
    fn a_delete_region_removes_rows_from_every_array_consistently() {
        let base = cloud();
        let mut doc = EditDocument::default();
        // Radius 1.1 about x = 1: catches points 0, 1 and 2.
        doc.add_region(&sphere("r-d", [1.0, 0.0, 0.0], 1.1, 0.0, Op::Delete), None)
            .unwrap();
        let composed = compose_trips_weights(&doc, &crate::edit::weights::widen(&base.xyz));
        let edited = edited_points(&base, &composed).unwrap();
        assert_eq!(edited.num_deleted, 3);
        assert_eq!(edited.points.len(), 1);
        assert_eq!(edited.points.xyz, vec![3.0, 0.0, 0.0]);
        assert_eq!(edited.points.size, vec![0.4]);
        assert_eq!(edited.points.conf, vec![0.6]);
        assert_eq!(edited.points.feat, vec![12.0, 13.0, 14.0, 15.0]);
        assert!(edited.probe.is_none(), "a delete needs no per-pixel weight");

        let scale = gaussian_opacity_scale(&composed).expect("deletes reach the Gaussians");
        assert_eq!(scale, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_blend_region_builds_a_probe_that_shares_the_main_passs_alphas() {
        let base = cloud();
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-b", [0.5, 0.0, 0.0], 0.6, 0.25, Op::Blend), None)
            .unwrap();
        let composed = compose_trips_weights(&doc, &crate::edit::weights::widen(&base.xyz));
        let edited = edited_points(&base, &composed).unwrap();

        // `conf` is untouched: the TRIPS pass still draws the points at full
        // alpha, and the mix happens per pixel. See "why the weight does not
        // multiply conf".
        assert_eq!(edited.points.conf, base.conf);
        let probe = edited.probe.expect("a blend region needs the probe pass");
        assert_eq!(probe.num_channels, PROBE_CHANNELS);
        assert_eq!(probe.xyz, base.xyz);
        assert_eq!(probe.conf, base.conf);
        assert_eq!(probe.size, base.size);
        // Points 0 and 1 are inside; 2 and 3 are not.
        assert_eq!(
            probe.feat,
            vec![
                0.25, 1.0, 1.0, //
                0.25, 1.0, 1.0, //
                1.0, 0.0, 1.0, //
                1.0, 0.0, 1.0,
            ]
        );
        assert_eq!(edited.num_touched, 2);
    }

    #[test]
    fn deleting_and_blending_at_once_keeps_the_two_sets_row_aligned() {
        let base = cloud();
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-b", [3.0, 0.0, 0.0], 0.5, 0.0, Op::Fade), None)
            .unwrap();
        doc.add_region(&sphere("r-d", [0.0, 0.0, 0.0], 0.5, 0.0, Op::Delete), None)
            .unwrap();
        let composed = compose_trips_weights(&doc, &crate::edit::weights::widen(&base.xyz));
        let edited = edited_points(&base, &composed).unwrap();
        let probe = edited.probe.expect("the fade region survives the delete");
        assert_eq!(edited.points.len(), 3, "point 0 was deleted");
        assert_eq!(probe.len(), edited.points.len());
        assert_eq!(probe.xyz, edited.points.xyz);
        // The surviving rows are 1, 2, 3; only row 3 is faded to 0.
        assert_eq!(
            probe.feat,
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn the_preview_tints_the_selection_and_dims_the_rest() {
        let base = cloud();
        let tinted = tinted_points(&base, &[1, 99]);
        assert_eq!(tinted.len(), base.len());
        // Row 0 is dimmed on its colour channels and untouched elsewhere.
        assert_eq!(&tinted.feat[0..4], &[0.0, PREVIEW_DIM, 2.0 * PREVIEW_DIM, 3.0]);
        // Row 1 is the tint, with its fourth (descriptor) channel intact.
        assert_eq!(&tinted.feat[4..8], &[1.0, 0.0, 1.0, 7.0]);
        // An id past the end of the cloud is ignored, not a panic.
        assert_eq!(&tinted.feat[8..12], &[8.0 * PREVIEW_DIM, 9.0 * PREVIEW_DIM, 10.0 * PREVIEW_DIM, 11.0]);
    }

    #[test]
    fn the_probe_background_is_zero_so_an_empty_pixel_is_an_unedited_one() {
        assert_eq!(PROBE_BACKGROUND, [0.0; PROBE_CHANNELS]);
        assert!(COVERAGE_EPS > 0.0 && COVERAGE_EPS < 1e-3);
    }
}
