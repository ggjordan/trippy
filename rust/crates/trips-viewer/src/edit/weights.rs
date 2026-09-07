//! Per-point blend-weight composition: the gate default plus ordered region overrides.
//!
//! Module: `trips_viewer::edit::weights`
//! Purpose: turn an [`EditDocument`]'s ordered, enabled regions into the scalar
//!     per-point weight `docs/EDITOR.md` §2 calls `w_edit`, plus the delete mask
//!     that removes points/Gaussians from the renderer's input before
//!     rasterisation. A **twin of `trippy/edit/weights.py`**: same starting
//!     default, same order, same three op rules, same "once deleted, stays
//!     deleted" latch — which is what `tests/fixtures/synthetic/edit_golden`
//!     and `--dump-weights` check to 1e-6.
//! Invariants:
//!     - Starts from `default` ([`super::model::GATE_DEFAULT_WEIGHT`] = 1.0 =
//!       pure TRIPS) and applies `edits.order()` in sequence, skipping disabled
//!       regions.
//!     - `blend` moves the running weight towards `mix`, graded by the region's
//!       own membership; `fade` multiplies towards `weight * mix`; `delete`
//!       zeroes the weight and latches the delete mask using the region's HARD
//!       membership. Byte-for-byte the Python's three branches.
//!     - A point once deleted is skipped by every later region, so a `delete`
//!       can override an earlier `blend` and never the reverse.
//!     - `pointset` regions index the TRIPS cloud's own row order, so
//!       [`compose_gaussian_weights`] skips them rather than guess a
//!       correspondence to a Gaussian PLY's rows.
//!     - [`ComposedWeights::per_region`] and [`compose_point_weights_solo`]'s
//!       `solo` argument are viewer-only too, and for the same reason as
//!       `touched`: the Named Objects panel has to say how many points each
//!       region actually claims, and "solo" has to show one region's effect
//!       alone. Neither changes what the shared code path computes — `solo =
//!       None` is the Python's own behaviour, byte for byte, and the counts are
//!       observed on the way through rather than computed by a second pass.
//!     - [`ComposedWeights::touched`] is the ONE quantity with no Python twin:
//!       it is the viewer's "did any region have an opinion here" mask, needed
//!       because a region edit must override the *gate* only where it applies
//!       (`docs/EDITOR.md` §2). It never changes `weight`, so Python parity is
//!       unaffected.
//! Units: weights are dimensionless in `[0, 1]` (0 = pure splat, 1 = pure TRIPS).
//! Related docs: `docs/EDITOR.md` §2 "Render integration"; `trippy/edit/weights.py`.

use std::collections::HashSet;

use super::model::{
    lid_membership, region_contains, region_weight, EditDocument, Kind, Op, Params, Region,
    GATE_DEFAULT_WEIGHT,
};

/// The result of composing an [`EditDocument`] against one point cloud.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposedWeights {
    /// `(N,)` in `[0, 1]`: the per-point `w_edit`. Always 0.0 where
    /// `delete_mask` is set — a deleted point has no weight because it does not
    /// render at all.
    pub weight: Vec<f64>,
    /// `(N,)` in `[0, 1]`: how strongly ANY enabled `blend`/`fade` region claims
    /// this point. 0 = untouched, so the gate keeps the pixel. Viewer-only; see
    /// the module invariants.
    pub touched: Vec<f64>,
    /// `(N,)`: true where a `delete`-op region removed the point.
    pub delete_mask: Vec<bool>,
    /// Per region, in paint order: its id and how many points it actually
    /// claimed *at the moment it applied* — i.e. after any earlier `delete`
    /// took its own points out of play. That is the number the Named Objects
    /// panel shows, because it is the number the frame reflects.
    pub per_region: Vec<RegionCount>,
}

/// One region's contribution, for the Named Objects panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionCount {
    /// The region's id.
    pub id: String,
    /// How many points it claimed (membership > 0, or the hard clip for `delete`).
    pub count: usize,
}

impl ComposedWeights {
    /// How many points a `delete` region removed.
    #[must_use]
    pub fn num_deleted(&self) -> usize {
        self.delete_mask.iter().filter(|d| **d).count()
    }

    /// How many points any enabled `blend`/`fade` region claims at all.
    #[must_use]
    pub fn num_touched(&self) -> usize {
        self.touched.iter().filter(|t| **t > 0.0).count()
    }

    /// Whether anything here changes what the renderer would otherwise draw.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.delete_mask.iter().all(|d| !d) && self.touched.iter().all(|t| *t == 0.0)
    }
}

/// A `pointset` region's ids as a set, so membership is O(1) per point rather
/// than the O(|ids|) linear scan [`region_weight`] falls back to.
struct Lookup<'a> {
    region: &'a Region,
    ids: Option<HashSet<u32>>,
}

impl Lookup<'_> {
    fn membership(&self, index: usize, p: [f64; 3]) -> f64 {
        match (&self.ids, &self.region.params) {
            (Some(ids), Params::Pointset { .. }) => f64::from(
                u32::try_from(index)
                    .ok()
                    .is_some_and(|i| ids.contains(&i)),
            ),
            _ => region_weight(self.region, index, p),
        }
    }

    fn contains(&self, index: usize, p: [f64; 3]) -> bool {
        match (&self.ids, &self.region.params) {
            (Some(_), Params::Pointset { .. }) => self.membership(index, p) > 0.0,
            (_, Params::Lid(lid)) => lid_membership(p, lid).0,
            _ => region_contains(self.region, index, p),
        }
    }
}

/// Compose `edits`' enabled regions against `xyz`, starting from `default`.
///
/// The `solo`-free entry point; see [`compose_point_weights_solo`].
///
/// # Arguments
/// - `edits`: the document; regions are applied in `edits.order()`.
/// - `xyz`: flat `(N, 3)` world-frame positions, row-major. For a `pointset`
///   region to mean anything, row `i` must be the point `point_ids` containing
///   `i` refers to.
/// - `default`: the weight before any region applies (the gate's own opinion;
///   1.0 = pure TRIPS).
/// - `skip_pointset`: when true, `pointset` regions are ignored entirely.
///
/// # Panics
/// Panics if `xyz.len()` is not a multiple of 3, which is a programming error at
/// the call site rather than a runtime condition.
#[must_use]
pub fn compose_point_weights(
    edits: &EditDocument,
    xyz: &[f64],
    default: f64,
    skip_pointset: bool,
) -> ComposedWeights {
    compose_point_weights_solo(edits, xyz, default, skip_pointset, None)
}

/// [`compose_point_weights`], with an optional "only this region" filter.
///
/// `solo = Some(id)` skips every OTHER region, enabled or not, so the frame
/// shows one region's effect alone — the Named Objects panel's solo button
/// (`docs/EDITOR.md` §4). `solo = None` is exactly the Python's behaviour and
/// is what every parity test runs.
///
/// # Panics
/// Panics if `xyz.len()` is not a multiple of 3, which is a programming error
/// at the call site rather than a runtime condition.
#[must_use]
pub fn compose_point_weights_solo(
    edits: &EditDocument,
    xyz: &[f64],
    default: f64,
    skip_pointset: bool,
    solo: Option<&str>,
) -> ComposedWeights {
    assert!(xyz.len() % 3 == 0, "xyz must be a flat (N, 3) array");
    let n = xyz.len() / 3;
    let mut weight = vec![default; n];
    let mut touched = vec![0.0_f64; n];
    let mut delete_mask = vec![false; n];
    let mut per_region: Vec<RegionCount> = Vec::with_capacity(edits.order().len());

    for id in edits.order() {
        let Some(region) = edits.region(id) else {
            continue;
        };
        if !region.enabled || solo.is_some_and(|only| only != region.id) {
            continue;
        }
        if skip_pointset && region.kind == Kind::Pointset {
            continue;
        }
        let lookup = Lookup {
            region,
            ids: match &region.params {
                Params::Pointset { point_ids } => Some(point_ids.iter().copied().collect()),
                _ => None,
            },
        };

        let mut claimed = 0_usize;
        for i in 0..n {
            // `active`: a point a previous region already deleted takes no
            // further part in composition (see the module's third invariant).
            if delete_mask[i] {
                continue;
            }
            let p = [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
            if region.op == Op::Delete {
                if lookup.contains(i, p) {
                    delete_mask[i] = true;
                    weight[i] = 0.0;
                    touched[i] = 0.0;
                    claimed += 1;
                }
                continue;
            }
            let m = lookup.membership(i, p);
            if m == 0.0 {
                continue;
            }
            claimed += 1;
            weight[i] = match region.op {
                Op::Blend => weight[i].mul_add(1.0 - m, region.mix * m),
                // `fade` multiplies: w *= 1 - m*(1 - mix).
                Op::Fade => weight[i] * m.mul_add(-(1.0 - region.mix), 1.0),
                Op::Delete => unreachable!("handled above"),
            };
            touched[i] = touched[i].mul_add(1.0 - m, m);
        }
        per_region.push(RegionCount {
            id: region.id.clone(),
            count: claimed,
        });
    }

    ComposedWeights {
        weight,
        touched,
        delete_mask,
        per_region,
    }
}

/// TRIPS points: every region kind applies (`pointset.point_ids` index these rows).
///
/// The twin of `trippy.edit.weights.compose_trips_weights`.
///
/// # Panics
/// As [`compose_point_weights`].
#[must_use]
pub fn compose_trips_weights(edits: &EditDocument, xyz: &[f64]) -> ComposedWeights {
    compose_point_weights(edits, xyz, GATE_DEFAULT_WEIGHT, false)
}

/// [`compose_trips_weights`] with the Named Objects panel's solo filter.
///
/// # Panics
/// As [`compose_point_weights`].
#[must_use]
pub fn compose_trips_weights_solo(
    edits: &EditDocument,
    xyz: &[f64],
    solo: Option<&str>,
) -> ComposedWeights {
    compose_point_weights_solo(edits, xyz, GATE_DEFAULT_WEIGHT, false, solo)
}

/// Gaussian splat centres: `pointset` regions skipped (a different row order).
///
/// The twin of `trippy.edit.weights.compose_gaussian_weights`.
///
/// # Panics
/// As [`compose_point_weights`].
#[must_use]
pub fn compose_gaussian_weights(edits: &EditDocument, xyz: &[f64]) -> ComposedWeights {
    compose_point_weights(edits, xyz, GATE_DEFAULT_WEIGHT, true)
}

/// [`compose_gaussian_weights`] with the Named Objects panel's solo filter.
///
/// # Panics
/// As [`compose_point_weights`].
#[must_use]
pub fn compose_gaussian_weights_solo(
    edits: &EditDocument,
    xyz: &[f64],
    solo: Option<&str>,
) -> ComposedWeights {
    compose_point_weights_solo(edits, xyz, GATE_DEFAULT_WEIGHT, true, solo)
}

/// Widen a `(N, 3)` f32 position array to f64, exactly as `numpy`'s
/// `astype(np.float64)` does (an f32 -> f64 conversion is lossless).
#[must_use]
pub fn widen(xyz: &[f32]) -> Vec<f64> {
    xyz.iter().map(|v| f64::from(*v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::model::{LidParams, Region};
    use serde_json::json;

    fn sphere(id: &str, radius: f64, mix: f64, op: Op) -> Region {
        Region::new(
            id.to_owned(),
            id.to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius,
            },
            mix,
            op,
        )
    }

    /// Three points: inside the small sphere, inside only the big one, outside both.
    const XYZ: [f64; 9] = [
        0.0, 0.0, 0.0, //
        1.5, 0.0, 0.0, //
        9.0, 0.0, 0.0,
    ];

    #[test]
    fn no_regions_leaves_every_point_at_the_gate_default() {
        let doc = EditDocument::default();
        let w = compose_trips_weights(&doc, &XYZ);
        assert_eq!(w.weight, vec![1.0, 1.0, 1.0]);
        assert!(w.is_identity());
    }

    #[test]
    fn blend_sets_the_weight_and_fade_multiplies_it() {
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-1", 2.0, 0.5, Op::Blend), None).unwrap();
        let w = compose_trips_weights(&doc, &XYZ);
        assert_eq!(w.weight, vec![0.5, 0.5, 1.0]);
        assert_eq!(w.touched, vec![1.0, 1.0, 0.0]);

        // A second `fade` at mix 0.5 multiplies the 0.5 the blend set.
        doc.add_region(&sphere("r-2", 1.0, 0.5, Op::Fade), None).unwrap();
        let w = compose_trips_weights(&doc, &XYZ);
        assert_eq!(w.weight, vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn later_regions_win_on_overlap() {
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-1", 10.0, 0.2, Op::Blend), None).unwrap();
        doc.add_region(&sphere("r-2", 2.0, 0.9, Op::Blend), None).unwrap();
        let w = compose_trips_weights(&doc, &XYZ);
        // Inside both: the later region's mix wins outright.
        assert_eq!(w.weight[0], 0.9);
        assert_eq!(w.weight[1], 0.9);
        // Inside only the first.
        assert_eq!(w.weight[2], 0.2);
    }

    #[test]
    fn a_deleted_point_cannot_be_un_deleted_by_a_later_slider() {
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-1", 1.0, 0.0, Op::Delete), None).unwrap();
        doc.add_region(&sphere("r-2", 10.0, 1.0, Op::Blend), None).unwrap();
        let w = compose_trips_weights(&doc, &XYZ);
        assert!(w.delete_mask[0]);
        assert_eq!(w.weight[0], 0.0, "the delete's zero survives the later blend");
        assert_eq!(w.touched[0], 0.0);
        assert!(!w.delete_mask[1]);
        assert_eq!(w.weight[1], 1.0);
    }

    #[test]
    fn a_disabled_region_does_nothing_at_all() {
        let mut doc = EditDocument::default();
        doc.add_region(&sphere("r-1", 10.0, 0.0, Op::Blend), None).unwrap();
        doc.update_region("r-1", json!({ "enabled": false })).unwrap();
        let w = compose_trips_weights(&doc, &XYZ);
        assert_eq!(w.weight, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn pointset_regions_apply_to_trips_points_and_never_to_gaussians() {
        let mut doc = EditDocument::default();
        doc.add_region(
            &Region::new(
                "r-pts".to_owned(),
                "shade".to_owned(),
                Params::Pointset { point_ids: vec![2] },
                0.0,
                Op::Fade,
            ),
            None,
        )
        .unwrap();
        let trips = compose_trips_weights(&doc, &XYZ);
        assert_eq!(trips.weight, vec![1.0, 1.0, 0.0]);
        let gauss = compose_gaussian_weights(&doc, &XYZ);
        assert_eq!(gauss.weight, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn the_lids_delete_uses_the_hard_clip_and_its_fade_uses_the_ramp() {
        let lid = LidParams {
            up: [0.0, -1.0, 0.0],
            height: 0.0,
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            falloff: 1.0,
            band: 1.0,
        };
        // y = 0.5 is half a band below the plane, well inside the radius.
        let xyz = [0.0, 0.5, 0.0];
        let mut hard = EditDocument::default();
        hard.add_region(
            &Region::new("r-l".to_owned(), "lid".to_owned(), Params::Lid(lid), 0.0, Op::Delete),
            None,
        )
        .unwrap();
        assert!(compose_trips_weights(&hard, &xyz).delete_mask[0]);

        let mut soft = EditDocument::default();
        soft.add_region(
            &Region::new("r-l".to_owned(), "lid".to_owned(), Params::Lid(lid), 0.0, Op::Fade),
            None,
        )
        .unwrap();
        let w = compose_trips_weights(&soft, &xyz);
        assert!(!w.delete_mask[0]);
        // fade to mix 0 with membership 0.5: w = 1 * (1 - 0.5 * 1) = 0.5.
        assert!((w.weight[0] - 0.5).abs() < 1e-12, "{}", w.weight[0]);
    }
}
