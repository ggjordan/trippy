//! The editor's platform-neutral half: `edits.json`, region maths, weights, the shade finder.
//!
//! Module: `trips_viewer::edit`
//! Purpose: everything `docs/EDITOR.md`'s E1 and E2 milestones need that is not
//!     a window — the data model ([`model`]), the per-point weight composition
//!     the renderer uploads ([`weights`]) and the shade-cloud finder
//!     ([`shade`]). The egui panels that drive it live in the binary
//!     (`src/edit_ui.rs`), for the same reason [`crate::blend`] keeps its
//!     arithmetic out of `app.rs`: this half must stay testable on the CPU and
//!     compilable for wasm.
//! Invariants:
//!     - Nothing here references Burn, wgpu, egui or eframe. [`cluster`] reads
//!       `brush_pyramid::scene::Camera` for its numbers only; that type is
//!       plain data and is available without the `gpu` feature.
//!     - [`model`], [`weights`] and [`cluster`] are twins of
//!       `trippy/edit/model.py`, `trippy/edit/weights.py` and
//!       `trippy/edit/cluster.py`. The golden test at the bottom of this file
//!     	is the thing that keeps them twins: it replays a synthetic
//!       `edits.json` against a synthetic point cloud and compares against
//!       weights Python computed, to 1e-6. If it fails, one side drifted.
//! Units: world units; weights are dimensionless in `[0, 1]`.
//! Related docs: `docs/EDITOR.md`; `docs/decisions/ADR-0007-viewer-editing.md`.

pub mod apply;
/// The `brush` region kind: a sparse voxel set painted with spheres.
/// See `docs/EDITOR.md` §1 "brush".
pub mod brush;
pub mod cluster;
/// The 3D drag gizmos' maths (translate / resize / rotate a region).
/// See `docs/EDITOR.md` §6's E1/E3 rows.
pub mod gizmo;
pub mod model;
/// A minimal numpy `.npz` writer: the brush sidecar `EditDocument::save`
/// externalises a large brush region's cells/weights into. Not `pub` beyond
/// the crate — `model::EditDocument::save` is the only caller, matching how
/// `trippy.edit.model._externalize_brush` is private to its own module too.
mod npz_write;
/// The SAM tool's geometry: render pixels -> the capture view's pixel grid.
/// See `docs/EDITOR.md` §3 "4. SAM 3 lift (E5)".
pub mod sam;
pub mod shade;
pub mod weights;

pub use apply::{edited_points, gaussian_opacity_scale, EditedPoints, COVERAGE_EPS};
pub use brush::{depth_anchor, BrushCells, ScreenGrid};
pub use cluster::{click_to_cluster, ClickCamera, ClickParams, ClickSelection, PointGrid};
pub use gizmo::{Drag, GizmoScreen};
pub use model::{
    auto_name_for, auto_region_name, EditDocument, Kind, LidParams, Op, Params, Region,
    EDITS_FILENAME, EDIT_BRUSH_NPZ_CELL_THRESHOLD,
};
pub use sam::{is_box_drag, nearest_view, render_pixel, view_box_from_render, view_pixel_from_render};
pub use weights::{compose_gaussian_weights, compose_trips_weights, ComposedWeights};

/// Default region size as a fraction of the scene's own diameter.
///
/// A new box/sphere has to be big enough to see and small enough to aim, and
/// the only scale the viewer trusts is `crate::bundle::SceneScale` (never the
/// point cloud's bounds — a TRIPS export's environment sphere makes those
/// thousands of units across; see `renderer.rs`'s `bounds` field).
pub const DEFAULT_REGION_SCENE_FRACTION: f32 = 0.08;

/// Fraction of the scene diameter one keyboard nudge moves a region.
///
/// The keyboard nudges are kept now that the 3D drag gizmos exist
/// (`docs/EDITOR.md` §4's key table): a gizmo needs a visible handle and a
/// mouse, and an exact 2 %-of-the-scene step needs neither.
pub const NUDGE_SCENE_FRACTION: f32 = 0.02;

/// Multiplier one `[`/`]` press applies to the selected region's size.
///
/// With the Brush tool focused the same two keys shrink/grow the BRUSH instead
/// (`BRUSH_RADIUS_STEP`) — there is no selected region to resize while
/// painting, and a brush without a radius key is unusable.
pub const RESIZE_STEP: f32 = 1.25;

/// Default brush radius as a fraction of the scene's own diameter.
///
/// The same scale a new box/sphere is born at, halved: a brush stroke is meant
/// to touch part of an object, not swallow it.
pub const DEFAULT_BRUSH_SCENE_FRACTION: f32 = 0.04;

/// Multiplier one `[`/`]` press applies to the brush radius.
pub const BRUSH_RADIUS_STEP: f64 = 1.25;

/// Voxel cells across one brush radius, fixing a new brush region's `cell_size`.
///
/// The grid is fixed when the region is born (painting into a region whose
/// cells were measured on a different grid would be meaningless), so this is
/// the one number that decides how blocky a brush edit can be. Two cells per
/// radius keeps a stroke's own shape recognisable while keeping the cell count
/// — and the `edits.json` it is written into — small.
pub const BRUSH_CELLS_PER_RADIUS: f64 = 2.0;

/// How far the pointer must travel (render pixels) before a drag paints again.
///
/// A stroke is a path of spheres; sampling every frame would paint hundreds of
/// overlapping spheres at the same place while the pointer sits still, and each
/// sample costs one depth-anchor scan over the cloud.
pub const BRUSH_SAMPLE_PX: f64 = 4.0;

#[cfg(test)]
mod golden {
    //! Python parity, pinned by a committed fixture.
    //!
    //! `tests/fixtures/synthetic/edit_golden/` is written by
    //! `trippy.edit.golden.write_golden_fixture` and checked from the Python
    //! side by `tests/test_edit_golden.py`, so the two halves of this test can
    //! never quietly agree on a stale file.

    use super::*;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/fixtures/synthetic/edit_golden")
    }

    fn read(name: &str) -> serde_json::Value {
        let path = fixture_dir().join(name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (regenerate: python -m trippy.edit.golden)", path.display()));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn floats(value: &serde_json::Value, key: &str) -> Vec<f64> {
        value[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} must be an array"))
            .iter()
            .map(|v| v.as_f64().expect("numbers"))
            .collect()
    }

    #[test]
    fn the_viewer_composes_the_same_weights_python_does() {
        let points = read("points.json");
        let expected = read("expected_weights.json");
        let doc = EditDocument::from_json(&read("edits.json")).expect("fixture edits.json");
        doc.validate(Some("trippy-bundle-1")).expect("fixture validates");

        let xyz = floats(&points, "xyz");
        let got = compose_trips_weights(&doc, &xyz);

        let want = floats(&expected, "weight");
        assert_eq!(got.weight.len(), want.len(), "point count");
        for (i, (a, b)) in got.weight.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "TRIPS weight[{i}]: viewer {a} vs python {b}"
            );
        }

        let want_deleted: Vec<bool> = expected["delete_mask"]
            .as_array()
            .expect("delete_mask")
            .iter()
            .map(|v| v.as_bool().expect("bools"))
            .collect();
        assert_eq!(got.delete_mask, want_deleted, "delete mask");

        // The Gaussian half of the same document: `pointset` regions skipped.
        let gaussian_xyz = floats(&points, "gaussian_xyz");
        let got = compose_gaussian_weights(&doc, &gaussian_xyz);
        let want = floats(&expected, "gaussian_weight");
        assert_eq!(got.weight.len(), want.len(), "gaussian count");
        for (i, (a, b)) in got.weight.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "gaussian weight[{i}]: viewer {a} vs python {b}"
            );
        }
    }

    /// Every case of `click.json`, replayed with the viewer's own clustering.
    ///
    /// This asserts an EXACT id-for-id match, not a tolerance: a selection is a
    /// set of integers, and the three places the two implementations could
    /// legitimately break a tie (`cluster`'s own "Tie-breaking" section) are
    /// all unreachable in this fixture — `tests/test_edit_golden.py`'s
    /// `test_each_click_case_pins_a_different_branch` is what keeps it that
    /// way.
    #[test]
    fn the_viewer_clusters_the_same_click_python_does() {
        let scene = read("click.json");
        let expected = read("expected_click.json");
        assert_eq!(
            scene["format"].as_str(),
            Some(cluster::CLICK_FIXTURE_FORMAT),
            "click.json format"
        );

        let xyz = floats(&scene, "xyz");
        let rgb = floats(&scene, "rgb");
        let camera = cluster::ClickCamera::from_json(&scene["camera"]).expect("fixture camera");
        let grid = cluster::PointGrid::build(&xyz);

        let cases = scene["cases"].as_array().expect("cases");
        let want_cases = expected["cases"].as_array().expect("expected cases");
        assert_eq!(cases.len(), want_cases.len(), "case count");
        assert!(!cases.is_empty(), "the fixture must carry at least one click");

        for (case, want) in cases.iter().zip(want_cases) {
            let name = case["name"].as_str().expect("case name");
            assert_eq!(name, want["name"].as_str().expect("name"), "case order");
            let params = cluster::ClickParams {
                radius_px: case["radius_px"].as_f64().expect("radius_px"),
                colour_tol: case["colour_tol"].as_f64().expect("colour_tol"),
                max_radius: case["max_radius"].as_f64().expect("max_radius"),
                max_points: usize::try_from(case["max_points"].as_u64().expect("max_points"))
                    .expect("fits"),
                knn_k: usize::try_from(case["knn_k"].as_u64().expect("knn_k")).expect("fits"),
                depth_gap_factor: case["depth_gap_factor"].as_f64().expect("depth_gap_factor"),
            };
            let px = case["px"].as_array().expect("px");
            let found = cluster::click_to_cluster(
                &grid,
                &xyz,
                &rgb,
                &camera,
                (
                    px[0].as_f64().expect("u"),
                    px[1].as_f64().expect("v"),
                ),
                &params,
            );

            let want_ids: Vec<u32> = want["point_ids"]
                .as_array()
                .expect("point_ids")
                .iter()
                .map(|v| u32::try_from(v.as_u64().expect("ids")).expect("fits"))
                .collect();
            assert_eq!(found.point_ids, want_ids, "click case {name:?}: selection");
            assert_eq!(
                found.n_candidates,
                usize::try_from(want["n_candidates"].as_u64().expect("n_candidates")).expect("fits"),
                "click case {name:?}: candidates"
            );
            assert_eq!(
                found.n_seed,
                usize::try_from(want["n_seed"].as_u64().expect("n_seed")).expect("fits"),
                "click case {name:?}: seed"
            );
            assert_eq!(
                found.hit_max_points,
                want["hit_max_points"].as_bool().expect("hit_max_points"),
                "click case {name:?}: cap"
            );
            match want["seed_depth_mean"].as_f64() {
                Some(depth) => {
                    let got = found.seed_depth_mean.expect("a hit reports a seed depth");
                    assert!(
                        (got - depth).abs() <= 1e-9,
                        "click case {name:?}: seed depth {got} vs python {depth}"
                    );
                }
                None => assert!(
                    found.seed_depth_mean.is_none(),
                    "click case {name:?}: a miss has no seed depth"
                ),
            }
            assert_eq!(
                found.warning.is_some(),
                !want["warning"].is_null(),
                "click case {name:?}: warning"
            );
        }
    }

    /// `brush.json`, replayed stroke for stroke.
    ///
    /// Not "does the committed cell list give the committed weights" — that
    /// would pass with a completely different voxelisation. The fixture records
    /// the three AUTHORING calls (`trippy.edit.golden._BRUSH_STROKES`), this
    /// replays them into an empty region, and the resulting cells, their
    /// weights, their ORDER and the per-point membership must all match what
    /// Python wrote.
    #[test]
    fn the_viewer_paints_the_same_brush_python_does() {
        let fixture = read("brush.json");
        assert_eq!(
            fixture["format"].as_str(),
            Some(brush::BRUSH_FIXTURE_FORMAT),
            "brush.json format"
        );

        // The empty region the strokes start from, with its own grid.
        let initial = Region::from_json(&fixture["initial"]).expect("fixture initial region");
        let Params::Brush {
            origin,
            cell_size,
            cells,
        } = initial.params.clone()
        else {
            panic!("the fixture's initial region must be a brush")
        };
        assert!(cells.is_empty(), "the strokes start from an empty brush");

        let mut painted = cells;
        let strokes = fixture["strokes"].as_array().expect("strokes");
        assert_eq!(strokes.len(), 3, "paint_sphere, paint_along and erase");
        for stroke in strokes {
            brush::apply_stroke_json(&mut painted, origin, cell_size, stroke)
                .expect("the fixture's strokes replay");
        }

        // The cells Python painted, in Python's own order.
        let expected = Region::from_json(&fixture["region"]).expect("fixture region");
        let Params::Brush {
            cells: want_cells, ..
        } = expected.params.clone()
        else {
            panic!("the fixture's region must be a brush")
        };
        assert_eq!(
            painted.len(),
            want_cells.len(),
            "cell count: viewer {} vs python {}",
            painted.len(),
            want_cells.len()
        );
        assert_eq!(painted.cells(), want_cells.cells(), "cells, in order");
        for (i, (a, b)) in painted.weights().iter().zip(want_cells.weights()).enumerate() {
            assert!((a - b).abs() <= 1e-12, "cell weight[{i}]: viewer {a} vs python {b}");
        }
        assert!(
            painted.weights().iter().any(|w| *w < 1.0),
            "the erase must leave graded cells behind, or this tests nothing"
        );

        // And the membership those cells produce, point by point.
        let xyz = floats(&fixture, "xyz");
        let want_weight = floats(&fixture, "expected_weight");
        let want_contains: Vec<bool> = fixture["expected_contains"]
            .as_array()
            .expect("expected_contains")
            .iter()
            .map(|v| v.as_bool().expect("bools"))
            .collect();
        assert_eq!(xyz.len() / 3, want_weight.len(), "query point count");
        for i in 0..want_weight.len() {
            let p = [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
            let got = model::region_weight(&expected, i, p);
            assert!(
                (got - want_weight[i]).abs() <= 1e-12,
                "brush weight[{i}]: viewer {got} vs python {}",
                want_weight[i]
            );
            assert_eq!(
                model::region_contains(&expected, i, p),
                want_contains[i],
                "brush contains[{i}]"
            );
        }
        assert!(
            want_weight.iter().any(|w| *w > 0.0) && want_weight.iter().any(|w| *w == 0.0),
            "the query grid must straddle the brush"
        );
    }

    /// `names.json`: the Named Objects panel numbers a region the way `trippy
    /// edits` does, case for case.
    #[test]
    fn the_viewer_auto_names_a_region_the_way_python_does() {
        let fixture = read("names.json");
        assert_eq!(
            fixture["format"].as_str(),
            Some(model::NAMES_FIXTURE_FORMAT),
            "names.json format"
        );
        let cases = fixture["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "the fixture must carry at least one case");
        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let existing: Vec<&str> = case["existing_names"]
                .as_array()
                .expect("existing_names")
                .iter()
                .map(|v| v.as_str().expect("names are strings"))
                .collect();
            let got = model::auto_region_name(
                existing.iter().copied(),
                case["tool"].as_str().expect("tool"),
                case["detail"].as_str(),
            );
            assert_eq!(
                got,
                case["expected"].as_str().expect("expected"),
                "name case {name:?}"
            );
        }
    }

    #[test]
    fn the_viewer_selects_the_same_shade_points_python_does() {
        let points = read("points.json");
        let expected = read("expected_shade.json");
        let sidecar = read("shade_views.json");

        let views: Vec<shade::ShadeView> = sidecar["views"]
            .as_array()
            .expect("views")
            .iter()
            .map(|v| shade::ShadeView::from_json(v).expect("shade view"))
            .collect();
        assert!(!views.is_empty(), "the fixture must carry shade frames");

        let t = shade::Thresholds {
            znear_frac: expected["thresholds"]["znear_frac"].as_f64().expect("znear_frac"),
            zfar_frac: expected["thresholds"]["zfar_frac"].as_f64().expect("zfar_frac"),
            lum_threshold: expected["thresholds"]["lum_threshold"].as_f64().expect("lum"),
            conf_threshold: expected["thresholds"]["conf_threshold"].as_f64().expect("conf"),
        };
        let found = shade::find(
            &views,
            &floats(&points, "xyz"),
            &floats(&points, "rgb"),
            &floats(&points, "conf"),
            t,
        );

        let want: Vec<u32> = expected["point_ids"]
            .as_array()
            .expect("point_ids")
            .iter()
            .map(|v| u32::try_from(v.as_u64().expect("ids")).expect("fits"))
            .collect();
        assert_eq!(found.point_ids, want, "shade selection");
        let want_fraction = expected["dark_mass_fraction"].as_f64().expect("fraction");
        assert!(
            (found.dark_mass_fraction - want_fraction).abs() <= 1e-6,
            "dark mass fraction: viewer {} vs python {want_fraction}",
            found.dark_mass_fraction
        );
    }
}
