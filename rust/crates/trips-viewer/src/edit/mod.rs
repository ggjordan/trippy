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
pub mod cluster;
pub mod model;
pub mod shade;
pub mod weights;

pub use apply::{edited_points, gaussian_opacity_scale, EditedPoints, COVERAGE_EPS};
pub use cluster::{click_to_cluster, ClickCamera, ClickParams, ClickSelection, PointGrid};
pub use model::{EditDocument, Kind, LidParams, Op, Params, Region, EDITS_FILENAME};
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
/// The 3D drag gizmos are not built (`docs/EDITOR.md` §6, E1); the Inspector's
/// numeric fields plus arrow-key nudging are what E1 ships instead.
pub const NUDGE_SCENE_FRACTION: f32 = 0.02;

/// Multiplier one `[`/`]` press applies to the selected region's size.
pub const RESIZE_STEP: f32 = 1.25;

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
