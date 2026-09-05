//! GPU parity: the CubeCL forward pass vs trippy's Python forward.
//!
//! Module: `brush-pyramid` integration test `parity_gpu`
//! Purpose: prove that [`brush_pyramid::gpu::render_pyramid`] reproduces
//!     `trippy.raster.render_pyramid` on identical inputs, for all three
//!     layer-selection modes and both pixel-centre conventions, and that it
//!     agrees with the CPU twin exactly where it should.
//! Invariants:
//!     - **This is GPU work.** It only compiles under the `gpu` feature and
//!       must be launched through the queue, never directly:
//!
//!       scripts/gpu_submit.sh --prio 12 --wait brush-pyramid-gpu-1 -- \
//!         bash -c 'cd rust && cargo test -p brush-pyramid --features gpu \
//!                  --release --test parity_gpu -- --nocapture'
//!
//!     - Integer results (`n_used`, fragment counts) must match the Python
//!       reference **exactly**; only the float images get a tolerance.
//! Related docs: `docs/ARCHITECTURE.md`; `rust/README.md`; `AGENTS.md` §6.

#![cfg(feature = "gpu")]

use brush_pyramid::cpu::render_pyramid_cpu;
use brush_pyramid::fixture::{fixture_dirs, Fixture};
use brush_pyramid::gpu::{block_on, default_device, render_pyramid};

/// Absolute tolerance for the feature images and `t_final`, the same one the
/// CPU parity test uses. Both sides are float32; the difference is the order
/// of accumulation only.
const TOL: f32 = 1e-4;

#[test]
fn gpu_forward_matches_python_on_every_fixture() {
    block_on(async {
        let device = default_device();
        let dirs = fixture_dirs().expect("list fixtures");
        assert_eq!(dirs.len(), 6, "expected 6 fixtures, found {}", dirs.len());

        for dir in dirs {
            let fixture = Fixture::load(&dir).unwrap_or_else(|e| panic!("loading {}: {e}", dir.display()));
            let render = render_pyramid(
                &device,
                &fixture.points,
                &fixture.camera,
                &fixture.params,
                Some(&fixture.background),
            )
            .await
            .unwrap_or_else(|e| panic!("rendering {}: {e}", fixture.name()));

            let got = render
                .to_host()
                .await
                .unwrap_or_else(|e| panic!("readback for {}: {e}", fixture.name()));

            let report = got
                .compare(&fixture.expected, TOL)
                .unwrap_or_else(|e| panic!("{} vs Python: {e}", fixture.name()));

            println!(
                "{:<34} frags={:<5} slots={:<5} per-layer={:?} max|feature|={:.3e} max|t_final|={:.3e}",
                fixture.name(),
                got.num_fragments,
                render.num_fragment_slots(),
                got.fragments_per_layer,
                report.max_feature,
                report.max_t_final
            );
            assert!(
                render.num_fragment_slots() >= got.num_fragments,
                "{}: reserved {} slots for {} fragments",
                fixture.name(),
                render.num_fragment_slots(),
                got.num_fragments
            );
        }
    });
}

#[test]
fn gpu_and_cpu_agree_with_each_other() {
    block_on(async {
        // A tighter check than either-vs-Python: the two implementations emit in
        // a different order (the CPU is layer-major, the GPU point-major), so
        // this is what proves the sort's tie-breaking really is equivalent.
        let device = default_device();
        for dir in fixture_dirs().expect("list fixtures") {
            let fixture = Fixture::load(&dir).expect("load fixture");
            let cpu = render_pyramid_cpu(
                &fixture.points,
                &fixture.camera,
                &fixture.params,
                Some(&fixture.background),
            )
            .expect("cpu render");
            let gpu = render_pyramid(
                &device,
                &fixture.points,
                &fixture.camera,
                &fixture.params,
                Some(&fixture.background),
            )
            .await
            .expect("gpu render")
            .to_host()
            .await
            .expect("readback");

            let report = gpu
                .compare(&cpu, TOL)
                .unwrap_or_else(|e| panic!("{} GPU vs CPU: {e}", fixture.name()));
            println!(
                "{:<34} GPU vs CPU max|feature|={:.3e}",
                fixture.name(),
                report.max_feature
            );
        }
    });
}

#[test]
fn an_empty_point_set_renders_a_pure_background() {
    block_on(async {
        use brush_pyramid::params::PyramidParams;
        use brush_pyramid::scene::{Camera, PointSet};

        let device = default_device();
        let points = PointSet::new(vec![], vec![], vec![], vec![], 4).expect("empty point set");
        let camera = Camera {
            width: 16,
            height: 12,
            fx: 10.0,
            fy: 10.0,
            cx: 8.0,
            cy: 6.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
        };
        let params = PyramidParams {
            num_layers: 2,
            ..PyramidParams::default()
        };
        let bg = [0.25f32, 0.5, 0.75, 1.0];
        let images = render_pyramid(&device, &points, &camera, &params, Some(&bg))
            .await
            .expect("render")
            .to_host()
            .await
            .expect("readback");

        assert_eq!(images.num_fragments, 0);
        for layer in &images.layers {
            for (i, &t) in layer.t_final.iter().enumerate() {
                assert!((t - 1.0).abs() < 1e-9, "t_final[{i}] = {t}");
            }
            for c in 0..layer.channels {
                for y in 0..layer.height {
                    for x in 0..layer.width {
                        let got = layer.at(c, y, x);
                        assert!((got - bg[c]).abs() < 1e-6, "channel {c} at ({y}, {x}) = {got}");
                    }
                }
            }
        }
    });
}

#[test]
fn gpu_slot_counts_match_the_cpu_budget_point_for_point() {
    block_on(async {
        // The most localising check available: stage 1 writes a per-point slot
        // budget, and a disagreement there means the layer *selection* differs —
        // mode `Trips`'s footprint gate and its `break` being the only part that
        // can — rather than something downstream in the sort or the blend.
        // Reported per point, with the projection that produced it.
        use brush_pyramid::cpu::slot_budgets;

        let device = default_device();
        for dir in fixture_dirs().expect("list fixtures") {
            let fixture = Fixture::load(&dir).expect("load fixture");
            let (expected, projected) =
                slot_budgets(&fixture.points, &fixture.camera, &fixture.params).expect("cpu budgets");
            let got = render_pyramid(
                &device,
                &fixture.points,
                &fixture.camera,
                &fixture.params,
                Some(&fixture.background),
            )
            .await
            .expect("gpu render")
            .slot_counts()
            .await
            .expect("counts readback");

            let mut mismatches = Vec::new();
            for (i, (&want, &have)) in expected.iter().zip(&got).enumerate() {
                if want != have {
                    let p = projected[i];
                    mismatches.push(format!(
                        "  point {i}: cpu {want} slots, gpu {have}; u={:.9} v={:.9} \
                         size_px={:.9} (bits {:#x}) depth={:.9} visible={}",
                        p.u, p.v, p.size_px, p.size_px.to_bits(), p.depth, p.visible
                    ));
                }
            }
            assert!(
                mismatches.is_empty(),
                "{}: {} of {} points disagree on their slot budget:\n{}",
                fixture.name(),
                mismatches.len(),
                expected.len(),
                mismatches.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
            );
            println!("{:<34} slot budgets agree for all {} points", fixture.name(), expected.len());
        }
    });
}

#[test]
fn gpu_and_cpu_agree_on_fragments_per_layer_pixel() {
    block_on(async {
        // Finest-grained localisation available without a debugger: if the two
        // implementations disagree, this names the pyramid pixel — and therefore
        // the layer and the (y, x) — where fragments were gained or lost, rather
        // than only reporting a whole-image total.
        use brush_pyramid::cpu::fragments_per_pixel;

        let device = default_device();
        for dir in fixture_dirs().expect("list fixtures") {
            let fixture = Fixture::load(&dir).expect("load fixture");
            let (want, grid) =
                fragments_per_pixel(&fixture.points, &fixture.camera, &fixture.params).expect("cpu");
            let got = render_pyramid(
                &device,
                &fixture.points,
                &fixture.camera,
                &fixture.params,
                Some(&fixture.background),
            )
            .await
            .expect("gpu render")
            .fragments_per_pixel()
            .await
            .expect("segment readback");

            let mut lines = Vec::new();
            for (flat, (&w, &g)) in want.iter().zip(&got).enumerate() {
                if w == g {
                    continue;
                }
                // Decode the flat layer-major index back to (layer, y, x).
                let layer = grid
                    .offsets()
                    .iter()
                    .rposition(|&o| o <= flat)
                    .unwrap_or(0);
                let (_, width) = grid.shapes()[layer];
                let within = flat - grid.offsets()[layer];
                lines.push(format!(
                    "  layer {layer} pixel ({}, {}) [flat {flat}]: cpu {w}, gpu {g}",
                    within / width,
                    within % width
                ));
            }
            assert!(
                lines.is_empty(),
                "{}: {} of {} layer-pixels disagree:\n{}",
                fixture.name(),
                lines.len(),
                want.len(),
                lines.iter().take(12).cloned().collect::<Vec<_>>().join("\n")
            );
            println!(
                "{:<34} per-pixel fragment counts agree across all {} layer-pixels",
                fixture.name(),
                want.len()
            );
        }
    });
}
