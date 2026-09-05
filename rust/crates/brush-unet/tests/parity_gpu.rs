//! GPU parity: the Burn U-Net + tone mapper vs trippy's PyTorch forward.
//!
//! Module: `brush-unet` integration test `parity_gpu`
//! Purpose: prove that `brush_unet::net::Unet` and
//!     `brush_unet::camera::NeuralCamera` reproduce
//!     `trippy.net.unet.MultiScaleUnet2dDecOnlySmallFixed` and
//!     `trippy.net.camera_model.NeuralCamera` bit-for-bit within float32
//!     noise, and that the whole `pyramid -> U-Net -> tone map` chain
//!     reproduces `trippy.render.parity`'s frame on the PUBLIC Zenodo horse
//!     scene.
//! Invariants:
//!     - **This is GPU work.** It only compiles under the `gpu` feature and
//!       must be launched through the queue, never directly:
//!
//!       scripts/gpu_submit.sh --prio 12 --wait brush-unet-gpu-1 -- \
//!         bash -c 'cd rust && cargo test -p brush-unet --features gpu \
//!                  --release --test parity_gpu -- --nocapture'
//!
//!     - The two fixture tests are self-contained (committed, synthetic,
//!       random weights). The horse test SKIPS with a printed note when the
//!       exports are absent, because they are large and live in
//!       `$TRIPPY_OUTPUT/brush/horse/` rather than in git.
//! Related docs: `rust/README.md`; `docs/TRIPS_REFERENCE.md` Sec. 5/6;
//!     `AGENTS.md` section 6.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brush_pyramid::gpu::{block_on, default_device, render_pyramid};
use brush_pyramid::params::{Mode, PixelCenter, PyramidHalving, PyramidParams};
use brush_pyramid::scene::{Camera, PointSet};
use brush_unet::camera::NeuralCamera;
use brush_unet::net::{upload, Unet};
use brush_unet::weights::{read_plain, HostTensor};
use brush_unet::{fixture_dir, repo_root, Weights};
use burn::tensor::{Device, Tensor};

/// Absolute tolerance for the small random-weight fixtures. Both sides are
/// float32; only the order of accumulation differs.
const FIXTURE_TOL: f32 = 1e-4;

/// Mean absolute tolerance for the 1920x1080 horse frame. Looser than the
/// fixture's because 2.2 M points composited 16-deep accumulate in a
/// different order on the two backends (see `research/trips-metal.md`).
const HORSE_MEAN_ABS_TOL: f32 = 1e-3;

/// Minimum PSNR between the two implementations' horse frames, in dB.
const HORSE_MIN_PSNR: f32 = 40.0;

fn device() -> Device {
    default_device().into()
}

/// Read a `Tensor<4>` back to host, in row-major order.
async fn read4(tensor: Tensor<4>) -> Vec<f32> {
    tensor
        .into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("f32 data")
}

/// Worst absolute difference, mean absolute difference and PSNR.
struct Diff {
    max_abs: f32,
    mean_abs: f32,
    psnr: f32,
}

fn compare(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_sq = 0f64;
    for (&a, &b) in got.iter().zip(want) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += f64::from(d);
        sum_sq += f64::from(d) * f64::from(d);
    }
    let n = got.len() as f64;
    let mse = sum_sq / n;
    // PSNR against a peak signal of 1.0 (display-referred RGB), which is the
    // same convention `trippy.render.parity.psnr` uses.
    let psnr = if mse <= 0.0 {
        f32::INFINITY
    } else {
        (-10.0 * mse.log10()) as f32
    };
    Diff {
        max_abs,
        mean_abs: (sum_abs / n) as f32,
        psnr,
    }
}

fn expect_tensor<'a>(io: &'a HashMap<String, HostTensor>, name: &str) -> &'a HostTensor {
    io.get(name)
        .unwrap_or_else(|| panic!("fixture io.safetensors has no {name:?}"))
}

#[test]
fn unet_matches_pytorch_on_the_random_fixture() {
    block_on(async {
        let device = device();
        let dir = fixture_dir();
        let weights = Weights::from_file(&dir.join("weights.safetensors")).expect("weights");
        let io = read_plain(&dir.join("io.safetensors")).expect("io");

        let net = Unet::load(&weights, &device).expect("build net");
        let inputs: Vec<Tensor<4>> = (0..weights.unet.num_layers)
            .map(|level| {
                upload::<4>(expect_tensor(&io, &format!("input.{level}")), &device).expect("upload")
            })
            .collect();

        let out = net.forward(&inputs).expect("forward");
        assert_eq!(out.dims(), [1, 3, 24, 32], "output keeps the finest level's size");

        let got = read4(out).await;
        let want = &expect_tensor(&io, "unet_out").data;
        let diff = compare(&got, want);
        println!(
            "unet fixture: max|diff| = {:.3e}  mean|diff| = {:.3e}  PSNR = {:.2} dB",
            diff.max_abs, diff.mean_abs, diff.psnr
        );
        assert!(
            diff.max_abs < FIXTURE_TOL,
            "U-Net differs from PyTorch by {:.3e} (tol {FIXTURE_TOL:.0e})",
            diff.max_abs
        );
    });
}

#[test]
fn camera_matches_pytorch_on_the_random_fixture() {
    block_on(async {
        let device = device();
        let dir = fixture_dir();
        let weights = Weights::from_file(&dir.join("weights.safetensors")).expect("weights");
        let io = read_plain(&dir.join("io.safetensors")).expect("io");
        let camera = NeuralCamera::load(&weights, &device).expect("build camera");
        // `tools/export_unet_safetensors.py` evaluates the fixture at frame 1
        // so the per-image exposure and white-balance lookup is exercised
        // (frame 0's white balance is pinned to 1 by ApplyConstraints).
        let frame = 1;

        // (a) an independent probe that deliberately straddles the response
        //     LUT's [0, 1] domain on both sides.
        let probe = upload::<4>(expect_tensor(&io, "camera_probe"), &device).expect("upload");
        let probe_out = camera.forward(probe, frame).expect("camera forward");
        let diff = compare(&read4(probe_out).await, &expect_tensor(&io, "camera_probe_out").data);
        println!(
            "camera probe: max|diff| = {:.3e}  mean|diff| = {:.3e}",
            diff.max_abs, diff.mean_abs
        );
        assert!(diff.max_abs < FIXTURE_TOL, "camera differs by {:.3e}", diff.max_abs);

        // (b) the real chain: PyTorch's own U-Net output through the Burn
        //     camera must give PyTorch's own final RGB.
        let raw = upload::<4>(expect_tensor(&io, "unet_out"), &device).expect("upload");
        let rgb = camera.forward(raw, frame).expect("camera forward");
        let diff = compare(&read4(rgb).await, &expect_tensor(&io, "rgb_out").data);
        println!(
            "camera on unet_out: max|diff| = {:.3e}  mean|diff| = {:.3e}",
            diff.max_abs, diff.mean_abs
        );
        assert!(diff.max_abs < FIXTURE_TOL, "camera differs by {:.3e}", diff.max_abs);
    });
}

#[test]
fn unet_and_camera_chain_matches_pytorch_end_to_end_on_the_fixture() {
    block_on(async {
        let device = device();
        let dir = fixture_dir();
        let weights = Weights::from_file(&dir.join("weights.safetensors")).expect("weights");
        let io = read_plain(&dir.join("io.safetensors")).expect("io");
        let net = Unet::load(&weights, &device).expect("net");
        let camera = NeuralCamera::load(&weights, &device).expect("camera");

        let inputs: Vec<Tensor<4>> = (0..weights.unet.num_layers)
            .map(|level| {
                upload::<4>(expect_tensor(&io, &format!("input.{level}")), &device).expect("upload")
            })
            .collect();
        let rgb = camera
            .forward(net.forward(&inputs).expect("unet"), 1)
            .expect("camera");
        let diff = compare(&read4(rgb).await, &expect_tensor(&io, "rgb_out").data);
        println!(
            "fixture chain: max|diff| = {:.3e}  mean|diff| = {:.3e}  PSNR = {:.2} dB",
            diff.max_abs, diff.mean_abs, diff.psnr
        );
        assert!(diff.max_abs < FIXTURE_TOL, "chain differs by {:.3e}", diff.max_abs);
    });
}

// --- end-to-end on the public Zenodo horse scene ----------------------------

/// Per-view render parameters written by
/// `tools/export_unet_safetensors.py horse-e2e`.
#[derive(serde::Deserialize)]
struct ViewParams {
    num_layers: usize,
    max_frags: u32,
    t_cutoff: f32,
    alpha_min: f32,
    znear: f32,
    frame_index: usize,
    background: Vec<f32>,
}

/// Where `tools/export_unet_safetensors.py horse-e2e` writes.
fn horse_dir() -> PathBuf {
    match std::env::var("TRIPPY_OUTPUT") {
        Ok(value) => PathBuf::from(value).join("brush/horse"),
        Err(_) => repo_root().join("output/brush/horse"),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn horse_frame_matches_the_python_parity_engine() {
    let dir = horse_dir();
    let view = std::env::var("BRUSH_UNET_HORSE_VIEW").unwrap_or_else(|_| "view_00008".to_owned());
    let points_path = dir.join(format!("{view}_points.npz"));
    // Every artefact must be present: a half-finished export (the point set
    // written, the expected frame not yet) would otherwise fail as a missing
    // file rather than skip.
    let required = [
        points_path.clone(),
        dir.join(format!("{view}_camera.json")),
        dir.join(format!("{view}_params.json")),
        dir.join(format!("{view}_expected.safetensors")),
        dir.join("horse_unet.safetensors"),
    ];
    if let Some(missing) = required.iter().find(|p| !p.exists()) {
        println!(
            "SKIP horse end-to-end: {} not found. Generate the exports with\n  \
             PYTHONPATH=. TRIPS_DEVICE=cpu python tools/export_unet_safetensors.py horse-e2e --index 8",
            missing.display()
        );
        return;
    }

    block_on(async {
        let device = device();
        let params: ViewParams = read_json(&dir.join(format!("{view}_params.json")));
        let points = PointSet::from_npz(&points_path).expect("points");
        let camera_geom = Camera::from_json(&dir.join(format!("{view}_camera.json"))).expect("camera");
        let weights = Weights::from_file(&dir.join("horse_unet.safetensors")).expect("weights");
        let expected = read_plain(&dir.join(format!("{view}_expected.safetensors"))).expect("expected");

        assert_eq!(weights.unet.num_layers, params.num_layers);
        let pyramid_params = PyramidParams {
            mode: Mode::Trips,
            num_layers: params.num_layers,
            pixel_center: PixelCenter::Integer,
            halving: PyramidHalving::Ceil,
            max_frags: params.max_frags,
            t_cutoff: params.t_cutoff,
            alpha_min: params.alpha_min,
            znear: params.znear,
        };

        let render = render_pyramid(
            &default_device(),
            &points,
            &camera_geom,
            &pyramid_params,
            Some(&params.background),
        )
        .await
        .expect("render pyramid");

        let net = Unet::load(&weights, &device).expect("net");
        let tone = NeuralCamera::load(&weights, &device).expect("camera");
        let raw = net.forward(&render.layer_tensors()).expect("unet");
        let rgb = tone.forward(raw.clone(), params.frame_index).expect("tone map");

        assert_eq!(
            rgb.dims(),
            [1, 3, camera_geom.height, camera_geom.width],
            "frame keeps the full render resolution"
        );

        let unet_diff = compare(
            &read4(raw).await,
            &expected
                .get("unet_out")
                .expect("expected unet_out")
                .data,
        );
        let rgb_diff = compare(
            &read4(rgb).await,
            &expected.get("rgb").expect("expected rgb").data,
        );
        println!(
            "horse {view} ({}x{}, {} points, L={}):\n  \
             U-Net: max|diff| = {:.3e}  mean|diff| = {:.3e}  PSNR = {:.2} dB\n  \
             RGB  : max|diff| = {:.3e}  mean|diff| = {:.3e}  PSNR = {:.2} dB",
            camera_geom.width,
            camera_geom.height,
            points.len(),
            params.num_layers,
            unet_diff.max_abs,
            unet_diff.mean_abs,
            unet_diff.psnr,
            rgb_diff.max_abs,
            rgb_diff.mean_abs,
            rgb_diff.psnr,
        );
        assert!(
            rgb_diff.mean_abs < HORSE_MEAN_ABS_TOL,
            "mean |diff| {:.3e} exceeds {HORSE_MEAN_ABS_TOL:.0e}",
            rgb_diff.mean_abs
        );
        assert!(
            rgb_diff.psnr > HORSE_MIN_PSNR,
            "PSNR {:.2} dB is below {HORSE_MIN_PSNR} dB",
            rgb_diff.psnr
        );
    });
}
