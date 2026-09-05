//! Render a complete TRIPS frame on the GPU: pyramid -> U-Net -> tone map.
//!
//! Module: `brush-unet` example `render_frame_full`
//! Purpose: the first end-to-end forward pass of the Rust port, and trippy's
//!     first honest Mac frame-time number for stage 3. Takes exactly what
//!     `tools/export_unet_safetensors.py horse-e2e` writes — a point set, a
//!     camera, the render parameters and a weight file — and produces the
//!     displayed RGB frame as a PNG, printing the wall time of each of the
//!     three stages.
//! Invariants:
//!     - **GPU work**: build and run it through the queue
//!       (`scripts/gpu_submit.sh`), never directly (AGENTS.md section 6).
//!     - Timings are taken after a warm-up iteration and each stage is
//!       forced to complete (one-element readback) before the clock is read,
//!       so the numbers are per-stage, not "everything at the last await".
//!     - The tone mapper's output is already display-referred in [0, 1]; the
//!       PNG writer only quantises, it does not tone-map again.
//! Units: milliseconds, wall clock.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example render_frame_full --features gpu -- \
//!     --points  $TRIPPY_OUTPUT/brush/horse/view_00008_points.npz \
//!     --camera  $TRIPPY_OUTPUT/brush/horse/view_00008_camera.json \
//!     --params  $TRIPPY_OUTPUT/brush/horse/view_00008_params.json \
//!     --weights $TRIPPY_OUTPUT/brush/horse/horse_unet.safetensors \
//!     --out     $TRIPPY_OUTPUT/brush/horse/frame.png --iters 10
//! ```
//!
//! Related docs: `rust/README.md`; `research/trips-metal.md`.

// Everything below needs Burn + wgpu. Without the `gpu` feature the example
// still has to *compile* (`cargo test` builds every example), so the whole
// body is gated and a stub `main` explains how to build it for real.
#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "render_frame_full needs the GPU path: rebuild with `--features gpu` \
         (and run it through scripts/gpu_submit.sh)."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
mod gpu_main {
use std::path::PathBuf;
use std::time::Instant;

use brush_pyramid::gpu::{block_on, default_device, render_pyramid, PyramidRender};
use brush_pyramid::params::{Mode, PixelCenter, PyramidHalving, PyramidParams};
use brush_pyramid::png;
use brush_pyramid::scene::{Camera, PointSet};
use brush_unet::camera::NeuralCamera;
use brush_unet::net::Unet;
use brush_unet::Weights;
use burn::tensor::Tensor;

const USAGE: &str = "\
render_frame_full — pyramid -> U-Net -> tone map, on wgpu

  --points   <file.npz>          point set (xyz/size/feat/conf), camera-space
  --camera   <file.json>         width/height/fx/fy/cx/cy/R/t
  --weights  <file.safetensors>  trippy.net.export_safetensors output
  --out      <file.png>          where to write the frame
  --params   <file.json>         per-view render params (background, thresholds)
  --frame    <n>                 image index for exposure/white balance (default: from --params)
  --iters    <n>                 timed iterations after one warm-up (default 5)
";

struct Args {
    points: PathBuf,
    camera: PathBuf,
    weights: PathBuf,
    out: PathBuf,
    params: Option<PathBuf>,
    frame: Option<usize>,
    iters: usize,
}

/// The JSON `tools/export_unet_safetensors.py horse-e2e` writes beside each
/// view. Every field has the TRIPS default the parity engine uses, so the
/// example still runs without it.
#[derive(serde::Deserialize)]
struct ViewParams {
    #[serde(default = "default_layers")]
    num_layers: usize,
    #[serde(default = "default_max_frags")]
    max_frags: u32,
    #[serde(default = "default_t_cutoff")]
    t_cutoff: f32,
    #[serde(default)]
    alpha_min: f32,
    #[serde(default = "default_znear")]
    znear: f32,
    #[serde(default)]
    frame_index: usize,
    #[serde(default)]
    background: Vec<f32>,
}

const fn default_layers() -> usize {
    8
}
const fn default_max_frags() -> u32 {
    16
}
const fn default_t_cutoff() -> f32 {
    1e-3
}
const fn default_znear() -> f32 {
    1e-6
}

impl Default for ViewParams {
    fn default() -> Self {
        Self {
            num_layers: default_layers(),
            max_frags: default_max_frags(),
            t_cutoff: default_t_cutoff(),
            alpha_min: 0.0,
            znear: default_znear(),
            frame_index: 0,
            background: Vec::new(),
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let (mut points, mut camera, mut weights, mut out, mut params) = (None, None, None, None, None);
    let mut frame = None;
    let mut iters = 5usize;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        match flag.as_str() {
            "--points" => points = Some(PathBuf::from(value()?)),
            "--camera" => camera = Some(PathBuf::from(value()?)),
            "--weights" => weights = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--params" => params = Some(PathBuf::from(value()?)),
            "--frame" => frame = Some(value()?.parse().map_err(|e| format!("--frame: {e}"))?),
            "--iters" => iters = value()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
    }
    Ok(Args {
        points: points.ok_or_else(|| format!("--points is required\n\n{USAGE}"))?,
        camera: camera.ok_or_else(|| format!("--camera is required\n\n{USAGE}"))?,
        weights: weights.ok_or_else(|| format!("--weights is required\n\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n\n{USAGE}"))?,
        params,
        frame,
        iters: iters.max(1),
    })
}

/// Force every queued operation feeding `tensor` to finish, by reducing it to
/// a scalar and reading that back.
///
/// A one-element *slice* would be cheaper but is not a safe barrier: a fusion
/// backend is free to narrow the work to the elements actually read. Summing
/// touches every element, so the stage really has run when this returns. The
/// reduction itself costs well under a millisecond at 1080p and is charged to
/// the stage being measured — i.e. the per-stage numbers are very slightly
/// pessimistic, never optimistic.
async fn barrier(tensor: &Tensor<4>) {
    let _ = tensor
        .clone()
        .sum()
        .into_data_async()
        .await
        .expect("barrier readback");
}

/// Same, for the pyramid's finest layer.
async fn barrier_pyramid(render: &PyramidRender) {
    barrier(&render.layer_tensor(0)).await;
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let view: ViewParams = match &args.params {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        None => ViewParams::default(),
    };
    let frame = args.frame.unwrap_or(view.frame_index);

    let wgpu_device = default_device();
    let device = wgpu_device.clone().into();
    eprintln!("backend: wgpu ({wgpu_device:?})");

    let points = PointSet::from_npz(&args.points)?;
    let geometry = Camera::from_json(&args.camera)?;
    let weights = Weights::from_file(&args.weights)?;
    let net = Unet::load(&weights, &device)?;
    let tone = NeuralCamera::load(&weights, &device)?;

    let params = PyramidParams {
        mode: Mode::Trips,
        num_layers: view.num_layers,
        pixel_center: PixelCenter::Integer,
        halving: PyramidHalving::Ceil,
        max_frags: view.max_frags,
        t_cutoff: view.t_cutoff,
        alpha_min: view.alpha_min,
        znear: view.znear,
    };
    let background = (!view.background.is_empty()).then_some(view.background.as_slice());

    eprintln!(
        "{} points, {}x{}, C={}, L={}, frame {frame}",
        points.len(),
        geometry.width,
        geometry.height,
        points.num_channels,
        params.num_layers
    );

    // Timing method: three CUMULATIVE prefixes of the pipeline, each timed
    // from scratch with a single barrier at its own end, interleaved round-robin
    // so GPU clock ramp cannot bias one prefix against another.
    //
    //   t_p   = pyramid
    //   t_pu  = pyramid + U-Net
    //   t_puc = pyramid + U-Net + tone map   (= the whole frame)
    //
    // Per-stage costs are the differences. Barriers *inside* one timed run were
    // tried first and are not trustworthy: they reported a U-Net cost that is
    // physically impossible (82 GFLOP of 1080p convolutions in 1.3 ms on a
    // 21.5 TFLOPS-peak GPU), i.e. the work was still landing outside the window
    // being measured. Cumulative prefixes have no intermediate barrier to get
    // wrong — the only thing each measurement waits on is its own final result.
    let mut prefix_pyramid = Vec::new();
    let mut prefix_unet = Vec::new();
    let mut prefix_frame = Vec::new();
    let mut final_rgb = None;

    // One warm-up frame: the first iteration pays for shader compilation and
    // buffer-pool growth, which is not a viewer's steady state. Reported
    // separately so the cost is visible rather than hidden.
    {
        let start = Instant::now();
        let render = render_pyramid(&wgpu_device, &points, &geometry, &params, background).await?;
        let rgb = tone.forward(net.forward(&render.layer_tensors())?, frame)?;
        barrier(&rgb).await;
        eprintln!(
            "warm-up frame (shader compilation included): {:.1} ms",
            (Instant::now() - start).as_secs_f64() * 1e3
        );
    }

    for _ in 0..args.iters {
        // pyramid only
        let start = Instant::now();
        let render = render_pyramid(&wgpu_device, &points, &geometry, &params, background).await?;
        barrier_pyramid(&render).await;
        prefix_pyramid.push((Instant::now() - start).as_secs_f64() * 1e3);

        // pyramid + U-Net
        let start = Instant::now();
        let render = render_pyramid(&wgpu_device, &points, &geometry, &params, background).await?;
        let raw = net.forward(&render.layer_tensors())?;
        barrier(&raw).await;
        prefix_unet.push((Instant::now() - start).as_secs_f64() * 1e3);

        // the whole frame
        let start = Instant::now();
        let render = render_pyramid(&wgpu_device, &points, &geometry, &params, background).await?;
        let rgb = tone.forward(net.forward(&render.layer_tensors())?, frame)?;
        barrier(&rgb).await;
        prefix_frame.push((Instant::now() - start).as_secs_f64() * 1e3);
        final_rgb = Some(rgb);
    }

    let rgb = final_rgb.expect("at least one iteration");
    let [_, channels, height, width] = rgb.dims();
    let data = rgb
        .into_data_async()
        .await
        .map_err(|e| format!("readback: {e:?}"))?
        .into_vec::<f32>()
        .map_err(|e| format!("expected f32: {e:?}"))?;
    let pixels = png::feature_to_rgb8(&data, channels, height, width, 1.0)?;
    png::write_rgb8(&args.out, &pixels, width, height)?;

    let (t_p, t_pu, t_puc) = (
        median(prefix_pyramid),
        median(prefix_unet),
        median(prefix_frame),
    );
    println!(
        "{width}x{height}  median over {} iters, cumulative prefixes: pyramid {t_p:.1} ms | \
         +unet {t_pu:.1} ms | +camera {t_puc:.1} ms",
        args.iters
    );
    println!(
        "{width}x{height}  per stage: pyramid {t_p:.1} ms | unet {:.1} ms | camera {:.1} ms | \
         frame {t_puc:.1} ms ({:.1} fps)",
        t_pu - t_p,
        t_puc - t_pu,
        1000.0 / t_puc
    );
    eprintln!("wrote {}", args.out.display());
    Ok(())
}

pub fn main() {
    if let Err(message) = block_on(run()) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
}

#[cfg(feature = "gpu")]
fn main() {
    gpu_main::main();
}
