//! Render one frame from an exported point set + camera JSON to a PNG.
//!
//! Module: `brush-pyramid` example `render_frame`
//! Purpose: the smallest end-to-end exercise of the crate — load what
//!     trippy's Python side exports, rasterise it, and write something a
//!     human can open. It is also the sanity check that the GPU path works
//!     outside the test harness.
//! Invariants:
//!     - Uses the GPU path when built with `--features gpu`, and the CPU
//!       reference otherwise, so the example builds and runs either way.
//!     - Writes only the requested pyramid level's first three channels.
//!     - No image-codec dependency: PNG encoding is [`brush_pyramid::png`].
//!
//! # Usage
//!
//! ```text
//! cargo run --example render_frame --features gpu -- \
//!     --points  tests/fixtures/synthetic/raster_fixture_trips_half/points.npz \
//!     --camera  tests/fixtures/synthetic/raster_fixture_trips_half/camera.json \
//!     --mode    trips \
//!     --out     /tmp/frame.png
//! ```
//!
//! Related docs: `rust/README.md`; `docs/GEOMETRY.md`.

use std::path::PathBuf;

use brush_pyramid::params::{Mode, PixelCenter, PyramidHalving, PyramidParams};
use brush_pyramid::scene::{Camera, PointSet};
use brush_pyramid::{png, PyramidImages};

/// Parsed command line.
struct Args {
    points: PathBuf,
    camera: PathBuf,
    out: PathBuf,
    level: usize,
    scale: f32,
    background: Option<Vec<f32>>,
    params: PyramidParams,
}

const USAGE: &str = "\
render_frame — rasterise one frame to a PNG

  --points  <file.npz>    point set (xyz/size/feat/conf, or trippy's size0/rgb0/conf0)
  --camera  <file.json>   camera (width, height, fx, fy, cx, cy, R, t)
  --out     <file.png>    where to write the image
  --mode    <name>        trips (default) | trilinear | broadcast
  --layers  <n>           pyramid layers (default 5)
  --level   <n>           which pyramid level to write (default 0, the finest)
  --pixel-center <name>   half (default) | integer
  --halving <name>        ceil (default) | floor
  --background <a,b,..>   one value per feature channel (default: none)
  --scale   <f>           exposure multiplier applied before clamping (default 1.0)
";

fn parse_args() -> Result<Args, String> {
    let mut points = None;
    let mut camera = None;
    let mut out = None;
    let mut level = 0usize;
    let mut scale = 1.0f32;
    let mut background = None;
    let mut params = PyramidParams::default();

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        match flag.as_str() {
            "--points" => points = Some(PathBuf::from(value()?)),
            "--camera" => camera = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--level" => level = value()?.parse().map_err(|e| format!("--level: {e}"))?,
            "--scale" => scale = value()?.parse().map_err(|e| format!("--scale: {e}"))?,
            "--layers" => {
                params.num_layers = value()?.parse().map_err(|e| format!("--layers: {e}"))?;
            }
            "--mode" => {
                params.mode = match value()?.as_str() {
                    "trips" => Mode::Trips,
                    "trilinear" => Mode::Trilinear,
                    "broadcast" => Mode::Broadcast,
                    other => return Err(format!("unknown --mode {other:?}\n\n{USAGE}")),
                };
            }
            "--pixel-center" => {
                params.pixel_center = match value()?.as_str() {
                    "half" => PixelCenter::Half,
                    "integer" => PixelCenter::Integer,
                    other => return Err(format!("unknown --pixel-center {other:?}\n\n{USAGE}")),
                };
            }
            "--halving" => {
                params.halving = match value()?.as_str() {
                    "ceil" => PyramidHalving::Ceil,
                    "floor" => PyramidHalving::Floor,
                    other => return Err(format!("unknown --halving {other:?}\n\n{USAGE}")),
                };
            }
            "--background" => {
                background = Some(
                    value()?
                        .split(',')
                        .map(|v| v.trim().parse::<f32>().map_err(|e| format!("--background: {e}")))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        points: points.ok_or_else(|| format!("--points is required\n\n{USAGE}"))?,
        camera: camera.ok_or_else(|| format!("--camera is required\n\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n\n{USAGE}"))?,
        level,
        scale,
        background,
        params,
    })
}

/// Rasterise, on the GPU. Both `await`s are single device readbacks, so the
/// example drives them with the crate's own executor rather than depending on
/// an async runtime.
#[cfg(feature = "gpu")]
fn render(
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
) -> Result<PyramidImages, String> {
    use brush_pyramid::gpu::{block_on, default_device, render_pyramid};
    let device = default_device();
    eprintln!("backend: wgpu ({device:?})");
    block_on(async {
        render_pyramid(&device, points, camera, params, background)
            .await?
            .to_host()
            .await
    })
}

/// Rasterise, on the CPU reference. Same signature, no GPU dependency.
#[cfg(not(feature = "gpu"))]
fn render(
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
) -> Result<PyramidImages, String> {
    eprintln!("backend: CPU reference (rebuild with --features gpu for wgpu)");
    brush_pyramid::cpu::render_pyramid_cpu(points, camera, params, background)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let points = PointSet::from_npz(&args.points)?;
    let camera = Camera::from_json(&args.camera)?;
    eprintln!(
        "{} points, {}x{}, C={}, mode {:?}, {} layers",
        points.len(),
        camera.width,
        camera.height,
        points.num_channels,
        args.params.mode,
        args.params.num_layers
    );

    let images = render(&points, &camera, &args.params, args.background.as_deref())?;

    let layer = images
        .layers
        .get(args.level)
        .ok_or_else(|| format!("--level {} but only {} layers", args.level, images.layers.len()))?;
    let rgb = png::feature_to_rgb8(
        &layer.feature,
        layer.channels,
        layer.height,
        layer.width,
        args.scale,
    )?;
    png::write_rgb8(&args.out, &rgb, layer.width, layer.height)?;

    let covered = layer.n_used.iter().filter(|&&n| n > 0).count();
    eprintln!(
        "{} fragments {:?} per layer; level {} is {}x{}, {:.0}% covered -> {}",
        images.num_fragments,
        images.fragments_per_layer,
        args.level,
        layer.width,
        layer.height,
        100.0 * covered as f64 / (layer.width * layer.height) as f64,
        args.out.display()
    );
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
