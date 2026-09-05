// blit.wgsl — put a device-resident TRIPS frame on the screen.
//
// Module: trips-viewer, the paint callback's fragment stage.
// Purpose: the viewer's three view modes all end in one flat, channel-last
//     f32 buffer of `height * width * channels` values, which this shader
//     reads directly out of the Burn/CubeCL allocation with no copy. Adapted
//     from Brush's `apps/brush-app/src/ui/shaders/splat_backbuffer.wgsl`
//     (Apache-2.0, ArthurBrussee), whose buffer is packed RGBA8 instead.
// Invariants:
//     - Sampling is bilinear so `--scale` below 1.0 upsamples smoothly
//       instead of showing blocks; at scale 1.0 the weights are 0 or 1 and it
//       degenerates to an exact copy.
//     - The network's output is already display-referred; nothing here
//       tone-maps, it only clamps for display.
// Related docs: rust/README.md; docs/decisions/ADR-0006-viewer-integration.md.

struct Uniforms {
    // Rendered image size, which is the WINDOW size times `render_scale`.
    img_width: u32,
    img_height: u32,
    // Values per pixel in `image_data`. 3 or 4 for features, 1 for coverage.
    channels: u32,
    // 0 = network RGB, 1 = raw level-0 features, 2 = coverage.
    mode: u32,
}

const MODE_NETWORK: u32 = 0u;
const MODE_RAW: u32 = 1u;
const MODE_COVERAGE: u32 = 2u;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> image_data: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle, same oversized-triangle trick Brush uses.
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// One pixel's first three stored values, or its single value broadcast.
//
// Two layouts, because the two producers have two natural ones and converting
// either would cost a full-image copy per frame:
//   MODE_NETWORK  planar CHW, straight out of the U-Net's last convolution
//                 (`[1, 3, H, W]`), so channel `c` starts at `c * H * W`.
//   otherwise     channel-last, which is how the rasteriser composites its
//                 flat `(P, C)` buffer.
fn fetch(px: u32, py: u32) -> vec3<f32> {
    let cx = min(px, uniforms.img_width - 1u);
    let cy = min(py, uniforms.img_height - 1u);
    let pixel = cy * uniforms.img_width + cx;

    if (uniforms.mode == MODE_NETWORK) {
        let plane = uniforms.img_width * uniforms.img_height;
        return vec3<f32>(
            image_data[pixel],
            image_data[plane + pixel],
            image_data[2u * plane + pixel],
        );
    }
    let base = pixel * uniforms.channels;
    if (uniforms.channels == 1u) {
        let v = image_data[base];
        return vec3<f32>(v, v, v);
    }
    return vec3<f32>(image_data[base], image_data[base + 1u], image_data[base + 2u]);
}

// The coverage ramp: black where nothing was drawn, warm where the raster
// covered the pixel. Deliberately not a photographic palette -- this view
// exists to answer "which pixels did the network invent?".
fn coverage_ramp(t: f32) -> vec3<f32> {
    let c = clamp(t, 0.0, 1.0);
    return vec3<f32>(c, c * c, c * c * c * 0.6);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Bilinear sample at the pixel centre convention `uv * size - 0.5`.
    let fx = in.uv.x * f32(uniforms.img_width) - 0.5;
    let fy = in.uv.y * f32(uniforms.img_height) - 0.5;
    let x0 = max(floor(fx), 0.0);
    let y0 = max(floor(fy), 0.0);
    let tx = clamp(fx - x0, 0.0, 1.0);
    let ty = clamp(fy - y0, 0.0, 1.0);
    let ix = u32(x0);
    let iy = u32(y0);

    let c00 = fetch(ix, iy);
    let c10 = fetch(ix + 1u, iy);
    let c01 = fetch(ix, iy + 1u);
    let c11 = fetch(ix + 1u, iy + 1u);
    let value = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);

    if (uniforms.mode == MODE_COVERAGE) {
        // The buffer holds `t_final`, remaining transmittance: 1.0 means the
        // rasteriser drew nothing there and every displayed pixel came from
        // the network. Show coverage = 1 - t_final.
        return vec4<f32>(coverage_ramp(1.0 - value.r), 1.0);
    }
    return vec4<f32>(clamp(value, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
