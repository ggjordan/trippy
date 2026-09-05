// blend_fwd.metal -- front-to-back alpha compositing of sorted fragments.
//
// Module: trippy.raster.metal_src.blend_fwd (loaded by trippy/raster/metal_lib.py)
// Purpose: the only GPU kernel in the pyramid rasteriser forward pass. One
//   thread per layer-pixel walks that pixel's contiguous, depth-sorted
//   fragment segment and accumulates `out += T * alpha * feature`,
//   `T *= (1 - alpha)`, exactly as TRIPS's FastSortAndBlend does
//   (third_party/TRIPS/src/lib/rendering/RenderForward.cu:3529-3559).
//
// Invariants:
//   - NO ATOMICS. Fragments are pre-sorted by (layer, pixel, depth) and each
//     thread owns one segment, so every write is to a distinct address. This
//     is trippy's redesign, not a port: Metal via torch.mps.compile_shader has
//     no 64-bit atomics (docs/LIMITATIONS.md).
//   - All buffers are contiguous; the host wrapper asserts dtype and
//     contiguity because compile_shader reinterprets raw storage and would
//     silently read garbage otherwise.
//   - NUM_CHANNELS and MAX_FRAGS are baked in at compile time (the source is
//     templated per (C, max_frags) and the compiled library cached), so the
//     inner loops unroll and `acc` stays in registers.
//   - Buffer order here must match the argument order in metal_lib.blend_fwd.
//
// Units / frames: `frag_depth` is camera-space z in world units (positive, in
//   front of the camera); `frag_alpha` is dimensionless in (0, 1); features
//   are whatever the point texture stores (linear RGB or learned channels).
//
// Related docs: docs/ARCHITECTURE.md (forward data flow),
//   docs/TRIPS_REFERENCE.md section 3, docs/GEOMETRY.md.

#include <metal_stdlib>
using namespace metal;

// Templated by trippy/raster/metal_lib.py before compilation.
constant constexpr int NUM_CHANNELS = TRIPPY_NUM_CHANNELS;
constant constexpr int MAX_FRAGS = TRIPPY_MAX_FRAGS;
constant constexpr float T_CUTOFF = TRIPPY_T_CUTOFF;

kernel void blend_fwd(
    device float* out                [[buffer(0)]],   // (P, NUM_CHANNELS) row-major
    device float* t_final            [[buffer(1)]],   // (P,)  transmittance left after the segment
    device int* n_used               [[buffer(2)]],   // (P,)  fragments actually composited
    device float* depth_sum          [[buffer(3)]],   // (P,)  sum of T * alpha * depth
    device const int* offsets        [[buffer(4)]],   // (P + 1,) segment starts, non-decreasing
    device const int* frag_point_id  [[buffer(5)]],   // (F,)  index into `feat`
    device const float* frag_alpha   [[buffer(6)]],   // (F,)  in (0, 1)
    device const float* frag_depth   [[buffer(7)]],   // (F,)  world units, > 0
    device const float* feat         [[buffer(8)]],   // (N, NUM_CHANNELS) row-major
    constant long& n_pixels          [[buffer(9)]],   // P
    uint gid                         [[thread_position_in_grid]])
{
    if (long(gid) >= n_pixels) {
        return;
    }

    const int start = offsets[gid];
    const int end = offsets[gid + 1];

    float acc[NUM_CHANNELS];
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        acc[c] = 0.0f;
    }
    float transmittance = 1.0f;
    float d_sum = 0.0f;
    int used = 0;

    for (int i = start; i < end; ++i) {
        // Both stopping rules are checked *before* consuming the fragment, so
        // `used` and `t_final` describe exactly the composited prefix.
        if (used >= MAX_FRAGS) {
            break;
        }
        if (transmittance < T_CUTOFF) {
            break;
        }
        const float alpha = frag_alpha[i];
        const float weight = transmittance * alpha;
        device const float* f = feat + long(frag_point_id[i]) * NUM_CHANNELS;
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            acc[c] += weight * f[c];
        }
        d_sum += weight * frag_depth[i];
        transmittance *= (1.0f - alpha);
        used += 1;
    }

    device float* out_pixel = out + long(gid) * NUM_CHANNELS;
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        out_pixel[c] = acc[c];
    }
    t_final[gid] = transmittance;
    n_used[gid] = used;
    depth_sum[gid] = d_sum;
}
