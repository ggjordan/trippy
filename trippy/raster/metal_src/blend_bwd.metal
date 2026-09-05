// blend_bwd.metal -- gradients of front-to-back alpha compositing.
//
// Module: trippy.raster.metal_src.blend_bwd (loaded by trippy/raster/metal_lib.py)
// Purpose: the backward twin of blend_fwd.metal. One thread per layer-pixel
//   replays that pixel's composited prefix and writes the two per-fragment
//   gradients the rest of the graph needs:
//       d_alpha[i]     = dL / d alpha_i          (F,)
//       d_feat[i][c]   = dL / d feat[pid_i][c]   (F, NUM_CHANNELS), *per
//                        fragment*; the host reduces it onto points with
//                        index_add_ (trippy/raster/blend_autograd.py).
//
// Maths (front-to-back, T_0 = 1, T_{i+1} = T_i * (1 - a_i)):
//       out      = sum_i T_i * a_i * f_i
//       T_final  = T_n
//   so, with U_i = sum_{j>i} ( prod_{i<k<j} (1 - a_k) ) * a_j * f_j  and
//            Q_i = prod_{i<k<n} (1 - a_k):
//       d out     / d a_i    = T_i * (f_i - U_i)
//       d T_final / d a_i    = -T_i * Q_i
//       d out     / d f_i[c] = T_i * a_i
//   U and Q are *division free* suffix recurrences evaluated back-to-front:
//       U_{i-1} = a_i * f_i + (1 - a_i) * U_i,      U_{n-1} = 0
//       Q_{i-1} = (1 - a_i) * Q_i,                  Q_{n-1} = 1
//   This is deliberately NOT the textbook form `colour_behind / (1 - a_i)`
//   that TRIPS uses (RenderBackward.cu:284-301, `dem = 1/(1 - alpha + 1e-9)`).
//   That form needs an epsilon guard as a_i -> 1 and loses precision for
//   large alpha; the suffix recurrence above is exact for every a_i in [0, 1]
//   and needs no guard at all. See docs/ARCHITECTURE.md "Backward pass".
//
// Invariants:
//   - NO ATOMICS: fragment i is touched by exactly one thread (the thread
//     that owns its segment), so d_alpha/d_feat writes never collide. The
//     reduction onto points happens in torch (docs/ARCHITECTURE.md).
//   - The composited prefix is taken from the forward's `n_used`, not
//     re-derived from T_CUTOFF, so forward and backward agree by
//     construction. Fragments outside that prefix keep the zero the host
//     allocated: they did not influence the forward output.
//   - T_i is replayed with the *same* `T *= (1 - a)` recurrence and the same
//     float32 arithmetic as blend_fwd, so the two agree bit-for-bit.
//   - All buffers contiguous, exact dtypes; the host wrapper asserts both
//     (compile_shader binds raw storage and would read garbage otherwise).
//   - Buffer order here must match the argument order in metal_lib.blend_bwd.
//
// Units / frames: `frag_alpha` is dimensionless in (0, 1); `grad_out` and
//   `grad_t_final` are dL/d(output) in whatever units the loss uses; the
//   gradients written carry those same units.
//
// Related docs: docs/ARCHITECTURE.md ("Backward pass data flow"),
//   docs/TRIPS_REFERENCE.md section 4, docs/LIMITATIONS.md.

#include <metal_stdlib>
using namespace metal;

// Templated by trippy/raster/metal_lib.py before compilation.
constant constexpr int NUM_CHANNELS = TRIPPY_NUM_CHANNELS;
constant constexpr int MAX_FRAGS = TRIPPY_MAX_FRAGS;

kernel void blend_bwd(
    device float* d_alpha            [[buffer(0)]],   // (F,)      dL/d alpha_i
    device float* d_feat             [[buffer(1)]],   // (F, NUM_CHANNELS) per-fragment dL/d feat
    device const float* grad_out     [[buffer(2)]],   // (P, NUM_CHANNELS) dL/d out
    device const float* grad_t_final [[buffer(3)]],   // (P,)      dL/d T_final
    device const int* offsets        [[buffer(4)]],   // (P + 1,)  segment starts
    device const int* frag_point_id  [[buffer(5)]],   // (F,)      index into `feat`
    device const float* frag_alpha   [[buffer(6)]],   // (F,)      in (0, 1)
    device const float* feat         [[buffer(7)]],   // (N, NUM_CHANNELS) row-major
    device const int* n_used         [[buffer(8)]],   // (P,)      forward's composited prefix
    constant long& n_pixels          [[buffer(9)]],   // P
    uint gid                         [[thread_position_in_grid]])
{
    if (long(gid) >= n_pixels) {
        return;
    }

    const int start = offsets[gid];
    int n = n_used[gid];
    if (n > MAX_FRAGS) {
        n = MAX_FRAGS;   // defensive: the forward never reports more than the cap
    }
    if (n <= 0) {
        return;          // nothing was composited here; the host zero-filled both outputs
    }

    // Pass 1 -- replay the forward transmittance recurrence and stash T_i.
    // MAX_FRAGS is a compile-time constant (16), so this array is a handful
    // of registers / a few bytes of thread-private stack, not a spill.
    float t_stack[MAX_FRAGS];
    float transmittance = 1.0f;
    for (int i = 0; i < n; ++i) {
        t_stack[i] = transmittance;
        transmittance *= (1.0f - frag_alpha[start + i]);
    }

    device const float* g_out = grad_out + long(gid) * NUM_CHANNELS;
    const float g_t = grad_t_final[gid];

    // Pass 2 -- back-to-front, carrying the suffix accumulators U (per
    // channel) and Q (scalar) described in the header.
    float suffix[NUM_CHANNELS];
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        suffix[c] = 0.0f;
    }
    float suffix_t = 1.0f;

    for (int i = n - 1; i >= 0; --i) {
        const int f = start + i;
        const float alpha = frag_alpha[f];
        const float t_i = t_stack[i];
        device const float* f_in = feat + long(frag_point_id[f]) * NUM_CHANNELS;
        device float* d_feat_frag = d_feat + long(f) * NUM_CHANNELS;

        const float weight = t_i * alpha;
        float g_alpha = 0.0f;
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            const float go = g_out[c];
            d_feat_frag[c] = weight * go;
            g_alpha += go * t_i * (f_in[c] - suffix[c]);
        }
        g_alpha -= g_t * t_i * suffix_t;
        d_alpha[f] = g_alpha;

        const float one_minus = 1.0f - alpha;
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            suffix[c] = alpha * f_in[c] + one_minus * suffix[c];
        }
        suffix_t *= one_minus;
    }
}
