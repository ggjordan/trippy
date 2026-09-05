# TRIPS Reference (porting source of truth)

Extracted from `third_party/TRIPS` at commit `a59a65b6d9a8b1c14c73bc004cc9a8956f054c24`. All values below are
taken from source; every number/formula has a `path:line`. Where the checked-out tree is missing code
(Saiga submodule not vendored — `External/saiga/` is an empty dir), that is stated explicitly rather than
guessed. Default values are those of `configs/train_normalnet.ini` (the config the README trains with)
layered over the `Settings.h` struct defaults; where the two differ, the ini value wins at runtime and is
called out.

Non-default render code paths (FUZZY_DT, FULL_BLEND, FUZZY_BLEND, and `use_layer_point_size=true`) exist in
the source but are **not exercised** by the default config. They are described only where necessary to
explain dead/optional fields.

## 1. Data flow overview

```
SceneData (COLMAP-derived ADOP scene dir)
  -> NeuralPointCloudCuda (positions [N,4]=xyz+dropout_radius, point_size [N,1])
  -> NeuralPointTexture   (texture [C,N], confidence_raw [1,N], background_color [C])
  -> PointRenderModule::forward -> BlendPointCloudForward (torch::autograd::PointRender::apply)
       per image, per batch:
         CountTiled<L>      -- project all points, atomically count fragments per (layer,pixel)
         (prefix-sum / scan of per_pixel_list_lengths -> scanned_countings)
         CollectTiled2<L> / CollectTiled2Pointsize<L,P> -- re-project, write (depth,index) into
                              full_list_buffer[batch, slot] as packed doubles, per-pixel-list slot
                              assigned via the same atomic counters + scan offsets
         FusedSortAndBlend2 -> FastSortAndBlendWPrep2WarpSharedWork<NUM_DESC,16,train>
                              -- one warp-group per (layer,pixel): bitonic-sorts the fragment list by
                                 depth in 32-wide chunks, keeps nearest 16, alpha-blends front-to-back,
                                 writes color to neural_out[layer]
  -> output_forward[layer]  (one RGBA-ish tensor per pyramid layer, layer i is layer i-1 downsampled 2x)
  -> MultiScaleUnet2dDecOnlySmallFixed (5-level gated-conv U-Net decoder, coarse-to-fine)
  -> NeuralCamera: exposure -> white balance -> vignette -> camera-response LUT (or clamp) -> [rolling shutter]
  -> Loss: VGG (weight 1) + L1 (weight 1) + SSIM (weight 1), started per only_start_vgg_after_epochs
```

Key structural fact for porters: **there is no single "select one pyramid level by footprint size and
splat" step in the default configuration.** With `use_layer_point_size=false` (the default, see §2/§3),
every point is independently rasterized into *every* pyramid layer's own downsampled resolution, each with
a full 2x2 bilinear footprint and alpha = `bilinear_weight * confidence` (no layer attenuation). The 5
resulting images are fused only inside the U-Net decoder via skip connections, not before it.

## 2. Point parametrisation

Point cloud fields (`src/lib/rendering/NeuralPointCloudCuda.h:82-115`, `.cpp:29-104`):
- `t_position` `[N,4]` = world xyz + `drop_out_radius` in `.w` (learnable, `register_parameter`,
  `NeuralPointCloudCuda.cpp:100`).
- `t_point_size` `[N,1]` = **raw pre-softplus** point size, registered as a parameter only if
  `use_pointsize=true` (`NeuralPointCloudCuda.cpp:107`, called with default `true` from
  `NeuralPointCloudCuda(model, ...)`).
- `t_index` `[N,1]` int32 — maps render-order point id -> texture column (buffer, not a parameter).

**Size init (kNN):** `SceneData::ComputeRadius(pc, n=4)` (`src/lib/data/SceneData.cpp:620-663`) builds a
`KDTree<3,vec3>` over point positions, for each point finds its `n+1=5` nearest neighbors, and sets
`pc.data[i](0) = dist(point_i, v.back())` — the distance to the farthest of those 5 (`SceneData.cpp:632-643`).
It also sets a *randomized* drop-out radius `d(3) = d(0) * sqrt(1/uniform(0,1))` (`SceneData.cpp:655-661`),
which becomes `t_position.w`. `d(0)` (the plain kNN radius) becomes the size input.
`NeuralPointCloudCudaImpl` ctor then does:
```
data_point_size.push_back(inverse_softplus(data[i](0) * 0.5f));   // NeuralPointCloudCuda.cpp:50
```
i.e. `t_point_size_raw = softplus^-1(0.5 * knn_radius)`, `inverse_softplus(x)=log(exp(x)-1)`
(`NeuralPointCloudCuda.cpp:19-24`, `beta=1, threshold=20`).
At render time the size is recovered with `_softplus` (`RenderForward.cu:154`, same beta/threshold), giving
`point_size_opt = softplus(t_point_size_raw)` — so **softplus is the forward parametrisation**, matching
the init exactly (`inverse_softplus` then `softplus` round-trips to `0.5*knn_radius`).
This world-space size is only used **if `use_layer_point_size=true`**, in which case it is further
converted to pixel units by `point_size_opt = K.fx * cam.crop_transform.fx * point_size_opt / z`
(`RenderForward.cu:270`, pinhole case). **`use_layer_point_size` defaults to `false`**
(`src/lib/data/Settings.h:67`) and is not present in `RenderParams::Params()` at all
(`Settings.h:87-101` — no `SAIGA_PARAM(use_layer_point_size)`), so **it cannot be set from any `.ini` file
and is always `false` in practice**, regardless of config. When `false`, `point_size_opt` is hardcoded to
`1.f` (`RenderForward.cu:247-248`), and the point-size parameter has no effect on rendering (only on the
otherwise-dead radius-based drop-out check via `t_position.w`, which is independent of `t_point_size`).

**Confidence:** `confidence_raw` `[1,N]` init to `0.5` everywhere (`NeuralTexture.cpp:42`). Rendered
confidence is precomputed once per forward pass (not per-fragment) via
```
confidence_value_of_point = sigmoid((10 + narrowing_param_times_epoch) * confidence_raw);  // NeuralTexture.h:42
```
`narrowing_param_times_epoch = sigmoid_narrowing_factor * current_epoch`
(`src/lib/models/Pipeline.cpp:251-253`), and `sigmoid_narrowing_factor = 0` in the default config
(`configs/train_normalnet.ini:139`), so in practice `confidence = sigmoid(10 * confidence_raw)`, i.e.
`sigmoid(5) ≈ 0.9933` at init — **not** a plain `sigmoid(confidence)` as a naive reading would assume.

**Texture:** `texture_raw` `[C,N]`, `C = pipeline_params.num_texture_channels = 4`
(`configs/train_normalnet.ini:105`). With `texture_random_init=true` (ini:34), init is
`texture_raw.uniform_(0,1)` (`NeuralTexture.cpp:28-29`, note the `factor`-based lines above it are dead —
overwritten on the next line). `background_color_raw = ones(C) * 0.25` (`NeuralTexture.cpp:37`,
`fac_init=0.25`). `PrepareTexture(abs)` (`NeuralTexture.h:45-57`) optionally takes `abs()` of both texture
and background before use; the `abs` flag is `!non_subzero_texture` in the pipeline (config
`non_subzero_texture=false` → texture is abs'd, so effectively non-negative).

**Gradients are not scaled going into autograd** for points/pose/intrinsics beyond the analytic Jacobians
in §4; `RenderParams.normalize_grads` (default `false`, ini:85) would divide by an accumulated per-point
count if enabled, but is off by default.

## 3. Forward rasteriser (default: `render_mode = TILED_BILINEAR_BLEND` = 4)

Render mode is selected from `pipeline_params.render_modes_start_epochs = {-2,-2,-2,-2,0}`
(`configs/train_normalnet.ini:109`) via `Pipeline.cpp:227-239` — index 4 (`TILED_BILINEAR_BLEND`) starts at
epoch 0, all others are permanently disabled (`-2`). Enum: `FUZZY_DT=0, FULL_BLEND=1, FUZZY_BLEND=2,
BILINEAR_BLEND=3, TILED_BILINEAR_BLEND=4` (`src/lib/rendering/PointRenderer.h:168-176`).

**Pyramid construction** (`src/lib/rendering/PointRenderer.cu:328-390`): `num_layers = net_params.
num_input_layers = 5` (`RenderModule.cpp:11`, ini `num_input_layers=5`). Layer 0 = full render resolution
`(h,w)` from the dataset; each subsequent layer is `h/=2; w/=2` (integer division, `PointRenderer.cu:378`),
`scale *= 0.5`. Each layer gets **its own** `output_forward[i]` tensor and is rasterized independently —
this is a genuine image pyramid of 5 separately-blended images, not one image with mixed-resolution splats.

**Projection** (`ProjectPointPinholeWoNormal`, `src/lib/rendering/PointRendererHelper.h:232-253`):
`view_p = V*world_p` (world-to-camera via `Sophus::SE3f`), `z=view_p.z()`, `norm_p = view_p.xy/z`,
`dist_p = distortNormalizedPoint(norm_p, distortion, dist_cutoff)` (Saiga 8-parameter distortion model —
`distortion.h` is in the un-vendored Saiga submodule, so the exact polynomial is not in this checkout;
comment at `src/lib/data/SceneData.h:229` says "8 parameter distortion model"), `image_p =
K.normalizedToImage(dist_p)`. Then `ip = cam.crop_transform.normalizedToImage(image_p)` and
`ip = rotateCropAroundCenter(ip, ...)` apply the per-frame train-time crop/rotation
(`RenderForward.cu:1521-1522`). OCam (fisheye) and spherical/ortho camera models are also implemented
(`PointRendererHelper.h`) but pinhole+distortion is the scene format actually produced by `colmap2adop`.

**CountTiled\<num_layers\>** (`RenderForward.cu:1402-1699`, launched from `PointRendererCache::CountTiled`,
`.cu:1635-1699`): for every point, for `layer in [0, num_layers)` (default path,
`use_layer_point_size=false`, `RenderForward.cu:1616-1634`): reject if `ip` out of bounds
(`ip(0)<0 || ip(0) >= layer_w-1 || ip(1)<0 || ip(1) >= layer_h-1`) or (`drop_out_points_by_radius &&
radius_pixels < drop_out_radius_threshold`); else `atomicAdd` 4 times into
`per_pixel_list_lengths[layer](batch, floor(ip.y())+{0,1}, floor(ip.x())+{0,1})` — i.e. **every point that
passes the layer-0-style bounds test contributes to the 2x2 bilinear footprint of every layer**, with
`ip *= 0.5f` and `radius_pixels *= 0.5f` each iteration (both count and radius test are halved per layer,
consistent with each layer being a 2x downsample). `radius_pixels = K.fx * crop_transform.fx *
drop_out_radius / z` (`RenderForward.cu:1481`); `drop_out_points_by_radius=false` in the default config
(`configs/train_normalnet.ini:74`), so this cull is **off by default** and only the image-bounds test
applies.

**CollectTiled2\<num_layers\>** (`RenderForward.cu:1707-2053`, host launcher
`PointRenderer.cu` via `RenderForward.cu:2524-2677`): re-does the same projection/loop, this time writing
into `full_list_buffer_data[batch, slot, {0:ip.x, 1:ip.y, 2:texture_index(as float bits), 3:point_size_opt
(only if use_layer_point_size), 4:point_id(as float bits, train only)}]` and packing the sort key as
```
double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], scanned_c);   // RenderForward.cu:1929
full_list_buffer[batch, scanned_c] = data_l;
```
i.e. the fragment's **depth `z` (as its raw float bit pattern) occupies the high 32 bits of a double**, and
the flat list slot index occupies the low 32 bits — sorting the doubles numerically sorts by depth. This
only produces a valid ascending order for `z > 0` (guaranteed by the earlier `z<=0` reject).
`slot = scanned_countings[layer](batch,y,x) + atomicAdd(per_pixel_list_lengths[layer](batch,y,x), 1) +
layer_offset` — `scanned_countings` is a prefix sum over `(layer,batch,y,x)` computed between the count and
collect passes (uploaded via `PointRendererCache::UploadLinkedListBuffers`,
`RenderForward.cu:139-166`), so there is **no explicit per-pixel list cap at collection time** — every
passing fragment gets a slot; the cap is applied only later, during sort+blend.

**Layer factor / epsilon rule** (`compute_point_size_fac`, `src/lib/rendering/PointBlending.h:81-149`,
called from `RenderForward.cu:898` and `RenderBackward.cu:371` — **only reached when
`use_layer_point_size=true`**, which is unreachable from any `.ini` per §2):
- `layer_lower = clamp(floor(log2(point_size_opt)), 0, max_layers-1)`,
  `layer_higher = clamp(ceil(log2(point_size_opt)), 0, max_layers-1)` (both `0` if `point_size_opt<=1`).
- If `layer < layer_lower`: factor `= 0`.
- If `layer_higher == 0` (point smaller than 1 pixel): **not a clamp on point size** —
  `layer_factor = (1 - 0.25) * exp(point_size_opt - 1) + 0.25` (`PointBlending.h:106`, `cutoff_value=0.25`
  hardcoded), i.e. an exponential floor of `0.25` on the *factor*, reaching `1.0` at `point_size_opt=1`.
- If `layer_lower == layer_higher`: factor `= 1`.
- Else (spans two layers): `l_l = 2^layer_lower`, `l_h = 2^layer_higher`,
  `layer_factor = (point_size_opt - l_l) / (l_h - l_l)`, flipped to `1 - that` if `layer == layer_lower`
  (`PointBlending.h:126-130`) — **linear interpolation in point-size units between the two powers of two**,
  not linear interpolation of the `log2` fraction.
- If `layer_lower == max_layers-1`: factor forced to `1`.

When `use_layer_point_size=false` (actual default), `point_size_opt=1` is passed into
`compute_point_size_fac`, which returns `layer_factor = 1` unconditionally for every layer
(`layer_higher==0` branch evaluates to `0.75*exp(0)+0.25 = 1.0`), confirming §1's claim that the default
path applies **no** layer attenuation.

**Bilinear write:** `compute_blending_fac(ip)` (`PointBlending.h:216-257`) — standard 2x2 bilinear weights
from the fractional part of `ip`; `blend_fac_index(ip, gid)` (`PointBlending.h:259-264`) selects which of
the 4 weights belongs to the destination pixel `(gx,gy)` currently being processed.
`alpha_bilin = bilinear_fac * confidence_value_of_point[texture_index]` (times `layer_factor` if enabled)
— `RenderForward.cu:3505-3512`. If the fetched `ip` differs from `(gx,gy)` by more than 1 pixel (can happen
because the point's local `ip` was stored once per point but the fragment loop can hand it to a pixel
outside the 2x2 support — a defensive check), `bilinear_fac` is forced to 0
(`RenderForward.cu:3502-3503`).

**Sort + cap (`FastSortAndBlendWPrep2WarpSharedWork<NUM_DESCRIPTORS,ELEMENTS_PER_PIXEL=16,train>`,
`RenderForward.cu:3080-3673`):** one 32-lane warp is split into `LISTS_PER_WARP = 32/16 = 2` groups of 16
lanes, each group ("warp leader stride" = `ELEMENTS_PER_PIXEL=16`) owns one `(layer,pixel)` list. Fragments
are streamed in from global memory 32 at a time, `Saiga::CUDA::bitonicWarpSort` bitonic-sorts the 32
candidates (16 already-kept + up to 16 new) by the packed depth key, and only the nearest 16
(`ELEMENTS_PER_PIXEL`) survive each round (`RenderForward.cu:3260-3399`) — **this is the per-pixel cap: 16,
not 8**, matching `PointRendererCache::max_pixels_per_list = 16` (`PointRenderer.h:178`) and
`DeviceBilinearAlphaParams::bw_sorted_maxed` shape `[..., max_pixels_per_list=16, 6 or 7]`
(`PointRenderer.cu:335-337`). Dispatch is per `num_texture_channels ∈ {3,4,8,16}`
(`RenderForward.cu:3808-3862`); default config uses `NUM_DESCRIPTORS=4`.

**Depth test:** there is **no `depth_accept` / `depth_accept_blend` usage anywhere in the
`TILED_BILINEAR_BLEND` kernels.** Both fields are copied into `DeviceRenderParams`
(`RenderConstants.h:49-50,123,135`) but never read inside `RenderForward.cu` or `RenderBackward.cu` —
`grep` confirms zero uses outside the struct definition/assignment. They are **dead in the default render
path** (plausibly consumed only by the disabled `FUZZY_DT`/`FUZZY_BLEND` kernels not covered here). Ordering
is purely "nearest 16 by depth, standard front-to-back alpha compositing," no soft depth window.

**Blend (front-to-back alpha compositing, `RenderForward.cu:3529-3559`):**
```
alpha_dest[0] = 1
alpha_dest[k] = alpha_dest[k-1] * (1 - alpha_bilin[k-1])          // via __shfl_up_sync, ELEMENTS_PER_PIXEL=16
                (or set to 0 once alpha_dest[k-1] < ALPHA_DEST_CUTOFF = 0.001)   // RenderForward.cu:3522
color_out += alpha_dest[k] * alpha_bilin[k] * texture[:, texture_index[k]]      // per descriptor channel
```
An alternate `saturated_alpha_accumulation` mode exists (`RenderForward.cu:3527-3546`, config default
`false`, `configs/train_normalnet.ini:88`) using additive/clamped alpha instead of multiplicative
transmittance — not used by default.

**Background:** after the sorted list is exhausted, if `alpha_dest >= ALPHA_DEST_CUTOFF` (i.e. still some
transmittance left) and `!environment_map`: `color_out[i] += background_color[i] * alpha_dest`
(`RenderForward.cu:3610-3620`). `enable_environment_map=false` in the default config
(`configs/train_normalnet.ini:97`), so the environment map path is unused; `background_color` is the
learned per-channel scene background (§2).

**Outputs / buffer layout pitfall:** `output_forward[i]` is allocated as `[batch, C, h, w]` (NCHW,
pre-filled with `background_color` broadcast) **whenever `train==true` or `render_mode !=
TILED_BILINEAR_BLEND`**, but as `[batch, h, w, C]` (NHWC) otherwise (`PointRenderer.cu:456-473`). The fused
kernel writes accordingly: `neural_out[layer](batch, i, gy, gx)` in train mode vs.
`neural_out[layer](batch, gy, gx, i)` at inference (`RenderForward.cu:3630-3646`). A port that only tests
the training path will silently miss this layout flip at inference time.

**Training-mode extra buffer:** when `train=true` and `alpha_dest >= ALPHA_DEST_CUTOFF`,
`bw_sorted_maxed[layer](batch,gy,gx,index_in_list, 0..6)` stores
`{texture_index, point_id, alpha_bilin, ip.x, ip.y, blend_index, point_size_opt}` per surviving fragment
(`RenderForward.cu:3568-3583`) — this is exactly the state `RenderBackward.cu` replays to avoid
re-projecting points on the backward pass.

## 4. Backward pass

Single kernel `BlendBackwardsBilinearTiled<num_descriptors, ELEMENTS_PER_PIXEL=16>`
(`src/lib/rendering/RenderBackward.cu:90-618`), one warp per `(layer,pixel)` ticket (grabbed via a global
`atomicAdd` counter, `RenderBackward.cu:132-136` — **work-stealing via atomics**, not a fixed grid).

**Forward re-walk to get per-fragment color/alpha gradients** (`RenderBackward.cu:189-309`): replays the
`alpha_dest` recursion using the saved `bw_sorted_maxed` list; for each fragment `i` (foreground) or the
implicit background (`alpha_val=1`, `is_foreground=false`):
```
g = alpha_dest * alpha_val * grad_in[ci]
atomicAdd(out_gradient_texture(ci, texture_index), g)          // if foreground
atomicAdd(background_grads[ci][lane], g)                       // if background (reduced later)
g_alpha += color[ci] * grad_in[ci]                              // accumulated over channels
conf_gradients[index_in_list] += alpha_dest * g_alpha
```
plus a second loop (`RenderBackward.cu:284-301`) that back-propagates the *transmittance* dependency of
every later fragment `j>i` on this fragment's alpha:
```
for j in [0, index_in_list):
    dem = 1 / (1 - alpha_val[j] + 1e-9)
    conf_gradients[j] -= grad_in[ci] * color[ci] * alpha_dest * alpha_val * dem   // summed over channels
```
i.e. the standard "every point occludes everything behind it" alpha-compositing gradient. A
`saturated_alpha_accumulation` variant exists in parallel (`RenderBackward.cu:227-259`) matching the forward
alt-mode.

**Confidence and layer-factor gradient** (`RenderBackward.cu:349-376`): `blend_factors = compute_blending_fac(uv)`
(re-derived from stored `ip`, not stored directly); `layer_factor, J_layerfactor_proj =
compute_point_size_fac(point_size_opt, layer, num_layers, &grad)` (only if `out_gradient_layer` buffer
exists, i.e. `t_point_size.requires_grad()` — false by default since the whole mechanism is disabled);
`grad_point_confidence = blend_factors[blend_index] * layer_factor * grad_alpha`, atomically added to
`out_gradient_confidence(0, texture_index)`.

**Point/pose/intrinsics gradients** (`RenderBackward.cu:384-465`, pinhole case via
`ProjectPointPinholeBackward`, `src/lib/rendering/PointRendererHelper.h:434-489`): chain rule through
`crop_transform.normalizedToImage -> K.normalizedToImage -> distortNormalizedPoint -> DivideByZ ->
TransformPoint(V,.)`, each stage returning its own Jacobian; final `g_point = J_point^T * (...)`,
`g_pose = J_pose^T * (...)` (6-vector: Sophus `SE3` tangent, translation+rotation), `g_k` (5: `fx,fy,cx,cy,s`),
`g_dis` (8-param distortion). All accumulated with `atomicAdd` directly to global buffers for points
(`out_gradient_points[point_id]`) and to **shared-memory accumulators** for pose/intrinsics/background
(`pose_grad[6][32]`, `intrinsics_K_grad[5][32]`, `intrinsics_dis_ocam_grad[8][32]`,
`background_grads[C][32]`, `RenderBackward.cu:161-187`), warp-reduced once at kernel exit
(`RenderBackward.cu:547-617`) and only then atomically added to the true global gradient tensors — an
optimization to avoid one atomic per fragment on the same global address.

**`distortion_gradient_factor` (0.005) and `K_gradient_factor` (0.5)** (`configs/train_normalnet.ini:81-82`,
`Settings.h:56-57`) are copied into `DeviceRenderParams` (`RenderConstants.h:54-55,127-128`) but **never
multiplied into any gradient in `PointRendererHelper.h` or `RenderBackward.cu`** — confirmed dead by
`grep -rn "distortion_gradient_factor\|K_gradient_factor"` returning only the declaration/copy sites. If a
port intends to reproduce a learning-rate-like damping of camera-intrinsic gradients, it is **not** coming
from this mechanism in this commit; check the optimizer step (`MyAdam.cu`/`Pipeline.cpp`) instead if such
damping is actually observed empirically.

**Atomics**: the reference implementation uses `atomicAdd` pervasively, both forward (fragment-count/slot
allocation in `CountTiled`/`CollectTiled2`) and backward (every per-fragment gradient reduction). This
directly contradicts an atomic-free design — see §10.

## 5. Network: `MultiScaleUnet2dDecOnlySmallFixed`

Config (`configs/train_normalnet.ini:201-220`): `num_input_layers=5, num_input_channels=4,
num_output_channels=3, num_layers=5, upsample_mode=bilinear, norm_layer_up=id, last_act=id,
conv_block_up=gated, activation=elu, filters_network=[32]*8` (only indices `0..4` used since
`num_layers=5`).

Class at `src/lib/models/Networks.h:1100-1208`. **The `gated` conv block itself
(`UnetBlockFromString("gated", ...)`) is implemented in Saiga
(`saiga/vision/torch/PartialConvUnet2d.h`), which is not vendored in this checkout
(`External/saiga/` is empty)** — the exact gated-conv formula (presumably `content_conv(x) *
sigmoid(gate_conv(x))`, a 3x3 conv pair) could not be verified from source and must be sourced from the
Saiga repo directly before porting; do not guess the formula.

Architecture (bottom expressed top-down, coarsest layer 4 -> finest layer 0; `4` = `num_input_channels`
throughout; all convs `kernel=3, stride=1, pad=1` unless noted, `norm=id`, `activation=elu` for every gated
block, `last=True` only for `up[0]`):

| Stage | Module | In (after concat) | Conv out | Skip-concat out |
|---|---|---|---|---|
| `start` (`SmallDecStartBlock`, `Networks.h:751-787`) | gated(4→24) on `inputs[4]` | 4 | 24 | `cat(inputs[4], conv_out)` = **28** |
| `up[3]` (`i=3`, `Networks.h:1123-1132`) | upsample(28→28, bilinear 2x, `align_corners=false`) then `cat(inputs[3], up)`=32; gated(32→24) | 32 | 24 | `cat(inputs[3], conv_out)` = **28** |
| `up[2]` | same pattern | 32 | 24 | 28 |
| `up[1]` | same pattern | 32 | 24 | 28 |
| `up[0]` (`last=true`, conv_out = `out_channels - num_input_channels` not `-2*`) | upsample(28→28) then `cat(inputs[0], up)`=32; gated(32→28) | 32 | 28 | `cat(inputs[0], conv_out)` = **32** |
| `final` (`Networks.h:1134-1136`) | `Conv2d(32→3, kernel=1)` + `Activation("id")` | 32 | 3 | — |

`inputs[i]` is the raw rasterizer output at pyramid layer `i` (§3), 4 channels each, concatenated as the
**skip connection at every level** (`UpsampleDecOnlySmallBlockFixedImpl::forward`,
`Networks.h:1068-1091`: `CombineBridge(features_input, upsample_input)` before the conv, and
`CombineBridge(features_input, conv_output)` again after). `masks[i]` are passed alongside every tensor
(`Networks.h:1152-1169`, `masks[i].requires_grad()==false` asserted) but with the `gated` block type and no
partial-conv variant selected, the mask channel is architecturally present but not doing partial-conv
masking (that's `conv_block="partial_multi"`, explicitly disallowed at
`Networks.h:1028,824`: `SAIGA_ASSERT(conv_block != "partial_multi")`).

**Parameter count**: cannot be given as a closed formula without the vendored gated-block definition (it
determines whether each block has 1 or 2 conv weight tensors of shape `[out,in,3,3]`). If it is the common
"gated conv = 2x parallel 3x3 convs" pattern, each block above costs `2 * (in*out*9 + out)` parameters; a
porter must confirm against the real Saiga source or against a loaded checkpoint's `state_dict` keys/shapes
before assuming this.

### 5a. UPDATE (feat/net, 2026-09-05): gated-conv formula fetched and verified against a real checkpoint

The above paragraph's "presumably" is resolved. `External/saiga/` is still an empty dir in this checkout,
but Saiga is public MIT source, so it was fetched directly from GitHub (no private data involved):

```
Source:  https://github.com/darglein/saiga
Commit:  ee7a4e6b65832433e2ca521353b7b7431c8e17a0  ("use namespace tinyeigen", 2026-03-20)
File:    src/saiga/vision/torch/PartialConvUnet2d.h:108-152 (GatedBlockImpl)
```

Exact formula (`GatedBlockImpl::forward`, PartialConvUnet2d.h:139-145):
```cpp
auto x_t = feature_transform->forward(x);   // Conv2d(in,out,3,pad=1) -> Activation (elu by config)
auto m_t = mask_transform->forward(x);      // Conv2d(in,out,3,pad=1) -> Sigmoid  (independent weights)
auto res = norm.forward(x_t * m_t);         // norm = Identity when norm_str == "id"
return {res, mask};                          // the incoming validity `mask` is passed through UNCHANGED
```
Both convs read the *same* input `x` (not each other's output, not `mask`); they have independent
weights but identical (in_channels, out_channels, kernel=3, stride=1, dilation=1, padding=1) — confirmed
against `NormFromString`/`ActivationFromString` in `src/saiga/vision/torch/TorchHelper.h:194-246` (same
commit; `"id"` -> `torch::nn::Identity()`, `"bn"` -> `BatchNorm2d(momentum=0.01)`, `"elu"` ->
`torch::nn::ELU()`). So each `GatedConvBlock(in, out)` costs exactly `2 * (9*in*out + out)` parameters
under `norm="id"` — the guess in the paragraph above was right, and is now source-verified, not assumed.

**Real-checkpoint verification** (trippy's `feat/net` port, `tests/test_net_unet.py` +
`third_party/zenodo/tt_checkpoints/checkpoint_horse/ep0600/render_net.pth`, extracted from the public
Zenodo record 10687419 checkpoint archive): loading that file via `torch.jit.load` (see correction to
Sec. 9 below) yields exactly 34 named tensors whose names (`start.conv.feature_transform.0.weight`,
`up7.convolution.mask_transform.0.bias`, `final.0.weight`, ...) and shapes match trippy's from-scratch
Python port **tensor-for-tensor, in registration order**, once `num_layers` is set to match that
checkpoint's own `params.ini` (see next paragraph) — the strongest possible confirmation of this section's
architecture table and the gated-conv formula above, independent of reading Networks.h a second time.

**IMPORTANT discrepancy discovered from the real checkpoint's `params.ini`**: the public Zenodo Tanks &
Temples checkpoints (all 8 scenes: family, francis, horse, lighthouse, m60, panther, playground, train)
were trained with `num_input_layers = 8` and `num_layers = 8`, **not** the `num_layers=5` shipped in this
checkout's `configs/train_normalnet.ini`. Every other relevant field matches exactly
(`num_input_channels=4, num_output_channels=3, filters_network=32 32 32 32 32 32 32 32, upsample_mode=
bilinear, norm_layer_up=id, last_act=id, conv_block_up=gated, activation=elu`) — only the pyramid depth
differs. `filters_network` already has 8 entries in the shipped ini (this section's opening paragraph
notes "only indices 0..4 used since num_layers=5"); with `num_layers=8`, all 8 entries are used, all still
equal to 32, so the "-2C"/"-C" channel-bookkeeping formula in this section's table is unchanged, just
applied 3 more times (up7..up2 non-last, up1 last). Total parameter count at `num_layers=8`:
`1776 (start) + 6*13872 (up7..up2) + 16184 (up1, last) + 99 (final) = 101291`. **A porter targeting
bit-exact compatibility with the released Tanks & Temples checkpoints must use `num_layers=8`, not the
`train_normalnet.ini` default of 5** — trippy's `NetworkConfig` defaults to 5 per this task's brief
(matching `train_normalnet.ini`, the config named in the task) but takes `num_layers` as a parameter for
exactly this reason. See `docs/LIMITATIONS.md` for the load attempt report.

## 6. Neural camera / tone mapping (`src/lib/models/NeuralCamera.{h,cpp}`)

Order of operations in `NeuralCameraImpl::forward` (`NeuralCamera.cpp:258-390`), applied to an already-3
channel RGB tensor `x` in `[0,1]`-ish linear-ish space coming out of the U-Net:

1. **Exposure** (`enable_exposure=true` by default, ini:191): `exposure = exposures_values[frame_index]`
   (per-image learned scalar, `[N,1,1,1]`, init from EXIF `EV_log2` computed in `colmap2adop.cpp:39-41`
   below); `x = x * (1 / 2^exposure)` (`NeuralCamera.cpp:307-309`; the `log_render` branch, `x = x -
   exposure`, is dead — `log_render` is a local `false` constant, `NeuralCamera.cpp:263`).
2. **White balance** (`enable_white_balance=true` in `configs/train_normalnet.ini:193`, though the
   `NeuralCameraParams` struct default is `false`, `Settings.h:128`): per-image learned `[N,3,1,1]`,
   `x = wb * x` (`NeuralCamera.cpp:334`). `ApplyConstraints` fixes image 0's WB to its reference value and
   forces the green channel to its reference (`NeuralCamera.cpp:418-424`) every step — a hard constraint,
   not a loss term.
3. **Vignette** (`enable_vignette=true`, ini:190): `VignetteNetImpl::forward`
   (`NeuralCamera.cpp:22-42`): `r2 = ||(uv - center) * (aspect_x_only, 1)||^2` (aspect only applied to the
   x/u coordinate, `NeuralCamera.cpp:33`), `factor = 1 + p0*r2 + p1*r4 + p2*r6` — a symmetric radial
   polynomial vignette, params init to `0` (no vignetting) (`NeuralCamera.cpp:16-19`); `x = factor * x`.
4. **Camera response function** (`enable_response=true`, ini:192): `CameraResponseNetImpl` — a learned 1D
   LUT per channel, `response_params=25` control points (ini:196), initialized as a gamma curve
   `MakeGamma(response_gamma=1/2.2≈0.4545)` then `normalize(1)` (`NeuralCamera.cpp:66-69`), applied via
   `grid_sample` (bilinear, `align_corners=true`, border padding) treating the (rescaled to `[-1,1]`) pixel
   value as a 1D lookup coordinate (`NeuralCamera.cpp:113-127`). A "leaky" extrapolation
   (`response_leak_factor=0.01`, ini:198) linearly/`1/sqrt`-extrapolates outside `[0,1]` during training
   only (`NeuralCamera.cpp:93-105`). If `enable_response=false`, falls back to `x = clamp(x,0,1)`
   (`NeuralCamera.cpp:366`).
5. **Rolling shutter** (`enable_rolling_shutter=false` by default, both struct and ini) — off; when on,
   applies a learned per-image 2-channel flow-field grid-sample warp (`NeuralCamera.cpp:189-212`).

`CameraResponseNetImpl::ParamLoss` (`NeuralCamera.cpp:137-157`) is an explicit smoothness regularizer on
the response LUT (`response_smoothness` config, weight `1e-5` folded into the MSE call, ini not shown but
struct default `Settings.h:187`), forcing each interior control point toward the mean of its neighbours and
the first point toward 0.

Exposure init: `colmap2adop.cpp:39-41`:
```
EV_log2 = log2(FNumber^2 / ExposureTime) + log2(ISO/100) - ExposureBiasValue
```
computed per-image from EXIF, `0` if EXIF is missing (`colmap2adop.cpp:32-36`); the scene-level mean is
stored as `dataset.ini`'s `scene_exposure_value` (`colmap2adop.cpp:105`).

## 7. Losses and training schedule (`configs/train_normalnet.ini`, section `[TrainParams]` unless noted)

- Batch: `batch_size=4, inner_batch_size=4, inner_sample_size=3, train_crop_size=512, num_epochs=600`.
- Loss weights: `loss_vgg=1, loss_l1=1, loss_mse=0, loss_ssim=1, loss_lpips=0` (ini:40-42,62-63).
  Combined in `src/lib/models/Pipeline.cpp:700-780`:
  `loss = w_vgg*VGG(x,target) + w_l1*L1(x,target) + w_mse*MSE(x,target) + w_ssim*(1-SSIM(x,target))/2 +
  w_lpips*LPIPS(x,target).sum()`. VGG only starts contributing after
  `only_start_vgg_after_epochs=100` (ini:66, `Pipeline.cpp:739-740`); there is also a global
  `warmup_epochs=20` (ini:64) whose consumer is not in `Pipeline.cpp`'s loss block (likely gates
  camera/point learning-rate ramp — not traced further here).
- **VGG**: `Saiga::PretrainedVGG19Loss` loaded from a **pre-traced TorchScript file**,
  `loss/traced_caffe_vgg_optim.pt` (ini `vgg_path`, `Pipeline.cpp:119-123`) — a Caffe-derived VGG19, put in
  `eval()` mode. The exact layer/weight breakdown lives inside the traced `.pt` graph, not in this source
  tree (`PretrainedVGG19Loss` class itself is in the un-vendored Saiga).
- **SSIM**: `Saiga::SSIM` (`Pipeline.h:238`) — window size and formula are in the un-vendored Saiga
  (`grep` found no local `SSIM` class definition or `.h` under `src/` or `External/`). Not verifiable from
  this checkout.
- **LPIPS**: present (`loss/traced_lpips.pt`) but weight `0` by default — unused.
- Optimizer LRs (`[OptimizerParams]`, ini:151-186): `lr_render_network=0.0002, lr_texture=0.1,
  lr_background_color=0.004, lr_points=0.0001, lr_poses=0.0001, lr_intrinsics=0.001, lr_confidence=0.001,
  lr_response=0.0001, lr_exposure=0.0005, lr_wb=0.0005`; `fix_intrinsics=true, fix_dynamic_refinement=true,
  fix_vignette=true, fix_wb=true, fix_motion_blur=true, fix_rolling_shutter=true` (these subsystems are not
  trained even though some are "enabled" for forward-pass purposes, e.g. white balance is applied but
  frozen).
- LR decay: `lr_decay_factor=0.85, lr_decay_patience=10` (plateau-style, consumer in `train.cpp`, not
  traced in detail here). `use_myadam_everywhere=true` selects the custom `MyAdam.cu` optimizer
  (not detailed in this pass — flagged as unexplored, see report).
- Crop augmentation: `min_zoom=0.75, max_zoom=1.5, crop_prefere_border=true, train_use_crop=true,
  train_mask_border=16`.
- Point-cloud/structure locking: `lock_camera_params_epochs=100, lock_structure_params_epochs=10,
  lock_dynamic_refinement_epochs=50` — points/poses/etc. are held fixed for these many initial epochs
  regardless of the `fix_*` flags above (exact interaction not traced further here).

## 8. ADOP scene format and `colmap2adop`

`src/apps/colmap2adop.cpp:53-164` produces, under `output_path`:
- `dataset.ini` (`SceneDatasetParams`, `src/lib/data/SceneData.h:105-179`): `image_dir`, `camera_files`
  (list), `render_scale` (CLI arg, default requested `1`), `scene_exposure_value` (mean EV over all
  images), `scene_up_vector = (0,-1,0)` hardcoded (`colmap2adop.cpp:108`), `znear/zfar/point_factor`
  (struct defaults, not set by the converter).
- `camera<i>.ini` per distinct COLMAP camera (`SceneCameraParams`, `SceneData.h:180-` ff.): `w,h`; `K` as
  `fx fy cx cy s` (5 floats, `SceneData.h:209-222`); `distortion` as an **8-float** vector (`SceneData.h:
  224-241`, "8 parameter distortion model, see distortion.h" — model itself in un-vendored Saiga); optional
  `ocam` (fisheye) affine+polynomial params if present.
- `images.txt`: one filename per line, in COLMAP image order (`colmap2adop.cpp:118-122`).
- `camera_indices.txt`: one integer per line, mapping each image to its `camera<i>.ini`
  (`colmap2adop.cpp:121`).
- `point_cloud.ply`: **copied verbatim** from the COLMAP dense `fused.ply` (`colmap2adop.cpp:125-141`,
  literally `cp`), not re-encoded.
- `poses.txt`: **camera-to-world**, one line per image, format
  `qx qy qz qw tx ty tz` — **xyzw quaternion order** (`SceneData.cpp:458-469`,
  `q.x() q.y() q.z() q.w() t.x() t.y() t.z()`), confirmed by both the writer and the reader
  (`SceneData.cpp:165`: `sstream >> q.x() >> q.y() >> q.z() >> q.w() >> ...`). Built from COLMAP's
  world-to-camera `SE3(q,t)` via `.inverse()` (`colmap2adop.cpp:143-148`). **This is xyzw, not wxyz** — see
  §10 contradiction with `docs/GEOMETRY.md`.
- `exposure.txt` (optional in general, but always written by `colmap2adop`): one EV float per line, from
  EXIF (§6).
- No `white_balance.txt` written by `colmap2adop` (optional, format `wb.x wb.y wb.z` per line,
  `SceneData.cpp:220-232`).

`scale_intrinsics` (CLI arg, required, no default) scales `w,h,K` by that factor before saving
(`colmap2adop.cpp:82-86`) — this is a resize of the *source* calibration (e.g. if you feed half-size
images), distinct from `render_scale`, which only affects the *rendered* buffer size at train/inference
time (`scenes/README.md:28-36`) and is stored, read back, and can be edited later directly in `dataset.ini`.

At first load, `SceneData::SceneData` (`SceneData.cpp:35-`) runs one-time point cloud preprocessing
(`RemoveDoubles`, `RemoveLonelyPoints`, `RemovePointsInCloseArea`, Morton-order + block shuffle,
`ComputeRadius` §2) and caches the result as `point_cloud_compressed` — so the effective point cloud
consumed by training is **not** the raw `fused.ply`, even though `point_cloud.ply` itself is a verbatim
copy.

## 9. Checkpoint layout

Per-epoch directory `experiments/<experiment_name>/ep<NNNN>[<point-add/remove suffix>]/`
(`src/apps/train.cpp:661`; `full_experiment_dir = experiment_dir/experiment_name/`, `train.cpp:484-485`).
On completion, the whole experiment dir is renamed to `experiments/_f_<experiment_name>/`
(`train.cpp:800-802`). Alongside per-epoch dirs: `log.txt`, `error.txt`, `params.ini` (full resolved
config, `train.cpp:575`), `git.txt` (commit hash), `tfevents.pb` (TensorBoard).

Inside each `ep<NNNN>/` (all via plain `torch::save`/`torch::load` on `torch::nn::Module`s — **these are
libtorch module-state archives, not `torch.jit.ScriptModule`s**; loading requires reconstructing the
identical C++ (or a faithful re-implementation's) module graph before `torch::load`, there is no
self-describing graph to trace):
- `render_net.pth` — `torch::save(*network, ...)` where `network` is the
  `MultiScaleUnet2dDecOnlySmallFixed` instance (`Pipeline.h:160,184`).
- `dynamic_refinement_module.pth` — only if a dynamic refinement MLP is active (`Pipeline.h:185-188`).
- `scene_<scene_name>_texture.pth` — the whole `NeuralPointTexture` module (texture, confidence, background
  color) — skipped if `reduced_check_point=true` (`NeuralScene.cpp:602-617`).
- `scene_<scene_name>_points.pth` — the whole `NeuralPointCloudCuda` module (positions, point sizes) —
  also skipped if reduced.
- `scene_<scene_name>_env.pth` — environment map, only if enabled (off by default, §3).
- `scene_<scene_name>_poses.pth` / `scene_<scene_name>_intrinsics.pth` — always saved, even in reduced mode
  (`NeuralScene.cpp:619-621`).
- `scene_<scene_name>_poses.txt` — human-readable re-export via `SceneData::SavePoses`, same xyzw
  camera-to-world format as §8 (`NeuralScene.cpp:623-624`).
- `scene_<scene_name>_vignette.pth`, `..._response.pth` (+ `.png` LUT visualization + `.csv` per-channel
  curve), `..._wb.pth` (+ `.txt`), `..._ex.pth` (+ `.txt`) — from `NeuralCameraImpl::SaveCheckpoint`
  (`NeuralCamera.cpp:542-599`), always written when the corresponding submodule exists (independent of
  `reduced_check_point`).

`params->optimizer_params.network_checkpoint_directory` allows loading only the render network from a
different location than the rest of the scene state (`Pipeline.cpp:74-80`).

### 9a. UPDATE (feat/net, 2026-09-05): `torch.jit.load` DOES read `render_net.pth` from Python

Correcting this section's earlier claim ("these are libtorch module-state archives, not
`torch.jit.ScriptModule`s... there is no self-describing graph to trace") and the same claim repeated in
Sec. 11: it is **half right**. Tested directly against a real file,
`third_party/zenodo/tt_checkpoints/checkpoint_horse/ep0600/render_net.pth` (extracted selectively from the
public Zenodo record 10687419 checkpoint archive, whole-zip `unzip -t` verified error-free at 2.65 GB
first, see `docs/LIMITATIONS.md`):

```python
>>> import torch
>>> m = torch.jit.load("render_net.pth", map_location="cpu")   # succeeds, no error
>>> len(m.state_dict())
34
```

`torch::save(*network, path)` on a plain (non-scripted) `torch::nn::Module` writes a
`torch::serialize::OutputArchive` zip container that Python's `torch.jit.load` can open and expose via
`.state_dict()` — it returns something usable for *reading the named tensors*, with names that exactly
match the C++ module's `register_module`/`register_parameter` hierarchy (e.g.
`start.conv.feature_transform.0.weight`, `up7.convolution.mask_transform.0.bias`, `final.0.weight`). What
remains true from the original claim: this is **not** a traced/scripted computation graph — the loaded
object cannot run `forward()` correctly as TRIPS's C++ network would (no graph, no `forward` method beyond
whatever `torch.jit.load`'s generic wrapper provides), so it is only useful as a **named-tensor bag** for
transplanting weights into an independently-built module (exactly what
`trippy.net.checkpoint.try_load_trips_network` does), not as a runnable model. `torch.load(path,
weights_only=False)` was not needed for this file in practice — `torch.jit.load` succeeded first — but the
loader still tries it as a fallback per the task brief, in case a different TRIPS build/version produces a
file `torch.jit.load` can't parse.

Verification result (see `docs/LIMITATIONS.md` for the full report): with `num_layers=8` (matching that
checkpoint's own `params.ini`, see Sec. 5a above), all 34 tensors from `render_net.pth` shape-match
trippy's `MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig(num_layers=8))` **exactly**, in registration
order — the network architecture (Sec. 5) is now checkpoint-verified, not just source-read.

## 10. Contradictions with `docs/ARCHITECTURE.md` / `docs/GEOMETRY.md`

1. **Forward pass "emit ≤8 fragments per point, 2 pyramid levels × 2×2 bilinear weights" (ARCHITECTURE.md
   line ~30) does not match TRIPS's actual default.** TRIPS's default (`use_layer_point_size=false`, the
   only value reachable from any shipped `.ini` — see §2/§3) writes **every point into every one of the 5
   pyramid layers**, each with its own full 2×2 footprint (up to `5*4=20` fragment-writes per point), with
   *no* layer attenuation. The "≤8 fragments across 2 levels via footprint size" description matches only
   TRIPS's `use_layer_point_size=true` code path, which exists in source but is unreachable via config in
   this commit. **Recommendation:** decide explicitly whether the port targets (a) TRIPS's actual default
   behavior (full multi-layer broadcast, needed to match any checkpoint trained with the shipped configs,
   including the public Zenodo scene — check that scene's own `params.ini` to be sure which mode it used),
   or (b) the footprint-based 2-level scheme as a deliberate, documented simplification. Update
   `docs/ARCHITECTURE.md` to say which, and drop "8 fragments" as if it were TRIPS's default.

2. **Pyramid level selection / layer factor formula is materially different from `docs/GEOMETRY.md`'s
   description**, even for the (unreachable-by-default) `use_layer_point_size=true` path:
   - GEOMETRY.md's "sub-pixel epsilon rule: clamp footprint size to 0.25" does not exist. The real rule
     (`PointBlending.h:99-111`) is `layer_factor = 0.75*exp(size-1) + 0.25` for sizes `<=~2px` — an
     exponential floor on the *blend factor*, not a clamp on the size feeding the log2 computation.
   - GEOMETRY.md's "linear layer factor... if log2(s)=1.3, weight 0.3 toward upper" is not what the code
     does. The real interpolation (`PointBlending.h:126-130`) is linear **in point-size units** between
     `2^layer_lower` and `2^layer_higher`, not linear in the `log2` fraction. Worked example: `s=2^1.3≈2.46`
     gives GEOMETRY.md's implied weight `0.3`, but the real code gives `(2.46-2)/(4-2)=0.23`.
   **Recommendation:** rewrite `docs/GEOMETRY.md`'s pyramid-level-selection section to quote the exact
   piecewise formula in `PointBlending.h:81-149`, or explicitly mark it as an intentional simplification the
   port is choosing over TRIPS's formula (and say why).

3. **`docs/ARCHITECTURE.md`'s "Core principle: No atomics anywhere" is a deliberate deviation from TRIPS,
   not a port of it** — TRIPS uses `atomicAdd` extensively in both the forward counting/collection kernels
   (`CountTiled`/`CollectTiled2`, list-slot allocation) and the entire backward pass (every gradient
   reduction in `RenderBackward.cu`). This is fine as an intentional Metal-driven redesign, but
   `docs/ARCHITECTURE.md` should say explicitly "TRIPS uses atomics; we do not, by design, because of the
   Metal 64-bit atomic limitation" rather than reading as if it were describing TRIPS's own algorithm.
   Recommendation: add one sentence acknowledging this is a redesign, so future readers don't assume
   TRIPS's sort-based numbers (fragment counts, list caps) map 1:1 onto an atomic-free reformulation without
   re-deriving them.

4. **`docs/GEOMETRY.md`'s 3DGS-export opacity formula (`opacity = logit(sigmoid(confidence))`) omits
   TRIPS's `×10` confidence scale.** TRIPS's actual confidence-to-alpha mapping is
   `sigmoid(10 * confidence_raw)` (`NeuralTexture.h:42`), not `sigmoid(confidence_raw)`. If a port ever
   needs bit-for-bit opacity parity with a TRIPS-trained checkpoint (e.g. for the honesty/coverage
   comparisons in `AGENTS.md` §7), this factor of 10 (plus the epoch-dependent narrowing term, `0` by
   default) must be included.

5. **Quaternion order**: `docs/GEOMETRY.md` mandates wxyz as *trippy's* internal convention (fine, no
   contradiction there), but the **ADOP `poses.txt` file format itself is xyzw** (`SceneData.cpp:463-469`).
   Any COLMAP/ADOP scene loader in the port must convert xyzw → wxyz on read and back on write; this isn't
   contradicted by GEOMETRY.md but isn't stated there either — worth a one-line addition so a future porter
   doesn't assume the on-disk format already matches trippy's convention.

## 11. Pitfalls a porter must know

- `use_layer_point_size` cannot be set from any `.ini` (`SAIGA_PARAM` omitted, `Settings.h:87-101`) — the
  entire per-point-size / layer-attenuation code path in `RenderForward.cu`/`RenderBackward.cu`/
  `PointBlending.h` is dead weight in every config shipped with this repo. Don't spend porting effort making
  it bit-exact unless you also plan to flip that flag (which currently requires a C++ code change, not a
  config change).
- `depth_accept` / `depth_accept_blend` are likewise dead in the default `TILED_BILINEAR_BLEND` kernel path
  (§3) — present in `RenderParams` and copied to the device struct, never read.
- `distortion_gradient_factor` / `K_gradient_factor` are dead in the kernels (§4) — don't assume they scale
  intrinsic/distortion gradients; if such damping is wanted, it must be added, not copied.
- Sort key packing (`__hiloint2double(z_bits, slot)`) only produces a correct ascending sort for `z > 0`;
  a symbolic/numeric reimplementation (e.g. in Rust/CubeCL) should use an explicit `(depth, slot)` tuple
  sort rather than relying on float-bits-in-double-high-word tricks.
- Forward output tensor layout flips between NCHW (train, or any non-`TILED_BILINEAR_BLEND` mode) and NHWC
  (`TILED_BILINEAR_BLEND` + eval) — see §3. A port that only validates the training path will not catch an
  inference-time transpose bug.
- `full_list_buffer`/`full_list_buffer_data` sizes must accommodate up to `num_layers * H * W * 4`
  (2x2 splat) fragment slots **per point that passes the bounds test in every layer**, not per point total —
  this is `5x` larger than a naive "each point contributes to one layer" assumption, because of finding #1
  in §10.
- The **gated conv block** (`conv_block=gated`/`conv_block_up=gated`, the only block type used by
  `MultiScaleUnet2dDecOnlySmallFixed`) is defined in Saiga (`saiga/vision/torch/PartialConvUnet2d.h`), which
  is **not present in this checkout** (`External/saiga/` is an empty directory — the submodule was never
  initialized here). Its exact formula must be pulled from the real Saiga source (or reverse-engineered from
  a loaded checkpoint's parameter shapes) before the U-Net can be ported bit-exact. The same applies to
  `Saiga::SSIM` and `Saiga::PretrainedVGG19Loss` (§7) and the Saiga 8-parameter lens distortion model (§3/§8).
- `texture_random_init` path in `NeuralTexture.cpp:24-29` has dead code above the line that actually executes
  (the `factor`-scaled uniform assignment is immediately overwritten by `texture_raw.uniform_(0,1)`) — the
  real init is plain `Uniform(0,1)`, not the commented "range [-factor/2, factor/2]" the code comments
  suggest.
- Checkpoints are plain libtorch module archives (`torch::save`/`torch::load` on live `nn::Module`s), **not
  TorchScript**. A Rust/Burn/CubeCL port cannot `torch.jit.load` them; it needs a tensor-name/shape-level
  loader (e.g. via `safetensors` conversion) matching each module's exact parameter names, which in turn
  requires knowing the un-vendored gated-block's registered submodule names.

## Not found / not verifiable from this checkout

- Exact gated-convolution formula (`UnetBlockFromString("gated", ...)`) — lives in Saiga, submodule not
  vendored at this commit's checkout (`External/saiga/` empty).
- Exact `SSIM` window size/formula and `PretrainedVGG19Loss` layer/weight breakdown — same reason (Saiga),
  plus the VGG weights are additionally baked into a binary traced `.pt` file
  (`loss/traced_caffe_vgg_optim.pt`), not human-readable source either way.
- Saiga's 8-parameter lens distortion polynomial definition (`distortion.h`) — same reason.
- `MyAdam.cu`'s custom optimizer update rule (`use_myadam_everywhere=true` is the default) — file exists in
  this checkout but was out of scope for this pass; flagging so a porter doesn't assume plain-Adam
  semantics for the point/texture/confidence learning rates in §7 without checking `src/lib/models/MyAdam.cu`
  directly.
- The exact consumer/semantics of `warmup_epochs` (ini `20`) — referenced in the config but its use site was
  not located inside `Pipeline.cpp`'s loss block in this pass.

---

## Corrections found by rendering a real checkpoint (feat/adop-parity, EXP-0002, 2026-09-06)

Everything below was found while reproducing the public Zenodo `checkpoint_horse` render end to end
(`trippy parity`, see `experiments/EXP-0002-horse-parity/README.md`). Each correction is stated against
the section it corrects, with the `path:line` that settles it. Three of them are the difference between
an 8.5 dB render and a 25.1 dB one, so they are not cosmetic.

### 2a. `PrepareTexture` is NOT called with `!non_subzero_texture` — Sec. 2 is wrong

Sec. 2 says: "`PrepareTexture(abs)` optionally takes `abs()` of both texture and background before use;
the `abs` flag is `!non_subzero_texture` in the pipeline (config `non_subzero_texture=false` → texture
is abs'd, so effectively non-negative)." The negation is not there. The two call sites pass the config
flag straight through:

```cpp
scene.texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
    // src/lib/models/Pipeline.cpp:257  (and data/NeuralScene.cpp:1292)
```

and `PrepareTexture(bool abs)` only takes `abs()` when its argument is *true* (`NeuralTexture.h:44-57`).
With `non_subzero_texture = false` — the value in `configs/train_normalnet.ini` **and** in every published
Tanks & Temples `params.ini` — the texture and the background colour are consumed **raw, negatives
intact**. The published horse texture ranges `[-107.6, +95.9]`; taking `abs()` roughly triples the
composited feature magnitude, pushes the U-Net's output to a median of 1.05 (it should be ~0.4), saturates
the response LUT, and costs 16.6 dB (measured: 8.46 dB with `abs()`, 25.10 dB without, same frame).

### 2b. `use_layer_point_size` IS reachable from a config — Sec. 2 and Sec. 11 are wrong

Sec. 2 says `use_layer_point_size` "cannot be set from any `.ini` file and is always `false` in practice",
and Sec. 11 repeats it. True that there is no `SAIGA_PARAM(use_layer_point_size)`; false that the value is
therefore fixed. `CombinedParams::Check` *derives* it from an optimizer flag that **is** a config field:

```cpp
render_params.use_layer_point_size = !optimizer_params.fix_point_size;
    // src/lib/data/Settings.cpp:39
```

`checkpoint_horse/params.ini` has `fix_point_size = false`, so **`use_layer_point_size = true` for every
published Tanks & Temples checkpoint**. Consequences: the point-size parameter, the `softplus`
parametrisation, `compute_point_size_fac` and the whole "layer factor" machinery Sec. 2/3/11 write off as
dead weight are all live, and `RenderParams::use_environment_map` is likewise derived, not configured:

```cpp
render_params.use_environment_map =
    pipeline_params.enable_environment_map && !pipeline_params.environment_map_params.use_points_for_env_map;
    // src/lib/data/Settings.cpp:34
```
The horse config has `enable_environment_map = true` **and** `use_points_for_env_map = true`, so
`use_environment_map` is **false** and the learned `background_color` *is* composited under the
transmittance (`RenderForward.cu:3620-3634`). The "environment map" is instead 4 x 100 000 = 400 000 extra
points added to the cloud on nested spheres (`NeuralScene.cpp:77-88` `AddNewRandomForEnvSphere`), which is
why the horse checkpoint has 2 218 471 points against the scene's own 1 818 471.

### 3a. The published checkpoints do not use the kernel Sec. 3 documents

With `use_layer_point_size = true`, `render_points_in_all_lower_resolutions = true` (`Settings.h:78`, also
in the horse `params.ini`) and `combine_lists = false`, `PointRenderer.cu:726-750` takes the `new_impl`
branch and calls `PointRendererCache::RenderFast16` (`RenderForward.cu:1065`) instead of the
`CountTiled` / `CollectTiled2` / `FusedSortAndBlend2` chain Sec. 3 describes. The forward kernel is
`CountAndCollectTiled<num_layers>` (`RenderForward.cu:168-368`) plus `SplattingPass` and
`FastSortFusedNew<NUM_DESCRIPTORS, 16, train>` (`RenderForward.cu:474`, dispatched at 1285-1389).
`ELEMENTS_PER_PIXEL` is still 16 and the blend is still front-to-back with `ALPHA_DEST_CUTOFF = 0.001`,
but the **layer selection is a third thing**, neither Sec. 3's "every point into every layer" nor
GEOMETRY.md's "two straddling layers":

```
point_size_opt = K.fx * crop_transform.fx * softplus(t_point_size) / z        // RenderForward.cu:268
layer_higher   = (point_size_opt > 1) ? min(ceil(log2(point_size_opt)), L-1) : 0   // :334-338
for (layer = 0; layer <= layer_higher; ++layer, ip *= 0.5f)                    // :340-352
    if (!valid_point(floor(ip), z, layer)) break;                              // note: break, not continue
    atomicAdd 4x into per_pixel_list_lengths[layer]
alpha_bilin = bilinear_fac * confidence * compute_point_size_fac(point_size_opt, layer, L)  // :3511-3517
```

and `compute_point_size_fac` returns **1.0** for every `layer < layer_lower` (`PointBlending.h:92-96`).
So a 5-pixel point writes layers 0..3 with factors 1, 1, 0.75, 0.25. trippy calls this mode `"trips"`.
Measured on three held-out horse frames: `trips` 22.27 dB, trippy's `trilinear` (two straddling layers
only) 21.47 dB, trippy's `broadcast` (all layers, factor 1 — Sec. 3's reading) **15.14 dB**.

### 3b. The pyramid halves with `ceil`, not integer division

Sec. 3 says "each subsequent layer is `h/=2; w/=2` (integer division, `PointRenderer.cu:378`)". That is
only the `else` branch. The live code is:

```cpp
if (info->scene->params->net_params.network_version != "MultiScaleUnet2d") { h = std::ceil(h/2.f); w = std::ceil(w/2.f); }
else                                                                       { h = h / 2;            w = w / 2; }
    // src/lib/rendering/PointRenderer.cu:385-391
```

Every published checkpoint sets `network_version = MultiScaleUnet2dDecOnlySmallFixed`, so **ceil** wins.
This matters: 1080 floor-halved 8 times gives 1080, 540, 270, 135, 67, 33, 16, 8, whose upsamples make the
U-Net's output 1024 rows; ceil gives 1080, 540, 270, 135, 68, 34, 17, 9 and the output stays 1080. It also
means `trippy.raster.emit.layer_grid`'s ceil halving — documented there as a *deviation* from TRIPS — is
in fact exactly what TRIPS does for this network. The `CombineBridge(below=features_input, skip=upsampled)`
crop then only ever trims the upsampled tensor by one row/column (136 -> 135), never the raw input.

### 6a. TRIPS's `ip` puts pixel centres on integers; trippy's `uv` puts them on half-integers

`compute_blending_fac` (`PointBlending.h:216-240`) takes `subpixel_pos = ip - floor(ip)` and writes the
2x2 footprint at `floor(ip)` and `floor(ip)+1`. So in TRIPS the *centre* of pixel `i` is at coordinate
`i`, whereas `docs/GEOMETRY.md` puts it at `i + 0.5`. Feeding trippy's rasteriser the raw `K` therefore
shifts the whole render by half a pixel per layer. `trippy.render.parity` corrects this by adding 0.5 to
`cx, cy` **and** rendering each pyramid layer with its own `num_layers=1` call at
`K_l = (fx, fy, cx, cy) / 2**l`, which reproduces TRIPS's `ip *= 0.5f` exactly (a single multi-layer call
cannot: `uv/2**l - 0.5` and `ip/2**l` differ by a layer-dependent amount for any fixed `cx`).

### 8a. `poses.txt` xyzw camera-to-world is confirmed against TRIPS's own buffer

Sec. 8/Sec. 10.5's claim is correct and now cross-checked against the trained state rather than the
writer: `PoseModuleImpl` stores `frame.pose.inverse()` as an `[N, 8]` float64 buffer
(`data/NeuralStructure.cpp:20-33`; a `Sophus::SE3d` is quaternion `x, y, z, w` + translation + one padding
double). For tt_horse frame 0, `poses.txt` line 1 is
`q_xyzw = (0.012484, 0.151535, 0.020931, 0.988151)`, `t = (-0.557297, 0.415945, -3.309434)`, and the
checkpoint's `poses_se3[0]` is `(-0.012484, -0.151535, -0.020931, 0.988151, -0.476943, -0.333750, 3.331228)`
— exactly the inverse. `trippy.scene.adop_io.pose_c2w_xyzw_to_w2c_wxyz` reproduces the checkpoint value to
1e-7 (`tests/test_scene_adop_io.py::test_read_poses_matches_manual_conversion`). Note also that although
`fix_poses = false`, the horse checkpoint's `tangent_poses` are all zero and `poses_se3` equals the scene
file to float64 precision — the poses did not move.

### 8b. `point_cloud.bin` is a Saiga zlib container, not a bespoke format

Sec. 8 mentions `point_cloud_compressed` without saying what it is. `SceneData.cpp:121` calls
`Saiga::UnifiedMesh::SaveCompressed` (`saiga/core/model/UnifiedMesh.cpp:508-528`), which is
`compress(BinaryOutputVector << position << normal << color << texture_coordinates << data << bone_info
<< triangles << lines << material_id)`. `Saiga::compress` (`saiga/core/util/zlib.cpp:68-88`) prepends a
24-byte header of three little-endian `size_t`s: magic `0x006712956A9725DE`, compressed size,
decompressed size, then a plain zlib stream. Each `std::vector<T>` is a `size_t` count followed by packed
elements (`saiga/core/util/BinaryFile.h:79-110`); `vec3` is 12 bytes, `vec4` 16, `vec2` 8.
**The header's compressed-size field is unreliable** — Saiga's `compress3` accumulates `stream.total_out`
without resetting it between `deflate` calls (`zlib.cpp:40-60`), so it over-counts for multi-chunk
streams; validate against the decompressed size instead. `trippy.scene.adop_io.read_point_cloud_bin`
implements this and `write_point_cloud_bin` round-trips it.

### 8c. Saiga's 8-parameter distortion model, fetched and verified

Sec. 3/8 list this as "not verifiable from this checkout". It is public MIT source
(https://github.com/darglein/saiga @ `ee7a4e6b65832433e2ca521353b7b7431c8e17a0`,
`src/saiga/vision/cameraModel/Distortion.h:130-171`):

```
r2 = x^2 + y^2
xd = x * (1 + k1 r2 + k2 r4 + k3 r6) / (1 + k4 r2 + k5 r4 + k6 r6) + (p1*2xy + p2*(r2 + 2x^2))
yd = y * (same radial)                                             + (p1*(r2 + 2y^2) + p2*2xy)
if (r2 > max_r^2) xd = yd = 100000            // max_r = RenderParams::dist_cutoff = 20 (Settings.h:61)
```

**Coefficient order in `camera<i>.ini` is `k1 k2 k3 k4 k5 k6 p1 p2`** (`Distortion.h:20-45`), *not*
OpenCV's `k1 k2 p1 p2 k3 k4 k5 k6`. tt_horse uses `k1 = -0.0640, k2 = +0.0444`, the rest zero: a 2.2%
radial shift at the image corner, ~24 px. Ignoring it is not an option. `K.normalizedToImage` is
`(fx*x + s*y + cx, fy*y + cy)` (`Intrinsics4.h:80`), and at eval time `crop_transform` is the default
`IntrinsicsPinholef()` = identity (`Dataset.cpp:276`, `Intrinsics4.h:33-38`), so `ip = image_p`.

### 9b. Field-by-field map of a real scene checkpoint

Observed in `checkpoint_horse/ep0600` (2 218 471 points, 151 frames, `num_texture_channels = 4`); every
file opens with `torch.jit.load` per Sec. 9a.

| file | tensor | shape | meaning |
|---|---|---|---|
| `scene_<s>_points.pth` | `t_position` | `[N,4]` | world xyz + learnable drop-out radius in `.w` |
| | `t_point_size` | `[N,1]` | **pre-softplus** size; `softplus(x)` (beta 1, threshold 20) is the world-unit size |
| | `t_index` | `[N,1]` i32 | render-order id -> texture column (identity in the published files) |
| | `t_original_color` | `[N,4]` | the input cloud's RGBA, unused at render time |
| `scene_<s>_texture.pth` | `texture` | `[C,N]` | **`texture_raw`** despite the name (`register_parameter("texture", texture_raw)`, NeuralTexture.cpp:53) |
| | `background_color` | `[C]` | **`background_color_raw`** (`.cpp:54`) |
| | `confidence_value_of_point` | `[1,N]` | **`confidence_raw`** (`.cpp:55`), pre-sigmoid; horse range `[-0.42, 1.17]` |
| `scene_<s>_poses.pth` | `poses_se3` | `[M,8]` f64 | world-to-camera, xyzw + t + 1 pad (see 8a) |
| | `tangent_poses` | `[M,6]` f64 | pending SE3 delta; zero in a saved checkpoint |
| `scene_<s>_intrinsics.pth` | `intrinsics` | `[num_cameras,13]` | `fx fy cx cy s` + 8 distortion coefficients |
| `scene_<s>_ex.pth` | `0` | `[M,1,1,1]` | per-frame exposure EV, applied as `x * 2**-EV` |
| `scene_<s>_wb.pth` | `0` | `[M,3,1,1]` | per-frame white balance (all 1.0 in the horse run: `fix_wb = true`) |
| `scene_<s>_response.pth` | `response` | `[1,3,1,25]` | per-channel response LUT |
| `scene_<s>_vignette.pth` | `vignette_params` | `[3]` | radial polynomial (all 0 in the horse run: `fix_vignette = true`) |
| | `vignette_center` | `[1,2,1,1]` | uv centre (0 in the horse run) |

The `ep<NNNN>/test/` subdirectory holds TRIPS's own rendered test images (`<scene>_<index>.jpg`) and their
targets (`..._gt.jpg`). **Those renders carry a blacked-out border of `train_mask_border = 16` pixels on
every side** (measured: 15-16 all-zero rows/columns per side on every frame checked). Comparing them to
the scene photographs without cropping that border costs about 10 dB — the authors' own horse render
scores 15.7 dB uncropped and 25.2 dB cropped on frame index 8. Any parity number quoted against those
files must say which.
