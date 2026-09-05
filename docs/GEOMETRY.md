# Geometry conventions

All transforms are implemented twice—independently in numpy (`xform_a`) and PyTorch (`xform_b`)—and must disagree until both are corrected. This redundancy has caught three historical bug classes in this project (rotation composition, depth sign, padded pixels).

## Coordinate systems and conventions

### COLMAP world-to-camera transform

```
x_c = R @ x_w + t
```

where `x_w` is a 3D point in world, `x_c` is the same point in camera coordinates, `R` is a 3×3 rotation matrix, and `t` is a 3×1 translation. This is the standard COLMAP convention.

### Quaternions

All quaternions in this project use **wxyz** order: `q = [w, x, y, z]` where `w` is the scalar part. Note that ADOP's own `poses.txt` stores **xyzw** (`SceneData.cpp:463-469`), so any ADOP scene reader/writer must swap on both sides.

### Camera coordinate system

The camera looks down the **+Z axis**. Positive X points right, positive Y points down (standard image convention). This is consistent with OpenGL and COLMAP.

### Image coordinates

- **Image origin**: top-left corner is (0, 0).
- **Pixel centre convention**: pixel at integer index `i` (row) and `j` (column) has its continuous-space centre at `(i + 0.5, j + 0.5)`. A pixel spans `[i, i+1) × [j, j+1)` in continuous coordinates.
- **Depth sign**: depth is positive in front of the camera (+Z direction). Any reprojection where depth is negative indicates the point is behind the camera and should be clipped.

### Camera intrinsics

Intrinsics are a 3×3 matrix `K`:

```
K = [fx   0  cx]
    [ 0  fy  cy]
    [ 0   0   1]
```

where `(cx, cy)` is the principal point (usually near image centre) and `(fx, fy)` are focal lengths in pixels. Projection: `u = K @ (x_c / z_c)` for a point `x_c = [x, y, z]^T` in camera coordinates.

### Image pyramid

A pyramid has `L` levels (default 5, `trippy.constants.RASTER_NUM_LAYERS`). At level `l`, the resolution is:

```
H_l = ceil(H_0 / 2^l)
W_l = ceil(W_0 / 2^l)
```

where `H_0`, `W_0` are the base (level 0) resolution. **Deviation from TRIPS:** TRIPS halves with integer division (`h /= 2; w /= 2`, `PointRenderer.cu:378`) and therefore silently drops the last row/column of an odd-sized level. We use `ceil` so every layer-0 pixel has a covering pixel in every layer. The cost is that the coarsest layer covers up to `2^(L-1) - 1` columns/rows *beyond* the image; the visibility cull in `raster/emit.py` accounts for that explicitly.

### Layer coordinates: halving convention

A point's continuous layer-0 coordinate `uv` (corner origin, from `K`) maps to layer `l` as

```
uv_l = uv / 2^l          # no offsets
```

This matches TRIPS exactly: the layer loop carries `ip *= 0.5f` per level with no additive term (`RenderForward.cu:1610`, and identically at `:348`, `:1556`, `:2001`). It is also the *only* correct halving in the corner-origin convention above — a 2× box downsample maps the continuous coordinate `x` to exactly `x/2`, because the corner at `x = 0` is shared by both grids. (Had we stored `uv` as pixel *indices*, i.e. centre origin, the map would be `(x + 0.5)/2 - 0.5`; we do not.)

The 2×2 bilinear footprint is anchored on pixel **centres**, so the base pixel of the splat is

```
base = floor(uv_l - 0.5)
frac = uv_l - 0.5 - base
weights = [(1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx·fy]   for pixels base + {(0,0),(1,0),(0,1),(1,1)}
```

TRIPS's `compute_blending_fac` (`PointBlending.h:216-240`) uses `floor(ip)` with no `-0.5`, i.e. it treats its `ip` as centre-origin. Our `-0.5` is the corner-origin equivalent of the same formula; the exact meaning of TRIPS's `ip` depends on Saiga's `normalizedToImage`, which is not vendored in `third_party/TRIPS` (`External/saiga/` is empty), so we follow this file's stated convention rather than guessing theirs.

Out-of-bounds fragments are **dropped, never clamped** (bug class 3 below). A footprint straddling the image border contributes only its in-bounds corners.

### Pyramid level selection and the layer factor

Implemented in `trippy/raster/emit.py` (`layer_bounds`, `layer_factor`) and re-derived independently in `trippy/raster/ref_numpy.py`. It is an exact port of TRIPS's `compute_point_size_fac` (`third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149`).

For a point whose projected size is `s = fx · size_world / z` pixels at layer 0 (TRIPS uses `fx` only, `RenderForward.cu:1489`):

```
if s <= 1:  lower = upper = 0
else:       lower = clamp(floor(log2 s), 0, L-1)
            upper = clamp(ceil(log2 s),  0, L-1)

layer_factor(s, layer):
    if layer < lower:            -> 1.0     # inert; see note
    if upper == 0:               -> (1 - 0.25) * exp(s - 1) + 0.25
    if lower == upper:           -> 1.0
    else:  f = (s - 2^lower) / (2^upper - 2^lower)
           -> 1 - f  if layer == lower, else f
```

Three points this file previously got wrong (docs/TRIPS_REFERENCE.md §10.2):

1. **There is no sub-pixel size clamp.** The old "clamp footprint size to 0.25" rule does not exist in TRIPS. The real rule is an exponential *floor on the blend factor*: a 0.01 px point still splats, with factor `0.75·exp(-0.99) + 0.25 ≈ 0.529`, never below 0.25 (`PointBlending.h:106`). The 0.25 was chosen so that 16 such fragments still leave `alpha_dest ≈ 0.99`.
2. **The interpolation is linear in point-size units, not in the log2 fraction.** For `s = 2^1.3 = 2.46`: the old text implies weight 0.3 toward the upper layer; the code gives `(2.46 - 2)/(4 - 2) = 0.23`.
3. **`layer < lower` returns `1.0` in the C++**, not 0 (`PointBlending.h:96-100`) — `docs/TRIPS_REFERENCE.md` §3 states 0. It is inert either way: TRIPS's point-size emission kernel `CollectTiled2Pointsize` writes only `layer_lower` and `layer_higher` (`RenderForward.cu:2296-2360`), and so does trippy's `"trilinear"` mode, so neither ever asks for a layer below `lower`. We match the source, not the doc.

In `mode="broadcast"` (TRIPS's shipped default, see docs/ARCHITECTURE.md) none of this runs: every point goes into every layer with factor 1.

## Undistortion and image cache

`trippy.scene.dataset.SceneDataset` undistorts every COLMAP image exactly
once per (scene, width) and caches the result to disk, so repeated
training runs never re-pay the cost.

- **Destination camera**: a pinhole camera (zero distortion) at the
  requested width, keeping the source camera's aspect ratio. Destination
  intrinsics are the source's `(fx, fy, cx, cy)` uniformly scaled by
  `width_dst / width_src` -- no re-centring, no field-of-view change (this
  is the simple "scaled pinhole" convention, not OpenCV's
  `getOptimalNewCameraMatrix` free-scaling).
- **Sampling direction**: inverse mapping, one sampling grid per (source
  camera, destination width), built once in
  `trippy.geom.camera.undistort_maps`. For every destination pixel, cast
  the pinhole ray, apply the source camera's *forward* OpenCV
  radial-tangential distortion (`OpenCVDistortion.distort`) to find where
  that ray lands in the as-captured (distorted) source image, then sample
  there with `torch.nn.functional.grid_sample(mode="bilinear",
  padding_mode="zeros", align_corners=False)`.
- **Pixel-centre convention carries through unchanged**: per "Image
  coordinates" above, pixel (row i, col j) has its centre at
  `(j + 0.5, i + 0.5)`. Combined with `align_corners=False`, `grid_sample`
  maps grid value `g in [-1, 1]` to continuous pixel coordinate
  `x = (g + 1) * size / 2`, i.e. `g = 2*x/size - 1` -- exactly this
  pixel-centre convention, with no extra `+/-0.5` shift anywhere in
  `undistort_maps`.
- **Pinhole sources skip the distortion step**: when a source camera's
  `(k1, k2, p1, p2)` are all zero (PINHOLE, SIMPLE_PINHOLE), the sampling
  grid degenerates to a plain resize -- `undistort_maps` is still used
  (one code path for all camera models), but the forward-distortion call
  is skipped.
- **Cache layout**: `<cache_root>/<scene_name>/w<width>/<image_name>.npy`
  (uint8 RGB, `(H, W, 3)`) plus one `meta.json` per (scene, width) holding
  each cached image's scaled intrinsics, pose, original size, and EXIF
  exposure/ISO if present. A second `SceneDataset` construction over the
  same `cache_root`/width recomputes each image's intrinsics from the live
  COLMAP model and asserts they match `meta.json`'s stored values before
  trusting the cached pixels -- a mismatch (e.g. the scene was re-run
  through COLMAP) raises loudly instead of silently serving stale pixels.
- **Crop padding (bug class 3, repeated here deliberately)**:
  `trippy.scene.dataset.crop` never treats a crop's overshoot past the
  source image edges as content -- `rgb == 0` and `mask == 0` there,
  exactly, not a blurred or extrapolated value.

## 3DGS PLY export mapping

When exporting a trained TRIPS model to 3DGS PLY format for use with `gsrender.py` and other tools:

- **Position** (`x`, `y`, `z`): learned point positions, in world coordinates.
- **Isotropic scale** (`scale_0`, `scale_1`, `scale_2` all equal): from learned `log(size)`. 3DGS stores `log(scale)` on export; we export `scale_0 = exp(log_size)`.
- **Opacity** (`opacity`): from learned `confidence`. 3DGS uses logit space on export; we export `opacity = logit(sigmoid(confidence))`.
- **Colour** (`f_dc_0`, `f_dc_1`, `f_dc_2`): from learned RGB colour, normalized to spherical harmonics level-0 (DC) component.
- **SH coefficients** (higher harmonics): set to zero on export.
- **Rotation** (not used in TRIPS; set to identity quaternion wxyz [1, 0, 0, 0]).

This mapping allows external tools to load and render TRIPS point sets as if they were standard Gaussians, enabling metrics like shade audit and extent gate to run unchanged.

## ADOP format for COLMAP export

ADOP (as used by TRIPS authors) stores poses as camera-to-world (the *inverse* of COLMAP's world-to-camera). Poses are represented as:
- **3×3 rotation** `R_cw` (camera-to-world rotation)
- **3×1 translation** `t_cw` (camera position in world)
- **Up vector** fixed at `(0, -1, 0)` (standard for outdoor captures with gravity pointing down).

When converting from COLMAP (world-to-camera), use: `R_cw = R^T`, `t_cw = -R^T @ t`.

## Historical bug classes (caught by the dual-implementation rule)

1. **Rotation composition**: multiplying quaternions in the wrong order or using conjugate instead of transpose when converting between active/passive rotations.
2. **Depth sign**: forgetting that world-to-camera `z > 0` means "in front" but then treating negative depth as valid, or flipping the camera look direction.
3. **Padded pixels unprojected as scene**: rasterisation pads images to pyramid-level resolutions. Fragments from padded regions must be dropped before unprojection; otherwise, padded pixels project as valid scene points, corrupting gradients. Enforced by `tests/test_raster_bounds.py`: the emitter drops, and never clamps, any footprint corner outside its own layer, and the pre-emission visibility cull is proven conservative (a point 3 px off the left edge is culled from layer 0 but still drawn at layer 3).

Each has been caught early by comparing `xform_a` (numpy) and `xform_b` (torch) on small CPU-only tests before GPU work. Make both implementations fail the same way before fixing.

## SE(3) exponential map (pose-refinement twists)

`trippy.geom.xform_b.se3_exp` maps a twist `xi = (rho, phi)` (Sophus/g2o
convention: `rho = delta[:3]` translation generator, `phi = delta[3:]`
rotation vector, axis * angle radians) to a homogeneous transform
`T = [[R, V @ rho], [0, 0, 0, 1]]`, using the standard Taylor-guarded
Rodrigues coefficients with `t = |phi|`:

```
A = sin(t)/t,  B = (1 - cos t)/t^2,  C = (1 - A)/t^2
R = I + A [phi]x + B [phi]x^2
V = I + B [phi]x + C [phi]x^2
```

`A`, `B`, `C` are computed from `t^2 = phi . phi` (never from `t = |phi|`
directly, which has an undefined gradient at `phi == 0`), with the
small-angle series `A ~= 1 - t^2/6`, `B ~= 1/2 - t^2/24`, `C ~= 1/6 - t^2/120`
below `t = 1e-4` (avoids the float64 cancellation in `1 - cos t` for tiny
`t`, not just the `0/0` at `t == 0`).

**History:** an earlier version built `R` from a normalized axis,
`a * |phi| * skew(phi / max(|phi|, eps))`. That expression is second order
in `phi` at the origin, so autograd returned an exactly-zero rotation
gradient for `delta[3:]` at `phi == 0` -- the common case of a pose delta
initialised at zero -- even though the true derivative there is the SO(3)
generator (magnitude 1). The fix builds `[phi]x` directly (linear in `phi`,
so its own gradient at the origin is the generator) and keeps `A`, `B`, `C`
as smooth even functions of `phi` computed via `t^2`, so every term is
differentiable through `phi == 0`. Regression test:
`tests/test_raster_bwd_ref.py::test_pose_delta_rotation_gradient_matches_generator_at_zero`;
formula coverage: `tests/test_xform_agreement.py`'s `test_se3_exp_*` tests
(gradcheck at `phi == 0`, at `|phi| = 1e-6`, and at `|phi| = 0.5`; agreement
with `torch.matrix_exp` of the 4x4 se(3) generator to `1e-12`).
