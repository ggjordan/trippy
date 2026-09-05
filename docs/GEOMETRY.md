# Geometry conventions

All transforms are implemented twice—independently in numpy (`xform_a`) and PyTorch (`xform_b`)—and must disagree until both are corrected. This redundancy has caught three historical bug classes in this project (rotation composition, depth sign, padded pixels).

## Coordinate systems and conventions

### COLMAP world-to-camera transform

```
x_c = R @ x_w + t
```

where `x_w` is a 3D point in world, `x_c` is the same point in camera coordinates, `R` is a 3×3 rotation matrix, and `t` is a 3×1 translation. This is the standard COLMAP convention.

### Quaternions

All quaternions in this project use **wxyz** order: `q = [w, x, y, z]` where `w` is the scalar part.

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

A pyramid has `L` levels (typically 5). At level `l`, the resolution is:

```
H_l = ceil(H_0 / 2^l)
W_l = ceil(W_0 / 2^l)
```

where `H_0`, `W_0` are the base (level 0) resolution.

### Pyramid level selection

For a point projected to size `s` (in pixels, measured as the major axis of the bilinear footprint):

- **Lower level**: `l_lower = max(0, floor(log2(s)))`
- **Upper level**: `l_upper = min(L-1, ceil(log2(s)))`
- Both are clamped to `[0, L-1]`.

Bilinear interpolation weights are computed between the two levels with a **linear layer factor**: if `log2(s) = 1.3`, we interpolate with weight 0.3 toward the upper level.

**Sub-pixel epsilon rule**: if the footprint size is < 0.25 pixels, clamp it to 0.25 to avoid numerical issues in bilinear weight computation.

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
3. **Padded pixels unprojected as scene**: rasterisation pads images to pyramid-level resolutions. Fragments from padded regions must be dropped before unprojection; otherwise, padded pixels project as valid scene points, corrupting gradients.

Each has been caught early by comparing `xform_a` (numpy) and `xform_b` (torch) on small CPU-only tests before GPU work. Make both implementations fail the same way before fixing.
