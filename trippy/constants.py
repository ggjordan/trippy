"""Named constants used across the trippy package.

Module: trippy.constants
Invariants: pure Python literals only (no numpy/torch import) so both
    trippy.geom.xform_a (numpy-only) and trippy.geom.xform_b (torch-only)
    can import it without pulling in each other's dependency.
Related docs: docs/SPEC.md "Technical design"; AGENTS.md review
    checklist item "no magic numbers outside constants.py".
"""

# --- config.py defaults ---

# Fallback SPLATS_ROOT when the SPLATS_ROOT env var is unset. trippy reads
# Jordan's Splats project read-only for scenes, GPU queue, and depth tools.
DEFAULT_SPLATS_ROOT = "/Users/nzbirdranch/Splats"

# Device pick_device() resolves to when neither an explicit --device flag nor
# TRIPS_DEVICE is set. CPU is always safe; MPS must be requested explicitly.
DEFAULT_DEVICE = "cpu"

# --- cli.py ---

# `git describe --tags --match <this>` restricts matches to trippy's own
# release-tag scheme (build-NNNN), so an unrelated tag never gets printed.
GIT_DESCRIBE_MATCH_PATTERN = "build-*"

# Length of the trivial tensor used by `trippy smoke --device mps` to prove
# torch.mps.compile_shader works end to end (spec: "add 1.0 to an 8-element
# tensor").
SMOKE_MPS_TEST_TENSOR_LEN = 8

# --- geom/xform_b.py numerical stability ---

# Below this quaternion vector-part norm, the rotation angle is treated as
# ~0 and an arbitrary axis is used (the axis is undefined at angle 0, but the
# resulting rotation matrix is the identity regardless of axis choice).
EPS_QUAT_AXIS = 1e-8

# Below this rotation-vector norm (radians), se3_exp uses the Taylor-series
# form of the Rodrigues coefficients instead of dividing by theta, avoiding
# a 0/0 at theta == 0 (the common case: a zero pose-refinement delta).
EPS_SE3_ANGLE = 1e-8

# --- points/ : PointSet provenance tags ---
# Every PointSet.provenance entry is one of these uint8 values, recording
# which PointSource produced a given point. Carried through training so
# per-source diagnostics (D4 experiments 1/2/3) can be computed post hoc.
PROVENANCE_GAUSSIAN = 1
PROVENANCE_MONODEPTH = 2
PROVENANCE_LIDAR = 3
PROVENANCE_COLMAP = 4

# Human-readable names for PointSet.summary()'s provenance histogram. Pure
# Python literal (see module invariant above: no numpy/torch import here).
PROVENANCE_NAMES = {
    PROVENANCE_GAUSSIAN: "gaussian",
    PROVENANCE_MONODEPTH: "monodepth",
    PROVENANCE_LIDAR: "lidar",
    PROVENANCE_COLMAP: "colmap",
}

# --- points/gaussian_ply.py ---

# Spherical-harmonics degree-0 basis constant: 1 / (2 * sqrt(pi)). 3DGS
# stores colour as SH coefficients; the DC term f_dc converts to RGB via
# rgb = 0.5 + SH_C0 * f_dc (see docs/GEOMETRY.md "3DGS PLY export mapping").
SH_C0 = 0.28209479177387814

# Default confidence-mass cutoff for GaussianPlySource: points with
# sigmoid(opacity) below this are dropped before training (near-transparent
# Gaussians contribute negligible density and are mostly noise).
DEFAULT_MIN_OPACITY = 0.05

# --- points/knn_size.py ---

# Number of nearest neighbours averaged for kNN-based point size estimation
# (GaussianPlySource size_mode="knn", ColmapSparseSource size0). Matches the
# "local point spacing" heuristic used by 3DGS-style density init.
KNN_SIZE_K = 4

# Query-side chunk size (rows per cKDTree.query call) for kNN size/summary
# computations, bounding peak memory on multi-million-point clouds (the
# author 3DGS PLYs run to 7.36M points; a single unchunked query would
# allocate an (N, k+1) distance array up front).
KNN_CHUNK = 500_000

# --- points/source.py : PointSet.summary() ---

# Random-subsample size for PointSet.summary()'s median nearest-neighbour
# distance. The subsample is queried against itself (not the full cloud) so
# summary() stays fast (O(sample log sample)) regardless of point count;
# this is a density estimate for eyeballing, not an exact per-point stat.
SUMMARY_NN_SAMPLE = 20_000

# --- points/colmap_sparse.py ---

# ColmapSparseSource has no learned/estimated opacity (points3D.txt carries
# no confidence signal), so every point gets this fixed mid-scale confidence.
COLMAP_DEFAULT_CONF0 = 0.5

# --- cli.py : `trippy density` default paths (relative to SPLATS_ROOT) ---

DEFAULT_DENSITY_GAUSSIAN_PLY = "output/Training-Data/karekare/kk-coherent/kkc_15000.ply"
DEFAULT_DENSITY_COLMAP_SPARSE_DIR = "scenes/karekare/kk-coherent/sparse_txt"

# --- net/gated.py : Saiga::GatedBlockImpl ---
# Source: https://github.com/darglein/saiga @ ee7a4e6b65832433e2ca521353b7b7431c8e17a0
# src/saiga/vision/torch/PartialConvUnet2d.h:108-150 (fetched over the network per task
# authorization -- External/saiga/ is an empty dir in the vendored TRIPS checkout).

# GatedBlockImpl only supports kernel_size==3 (SAIGA_ASSERT(kernel_size == 3),
# PartialConvUnet2d.h:116) -- MultiScaleUnet2dDecOnlySmallFixed never calls it with anything
# else (Networks.h UnetBlockFromString(..., 3, 1, 1, ...) at every call site).
GATED_CONV_KERNEL_SIZE = 3

# --- net/unet.py : MultiScaleUnet2dDecOnlySmallFixed defaults ---
# configs/train_normalnet.ini:202-219 (TRIPS @ a59a65b6d9a8b1c14c73bc004cc9a8956f054c24).
NET_DEFAULT_NUM_INPUT_LAYERS = 5  # ini:202 num_input_layers
NET_DEFAULT_NUM_INPUT_CHANNELS = 4  # ini:203 num_input_channels
NET_DEFAULT_NUM_OUTPUT_CHANNELS = 3  # ini:204 num_output_channels
NET_DEFAULT_NUM_LAYERS = 5  # ini:206 num_layers
NET_DEFAULT_UPSAMPLE_MODE = "bilinear"  # ini:210 upsample_mode
NET_DEFAULT_NORM = "id"  # ini:211-212 norm_layer_down / norm_layer_up (both "id")
NET_DEFAULT_LAST_ACT = "id"  # ini:213 last_act
NET_DEFAULT_ACTIVATION = "elu"  # ini:217 activation (every gated block's feature activation)
# ini:219 `filters_network = 32 32 32 32 32 32 32 32` -- 8 entries (max num_layers the C++
# struct supports); only indices 0..num_layers-1 are read when num_layers=5, and they are
# all equal, so trippy's NetworkConfig simplifies this to one scalar `filters` field. TRIPS's
# struct technically allows a per-level list; that generality is unused by any shipped .ini
# and is out of scope here (documented, not guessed -- see docs/TRIPS_REFERENCE.md Sec. 5).
NET_DEFAULT_FILTERS = 32

# Upsample scale factor for every UpsampleDecOnlySmallBlockFixed / SmallDecStartBlock stage
# (Networks.h:1009 `std::vector<double> scale = {2.0, 2.0}`).
NET_UPSAMPLE_SCALE_FACTOR = 2.0

# --- net/camera_model.py : NeuralCameraParams / CameraResponseNetImpl defaults ---
# Sources: third_party/TRIPS/src/lib/data/Settings.h:110-141, configs/train_normalnet.ini:190-198.
CAMERA_DEFAULT_ENABLE_EXPOSURE = True  # ini:191
CAMERA_DEFAULT_ENABLE_WHITE_BALANCE = True  # ini:193 (Settings.h struct default is False; ini wins)
CAMERA_DEFAULT_ENABLE_VIGNETTE = True  # ini:190 (module always present; params init to 0 => no-op at init)
CAMERA_DEFAULT_ENABLE_RESPONSE = True  # ini:192
CAMERA_DEFAULT_RESPONSE_PARAMS = 25  # ini:196, CameraResponseNetImpl LUT control-point count
CAMERA_DEFAULT_RESPONSE_GAMMA = 1.0 / 2.2  # ini:197 (0.4545454681, a float32 round-trip of 1/2.2)
CAMERA_DEFAULT_RESPONSE_LEAK_FACTOR = 0.01  # ini:198 (0.009999999776, float32 round-trip of 0.01)

# CameraResponseNetImpl::ParamLoss's internal MSE weighting (NeuralCamera.cpp:154
# `double smoothness_factor = 1e-5;`), squared into the loss via `factor = n*sqrt(1e-5)`
# applied to both operands before torch::mse_loss(..., Reduction::Sum). This is a different
# constant from the ini's `response_smoothness` (=1, an outer multiplier applied at the
# Pipeline.cpp:804 call site, `params->optimizer_params.response_smoothness * camera->ParamLoss()`)
# -- both are folded into NeuralCamera.regularizer() in trippy's port for a single call site.
CAMERA_RESPONSE_SMOOTHNESS_INTERNAL_FACTOR = 1e-5
CAMERA_RESPONSE_SMOOTHNESS_OUTER_WEIGHT = 1.0  # ini response_smoothness (OptimizerParams section)

# CameraResponseNetImpl::forward's leaky-extrapolation epsilon (NeuralCamera.cpp:104
# `leaky_value / (image.abs() + 1e-4).sqrt()`), guarding the 1/sqrt(|x|) singularity at x=0.
CAMERA_RESPONSE_LEAK_SQRT_EPS = 1e-4

# --- net/losses.py : Saiga::SSIMImpl defaults ---
# Source: https://github.com/darglein/saiga @ 5fb87057f09f518b1ecf7de1a486420681455892
# src/saiga/vision/torch/ImageSimilarity.h:73-126 (fetched over the network; confirmed as the
# exact class TRIPS instantiates via `SSIM loss_ssim = SSIM();`, Pipeline.h:238, i.e. with the
# class's own defaults radius=2, max_value=1 -- NOT the generic "11x11" Wang et al. window).
SSIM_GAUSSIAN_RADIUS = 2  # window size = 2*radius + 1 = 5x5
SSIM_GAUSSIAN_SIGMA = 1.5
SSIM_MAX_VALUE = 1.0  # images are compared in [0, 1]; C1 = (0.01*max)^2, C2 = (0.03*max)^2
SSIM_C1_COEFF = 0.01
SSIM_C2_COEFF = 0.03

# --- net/losses.py : TripsLoss default weights ---
# configs/train_normalnet.ini:40-42,62-63 (`[TrainParams]`).
LOSS_DEFAULT_WEIGHT_VGG = 1.0  # ini:40 loss_vgg
LOSS_DEFAULT_WEIGHT_L1 = 1.0  # ini:41 loss_l1
LOSS_DEFAULT_WEIGHT_MSE = 0.0  # ini:42 loss_mse (unused by trippy's TripsLoss; kept for parity)
LOSS_DEFAULT_WEIGHT_SSIM = 1.0  # ini:62 loss_ssim
LOSS_DEFAULT_WEIGHT_LPIPS = 0.0  # ini:63 loss_lpips (off by default; VGG carries perceptual loss)
# --- scene/colmap_io.py binary format ---

# COLMAP's fixed camera_model_id -> (model_name, num_params) table, as used
# by cameras.bin (and by COLMAP's own scripts/python/read_write_model.py).
# Verified byte-for-byte against a real cameras.bin (6 OPENCV cameras,
# model_id=4, num_params=8) in ~/Splats/scenes/karekare/kk-coherent.
COLMAP_CAMERA_MODEL_TABLE: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

# --- scene/dataset.py cache ---

# Filename of the per-(scene, width) cache sidecar written by SceneDataset,
# holding scaled intrinsics/poses/EXIF for every cached image (used to
# validate the cache is still current on the next construction).
SCENE_CACHE_META_FILENAME = "meta.json"

# EXIF tag ids (TIFF/Exif spec, not COLMAP-specific) SceneDataset reads for
# the tone-mapper's per-image exposure/ISO init. Missing tags are fine.
EXIF_TAG_EXPOSURE_TIME = 33434
EXIF_TAG_ISO = 34855

# TIFF/Exif tag id of the "Exif IFD Pointer" -- the sub-IFD that actually
# holds ExposureTime/ISO (the base IFD PIL's Image.getexif() returns does
# not), per the Exif spec.
EXIF_TAG_EXIF_IFD_POINTER = 0x8769

# --- scene/splits.py ---

# Default modulo-split stride: 1 in MODULO_SPLIT_DEFAULT_K images is held
# out, the rest train. Matches the "held-out" fraction used for v0.1.0/
# v0.2.0 PSNR gates in docs/SPEC.md.
MODULO_SPLIT_DEFAULT_K = 8
MODULO_SPLIT_DEFAULT_OFFSET = 0

# Six consecutive kk-coherent frames that pass through the shade region
# under the trees (the project's core defect, see docs/SPEC.md "Context"
# and docs/EXPERIMENTS.md's dolly camera note, which anchors on
# IMG_3830.jpg as the centre of the shade region). Forced into the
# held-out set so every eval reports a shade-region number, not just an
# average over easy frames. See docs/EXPERIMENTS.md ("shade frames
# IMG_3828-3833").
SHADE_FRAMES_KK = [
    "IMG_3828.jpg",
    "IMG_3829.jpg",
    "IMG_3830.jpg",
    "IMG_3831.jpg",
    "IMG_3832.jpg",
    "IMG_3833.jpg",
]

# --- train/export.py : 3DGS PLY export (docs/GEOMETRY.md "3DGS PLY export
# mapping"; must invert exactly against trippy.points.gaussian_ply's read
# side, which uses SH_C0 above and its own sigmoid/exp) ---

# Clamp bound for conf0 before taking logit() to produce the exported
# `opacity` field: conf0 is only guaranteed to lie in the open interval
# (0, 1) (see PointSet docstring), but a value within float32 epsilon of
# 0 or 1 makes logit() +/-inf. Clamping to [eps, 1-eps] keeps opacity
# finite; DEFAULT_MIN_OPACITY's read-side filter (0.05) is far above this
# eps, so no point that survives that filter is ever affected.
EXPORT_OPACITY_CLAMP_EPS = 1e-4

# TRIPS has no learned rotation (points are isotropic splats); exported
# 3DGS `rot_0..3` (wxyz, per docs/GEOMETRY.md "Quaternions") is always the
# identity rotation.
EXPORT_IDENTITY_ROT = (1.0, 0.0, 0.0, 0.0)

# sh_degree flag values accepted by write_gaussian_ply(): 0 (default, no
# f_rest_* properties written -- TRIPS colour is DC-only) or 3 (writes
# zero-filled f_rest_0..44 so strict 3DGS viewers expecting a full SH
# basis don't choke on a missing property).
SH_DEGREE_ZERO = 0
SH_DEGREE_THREE = 3

# Number of "rest" (non-DC) SH coefficients for degree-3, 3-channel colour:
# 3 channels * ((3 + 1)**2 - 1) = 3 * 15 = 45, i.e. f_rest_0..44.
SH_DEGREE_THREE_NUM_REST_COEFFS = 45

# --- render/sheets.py : contact_sheet() / colorize() ---

# contact_sheet() defaults: each thumbnail is resized to fit within a
# cell_max x cell_max box; pad is the gutter (pixels) around every cell
# and between the sheet border and its content; bg fills the space a
# non-square thumbnail doesn't cover plus the inter-cell gutters.
CONTACT_SHEET_CELL_MAX = 512
CONTACT_SHEET_PAD = 8
CONTACT_SHEET_BG = (20, 20, 20)

# Vertical band (pixels) reserved under each thumbnail for its PIL
# default-font label. PIL's built-in bitmap font is ~11px tall; this
# leaves comfortable clearance without measuring text per-call.
CONTACT_SHEET_LABEL_BAND_PX = 18

# colorize()'s 5-stop RGB ramp (0, uint8), sampled at t = 0, 0.25, 0.5,
# 0.75, 1.0 -- a hand-picked approximation of matplotlib's viridis
# colormap (dark purple -> blue -> teal -> green -> yellow), used so
# depth/coverage panels get a perceptually-monotonic colour map without
# adding a matplotlib dependency (AGENTS.md: no new dependencies).
COLORMAP_VIRIDIS_STOPS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)

# --- render/video.py : write_video() ---

# Default output frame rate (frames/sec) when the caller doesn't specify
# one; matches the dolly render cadence in docs/EXPERIMENTS.md ("Dolly
# camera paths", typically 24 fps, but 30 is the more common default for
# quick-look contact videos of arbitrary frame sequences).
VIDEO_DEFAULT_FPS = 30

# ffmpeg raw-frame input pixel format: frames are always 8-bit RGB,
# channel-interleaved, matching the uint8 HxWx3 arrays this module
# accepts everywhere else.
FFMPEG_RAW_PIX_FMT = "rgb24"

# ffmpeg encoded-output pixel format (yuv420p): the most broadly
# compatible H.264 output format for playback outside ffmpeg itself.
FFMPEG_OUTPUT_PIX_FMT = "yuv420p"

# Hardware H.264 encoder used when available (Apple Silicon VideoToolbox
# is a free win on this machine per Splats' PROJECT.md notes) and its
# software fallback.
FFMPEG_VIDEOTOOLBOX_ENCODER = "h264_videotoolbox"
FFMPEG_LIBX264_ENCODER = "libx264"
# --- raster/ : pyramid rasteriser (forward pass) ---
# Sources for the TRIPS-derived values below are quoted in
# docs/TRIPS_REFERENCE.md sections 3 and 10; `path:line` refers to
# third_party/TRIPS @ a59a65b.

# Number of pyramid layers rendered per image. TRIPS: net_params.
# num_input_layers = 5 (configs/train_normalnet.ini, RenderModule.cpp:11).
RASTER_NUM_LAYERS = 5

# Points with camera-space depth z <= this (world units) are culled before
# projection. TRIPS rejects z <= 0 only; we use a small positive near plane
# so that s = fx * size / z cannot overflow float32 for points sitting on
# the optical centre.
RASTER_ZNEAR = 1e-3

# Slack, in *coarsest-layer* pixels, added around the image when deciding
# whether a point is fully outside it. The coarse cull must never remove a
# point that any layer's exact per-fragment bounds test would have kept: a
# fragment at layer l needs uv/2**l within (-1.5, w_l + 0.5), so 2 coarsest
# pixels of slack (= 2 * 2**(L-1) layer-0 pixels) is strictly conservative.
RASTER_CULL_MARGIN_COARSE_PX = 2.0

# Fragments whose alpha is below this are dropped at emission time. TRIPS has
# no such rule (every in-bounds bilinear corner is written); we drop them
# because they cost a slot in the 16-deep per-pixel list while changing the
# composite by at most this value times the feature magnitude.
RASTER_ALPHA_MIN = 1e-5

# Transmittance below which front-to-back compositing stops for a pixel.
# TRIPS ALPHA_DEST_CUTOFF = 0.001 (RenderForward.cu:3522).
RASTER_T_CUTOFF = 1e-3

# Maximum number of fragments composited per layer-pixel. TRIPS
# ELEMENTS_PER_PIXEL / max_pixels_per_list = 16 (PointRenderer.h:178).
RASTER_MAX_FRAGS = 16

# Floor on the layer blend factor for sub-pixel points:
# layer_factor = (1 - c) * exp(s - 1) + c with c = this value
# (PointBlending.h:106, `cutoff_value = 0.25f`).
RASTER_SMALL_POINT_CUTOFF = 0.25

# Alpha is clamped below 1 by this epsilon before log1p(-alpha) in the
# vectorised float64 reference, so a fragment with alpha == 1 cannot produce
# log(0) = -inf. 1 - 1e-12 is still exactly representable in float64.
RASTER_ALPHA_MAX_EPS = 1e-12

# Feature channel counts the Metal blend_fwd kernel is templated for. TRIPS
# dispatches per num_texture_channels in {3,4,8,16} (RenderForward.cu:3808);
# we ship {3,4,8} (4 is the shipped TRIPS default).
RASTER_SUPPORTED_CHANNELS = (3, 4, 8)

# Layer-selection modes the rasteriser implements (trippy.raster.emit.EMIT_MODES).
# "trips"     -- the rule the published TRIPS checkpoints actually render with:
#                use_layer_point_size = !fix_point_size = true, so each point is
#                written into layers 0..layer_higher and weighted by
#                compute_point_size_fac (RenderForward.cu:334-352 + :3511-3517).
# "trilinear" -- only the two layers the projected size straddles
#                (CollectTiled2Pointsize, RenderForward.cu:2296-2360).
# "broadcast" -- every layer with factor 1 (use_layer_point_size = false).
# Defined here, and not in trippy.raster.emit, so that trippy.train.config can
# validate a YAML `mode:` without importing torch.
RASTER_MODES = ("trilinear", "broadcast", "trips")

# Where the rasteriser puts the centre of pixel index i on the continuous
# coordinate axis, i.e. where the 2x2 bilinear footprint is anchored.
# "half"    -- centre at i + 0.5 (docs/GEOMETRY.md, trippy's own convention;
#              base = floor(uv/2**l - 0.5)).
# "integer" -- centre at i (TRIPS's `ip`, PointBlending.h:216-240;
#              base = floor(ip/2**l)). Needed only to reproduce a TRIPS
#              checkpoint bit-for-bit; trippy's own scenes use "half".
RASTER_PIXEL_CENTERS = ("half", "integer")

# How a pyramid layer's size is derived from the one below.
# "ceil"  -- h_{l+1} = ceil(h_l / 2). TRIPS for every network_version except the
#            literal "MultiScaleUnet2d", i.e. for every published checkpoint
#            (PointRenderer.cu:385-391), and trippy's default.
# "floor" -- h_{l+1} = h_l // 2, TRIPS's `MultiScaleUnet2d` branch; loses the
#            last row/column of an odd-sized layer.
RASTER_PYRAMID_HALVINGS = ("ceil", "floor")

# Bits reserved for the depth part of the composite int64 sort key. A float32
# depth's IEEE bit pattern is monotonically increasing for positive values, so
# key = layer_pixel * 2**32 + depth_bits sorts by (layer, pixel, depth).
RASTER_SORT_DEPTH_BITS = 32

# Largest flat layer-pixel index the composite sort key can hold without
# overflowing int64: 2**63 / 2**RASTER_SORT_DEPTH_BITS.
RASTER_SORT_MAX_LAYER_PIXELS = 1 << 31

# --- render/pyramid_render.py : `trippy render` orchestration ---

# Subdirectory of TRIPPY_OUTPUT that SceneDataset's undistortion cache is
# written under by `trippy render` (see trippy.config.load_settings).
RENDER_CACHE_SUBDIR = "cache"

# A layer-0 pixel with (1 - t_final) at or below this is treated as
# "nothing was drawn here" for the honesty coverage/depth maps -- below
# this the expected-depth divide (depth_sum / (1 - t_final)) is numerically
# unstable and the pixel is rendered black instead of a fabricated depth.
# One order of magnitude above RASTER_T_CUTOFF's compositing stop threshold.
RENDER_COVERAGE_EPS = 1e-2

# Percentile clamp (of the *covered* pixels' expected depth) used as
# colorize()'s vmin/vmax for the depth panel, so a handful of far outlier
# points (e.g. sky, mis-triangulated Gaussians) don't wash out the ramp.
RENDER_DEPTH_PERCENTILE_LOW = 1.0
RENDER_DEPTH_PERCENTILE_HIGH = 99.0

# Central crop fraction (of height and width) used for metrics.json's
# "mean_center" coverage number: a numeric proxy for "is the middle of the
# frame covered", computed directly from the T_final tensor (never from a
# rendered image) so a shade-region coverage verdict never requires opening
# a photo-derived image (AGENTS.md privacy rule).
RENDER_CENTER_REGION_FRAC = 0.5
# --- points/depth_io.py : Splats' DepthPro (depth_batch.py) integration ---
# Source: ~/Splats/tools/ldi/depth_batch.py (READ ONLY -- never copied or
# edited; trippy only shells out to it). Chosen over the alternative
# ~/Splats/tools/va_depth/run_depthpro.py because depth_batch.py already
# writes exactly the per-image <id>_depth.npy / <id>_mask.npy /
# <id>_meta.json triple this module needs, driven by a plain JSON manifest
# -- no extra per-scene glue script required. Verified working end to end
# via the existing tools/ldi/jobs/run_depth_job.sh GPU-queue job (see
# ~/Splats/tools/gpu_queue/logs/15-ldi-depth.log): ~1.5s/image on MPS,
# apple/DepthPro-hf weights already in the local HF cache (HF_HUB_OFFLINE=1
# set by depth_batch.py itself, so no network access).
DEPTHPRO_VENV_PYTHON_REL = "tools/vggt/.venv/bin/python3"  # relative to SPLATS_ROOT
DEPTHPRO_SCRIPT_REL = "tools/ldi/depth_batch.py"  # relative to SPLATS_ROOT

# depth_batch.py's fixed output filename suffixes (per its own module
# docstring) -- one instance of each per manifest record "id".
DEPTHPRO_DEPTH_SUFFIX = "_depth.npy"
DEPTHPRO_MASK_SUFFIX = "_mask.npy"
DEPTHPRO_META_SUFFIX = "_meta.json"

# --- points/monodepth.py ---

# Every stride-th valid depth pixel (in both row and column) is
# backprojected, so point density scales as ~1/stride^2 -- the main
# density/runtime knob for MonoDepthSource.
MONODEPTH_DEFAULT_STRIDE = 6

# World-unit voxel edge length for MonoDepthSource's own dedupe pass over
# its (potentially heavily overlapping-by-frame) backprojected points.
MONODEPTH_DEFAULT_VOXEL = 0.03

# Fixed per-point confidence -- points3D.txt carries no per-point
# uncertainty and DepthPro's output here carries no calibrated per-pixel
# uncertainty channel either, so every backprojected point gets this fixed
# value, same pattern as points.colmap_sparse.COLMAP_DEFAULT_CONF0 but
# deliberately lower (0.35 vs 0.5): monocular depth is a noisier signal
# than triangulated SfM points.
MONODEPTH_DEFAULT_CONF0 = 0.35

# Only "median_ratio" (median of per-point z_colmap / d_pred) is
# implemented; kept as a named mode string (not a bare literal) so a
# future alternative (e.g. a least-squares affine scale+shift fit) has an
# obvious place to plug in -- see research/README.md parked ideas.
MONODEPTH_SCALE_MODE_MEDIAN_RATIO = "median_ratio"

# Minimum number of COLMAP sparse points landing on valid depth pixels
# needed before an image's median-ratio scale is trusted; frames below
# this are dropped from the union and reported (not silently) in
# MonoDepthSource.describe()'s per-image stats.
MONODEPTH_MIN_SCALE_MATCHES = 5

# `trippy depth-points --run-depth` exit code: "depth outputs are missing,
# go run the printed GPU job first" -- distinct from argparse's own exit(2)
# for bad arguments and the normal exit(0) success path.
DEPTH_POINTS_MISSING_DEPTH_EXIT_CODE = 3
# --- train/ : TrainConfig defaults, PointParams parametrisation, trainer schedule ---
# Sources are configs/train_normalnet.ini (TRIPS @ a59a65b6d9a8b1c14c73bc004cc9a8956f054c24)
# unless noted; every value below is scaled or substituted deliberately for trippy's smaller
# compute budget / different optimizer, and says so inline. See docs/ARCHITECTURE.md "train/"
# and docs/EXPERIMENTS.md "Training runs" for the narrative version of these decisions.

# Dataset width: half of the 2016px "wide" undistort cache (docs/SPEC.md v0.1.0), keeping the
# per-crop memory footprint small while still resolving the shade region.
TRAIN_DEFAULT_WIDTH = 1008

# ini:422 train_crop_size=512; trippy trains on smaller square crops (memory-safe first,
# per task brief) since a full 512 crop rasterises 512*512*RASTER_MAX_FRAGS fragments per
# pyramid layer on a CPU-testable, single-crop-per-step budget.
TRAIN_DEFAULT_CROP = 384

# ini:447 min_zoom=0.75, max_zoom=1.5 (crop augmentation range).
TRAIN_DEFAULT_ZOOM_MIN = 0.75
TRAIN_DEFAULT_ZOOM_MAX = 1.5

# ini batch_size=4, inner_batch_size=4; trippy starts at one crop per optimizer step
# (memory-safe first, per task brief) rather than reproducing TRIPS's 4x4 batching.
TRAIN_DEFAULT_CROPS_PER_STEP = 1

# ini:422 num_epochs=600, scaled down for trippy's much smaller per-epoch step count (see
# TRAIN_DEFAULT_TRAIN_FACTOR) and shorter overnight training-job budget.
TRAIN_DEFAULT_EPOCHS = 150

# Not an ini field: ADOP/TRIPS's own "epoch" is itself a fixed step count from
# [TrainParams], not one pass over the dataset (this checkout's Pipeline/train.cpp epoch
# loop was out of scope for docs/TRIPS_REFERENCE.md's extraction). trippy defines one epoch
# as a fraction of the training set so `steps_per_epoch = ceil(train_factor * n_train)`
# scales sensibly across scenes of very different image counts; 0.125 means roughly one
# in 8 training images gets a crop step per epoch.
TRAIN_DEFAULT_TRAIN_FACTOR = 0.125

# Fractions of `lock_*_epochs` / `num_epochs` and `only_start_vgg_after_epochs` /
# `num_epochs` from the ini (lock_camera_params_epochs=100, lock_structure_params_epochs=10,
# only_start_vgg_after_epochs=100, num_epochs=600). TrainConfig multiplies these fractions by
# its own `epochs` field, so the *proportion* of a run spent locked/pre-VGG matches TRIPS
# regardless of how many epochs trippy actually runs.
TRAIN_LOCK_CAMERAS_FRAC = 100 / 600
TRAIN_LOCK_STRUCTURE_FRAC = 10 / 600
TRAIN_VGG_START_FRAC = 100 / 600

# ini:151 lr_render_network=0.0002 -- resolution/optimizer-independent, used unscaled.
TRAIN_DEFAULT_LR_NETWORK = 2e-4

# ini lr_texture=0.1, tuned for TRIPS's own custom `MyAdam` optimizer (src/lib/models/
# MyAdam.cu), whose update rule is not detailed in docs/TRIPS_REFERENCE.md Sec. 7 ("not
# traced in detail here"). trippy uses plain torch.optim.Adam, whose effective step size at
# a given lr differs from MyAdam's; empirically 0.1 way overshoots a per-point feature
# vector under vanilla Adam's fixed learning rate (no visitation-count normalisation).
# Started 10x lower; re-tune from the first real training run (see EXP-0003).
TRAIN_DEFAULT_LR_TEXTURE = 0.01

# ini lr_background_color=0.004.
TRAIN_DEFAULT_LR_BACKGROUND = 0.004

# ini lr_points=0.0001.
TRAIN_DEFAULT_LR_POINTS = 1e-4

# Not an ini field: TRIPS's own point-size parameter is dead code under the shipped default
# (use_layer_point_size=false is unreachable from any .ini, docs/TRIPS_REFERENCE.md Sec. 2/
# 11), so no lr_pointsize exists to copy. trippy's "trilinear" render mode does use size for
# real (layer selection), so it needs its own rate; reuses lr_points' value since both
# parameters affect point geometry at a similar physical scale.
TRAIN_DEFAULT_LR_SIZE = 1e-4

# ini lr_poses=0.0001.
TRAIN_DEFAULT_LR_POSES = 1e-4

# ini lr_confidence=0.001.
TRAIN_DEFAULT_LR_CONFIDENCE = 1e-3

# ini lr_exposure=0.0005. Exposure is not in the ini's fix_* list (fix_intrinsics,
# fix_dynamic_refinement, fix_vignette, fix_wb, fix_motion_blur, fix_rolling_shutter --
# docs/TRIPS_REFERENCE.md Sec. 7), so it is trained by default.
TRAIN_DEFAULT_LR_EXPOSURE = 5e-4

# ini lr_response=0.0001. Also absent from the fix_* list, so trained by default (unlike
# vignette/white-balance, which trippy freezes -- see trainer.py).
TRAIN_DEFAULT_LR_RESPONSE = 1e-4

# ini lr_decay_factor=0.85, lr_decay_patience=10 (plateau-style LR decay, driven here by
# held-out PSNR rather than TRIPS's untraced train.cpp consumer).
TRAIN_LR_DECAY_FACTOR = 0.85
TRAIN_LR_DECAY_PATIENCE = 10

# "trips" is the layer rule the published TRIPS checkpoints were trained and rendered with
# (`use_layer_point_size = !fix_point_size = true`, Settings.cpp:39; docs/TRIPS_REFERENCE.md
# Sec. 3a). Measured on three held-out tt_horse frames it beats trippy's older "trilinear"
# reading by 0.8 dB and the "broadcast" reading by 7.1 dB
# (experiments/EXP-0002-horse-parity/README.md), so it is the trainer default.
TRAIN_DEFAULT_MODE = "trips"

# Pixel-centre and pyramid-halving conventions the trainer renders with. trippy's own
# scenes are undistorted with pixel centres at i + 0.5 (docs/GEOMETRY.md), so training
# stays on "half"; "integer" exists only to reproduce a TRIPS checkpoint (see
# RASTER_PIXEL_CENTERS). "ceil" is both trippy's and TRIPS's halving for this network.
TRAIN_DEFAULT_PIXEL_CENTER = "half"
TRAIN_DEFAULT_PYRAMID_HALVING = "ceil"

# ini num_input_layers=num_layers=5 (matches NET_DEFAULT_NUM_LAYERS/RASTER_NUM_LAYERS).
TRAIN_DEFAULT_LAYERS = 5

# ini num_texture_channels=4 (matches NET_DEFAULT_NUM_INPUT_CHANNELS).
TRAIN_DEFAULT_FEATURE_CHANNELS = 4

# NeuralTexture.cpp:37 `background_color_raw = ones(C) * 0.25` (fac_init=0.25), per-channel.
TRAIN_DEFAULT_BACKGROUND = 0.25

# Matches MODULO_SPLIT_DEFAULT_K -- trippy's held-out fraction (1 in 8 images).
TRAIN_DEFAULT_HELDOUT_K = 8

# Not an ini field (TRIPS logs to tfevents continuously); chosen so a 150-epoch run gets
# 15 held-out evaluations (also the default checkpoint cadence, so every checkpoint has a
# metrics reading next to it).
TRAIN_DEFAULT_EVAL_EVERY = 10
TRAIN_DEFAULT_CHECKPOINT_EVERY = 10

TRAIN_DEFAULT_SEED = 0

# docs/SPEC.md v0.2.0 acceptance / task brief: "up to 6 held-out images" per honesty/eval
# contact sheet.
TRAIN_EVAL_MAX_SHEET_IMAGES = 6

# NeuralTexture.h:42 `confidence = sigmoid((10 + narrowing_param_times_epoch) *
# confidence_raw)`; sigmoid_narrowing_factor=0 in the default config (ini:139), so this
# simplifies to sigmoid(10 * confidence_raw) at every epoch. docs/TRIPS_REFERENCE.md Sec.
# 10.4 flags dropping this factor as a fidelity gap (e.g. for 3DGS-opacity comparisons);
# trippy's decision (this task): KEEP the x10 scale in PointParams.conf(), matching TRIPS
# exactly, rather than a plain sigmoid(confidence_raw).
CONF_SIGMOID_SCALE = 10.0

# torch.nn.functional.softplus's own default threshold (beta=1, above which softplus(x)
# linearises to x) -- named here because PointParams.raw_size's init (inverse_softplus)
# must use the exact same threshold to round-trip, matching NeuralPointCloudCuda.cpp:19-24's
# "beta=1, threshold=20" citation in docs/TRIPS_REFERENCE.md Sec. 2.
TRAIN_SOFTPLUS_THRESHOLD = 20.0

# Not in TRIPS (NeuralTexture.cpp's texture_raw init is plain Uniform(0,1) on every
# channel, docs/TRIPS_REFERENCE.md Sec. 2). Task brief: PointParams.feat's first 3 channels
# are seeded with rgb0 (so the untrained net sees real colour immediately) and any
# remaining channels get small random noise at this standard deviation, rather than
# TRIPS's full-range Uniform(0,1) init.
TRAIN_FEAT_EXTRA_INIT_STD = 0.01

# New trippy addition, not present in TRIPS: a soft penalty keeping trained point positions
# near the initial point cloud's bounding box, guarding docs/SPEC.md's "extent inflation"
# risk (scene sprawl making rendering/export unusable). Small by design so it only engages
# once points drift meaningfully past the (margin-padded) initial bbox.
TRAIN_EXTENT_PENALTY_WEIGHT_DEFAULT = 1e-3

# Matches docs/EXPERIMENTS.md's extent gate wording ("should not exceed the extent of the
# original sparse COLMAP points by >20%") -- the soft penalty's free margin before it starts
# pushing back.
TRAIN_EXTENT_MARGIN_FRAC = 0.2

# eval.py's held-out LPIPS metric backbone -- matches TripsLoss's own vgg-substitute choice
# (trippy.net.losses module docstring) so the training loss and the reported metric agree.
TRAIN_LPIPS_METRIC_NET = "vgg"

# Guards -10*log10(mse) against mse == 0 (a pred/target exact match, e.g. a degenerate
# all-zero synthetic test image) producing +inf.
TRAIN_PSNR_EPS = 1e-10

# --- hybrid/ : Design C, render->photo U-Net refinement (docs/PLAN-2026-09-05.md
# "Hybrid (v0.3): (C) render->photo U-Net refinement on gsrender.py outputs first (cheap,
# validates net/losses)"; docs/EXPERIMENTS.md "Training runs" for the sibling point-based
# trainer this one deliberately does NOT reuse (Trainer is point-based; render_splat_views
# + this section's constants back trippy.hybrid's own small image->image trainer instead) ---

# ~/Splats/tools/gsrender.py's own `max_hw` kwarg defaults to 32, which corrupts near-camera
# Gaussian footprints (Splats' PROJECT.md note, also cited in this task's brief); every
# trippy.hybrid.render_splat_views call overrides it with this value.
HYBRID_C_GSRENDER_MAX_HW = 400

# gsrender.render()'s own opacity-cutoff default (min_opacity kwarg), used unscaled here so
# trippy.hybrid.render_splat_views reproduces "the existing best Gaussian splat" render
# faithfully rather than silently re-filtering it a second time.
HYBRID_C_GSRENDER_MIN_OPACITY = 0.02

# Subdirectory name (under output/hybrid-c/renders/) rendered rgb/depth/alpha triples are
# written under, keyed by the dataset width so multiple resolutions never collide on disk.
HYBRID_C_RENDER_SUBDIR_FMT = "w{width}"

# Default render feature-channel count fed to the U-Net: rgb (3) + alpha (1) -- matches
# NET_DEFAULT_NUM_INPUT_CHANNELS/TRAIN_DEFAULT_FEATURE_CHANNELS exactly, so the same
# NetworkConfig applies unchanged to this design (task brief: "make channels configurable,
# default 4 so the same net config as TRIPS applies").
HYBRID_C_DEFAULT_CHANNELS = 4

# Channel count when depth is included as an optional 5th input channel (rgb + alpha + depth).
HYBRID_C_CHANNELS_WITH_DEPTH = 5

# Rough normalisation divisor for the optional depth channel: gsrender's depth output is
# alpha-weighted expected camera-space z in COLMAP/world metres (see gsrender.py docstring);
# dividing by this constant keeps typical kk-coherent depths (a handful to a few dozen world
# units) in an order-1 range before concatenation with the [0, 1] rgb/alpha channels. This is
# a coarse heuristic for this "cheap side-experiment" (docs/PLAN-2026-09-05.md), not a
# per-scene calibrated fit -- unlike, say, MonoDepthSource's median-ratio scale alignment.
HYBRID_C_DEPTH_NORM_SCALE = 20.0

# Number of held-out frames written to the honesty sheet / delivered as standalone PNGs.
# Mirrors TRAIN_EVAL_MAX_SHEET_IMAGES's "up to 6" convention as its own constant (rather than
# importing it) so trippy/hybrid never depends on trippy/train/trainer.py internals -- the two
# trainers are deliberately independent per this task's brief ("write a small separate
# trainer for image->image").
HYBRID_C_EVAL_MAX_SHEET_IMAGES = 6

# --- train/checkpoint_io.py, trainer.py : on-disk run layout ---
# docs/EXPERIMENTS.md "Run location": output/runs/<exp>/<run>/{...}.
TRAIN_CHECKPOINT_DIRNAME = "checkpoints"
TRAIN_CHECKPOINT_FILENAME_FMT = "checkpoint_ep{epoch:04d}.pt"
TRAIN_CHECKPOINT_LATEST_FILENAME = "checkpoint_latest.pt"
TRAIN_LOG_FILENAME = "log.txt"
TRAIN_METRICS_FILENAME = "metrics.jsonl"
TRAIN_EVAL_DIRNAME_FMT = "eval_ep{epoch:04d}"
TRAIN_EVAL_SHEET_FILENAME = "sheet.png"
TRAIN_EVAL_METRICS_FILENAME = "metrics.json"
TRAIN_EXPORT_FILENAME = "export.ply"
# --- scene/adop_io.py : ADOP scene directory + Saiga containers ---
# Sources: third_party/TRIPS/src/lib/data/SceneData.{h,cpp} and the public MIT
# Saiga tree at https://github.com/darglein/saiga @ ee7a4e6b658 (External/saiga/
# is an empty dir in the vendored TRIPS checkout, so the Saiga files below were
# fetched over the network -- same authorization as net/gated.py's constants).

# Magic word at the head of every Saiga `compress()` container, e.g.
# point_cloud.bin (saiga/core/util/zlib.cpp:16 `magic_value`).
ADOP_POINT_CLOUD_MAGIC = 0x006712956A9725DE

# Saiga's compressed-container header: three little-endian size_t values
# (magic, compressed_size, decompressed_size), zlib.cpp:15 `header_size`.
ADOP_COMPRESSED_HEADER_BYTES = 24

# Vertex arrays a Saiga UnifiedMesh dumps, in serialization order, with their
# float width (saiga/core/model/UnifiedMesh.{h:44-48,cpp:508-528}). Every
# std::vector<T> is written as a size_t count followed by packed elements.
ADOP_UNIFIED_MESH_VERTEX_FIELDS = (
    ("position", 3),
    ("normal", 3),
    ("color", 4),
    ("texture_coordinates", 2),
    ("data", 4),
)

# Saiga's lens model carries exactly 8 coefficients, stored as
# `k1 k2 k3 k4 k5 k6 p1 p2` (saiga/vision/cameraModel/Distortion.h:20-45) --
# NOT OpenCV's `k1 k2 p1 p2 k3 k4 k5 k6` order.
ADOP_DISTORTION_COEFFS = 8

# RenderParams::dist_cutoff (third_party/TRIPS/src/lib/data/Settings.h:61):
# distortNormalizedPoint maps any point with r2 > dist_cutoff**2 to the
# sentinel 1e5, which the caller turns into z = 0 (i.e. "cull this point"),
# PointRendererHelper.h:244-249.
ADOP_DIST_CUTOFF = 20.0

# Sentinel distortNormalizedPoint writes when r2 exceeds max_r**2
# (Distortion.h:166-170).
ADOP_DISTORTION_SENTINEL = 100000.0

# --- net/checkpoint.py : TRIPS scene-checkpoint parametrisation ---

# NeuralPointTextureImpl::PrepareConfidence (models/NeuralTexture.h:38-42):
# `confidence = sigmoid((10 + sigmoid_narrowing_factor * epoch) * confidence_raw)`.
# The narrowing term is 0 in every shipped config and in the public Tanks &
# Temples checkpoints (params.ini `sigmoid_narrowing_factor = 0`).
TRIPS_CONFIDENCE_SIGMOID_SCALE = 10.0

# A Sophus::SE3d occupies 8 doubles: quaternion (x, y, z, w) + translation
# (x, y, z) + one padding double, and PoseModuleImpl stores the batch as an
# [N, 8] float64 tensor (data/NeuralStructure.cpp:25-33). The stored pose is
# `frame.pose.inverse()`, i.e. WORLD-TO-CAMERA.
TRIPS_SE3_DOUBLES_PER_POSE = 8

# Softplus parameters used for the point-size parametrisation
# (rendering/NeuralPointCloudCuda.cpp:19-24, RenderForward.cu:154):
# `size_world = softplus(t_point_size)` with beta = 1, threshold = 20.
TRIPS_SOFTPLUS_BETA = 1.0
TRIPS_SOFTPLUS_THRESHOLD = 20.0

# Layout of IntrinsicsModule's `intrinsics` row: fx fy cx cy s followed by the
# 8 distortion coefficients (data/NeuralStructure.cpp, SceneData.h:209-241).
TRIPS_INTRINSICS_ROW_LEN = 13

# Filenames inside an `ep<NNNN>/` checkpoint directory (Sec. 9 of
# docs/TRIPS_REFERENCE.md); `{scene}` is the scene name from params.ini.
TRIPS_CKPT_NETWORK_FILE = "render_net.pth"
TRIPS_CKPT_SCENE_FILES = {
    "points": "scene_{scene}_points.pth",
    "texture": "scene_{scene}_texture.pth",
    "poses": "scene_{scene}_poses.pth",
    "intrinsics": "scene_{scene}_intrinsics.pth",
    "exposure": "scene_{scene}_ex.pth",
    "white_balance": "scene_{scene}_wb.pth",
    "response": "scene_{scene}_response.pth",
    "vignette": "scene_{scene}_vignette.pth",
}

# --- render/parity.py : TRIPS-faithful forward render ---

# num_input_layers / num_layers in the public Tanks & Temples checkpoints'
# own params.ini (docs/TRIPS_REFERENCE.md Sec. 5a) -- not the 5 of
# configs/train_normalnet.ini.
PARITY_DEFAULT_NUM_LAYERS = 8

# TRIPS indexes its bilinear footprint straight off `ip` (the 2x2 corners are
# floor(ip) and floor(ip)+1, PointBlending.h:216-240), i.e. pixel *centres*
# sit at integer coordinates. trippy's rasteriser uses the corner-origin
# convention where pixel i's centre is at i + 0.5 (docs/GEOMETRY.md), so the
# intrinsics handed to render_pyramid get cx, cy shifted by this much.
PARITY_PIXEL_CENTRE_OFFSET = 0.5

# Peak signal value for PSNR: images are compared in [0, 1].
PARITY_PSNR_MAX = 1.0

# PSNR of a numerically identical pair is +inf; report this instead so the
# metrics JSON stays valid JSON.
PARITY_PSNR_CAP_DB = 99.0

# Held-out (test) frame indices this experiment renders by default. The
# checkpoint's own test split is `modulo` with train_factor 0.125, i.e.
# indices 0, 8, 16, ... 144 (checkpoint_horse/test_indices_tt_horse.txt);
# these three sample the start, middle and end of that split.
PARITY_DEFAULT_INDICES = (8, 120, 144)

# Border (pixels, each side) TRIPS blacks out in the test images it writes at
# checkpoint time -- `train_mask_border = 16` in the published
# checkpoint_horse/params.ini, and measured directly on
# checkpoint_horse/ep0600/test/*.jpg (15-16 all-zero rows/columns per side on
# every frame checked). Any PSNR against those files must exclude it, or the
# authors' own render scores 15 dB against its own ground truth.
PARITY_EVAL_BORDER_PX = 16

# --- render/dolly.py : shade dolly camera path ---
# Same construction as ~/Splats/tools/depthprior_shade_dolly.py: pose_name's
# own COLMAP orientation is frozen and the camera centre slides along that
# pose's forward ray. Defaults below cite the exact source lines.

# depthprior_shade_dolly.py:33 `--pose-name` default -- IMG_3830.jpg is the
# centre of the karekare shade region (docs/EXPERIMENTS.md "Dolly camera
# paths", SHADE_FRAMES_KK above).
DOLLY_DEFAULT_POSE_NAME = "IMG_3830.jpg"

# depthprior_shade_dolly.py:38-39 `--t-start`/`--t-end` defaults: walk from
# before the shade frame's own position (-0.35x local depth) to well past
# it (+1.20x).
DOLLY_DEFAULT_T_START = -0.35
DOLLY_DEFAULT_T_END = 1.20

# Not a Splats default (that script's `--frames` defaults to 9, sized for a
# contact-sheet grid); trippy's dolly is a video, so it defaults to enough
# frames for a smooth ~2s clip at VIDEO_DEFAULT_FPS.
DOLLY_DEFAULT_N_FRAMES = 48

# render_offpath.py:137 `--width` default (both the dolly and off-path
# generators share this).
DOLLY_DEFAULT_WIDTH = 640

# depthprior_shade_dolly.py:59 `infront.sum() > 50` -- below this count the
# local-depth estimate falls back to using every sparse point (not just
# ones in front of the camera).
DOLLY_DEPTH_INFRONT_MIN_COUNT = 50

# depthprior_shade_dolly.py:60 `np.percentile(dsel, [5, 95])` -- trims
# outlier near/far points before taking the median.
DOLLY_DEPTH_PERCENTILE_LOW = 5.0
DOLLY_DEPTH_PERCENTILE_HIGH = 95.0

# depthprior_shade_dolly.py:62 `else 3.0` -- fallback local depth (world
# units) when a scene has no usable sparse points at all (e.g. an empty
# points3D.txt in a minimal synthetic test scene).
DOLLY_FALLBACK_DEPTH = 3.0

# --- render/offpath.py : off-path honesty poses ---
# Same construction as ~/Splats/research/visual/render_offpath.py.

# render_offpath.py:139 `--lateral-factor` default.
OFFPATH_DEFAULT_LATERAL_FRAC = 0.5

# render_offpath.py:140 `--elevate-factor` default.
OFFPATH_DEFAULT_ELEVATE_FRAC = 0.7

# render_offpath.py:237 `- fwd0 * depth * 0.35` -- the oblique pose also
# pulls back along the (negative) forward direction, not just up.
OFFPATH_OBLIQUE_BACK_FRAC = 0.35

# render_offpath.py:137 `--width` default (shared with the dolly generator).
OFFPATH_DEFAULT_WIDTH = 640

# --- render/candidate.py : per-checkpoint render artifacts ---

CANDIDATE_FRAMES_DIRNAME = "frames"
CANDIDATE_RAW_FILENAME = "raw_level0.png"
CANDIDATE_NET_FILENAME = "net.png"
CANDIDATE_COVERAGE_FILENAME = "coverage.png"
CANDIDATE_HONESTY_FRAME_FILENAME = "honesty.png"
CANDIDATE_NET_VIDEO_FILENAME = "dolly.mp4"
CANDIDATE_RAW_VIDEO_FILENAME = "dolly_raw.mp4"
CANDIDATE_HONESTY_SHEET_FILENAME = "honesty_sheet.png"
CANDIDATE_METRICS_FILENAME = "metrics.json"

# docs/EXPERIMENTS.md "Mandatory honesty sheet": pixels with coverage <0.3
# are called out as likely-hallucinated; candidate.py outlines them in
# white on the network-output panel (rather than only the coverage panel)
# so a viewer sees exactly which part of the pretty render is inferred.
CANDIDATE_LOW_COVERAGE_THRESHOLD = 0.3
CANDIDATE_OUTLINE_COLOR = (255, 255, 255)

# Matches TRAIN_EVAL_MAX_SHEET_IMAGES's row cap, applied here to the
# honesty_sheet.png contact sheet (3 columns -- raw/net/coverage -- per
# frame instead of that constant's 4).
CANDIDATE_HONESTY_MAX_SHEET_FRAMES = 6

# --- eval/audits.py : Splats' shade audit + extent gate, via subprocess ---
# Both tools live in ~/Splats (read-only) and are run with Splats' own
# ml-sharp venv interpreter, never trippy's own venv (they import Splats-
# local helper modules trippy does not have, e.g. ply_extract.py).

AUDIT_VENV_PYTHON_REL = "tools/ml-sharp/.venv/bin/python"  # relative to SPLATS_ROOT
AUDIT_SHADE_SCRIPT_REL = "tools/depthprior_shade_audit.py"  # relative to SPLATS_ROOT
AUDIT_EXTENT_SCRIPT_REL = "tools/tmp/extent-audit/extent_gate.py"  # relative to SPLATS_ROOT

# Generous: depthprior_shade_audit.py reads a full points3D.txt plus one or
# more (potentially multi-GB) PLYs.
AUDIT_SUBPROCESS_TIMEOUT_S = 600.0

# --- cli.py : `trippy candidate-report` ---

CANDIDATE_REPORT_DOLLY_DIRNAME = "dolly"
CANDIDATE_REPORT_OFFPATH_DIRNAME = "offpath"
CANDIDATE_REPORT_JSON_FILENAME = "report.json"
CANDIDATE_REPORT_README_FILENAME = "README.md"
