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
