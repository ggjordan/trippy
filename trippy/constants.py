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

# How `split_with_forced_heldout` treats the forced (shade) frames -- the two protocols
# answer different questions (docs/EXPERIMENTS.md "Forced hold-out protocols"):
#   "all"       every forced frame is held out. The shade region then has NO photo in the
#               training set at all, so its numbers measure novel view synthesis of an
#               entirely unobserved region.
#   "alternate" every other forced frame (by sorted name, starting at
#               FORCED_HELDOUT_ALTERNATE_OFFSET) is held out and the rest are forced INTO
#               training -- interpolation *inside* an observed region, which is the protocol
#               the Gaussian baseline (trained on every frame) is implicitly measured under.
FORCED_HELDOUT_MODE_ALL = "all"
FORCED_HELDOUT_MODE_ALTERNATE = "alternate"
FORCED_HELDOUT_MODES = (FORCED_HELDOUT_MODE_ALL, FORCED_HELDOUT_MODE_ALTERNATE)
# Index parity (over the sorted forced list) that stays held out in "alternate" mode: 0 holds
# out IMG_3828/3830/3832 and trains IMG_3829/3831/3833.
FORCED_HELDOUT_ALTERNATE_OFFSET = 0

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

# Disk retention policy (task brief 2026-09-06, "disk at 94%, trainer keeps every
# checkpoint"): a 300-epoch run at checkpoint_every=10 writes ~30 epoch files of ~800 MB
# each. `Trainer.save_checkpoint` keeps checkpoint_latest.pt, checkpoint_best.pt, every
# epoch that is a multiple of `checkpoint_keep_every`, and the `checkpoint_keep_last` most
# recent epoch files, deleting the rest (see `trippy.train.retention`). 100 keeps a handful
# of milestone checkpoints across a 300-epoch run (ep0100, ep0200, ep0300) without the full
# ~24 GB/run cost; 1 keeps only the epoch just written (checkpoint_latest.pt is already a
# duplicate of it, so keep_last=1 is not wasted -- it survives independently of whatever
# `--resume` later overwrites `checkpoint_latest.pt` with).
TRAIN_DEFAULT_CHECKPOINT_KEEP_EVERY = 100
TRAIN_DEFAULT_CHECKPOINT_KEEP_LAST = 1

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

# --- test-time photometric calibration at eval (trippy.train.trainer.Trainer.calibrate_frame) ---
# A held-out image's per-image exposure/white balance is NEVER trained (only train frames'
# rows of NeuralCamera.exposures_values get gradients), so a held-out frame renders through
# whatever its EXIF initialisation happened to be. TRIPS has the same problem and ships two
# opt-in answers, both off in configs/train_normalnet.ini:48-49 and in the released horse
# checkpoint (third_party/zenodo/tt_checkpoints/checkpoint_horse/params.ini:53-54):
#   optimize_eval_camera   an extra per-epoch "EvalRefine" gradient pass over the TEST crops
#                          that steps the camera/pose optimisers with texture+network frozen
#                          (src/apps/train.cpp:591-596, 693-697; NeuralScene.cpp:1473-1503).
#   interpolate_eval_settings  copy each test frame's exposure/WB from its two neighbouring
#                          train frames (NeuralCamera.cpp:481-520, train.cpp:1604-1611).
# trippy's `eval_calibrate_camera` is the first of those two, cut down to the photometric
# scalars only: exposure (and optionally white balance), fitted per image by Adam on L1
# against that image's own photo, with the network, the points, the poses and the response
# LUT all frozen, and the fitted value never written back into the module or the checkpoint.
# Default OFF, so every training-time number stays the strict protocol.
# Adam moves an exposure by at most ~lr per step, so steps*lr caps the total EV travel: 200 x
# 0.05 = 10 EV of headroom. That matters because a kk-coherent frame with no EXIF starts 5.87
# EV (58x) away from neutral -- a 60-step budget could not have reached it. The fit is over
# 1-2 scalars on an already-rendered image, so the whole schedule costs far less than the
# render it corrects.
EVAL_CALIBRATE_DEFAULT_STEPS = 200
EVAL_CALIBRATE_DEFAULT_LR = 0.05

# --- eval_exposure_mode: which exposure/WB a HELD-OUT frame renders through at eval
# (trippy.train.trainer.Trainer.evaluate, trippy.net.camera_model.
# interpolate_from_train_neighbours) ---
# The 8.49 dB EXP-0003 shade number was largely this artefact, not a reconstruction failure
# (STATE.md 2026-09-05 correction; docs/EXPERIMENTS.md "Test-time camera calibration"): a
# held-out row of `NeuralCamera.exposures_values`/`white_balance_values` never receives a
# gradient (only sampled TRAIN frames do), so it renders through whatever EXIF/zero
# initialisation it happened to get. TRIPS's own `interpolate_eval_settings`
# (NeuralCamera.cpp:481-520, called from `TestEpoch`, train.cpp:1604-1611) fixes this WITHOUT
# touching the held-out photo: it overwrites each test frame's exposure/WB with a value
# derived only from its TRAINED neighbours' rows -- legitimate because no pixel of the
# held-out image is ever read to produce it.
#   "own"        the frame's own (never-trained) row, unmodified -- the old, only behaviour.
#   "neighbours" (default) interpolated from the nearest TRAINING frames by dataset index
#                (capture order) -- see camera_model.interpolate_from_train_neighbours for the
#                exact rule and how it relates to TRIPS's own iterative version.
#   "calibrate"  the existing `eval_calibrate_camera` per-image Adam fit
#                (`Trainer.calibrate_frame`) promoted to be the row's primary number instead
#                of only a side column -- this is the one mode that DOES use the held-out
#                photo (see calibrate_frame's own docstring for why that is still legitimate
#                but answers a different question).
# `Trainer.evaluate`'s own strict `psnr`/`ssim`/`lpips`/`psnr_mean`/`shade`/`other` fields are
# UNCHANGED by this setting -- always the frame's own raw exposure, exactly as before this
# feature existed (tests/test_train_regression.py, tests/test_train_eval_calibrate.py pin
# that meaning down). This feature instead adds parallel `_eval`-suffixed fields
# (`psnr_eval`, `psnr_mean_eval`, `shade_eval`, `other_eval`, ...) plus a per-image
# `"exposure_mode"` key recording which mode actually produced that row (a training frame's
# row is always "own", regardless of the requested mode -- TRIPS's `InterpolateFromNeighbors`
# is likewise only ever called on `not_training_indices`). Callers that want "the" held-out
# number under this feature should read the `_eval` fields.
EVAL_EXPOSURE_MODE_OWN = "own"
EVAL_EXPOSURE_MODE_NEIGHBOURS = "neighbours"
EVAL_EXPOSURE_MODE_CALIBRATE = "calibrate"
EVAL_EXPOSURE_MODES = (EVAL_EXPOSURE_MODE_OWN, EVAL_EXPOSURE_MODE_NEIGHBOURS, EVAL_EXPOSURE_MODE_CALIBRATE)
EVAL_EXPOSURE_MODE_DEFAULT = EVAL_EXPOSURE_MODE_NEIGHBOURS

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

# --- hybrid/ : Design A (this repo's "hybrid A") -- the Gaussian splat render fed to the
# TRIPS U-Net ALONGSIDE the TRIPS point pyramid, trained end to end (docs/EXPERIMENTS.md
# "Hybrid design A"; trippy.hybrid.gaussian_input, trippy.hybrid.gsrender_live). Design C
# above replaces the point pyramid with the render; design A concatenates the two, so the
# network can keep the Gaussians where they are good and fall back on TRIPS points where
# they fail. ---

# Canonical channel-group order of the Gaussian block appended to every U-Net input level.
# Fixed (not the order the config happens to list) so a checkpoint's channel layout is a
# function of the *set* of groups only -- reordering `hybrid.channels:` in a YAML file can
# never silently invalidate a trained checkpoint.
HYBRID_A_CHANNEL_ORDER = ("rgb", "alpha", "depth")

# Width in channels of each group in HYBRID_A_CHANNEL_ORDER.
HYBRID_A_CHANNEL_WIDTHS = {"rgb": 3, "alpha": 1, "depth": 1}

# The two ways the Gaussian block reaches the U-Net (TrainConfig.hybrid.mode):
#   "all_levels"     -- the block is area-averaged down to every pyramid level's own (h, w)
#                       and concatenated there. The default, because TRIPS's CombineBridge
#                       re-concatenates the *raw* input of every level twice per level
#                       (trippy.net.unet module docstring), so a level-0-only signal is
#                       structurally unavailable to the coarse blocks that decide large-scale
#                       structure -- exactly the decision "trust the Gaussians here" needs.
#   "concat_level0"  -- only level 0 carries the real block; coarser levels get zeros in
#                       those channels (the U-Net requires every level to have the same
#                       channel count, trippy.net.unet.forward's own validation).
HYBRID_A_MODES = ("all_levels", "concat_level0")

# TrainConfig.hybrid.dropout_gaussian_p default: fraction of training crops whose Gaussian
# channels are zeroed wholesale (the TRIPS channels are untouched). A regulariser, not an
# augmentation: at p = 0 the network is free to become a thin residual on top of the Gaussian
# render everywhere, which is precisely the failure design C already measured (EXP-0005:
# +0.45 dB non-shade, -1.96 dB shade). 0.2 keeps a fifth of the steps TRIPS-only so the point
# branch stays a working renderer on its own.
HYBRID_A_DEFAULT_DROPOUT_P = 0.2

# TrainConfig.hybrid.mask_by_alpha default: multiply the Gaussian rgb by its own alpha before
# concatenation, so an uncovered pixel (alpha ~ 0) presents as black-with-zero-alpha rather
# than as gsrender's composited-against-background colour, which carries no scene information
# but does look like content to a conv net.
HYBRID_A_DEFAULT_MASK_BY_ALPHA = True

# Alpha above which a render pixel counts as "covered" when measuring the scene's median
# camera-to-Gaussian depth (the depth-channel normaliser, see HYBRID_A_DEPTH_SCALE_FRAMES).
# gsrender writes depth 0 where nothing was hit, so an unmasked median would mostly measure
# holes.
HYBRID_A_DEPTH_SCALE_ALPHA_MIN = 0.5

# How many rendered frames the median camera-to-Gaussian depth is estimated from when
# `hybrid.depth_scale` is not given explicitly. Evenly spaced over the scene's frame list; a
# median over 12 full 1008-wide depth maps is ~9 M samples, far more than needed for a scale.
HYBRID_A_DEPTH_SCALE_FRAMES = 12

# Fallback depth normaliser when no rendered frame has any covered pixel at all (an empty or
# entirely-missed Gaussian cloud). Reuses design C's coarse constant rather than inventing a
# second magic number; only ever reached on a degenerate scene.
HYBRID_A_DEPTH_SCALE_FALLBACK = HYBRID_C_DEPTH_NORM_SCALE

# Frames held in trippy.hybrid.gaussian_input.RenderCache's LRU. A 1008x756 five-channel
# float32 frame is ~15 MB, so 16 frames is ~240 MB of host RAM -- enough that a training
# epoch over a shuffled 186-image split still re-reads from disk (the point: bounded memory),
# but enough to make an eval pass over the held-out split effectively free.
HYBRID_A_RENDER_CACHE_FRAMES = 16

# EXP-0005's measured held-out PSNR for the RAW Gaussian render (kkc_15000.ply) against the
# photo, on kk-coherent's modulo-8 held-out split: the "plain Gaussians" baseline any hybrid
# must beat. `all` = 33 frames, `shade` = the 6 SHADE_FRAMES_KK frames.
# Source: experiments/EXP-0005-hybrid-c/README.md "Verdict" table.
HYBRID_A_BASELINE_GAUSSIAN_PSNR_ALL = 15.53
HYBRID_A_BASELINE_GAUSSIAN_PSNR_SHADE = 14.94

# EXP-0003 full1-broadcast's held-out PSNR (40 epochs, still rising when it stopped): the
# "plain TRIPS" baseline. Recorded here so the hybrid is judged against both plain Gaussians
# and plain TRIPS, not just against itself. Source: STATE.md / experiments/EXP-0003-*/README.md.
HYBRID_A_BASELINE_TRIPS_PSNR_ALL = 14.42

# --- train/checkpoint_io.py, trainer.py : on-disk run layout ---
# docs/EXPERIMENTS.md "Run location": output/runs/<exp>/<run>/{...}.
TRAIN_CHECKPOINT_DIRNAME = "checkpoints"
TRAIN_CHECKPOINT_FILENAME_FMT = "checkpoint_ep{epoch:04d}.pt"
TRAIN_CHECKPOINT_LATEST_FILENAME = "checkpoint_latest.pt"
# Retention policy additions (trippy.train.retention): the best-held-out-PSNR-so-far
# checkpoint is kept alongside checkpoint_latest.pt regardless of the keep_every/keep_last
# thinning, plus a tiny sidecar recording which epoch/PSNR it is (so `trippy prune-run` and
# a human `cat`-ing the file both know without loading the .pt).
TRAIN_CHECKPOINT_BEST_FILENAME = "checkpoint_best.pt"
TRAIN_CHECKPOINT_BEST_JSON_FILENAME = "best.json"
TRAIN_LOG_FILENAME = "log.txt"
TRAIN_METRICS_FILENAME = "metrics.jsonl"
TRAIN_EVAL_DIRNAME_FMT = "eval_ep{epoch:04d}"
# `trippy eval --checkpoint` (trippy.train.eval.evaluate_checkpoint) writes here instead of
# `TRAIN_EVAL_DIRNAME_FMT` -- a standalone re-evaluation isn't tied to one epoch (it may run long
# after training finished, against a checkpoint whose own epoch dir already exists), and a
# timestamp keeps repeated manual re-evals from clobbering each other's output.
TRAIN_EVAL_MANUAL_DIRNAME_FMT = "eval_manual_{ts}"
TRAIN_EVAL_SHEET_FILENAME = "sheet.png"
# JPEG replaces PNG for the per-epoch eval contact sheet only (task brief 2026-09-06,
# "make eval image dumps lighter"): it is a quick progress-check artifact, not a pixel-exact
# honesty artifact -- those (candidate-report's honesty_sheet.png/raw/net/coverage PNGs,
# trippy.render.candidate) are unaffected and stay PNG. Quality 85 is a standard
# "visually lossless for photographic content, ~4-5x smaller than PNG" choice; not tuned
# per-scene.
TRAIN_EVAL_SHEET_JPEG_FILENAME = "sheet.jpg"
TRAIN_EVAL_SHEET_JPEG_QUALITY = 85
TRAIN_EVAL_METRICS_FILENAME = "metrics.json"
TRAIN_EXPORT_FILENAME = "export.ply"

# `trippy prune-run`'s default "never delete a file this fresh" window: guards a checkpoint
# written by a still-running training job against a concurrent prune pass racing it (task
# brief: "must skip files modified in the last 2 minutes").
PRUNE_RUN_DEFAULT_PROTECT_SECONDS = 120.0
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

# Splats' own depthprior_shade_dolly.py walks t in [-0.35, 1.20] of local depth
# because Gaussians have volume everywhere along that range. TRIPS points do
# not: past the shade volume's far surface there is nothing to render, so the
# raw (level-0, no U-Net) centre coverage collapses towards zero (observed on
# EXP-0003 full1-broadcast: 0.46 at t=-0.35 -> 0.08 at t=+0.51 -> 0.0001 at
# t=+1.20, docs/EXPERIMENTS.md "Dolly camera paths"). `dolly_stop_index`
# (trippy.render.dolly) finds the last frame whose centre coverage is still
# at or above this threshold, so `render_candidate`'s dolly video stops
# before the camera visibly exits the point cloud instead of drifting
# through empty space for the rest of `DOLLY_DEFAULT_T_END`.
DOLLY_COVERAGE_STOP_THRESHOLD = 0.05

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

# `depthprior_shade_audit.py --json-out` field for the darkest bucket its
# `--json-out` payload reports (docs/EXPERIMENTS.md "Shade audit"); used as
# the single dark-mass-fraction number in the train/candidate report's
# baseline-vs-candidate comparison table (trippy.render.report).
SHADE_AUDIT_DARK_MASS_LUM_KEY = "dark_mass_lum0.25"

# `run_shade_audit`/`run_extent_gate` results for a given PLY never change
# unless the file itself changes, so `trippy.eval.audits.cached_baseline_audit`
# keys its cache filename on path + mtime + size and reuses a hit rather than
# re-running Splats' (slow, full points3D.txt + multi-GB PLY) tools on every
# `trippy train --report` for the same unchanged source PLY. Cache lives
# under `$TRIPPY_OUTPUT/<this>/`.
AUDIT_CACHE_SUBDIR = "audits"

# --- cli.py : `trippy candidate-report` ---

CANDIDATE_REPORT_DOLLY_DIRNAME = "dolly"
CANDIDATE_REPORT_OFFPATH_DIRNAME = "offpath"
CANDIDATE_REPORT_JSON_FILENAME = "report.json"
CANDIDATE_REPORT_README_FILENAME = "README.md"

# --- render/report.py : `trippy train --report` self-reporting ---

# Sub-directory of a training run's own `run_dir` that holds the candidate
# report artifacts (export.ply/dolly/offpath/report.json) `train --report`
# builds from the final checkpoint -- mirrors `candidate-report`'s own
# `--out` layout (docs/EXPERIMENTS.md "Candidate report") but nested under
# the run instead of a separate directory the Orchestrator has to remember.
TRAIN_REPORT_DIRNAME = "report"

# Written under `run_dir` (never raised past `trippy.cli._cmd_train`) when
# `--report` is passed and reporting itself throws -- training already
# succeeded and must still exit 0 (this task's brief, requirement 1); this
# file is the only visible trace that the report step did not complete.
TRAIN_REPORT_FAILED_FILENAME = "REPORT_FAILED.txt"

# `scripts/deliver.sh` subprocess timeout, seconds -- generous but bounded
# (it mostly just copies a path reference into Splats' review queue and
# appends one line to research/trips-metal.md, no heavy I/O).
DELIVER_SUBPROCESS_TIMEOUT_S = 120.0

# Sub-directory of a training run's own `run_dir` that `train --report` (and the
# standalone `trippy bundle-launcher` command, for a run-shaped checkpoint) writes the
# free-navigation viewer bundle into (bundle.json/points.npz/weights.safetensors --
# `trippy.render.bundle.export_bundle`), alongside the existing dolly/offpath/report.json
# layout -- Jordan: "fixed dolly paths are hard to judge, I want to navigate freely."
TRAIN_REPORT_BUNDLE_DIRNAME = "bundle"

# Written under `<run_dir>/report/` (never raised past the caller) when
# `scripts/open_mac_viewer.sh` cannot produce a launcher for the exported bundle --
# most commonly because the viewer binary hasn't been built yet. Mirrors
# TRAIN_REPORT_FAILED_FILENAME's "leave a visible trace, never fail the run" pattern,
# scoped to just this one step so a missing binary never masks the rest of an
# otherwise-successful report.
VIEWER_LAUNCHER_FAILED_FILENAME = "VIEWER_LAUNCHER_FAILED.txt"

# Appended (after "; ") to the honest one-line summary when delivering the Mac viewer
# launcher, so the delivery queue entry says what makes this artifact different from
# the dolly video: free navigation, not a fixed path, plus the keys (N/P) to jump
# between the capture's own real cameras (docs/USER_GUIDE.md "How to open the native
# TRIPS viewer").
VIEWER_DELIVERY_WHY_SUFFIX = "open in the free-navigation viewer; N/P step capture views"

# --- distill/ : design-B fallback pipeline (docs/SPEC.md D2, "A plain splat that
# incorporates TRIPS learning (Design B) is a valid fallback path"; Quest honesty
# note: "ship fallback: distilled Gaussians via the existing ~/Splats/tools/publish/
# path"). Distils a trained TRIPS checkpoint into a plain-Gaussian PLY any existing
# 3DGS viewer (Brush, Splats' publish path, Quest) can open unchanged, by (1)
# rendering the TRIPS network output at the training cameras plus a small number of
# near-path interpolated cameras and writing a COLMAP-text image set from those
# renders, then (2) training an ordinary Brush/3DGS model on that image set.

# trippy.distill.cameras.slerp: above this quaternion dot product (near-identical
# rotations), fall back to a normalised linear interpolation to avoid a 0/0 in
# sin(theta)/theta -- same pattern as EPS_QUAT_AXIS/EPS_SE3_ANGLE above, own constant
# since it lives in a different module with its own precision requirements.
DISTILL_SLERP_NEAR_IDENTICAL_DOT = 1.0 - 1e-6

# Number of interpolated cameras generated between each pair of consecutive
# registered training cameras (slerp rotation + lerp camera-centre translation).
# 2 gives a denser camera path than the raw capture without inventing anything far
# off it -- see DISTILL_MAX_JUMP_MULTIPLIER for the honesty guard that skips pairs
# that are not really "consecutive along one continuous walk".
DISTILL_DEFAULT_INTERP_K = 2

# Honesty guard (AGENTS.md "Honesty rule"; task brief "only cameras close to the
# capture path; no far off-path invention"): a consecutive pair of registered
# cameras is only bridged with interpolated poses if their centre-to-centre
# distance is at most this many times the scene's own median consecutive-pair
# distance. A pair further apart than that (two different loops/sweeps of the same
# scene, a registration gap, ...) is not really "consecutive along one continuous
# walk" and interpolating between them would invent a camera path that was never
# walked -- skipped instead, and recorded in trippy.distill.cameras.SkippedPair.
DISTILL_MAX_JUMP_MULTIPLIER = 4.0

# Filename template (no extension -- trippy.distill.cameras.image_filename appends
# ".png") for a synthetic interpolated-camera pose: `{a}`/`{b}` are the two anchor
# image stems and `{j:02d}` the 1-based intermediate index between them.
DISTILL_INTERP_NAME_FMT = "INTERP_{a}_{b}_{j:02d}"

# On-disk layout of one distill run's output directory (trippy.distill.render_set).
DISTILL_IMAGES_DIRNAME = "images"
DISTILL_SPARSE_DIRNAME = "sparse_txt"
DISTILL_RENDERS_DIRNAME = "renders"  # render_candidate's own per-pose frames/ tree
DISTILL_TRIPS_EXPORT_FILENAME = "trips_export.ply"
DISTILL_REPORT_FILENAME = "distill_report.json"
DISTILL_BRUSH_JOB_FILENAME = "brush_train_job.sh"
DISTILL_BRUSH_OUT_DIRNAME = "brush_out"
DISTILL_COMPARE_FILENAME = "compare.md"

# COLMAP camera model the distilled image set is written with: every image
# trippy.distill.render_set produces is already an undistorted pinhole render
# (SceneDataset's own undistortion for anchor poses, or a synthetic interpolated
# pose reusing the same intrinsics convention), so PINHOLE (no distortion
# coefficients) is exact, never an approximation of the scene's real (usually
# OPENCV) capture lenses.
DISTILL_CAMERA_MODEL = "PINHOLE"

# Decimal places intrinsics (fx, fy, cx, cy) are rounded to when grouping
# CameraPoses into COLMAP "cameras" (trippy.distill.colmap_writer._camera_key) --
# far finer than any real difference between two poses meant to share one camera,
# just enough to absorb float roundoff from repeated K-matrix scaling.
DISTILL_CAMERA_KEY_DECIMALS = 6

# Dummy reprojection-error field written into every points3D.txt row (the COLMAP
# text spec includes this column; trippy.geom.xform_a's reader parses but never
# uses it, and Brush's colmap loader doesn't even read it -- see
# trippy/distill/colmap_writer.py). Any finite value works; 0.0 rather than a
# fabricated "confidence" number that could be misread as one.
DISTILL_POINTS3D_DUMMY_ERROR = 0.0

# Points3D.txt row cap: trippy.distill.colmap_writer randomly subsamples (seeded,
# reproducible) the TRIPS export point cloud down to at most this many rows before
# writing points3D.txt. A COLMAP sparse init cloud for a 3DGS trainer only needs to
# seed initial gaussian positions (Brush densifies from there); writing all ~5-7M
# TRIPS points would bloat points3D.txt for no benefit and slow Brush's dataset load.
DISTILL_DEFAULT_MAX_INIT_POINTS = 300_000

# `trippy distill`'s default `--brush-iters`: within the task brief's "5k-8k steps"
# budget for a queue job that already sits behind several prio-70 trainings.
DISTILL_DEFAULT_BRUSH_ITERS = 6000

# Brush eval-split-every / eval-every defaults passed to brush-cli (docs/
# EXPERIMENTS.md's own modulo-8 convention, MODULO_SPLIT_DEFAULT_K): 1 in 8
# rendered frames held out from Brush's own training, evaluated every 1000 steps.
DISTILL_BRUSH_EVAL_SPLIT_EVERY = 8
DISTILL_BRUSH_EVAL_EVERY = 1000

# sh_degree passed to brush-cli/brush: 0 (view-independent colour only). The TRIPS
# network output already bakes tone-mapping/appearance per rendered view, and a
# single-checkpoint distillation gives Brush no multi-view specular signal to
# recover with higher SH orders, so degree 0 is the honest choice, not merely the
# cheap one.
DISTILL_BRUSH_SH_DEGREE = 0

# Default export filename template handed to brush-cli/brush's --export-name
# (its own `{iter}` interpolation, not a Python format field).
DISTILL_BRUSH_EXPORT_NAME = "distilled_{iter}.ply"

# Paths (relative to this repo's rust/ dir) to the two binaries brush_runner.py
# will use, in preference order: the lean headless brush-cli over the full
# brush-app GUI binary (rust/README.md "Building and testing"; both share the same
# `Cli`/`TrainStreamConfig` flags, apps/brush-cli/src/lib.rs).
DISTILL_BRUSH_CLI_BINARY_REL = "brush-trips/target/release/brush-cli"
DISTILL_BRUSH_APP_BINARY_REL = "brush-trips/target/release/brush"

# --- render/leaderboard.py : `trippy leaderboard`, one comparison table across every run ---

# Convention used by the real EXP-0003 full1-broadcast candidate-report invocation
# (docs/EXPERIMENTS.md "Candidate report") -- `trippy candidate-report --out
# <run_dir>/candidate` -- so `trippy leaderboard`'s scan checks exactly this
# relative path (alongside `TRAIN_REPORT_DIRNAME`/report.json for `train --report`
# runs) rather than an unbounded recursive search for any file named report.json.
LEADERBOARD_CANDIDATE_REPORT_DIRNAME = "candidate"

LEADERBOARD_OUT_DIRNAME = "leaderboard"
LEADERBOARD_MARKDOWN_FILENAME = "leaderboard.md"
LEADERBOARD_PNG_FILENAME = "leaderboard.png"

# Delivered artifact name: fixed so `scripts/deliver.sh`'s `ln -sfn` always replaces
# the same symlink (`review_add.sh`) instead of accumulating one per training run --
# Jordan always has exactly one up-to-date "trips-leaderboard" sheet to open.
LEADERBOARD_DELIVER_NAME = "trips-leaderboard"
LEADERBOARD_DELIVER_WHY = (
    "One table of every TRIPS run so far vs the Gaussian baseline: held-out PSNR, "
    "shade dark-mass, extent, coverage. Regenerated after every training."
)

# Row sort: shade dark-mass fraction ascending (lower is better -- closer to/below
# the Gaussian baseline's 19.9%), then held-out PSNR descending within ties/unknowns.
# Rows missing either number (a failed audit, or a baseline with no held-out concept)
# sort to the end of their respective key rather than crashing the sort or silently
# dropping the row.
LEADERBOARD_SORT_MISSING_KEY = float("inf")

# --- Fixed baseline rows (not scanned from $TRIPPY_OUTPUT/runs -- neither is a
# trippy-native training run with its own metrics.jsonl/report.json). Numbers are
# real, already-published results copied verbatim from the cited experiment READMEs
# (AGENTS.md honesty rule: no re-derivation, no rounding beyond the source's own).

# Gaussian point-source baseline (`kkc_15000.ply`, the trained-3DGS PLY every
# EXP-0003/EXP-0005/EXP-0009 point source 1 config reads). PSNR/SSIM/LPIPS are
# EXP-0005's own "Baseline (raw render vs photo)" row (docs/EXPERIMENTS.md hybrid
# design C section; experiments/EXP-0005-hybrid-c/README.md "Verdict" table) --
# the plain Gaussian rendered with Splats' gsrender.py, no TRIPS/U-Net involved.
# Dark-mass fraction and extent are Splats' own shade-audit/extent-gate numbers on
# the same PLY (experiments/EXP-0003-kk-trips-train/README.md "full1-broadcast
# candidate numbers", baseline column; extent p99/max per this task's brief).
LEADERBOARD_BASELINE_GAUSSIAN_NAME = "Gaussians kkc_15000 (baseline PLY)"
LEADERBOARD_BASELINE_GAUSSIAN_PSNR_ALL = 15.53
LEADERBOARD_BASELINE_GAUSSIAN_SSIM_ALL = 0.431
LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_ALL = 0.477
LEADERBOARD_BASELINE_GAUSSIAN_PSNR_SHADE = 14.94
LEADERBOARD_BASELINE_GAUSSIAN_SSIM_SHADE = 0.427
LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_SHADE = 0.526
LEADERBOARD_BASELINE_GAUSSIAN_DARK_MASS = 0.199
LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_P99 = 52.2
LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_MAX = 133.4

# Design C (EXP-0005): render->photo U-Net refinement of the same kkc_15000.ply
# render. Numbers are EXP-0005's own "Refined (U-Net vs photo)" row, final eval
# (epoch 1125). No point cloud/extent of its own to audit (experiments/
# EXP-0005-hybrid-c/README.md: "Design C has no points/extent to audit").
LEADERBOARD_BASELINE_DESIGN_C_NAME = "Design C: render->photo U-Net (EXP-0005)"
LEADERBOARD_BASELINE_DESIGN_C_PSNR_ALL = 15.54
LEADERBOARD_BASELINE_DESIGN_C_SSIM_ALL = 0.476
LEADERBOARD_BASELINE_DESIGN_C_LPIPS_ALL = 0.461
LEADERBOARD_BASELINE_DESIGN_C_PSNR_SHADE = 12.97
LEADERBOARD_BASELINE_DESIGN_C_SSIM_SHADE = 0.442
LEADERBOARD_BASELINE_DESIGN_C_LPIPS_SHADE = 0.519

# --- PNG table rendering (PIL only, no matplotlib -- AGENTS.md "no new dependencies",
# same rule trippy.render.sheets follows). A monospace system font is preferred so
# columns of numbers align visually; PIL's built-in bitmap font is the fallback on a
# machine without one, since `render_table_png` must never raise for a missing font.
LEADERBOARD_PNG_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
)
LEADERBOARD_PNG_FONT_SIZE = 16
LEADERBOARD_PNG_TITLE_FONT_SIZE = 22
LEADERBOARD_PNG_CELL_PAD_X = 12
LEADERBOARD_PNG_CELL_PAD_Y = 8
LEADERBOARD_PNG_MARGIN = 18
LEADERBOARD_PNG_BG = (255, 255, 255)
LEADERBOARD_PNG_HEADER_BG = (32, 38, 58)
LEADERBOARD_PNG_HEADER_FG = (255, 255, 255)
LEADERBOARD_PNG_ROW_BG = (255, 255, 255)
LEADERBOARD_PNG_ROW_BG_ALT = (237, 240, 247)
LEADERBOARD_PNG_BASELINE_ROW_BG = (255, 244, 224)  # tinted so fixed baselines read as "not scanned"
LEADERBOARD_PNG_TEXT_COLOR = (25, 25, 30)
LEADERBOARD_PNG_TITLE_COLOR = (20, 20, 24)
