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
