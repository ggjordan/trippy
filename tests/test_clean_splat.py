"""Tests for trippy.clean: Design B, deleting fog Gaussians from a splat.

Module: tests.test_clean_splat
Invariants under test:
  - `trippy.clean.mapping.recover_mapping` rediscovers the TRIPS point ->
    PLY row correspondence from the opacity filter alone, and falls back to
    a nearest neighbour (and says so) when training removed points; both
    routes are verified against the `init_conf` fingerprint, and a mapping
    that does not reproduce it is refused rather than returned.
  - `trippy.clean.ply_filter.filter_ply` copies a survivor's row BYTES
    unchanged, including properties trippy itself never interprets, and
    rewrites only the header's vertex count.
  - The end-to-end CLI deletes exactly the planted fog Gaussians and
    nothing else -- the acceptance criterion for this feature.
  - `trippy.clean.freespace` calls a point in front of a surface "front"
    and a point on it "on", on a scene whose geometry is known by
    construction.
  - `trippy.clean.layers.export_splat_layers` (ADR-0008-supersplat.md Stage 1
    task 3) partitions the source PLY into `-keep.ply` / `-fog.ply` such that
    every planted fog row lands in `-fog.ply`, every other row lands in
    `-keep.ply` (byte for byte, same as `filter_ply`), and the two counts
    sum to the source row count.
Fixture: a synthetic Gaussian set with planted low-confidence "fog" points
    plus a hand-built `point_params` checkpoint (AGENTS.md: test fixtures
    must be synthetic; nothing here reads a scene, a photo or a real PLY).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trippy.clean.freespace import classify_against_surface, first_hit_depth
from trippy.clean.layers import export_splat_layers
from trippy.clean.mapping import recover_mapping
from trippy.clean.ply_filter import filter_ply, read_columns, read_ply_layout, rewrite_header
from trippy.clean.run import clean_splat, load_frames
from trippy.clean.score import load_point_scores
from trippy.clean.select import VARIANTS, deletion_mask, ply_keep_mask, variant_by_name
from trippy.cli import main as cli_main
from trippy.constants import CONF_SIGMOID_SCALE, EXPORT_OPACITY_CLAMP_EPS
from trippy.train.prune import ShadeView

# The synthetic splat: N_SURFACE Gaussians on a plane, N_FOG floating in
# front of it, plus N_WEAK Gaussians whose opacity is below the source's
# min_opacity so they never became TRIPS points at all.
N_SURFACE = 300
N_FOG = 60
N_WEAK = 40
MIN_OPACITY = 0.05
# One extra vertex property trippy never interprets, to prove the filter
# copies whatever the file happens to carry.
EXTRA_PROP = "vendor_tag"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EXPORT_OPACITY_CLAMP_EPS, 1.0 - EXPORT_OPACITY_CLAMP_EPS)
    return np.log(p / (1.0 - p))


def _write_ply(path: Path, xyz: np.ndarray, opacity_logit: np.ndarray, tag: np.ndarray) -> None:
    """Write a small binary_little_endian PLY carrying one extra property."""
    props = [
        ("x", "float"), ("y", "float"), ("z", "float"),
        ("f_dc_0", "float"), ("f_dc_1", "float"), ("f_dc_2", "float"),
        ("opacity", "float"),
        ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
        ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float"),
        (EXTRA_PROP, "int"),
    ]  # fmt: skip
    dtype = np.dtype([(n, "<i4" if t == "int" else "<f4") for n, t in props])
    rows = np.zeros(xyz.shape[0], dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rows["opacity"] = opacity_logit
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = -3.0
    rows["rot_0"] = 1.0
    rows[EXTRA_PROP] = tag
    header = ["ply", "format binary_little_endian 1.0", "comment synthetic fixture", f"element vertex {xyz.shape[0]}"]
    header += [f"property {t} {n}" for n, t in props]
    header += ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode())
        rows.tofile(f)


@pytest.fixture
def synthetic_splat(tmp_path: Path) -> dict:
    """A splat + a checkpoint whose confidences condemn exactly the planted fog.

    Layout (all in COLMAP world units): the surface is the plane z = 4,
    the fog floats at z = 1.5 (i.e. between a camera at the origin and
    that surface), and the weak Gaussians sit on the surface but below
    `MIN_OPACITY` so `GaussianPlySource` would never have seeded them.
    Rows are INTERLEAVED, so a mapping that merely preserved order without
    applying the opacity filter would fail every assertion below.
    """
    rng = np.random.default_rng(7)
    n = N_SURFACE + N_FOG + N_WEAK
    kind = np.array([0] * N_SURFACE + [1] * N_FOG + [2] * N_WEAK)
    kind = rng.permutation(kind)

    xyz = np.zeros((n, 3), dtype=np.float32)
    # The plane is wide enough to fill `_view()`'s frustum at z = 4 (half-width
    # 4.0 at focal 32 over a 64 px image), so every fog point below has a
    # surface BEHIND it and the free-space test is actually exercised.
    xyz[:, :2] = rng.uniform(-4.0, 4.0, size=(n, 2))
    xyz[kind == 1, :2] = rng.uniform(-1.0, 1.0, size=(int((kind == 1).sum()), 2))
    xyz[kind == 0, 2] = 4.0
    xyz[kind == 1, 2] = 1.5
    xyz[kind == 2, 2] = 4.0

    seed_conf = np.where(kind == 2, rng.uniform(0.001, 0.04, n), rng.uniform(0.2, 0.9, n))
    opacity_logit = _logit(seed_conf).astype(np.float32)
    tag = np.arange(n, dtype=np.int32) * 3 + 11

    ply = tmp_path / "synthetic.ply"
    _write_ply(ply, xyz, opacity_logit, tag)

    # TRIPS saw only the rows the opacity filter kept, in file order.
    keep_rows = np.flatnonzero(1.0 / (1.0 + np.exp(-opacity_logit.astype(np.float64))) >= MIN_OPACITY)
    init_conf = (1.0 / (1.0 + np.exp(-opacity_logit[keep_rows].astype(np.float64)))).astype(np.float32)
    is_fog = kind[keep_rows] == 1
    # Training drove the fog's confidence to ~0.01 and everything else to ~0.8.
    conf = np.where(is_fog, 0.01, 0.8).astype(np.float32)
    raw_conf = (np.log(conf / (1.0 - conf)) / CONF_SIGMOID_SCALE).astype(np.float32)
    point_xyz = xyz[keep_rows] + rng.normal(0.0, 0.002, size=(keep_rows.size, 3)).astype(np.float32)

    ckpt = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 42,
            "points_removed_total": 0,
            "cfg": {"point_source": {"type": "gaussian", "path": str(ply), "min_opacity": MIN_OPACITY}},
            "point_params": {
                "xyz": torch.from_numpy(point_xyz),
                "raw_size": torch.zeros(keep_rows.size),
                "raw_conf": torch.from_numpy(raw_conf),
                "feat": torch.full((keep_rows.size, 4), 0.1),
                "provenance": torch.zeros(keep_rows.size, dtype=torch.uint8),
                "init_conf": torch.from_numpy(init_conf),
                "bbox_min": torch.zeros(3),
                "bbox_max": torch.ones(3),
            },
        },
        ckpt,
    )
    return {
        "ply": ply,
        "checkpoint": ckpt,
        "kind": kind,
        "xyz": xyz,
        "keep_rows": keep_rows,
        "fog_rows": np.flatnonzero(kind == 1),
        "tag": tag,
        "tmp_path": tmp_path,
    }


def test_the_opacity_filter_recovers_the_mapping_and_proves_it(synthetic_splat):
    """The deterministic route finds the right rows and scores a perfect fingerprint."""
    scores = load_point_scores(synthetic_splat["checkpoint"])
    layout = read_ply_layout(synthetic_splat["ply"])
    mapping = recover_mapping(layout, scores.init_conf, scores.xyz, min_opacity=MIN_OPACITY)

    assert mapping.method == "opacity-filter"
    assert np.array_equal(mapping.ply_row, synthetic_splat["keep_rows"])
    assert mapping.evidence["fingerprint_max_abs_err"] == 0.0
    assert mapping.evidence["fingerprint_exact_frac"] == 1.0
    assert mapping.evidence["n_ply"] == N_SURFACE + N_FOG + N_WEAK
    assert mapping.evidence["n_opacity_keep"] == N_SURFACE + N_FOG


def test_a_wrong_min_opacity_is_refused_not_silently_accepted(synthetic_splat):
    """A mapping whose fingerprint disagrees must raise, never be returned."""
    scores = load_point_scores(synthetic_splat["checkpoint"])
    layout = read_ply_layout(synthetic_splat["ply"])
    with pytest.raises(ValueError, match="does not reproduce the checkpoint's init_conf"):
        recover_mapping(layout, scores.init_conf, scores.xyz, min_opacity=0.5)


def test_removed_points_fall_back_to_nearest_neighbour_on_seed_positions(synthetic_splat):
    """Point removal breaks the 1:1 count; the geometric fallback repairs it and says so."""
    scores = load_point_scores(synthetic_splat["checkpoint"])
    layout = read_ply_layout(synthetic_splat["ply"])
    survivors = np.arange(scores.n)[::2]  # pretend training dropped every other point

    mapping = recover_mapping(
        layout,
        scores.init_conf[survivors],
        scores.xyz[survivors],
        min_opacity=MIN_OPACITY,
    )
    assert mapping.method == "nearest-neighbour"
    assert np.array_equal(mapping.ply_row, synthetic_splat["keep_rows"][survivors])
    assert mapping.evidence["nn_duplicate_targets"] == 0
    assert mapping.evidence["fingerprint_max_abs_err"] == 0.0


def test_the_filter_copies_survivor_rows_byte_for_byte(synthetic_splat):
    """Every property survives, including one trippy has never heard of."""
    layout = read_ply_layout(synthetic_splat["ply"])
    keep = np.ones(layout.count, dtype=bool)
    keep[synthetic_splat["fog_rows"]] = False
    dest = synthetic_splat["tmp_path"] / "filtered.ply"

    n_written = filter_ply(layout, keep, dest, chunk_rows=37)
    assert n_written == int(keep.sum())

    out = read_ply_layout(dest)
    assert out.count == n_written
    assert out.props == layout.props
    src_rows = np.fromfile(synthetic_splat["ply"], dtype=layout.dtype, offset=layout.data_offset)
    dst_rows = np.fromfile(dest, dtype=out.dtype, offset=out.data_offset)
    assert np.array_equal(dst_rows, src_rows[keep])
    assert np.array_equal(read_columns(out, [EXTRA_PROP])[EXTRA_PROP], synthetic_splat["tag"][keep])


def test_rewrite_header_touches_only_the_vertex_count():
    header = b"ply\nformat binary_little_endian 1.0\ncomment keep me\nelement vertex 99\nproperty float x\nend_header\n"
    out = rewrite_header(header, 7)
    assert b"element vertex 7" in out
    assert b"comment keep me" in out
    assert out.count(b"\n") == header.count(b"\n")


def test_a_header_without_a_vertex_element_raises():
    with pytest.raises(ValueError, match="no 'element vertex"):
        rewrite_header(b"ply\nend_header\n", 1)


def test_deletion_never_touches_a_gaussian_trips_never_saw(synthetic_splat):
    """The below-min_opacity Gaussians have no TRIPS twin, so they always survive."""
    scores = load_point_scores(synthetic_splat["checkpoint"])
    layout = read_ply_layout(synthetic_splat["ply"])
    mapping = recover_mapping(layout, scores.init_conf, scores.xyz, min_opacity=MIN_OPACITY)
    keep = ply_keep_mask(deletion_mask(scores.conf, 0.99), mapping.ply_row, mapping.n_ply)
    weak = synthetic_splat["kind"] == 2
    assert keep[weak].all()
    assert not keep[~weak].any()


def test_the_cli_deletes_exactly_the_planted_fog(synthetic_splat, capsys):
    """End-to-end acceptance: `trippy splat-clean` removes the fog rows and only those."""
    out_dir = synthetic_splat["tmp_path"] / "clean"
    rc = cli_main(
        [
            "splat-clean",
            "--checkpoint", str(synthetic_splat["checkpoint"]),
            "--ply", str(synthetic_splat["ply"]),
            "--out", str(out_dir),
            "--variant", "005",
            "--no-audit",
            "--no-freespace",
        ]
    )  # fmt: skip
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    entry = summary["variants"]["005"]
    assert entry["n_deleted"] == N_FOG
    assert entry["n_kept"] == N_SURFACE + N_WEAK
    assert summary["mapping"]["method"] == "opacity-filter"
    assert summary["points_removed_total"] == 0

    out = read_ply_layout(entry["ply"])
    surviving_tags = read_columns(out, [EXTRA_PROP])[EXTRA_PROP]
    expected = synthetic_splat["tag"][synthetic_splat["kind"] != 1]
    assert np.array_equal(np.sort(surviving_tags), np.sort(expected))

    heat = out_dir / "kklid-tripsclean-005-deleted-density.png"
    assert heat.exists() and heat.stat().st_size > 0
    assert "deleted 60" in capsys.readouterr().out


def test_the_shade_variant_only_deletes_inside_the_region(synthetic_splat):
    """`shade` leaves fog outside the region alone; `005` does not."""
    scores = load_point_scores(synthetic_splat["checkpoint"])
    inside = np.zeros(scores.n, dtype=bool)
    inside[: scores.n // 2] = True
    fog = scores.conf < 0.05

    everywhere = deletion_mask(scores.conf, 0.05)
    shade_only = deletion_mask(scores.conf, 0.05, inside)
    assert everywhere.sum() == fog.sum()
    assert shade_only.sum() == int((fog & inside).sum()) < everywhere.sum()
    assert not shade_only[~inside].any()


def _view(distance_scale: float = 1.0) -> ShadeView:
    """A camera at the origin looking down +z, 64x64, focal 32."""
    return ShadeView(
        name="synthetic",
        R=np.eye(3),
        C=np.zeros(3),
        d=4.0 * distance_scale,
        fx=32.0,
        fy=32.0,
        cx=32.0,
        cy=32.0,
        width=64,
        height=64,
        nobs=10,
        znear=0.2,
        zfar=2.0,
    )


def test_first_hit_depth_takes_the_nearest_point_per_pixel():
    depth = first_hit_depth(np.array([2, 2, 5]), np.array([9.0, 3.0, 7.0]), 8)
    assert depth[2] == 3.0
    assert depth[5] == 7.0
    assert np.isinf(depth[0])


def test_freespace_calls_fog_front_and_surface_points_on(synthetic_splat):
    """A low-confidence point at z=1.5 in front of a confident plane at z=4 is `front`.

    Run at the shipped `scale` (8), i.e. an 8x8 buffer for this 64x64 view:
    300 surface points fill every cell of it, whereas at full resolution
    they would be 300 points scattered over 4096 pixels and most fog points
    would land on a pixel that never saw a surface at all.
    """
    scores = load_point_scores(synthetic_splat["checkpoint"])
    fog = scores.conf < 0.05
    stats = classify_against_surface([_view()], scores.xyz, scores.conf, fog, scale=8)
    assert stats["n_candidates"] == N_FOG
    assert stats["front"] == stats["n_seen"] == N_FOG
    assert stats["on"] == stats["behind"] == 0
    # Deleting the fog punches no hole -- the plane is still behind it --
    # and every pixel the fog covered now sees further, onto that plane.
    assert stats["pixels_emptied"] == 0
    assert stats["pixels_revealed"] > 0
    assert stats["pixels_revealed_confident"] == stats["pixels_revealed"]

    # The opposite case, and the failure this check exists to catch:
    # condemning the SURFACE itself is reported as `on`, and it empties
    # pixels that nothing else covers.
    surface = ~fog
    bad = classify_against_surface([_view()], scores.xyz, scores.conf, surface, scale=8)
    assert bad["on"] > bad["front"]
    assert bad["pixels_emptied"] > 0


def test_load_frames_reads_a_list_a_named_dict_and_refuses_neither(tmp_path: Path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(["a.jpg", "b.jpg"]))
    assert load_frames(None, plain, "big_tree") == ["a.jpg", "b.jpg"]

    named = tmp_path / "named.json"
    named.write_text(json.dumps({"big_tree": ["c.jpg"], "kkc": ["d.jpg"]}))
    assert load_frames(None, named, "big_tree") == ["c.jpg"]
    assert load_frames(None, named, "kkc") == ["d.jpg"]
    assert load_frames(["explicit.jpg"], named, "big_tree") == ["explicit.jpg"]

    with pytest.raises(ValueError, match="no key"):
        load_frames(None, named, "nope")
    with pytest.raises(ValueError, match="need --frames"):
        load_frames(None, None, "big_tree")


def test_a_shade_variant_without_a_region_is_an_error_not_a_scene_wide_delete(synthetic_splat):
    with pytest.raises(ValueError, match="needs --scene"):
        clean_splat(
            checkpoint=synthetic_splat["checkpoint"],
            ply=synthetic_splat["ply"],
            out_dir=synthetic_splat["tmp_path"] / "nope",
            variants=[variant_by_name("shade")],
            freespace=False,
            audit=False,
            log=lambda *_: None,
        )


def test_variant_names_and_thresholds_are_the_three_shipped_levels():
    assert [v.name for v in VARIANTS] == ["shade", "005", "015"]
    assert [v.conf_threshold for v in VARIANTS] == [0.05, 0.05, 0.15]
    assert [v.shade_only for v in VARIANTS] == [True, False, False]
    with pytest.raises(KeyError, match="unknown clean variant"):
        variant_by_name("nope")


def test_export_splat_layers_partitions_keep_and_fog_byte_for_byte(synthetic_splat):
    """ADR-0008-supersplat.md Stage 1 task 3: fog rows land in -fog.ply, everything
    else in -keep.ply, every survivor byte-identical to its source row, and the two
    files' row counts sum to the source PLY's."""
    out_dir = synthetic_splat["tmp_path"] / "layers"
    result = export_splat_layers(
        checkpoint=synthetic_splat["checkpoint"],
        ply=synthetic_splat["ply"],
        out_dir=out_dir,
        variant=variant_by_name("005"),
        name="synthetic",
    )

    assert result["variant"] == "005"
    assert result["n_ply"] == N_SURFACE + N_FOG + N_WEAK
    assert result["n_fog"] == N_FOG
    assert result["n_kept"] == N_SURFACE + N_WEAK
    assert result["n_kept"] + result["n_fog"] == result["n_ply"]
    assert result["mapping"]["method"] == "opacity-filter"

    keep_path = Path(result["keep_ply"])
    fog_path = Path(result["fog_ply"])
    layers_txt = Path(result["layers_txt"])
    assert keep_path.name == "synthetic-005-keep.ply"
    assert fog_path.name == "synthetic-005-fog.ply"
    assert layers_txt.exists()

    keep = read_ply_layout(keep_path)
    fog = read_ply_layout(fog_path)
    assert keep.count == N_SURFACE + N_WEAK
    assert fog.count == N_FOG

    src_layout = read_ply_layout(synthetic_splat["ply"])
    src_rows = np.fromfile(synthetic_splat["ply"], dtype=src_layout.dtype, offset=src_layout.data_offset)
    is_fog = synthetic_splat["kind"] == 1

    keep_rows = np.fromfile(keep_path, dtype=keep.dtype, offset=keep.data_offset)
    fog_rows = np.fromfile(fog_path, dtype=fog.dtype, offset=fog.data_offset)
    assert np.array_equal(keep_rows, src_rows[~is_fog])
    assert np.array_equal(fog_rows, src_rows[is_fog])
    # Byte equality on the one property trippy never interprets, proving the
    # partition copies whole rows verbatim, not a recomputation.
    assert np.array_equal(np.sort(keep_rows[EXTRA_PROP]), np.sort(synthetic_splat["tag"][~is_fog]))
    assert np.array_equal(np.sort(fog_rows[EXTRA_PROP]), np.sort(synthetic_splat["tag"][is_fog]))

    manifest = layers_txt.read_text()
    assert "005" in manifest
    assert "synthetic-005-keep.ply" in manifest
    assert "synthetic-005-fog.ply" in manifest


def test_export_splat_layers_shade_variant_needs_a_region(synthetic_splat):
    with pytest.raises(ValueError, match="needs --scene"):
        export_splat_layers(
            checkpoint=synthetic_splat["checkpoint"],
            ply=synthetic_splat["ply"],
            out_dir=synthetic_splat["tmp_path"] / "nope",
            variant=variant_by_name("shade"),
        )


def test_export_splat_layers_default_name_is_the_ply_stem(synthetic_splat):
    result = export_splat_layers(
        checkpoint=synthetic_splat["checkpoint"],
        ply=synthetic_splat["ply"],
        out_dir=synthetic_splat["tmp_path"] / "layers2",
        variant=variant_by_name("015"),
    )
    assert Path(result["keep_ply"]).name == "synthetic-015-keep.ply"
    assert Path(result["fog_ply"]).name == "synthetic-015-fog.ply"


def test_the_export_splat_layers_cli(synthetic_splat, capsys):
    """End-to-end acceptance: `trippy export-splat-layers` writes both files and reports counts."""
    out_dir = synthetic_splat["tmp_path"] / "cli-layers"
    rc = cli_main(
        [
            "export-splat-layers",
            "--checkpoint", str(synthetic_splat["checkpoint"]),
            "--ply", str(synthetic_splat["ply"]),
            "--out", str(out_dir),
            "--variant", "005",
            "--name", "kklid_test",
        ]
    )  # fmt: skip
    assert rc == 0

    keep = out_dir / "kklid_test-005-keep.ply"
    fog = out_dir / "kklid_test-005-fog.ply"
    assert keep.exists() and fog.exists()
    assert (out_dir / "kklid_test-005-layers.txt").exists()

    out = capsys.readouterr().out
    assert f"kept {N_SURFACE + N_WEAK:,}" in out
    assert f"fog {N_FOG:,}" in out
