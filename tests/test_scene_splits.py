"""Tests for trippy.scene.splits: deterministic train/held-out splits.

Module: tests.test_scene_splits
Invariants under test: `modulo_split` is a pure function of the sorted name
    list (order-independent input, deterministic output);
    `split_with_forced_heldout` pins the forced names into heldout in the
    default "all" mode, and in "alternate" mode holds out every other
    forced name while forcing the rest INTO train (they can never leak back
    into heldout via the modulo stride).
Related docs: docs/SPEC.md v0.1.0 milestone ("train/held-out split");
    trippy.constants.SHADE_FRAMES_KK.
"""

from __future__ import annotations

import pytest

from trippy.constants import SHADE_FRAMES_KK
from trippy.scene import splits


def test_modulo_split_deterministic_and_covers_all() -> None:
    names = [f"IMG_{i:04d}.jpg" for i in range(23)]
    train, heldout = splits.modulo_split(names, k=8, offset=0)

    assert sorted(train + heldout) == sorted(names)
    assert set(train).isdisjoint(heldout)
    # every 8th name (indices 0, 8, 16 of the sorted list) is held out.
    ordered = sorted(names)
    expected_heldout = [ordered[i] for i in range(0, len(ordered), 8)]
    assert heldout == expected_heldout


def test_modulo_split_order_independent() -> None:
    names = [f"IMG_{i:04d}.jpg" for i in range(17)]
    shuffled = list(reversed(names))
    train_a, heldout_a = splits.modulo_split(names, k=5, offset=1)
    train_b, heldout_b = splits.modulo_split(shuffled, k=5, offset=1)
    assert train_a == train_b
    assert heldout_a == heldout_b


def test_modulo_split_offset_selects_different_residue() -> None:
    names = [f"IMG_{i:04d}.jpg" for i in range(10)]
    _, heldout_0 = splits.modulo_split(names, k=5, offset=0)
    _, heldout_1 = splits.modulo_split(names, k=5, offset=1)
    assert heldout_0 != heldout_1
    assert set(heldout_0).isdisjoint(heldout_1)


def test_modulo_split_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        splits.modulo_split(["a"], k=0)
    with pytest.raises(ValueError):
        splits.modulo_split(["a"], k=4, offset=4)


def test_split_with_forced_heldout_pins_shade_frames() -> None:
    names = [f"IMG_{i:04d}.jpg" for i in range(3800, 3840)]
    forced = SHADE_FRAMES_KK
    train, heldout = splits.split_with_forced_heldout(names, forced, k=8, offset=0)

    assert sorted(train + heldout) == sorted(names)
    assert set(train).isdisjoint(heldout)
    for name in forced:
        assert name in heldout
        assert name not in train


def test_split_with_forced_heldout_ignores_absent_names() -> None:
    names = ["a.jpg", "b.jpg", "c.jpg"]
    forced = ["not_present.jpg"]
    train, heldout = splits.split_with_forced_heldout(names, forced, k=2, offset=0)
    assert sorted(train + heldout) == sorted(names)
    assert "not_present.jpg" not in train and "not_present.jpg" not in heldout


# --- forced hold-out protocols ("all" vs "alternate", trippy.constants.FORCED_HELDOUT_MODES) ---


def test_partition_forced_all_holds_every_forced_frame_out() -> None:
    heldout, train = splits.partition_forced(SHADE_FRAMES_KK, mode="all")
    assert heldout == sorted(SHADE_FRAMES_KK)
    assert train == []


def test_partition_forced_alternate_splits_the_shade_frames_in_half() -> None:
    heldout, train = splits.partition_forced(SHADE_FRAMES_KK, mode="alternate")
    assert heldout == ["IMG_3828.jpg", "IMG_3830.jpg", "IMG_3832.jpg"]
    assert train == ["IMG_3829.jpg", "IMG_3831.jpg", "IMG_3833.jpg"]
    assert sorted(heldout + train) == sorted(SHADE_FRAMES_KK)


def test_partition_forced_alternate_offset_flips_the_parity() -> None:
    heldout, train = splits.partition_forced(SHADE_FRAMES_KK, mode="alternate", alternate_offset=1)
    assert heldout == ["IMG_3829.jpg", "IMG_3831.jpg", "IMG_3833.jpg"]
    assert train == ["IMG_3828.jpg", "IMG_3830.jpg", "IMG_3832.jpg"]


def test_partition_forced_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError):
        splits.partition_forced(SHADE_FRAMES_KK, mode="every-other-one")


def test_split_with_forced_heldout_alternate_trains_half_the_shade_frames() -> None:
    names = [f"IMG_{i}.jpg" for i in range(3800, 3860)]
    train, heldout = splits.split_with_forced_heldout(names, SHADE_FRAMES_KK, k=8, mode="alternate")

    assert sorted(train + heldout) == sorted(names)
    assert set(train).isdisjoint(heldout)
    assert set(heldout) & set(SHADE_FRAMES_KK) == {"IMG_3828.jpg", "IMG_3830.jpg", "IMG_3832.jpg"}
    # The trained shade frames are pulled OUT of the modulo stride, so none of them can be
    # held out by accident -- the point of the protocol is that the shade region is observed.
    assert {"IMG_3829.jpg", "IMG_3831.jpg", "IMG_3833.jpg"} <= set(train)


def test_split_with_forced_heldout_defaults_to_the_all_protocol() -> None:
    names = [f"IMG_{i}.jpg" for i in range(3800, 3860)]
    default_split = splits.split_with_forced_heldout(names, SHADE_FRAMES_KK, k=8)
    explicit_split = splits.split_with_forced_heldout(names, SHADE_FRAMES_KK, k=8, mode="all")
    assert default_split == explicit_split
    assert set(SHADE_FRAMES_KK) <= set(default_split[1])
