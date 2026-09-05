"""Tests for trippy.scene.splits: deterministic train/held-out splits.

Module: tests.test_scene_splits
Invariants under test: `modulo_split` is a pure function of the sorted name
    list (order-independent input, deterministic output), and
    `split_with_forced_heldout` always pins the forced names into heldout.
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
