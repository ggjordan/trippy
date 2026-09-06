"""Deterministic train/held-out splits over a scene's image names.

Module: trippy.scene.splits
Invariants: pure functions over `list[str]`, no I/O, no torch/numpy -- the
    split is a function of sorted image names only, so it is reproducible
    across machines/reruns given the same image name set (see
    docs/SPEC.md D10: held-out PSNR/LPIPS is a ranking metric, and it must
    be computed over a stable held-out set run to run).
Related docs: docs/SPEC.md v0.1.0 milestone ("train/held-out split");
    docs/EXPERIMENTS.md's dolly-camera note (IMG_3830.jpg, shade centre);
    trippy.constants.SHADE_FRAMES_KK.
"""

from __future__ import annotations

from trippy.constants import (
    FORCED_HELDOUT_ALTERNATE_OFFSET,
    FORCED_HELDOUT_MODE_ALL,
    FORCED_HELDOUT_MODES,
    MODULO_SPLIT_DEFAULT_K,
    MODULO_SPLIT_DEFAULT_OFFSET,
)


def modulo_split(
    names: list[str],
    k: int = MODULO_SPLIT_DEFAULT_K,
    offset: int = MODULO_SPLIT_DEFAULT_OFFSET,
) -> tuple[list[str], list[str]]:
    """Split `names` into (train, heldout) by index modulo `k` over sorted order.

    Deterministic: `names` is sorted first, so the split does not depend on
    input order. Every `k`-th name (index `i` with `i % k == offset`) is
    held out; the rest train.

    Args:
        names: image names (or any hashable identifiers), order-independent.
        k: held-out stride; 1 in k names is held out.
        offset: which residue class (0 <= offset < k) is held out.

    Returns:
        (train_names, heldout_names), both sorted, disjoint, and covering
        all of `names`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0 <= offset < k):
        raise ValueError(f"offset must be in [0, k), got offset={offset}, k={k}")

    ordered = sorted(names)
    train = [name for i, name in enumerate(ordered) if i % k != offset]
    heldout = [name for i, name in enumerate(ordered) if i % k == offset]
    return train, heldout


def partition_forced(
    forced: list[str],
    mode: str = FORCED_HELDOUT_MODE_ALL,
    alternate_offset: int = FORCED_HELDOUT_ALTERNATE_OFFSET,
) -> tuple[list[str], list[str]]:
    """Split the forced (shade) frames into (held out, forced into training) per `mode`.

    The two modes answer different questions -- see
    `trippy.constants.FORCED_HELDOUT_MODES` and docs/EXPERIMENTS.md
    "Forced hold-out protocols":

      - "all" (the strict protocol used by every EXP-0003 run so far):
        every forced frame is held out, so the shade region has no photo
        in training at all and its PSNR measures novel-view synthesis of
        an unobserved region.
      - "alternate": every other forced frame (sorted by name, keeping the
        ones whose index parity equals `alternate_offset`) is held out and
        the remainder are pushed into training -- interpolation inside an
        observed region.

    Args:
        forced: forced-held-out names (order-independent; sorted here).
        mode: one of `trippy.constants.FORCED_HELDOUT_MODES`.
        alternate_offset: which index parity stays held out in
            "alternate" mode (0 = the first, third, fifth ... name).

    Returns:
        `(heldout_forced, train_forced)`, both sorted, disjoint, covering
        `forced`. `train_forced` is always empty in "all" mode.
    """
    if mode not in FORCED_HELDOUT_MODES:
        raise ValueError(f"mode must be one of {FORCED_HELDOUT_MODES}, got {mode!r}")
    ordered = sorted(set(forced))
    if mode == FORCED_HELDOUT_MODE_ALL:
        return ordered, []
    parity = alternate_offset % 2
    heldout = [n for i, n in enumerate(ordered) if i % 2 == parity]
    train = [n for i, n in enumerate(ordered) if i % 2 != parity]
    return heldout, train


def split_with_forced_heldout(
    names: list[str],
    forced: list[str],
    k: int = MODULO_SPLIT_DEFAULT_K,
    offset: int = MODULO_SPLIT_DEFAULT_OFFSET,
    mode: str = FORCED_HELDOUT_MODE_ALL,
    alternate_offset: int = FORCED_HELDOUT_ALTERNATE_OFFSET,
) -> tuple[list[str], list[str]]:
    """Like `modulo_split`, but always holds out every name in `forced`.

    Used to guarantee shade-region frames (trippy.constants.SHADE_FRAMES_KK)
    are always in the held-out set, regardless of where the modulo stride
    would otherwise place them -- so every eval reports a shade-region
    number rather than one that got lucky and landed in train.

    Args:
        names: full image name set (order-independent).
        forced: names that must end up in `heldout` (in `mode="all"`).
            Entries not present in `names` are ignored (so this is safe to
            call with a fixed constant list across scenes that don't have
            those frames).
        k, offset: passed to `modulo_split` for the remaining names.
        mode, alternate_offset: passed to `partition_forced`. With
            `mode="alternate"` only half the forced frames are held out and
            the other half are forced into `train` (they are *removed* from
            the modulo split, so they can never land in `heldout` by
            accident) -- see `partition_forced` for why.

    Returns:
        (train_names, heldout_names), both sorted, disjoint, covering all
        of `names`, with every `name in forced` (that is also in `names`)
        in `heldout` when `mode="all"`, and every other one of them in
        `heldout` when `mode="alternate"`.
    """
    forced_set = set(forced) & set(names)
    forced_heldout, forced_train = partition_forced(
        sorted(forced_set), mode=mode, alternate_offset=alternate_offset
    )
    remaining = [name for name in names if name not in forced_set]
    train, heldout = modulo_split(remaining, k=k, offset=offset)
    return sorted(train + forced_train), sorted(heldout + forced_heldout)
