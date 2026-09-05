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

from trippy.constants import MODULO_SPLIT_DEFAULT_K, MODULO_SPLIT_DEFAULT_OFFSET


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


def split_with_forced_heldout(
    names: list[str],
    forced: list[str],
    k: int = MODULO_SPLIT_DEFAULT_K,
    offset: int = MODULO_SPLIT_DEFAULT_OFFSET,
) -> tuple[list[str], list[str]]:
    """Like `modulo_split`, but always holds out every name in `forced`.

    Used to guarantee shade-region frames (trippy.constants.SHADE_FRAMES_KK)
    are always in the held-out set, regardless of where the modulo stride
    would otherwise place them -- so every eval reports a shade-region
    number rather than one that got lucky and landed in train.

    Args:
        names: full image name set (order-independent).
        forced: names that must end up in `heldout`. Entries not present in
            `names` are ignored (so this is safe to call with a fixed
            constant list across scenes that don't have those frames).
        k, offset: passed to `modulo_split` for the remaining names.

    Returns:
        (train_names, heldout_names), both sorted, disjoint, covering all
        of `names`, with every `name in forced` (that is also in `names`)
        in `heldout`.
    """
    forced_set = set(forced) & set(names)
    remaining = [name for name in names if name not in forced_set]
    train, heldout = modulo_split(remaining, k=k, offset=offset)
    heldout = sorted(heldout + list(forced_set))
    return sorted(train), heldout
