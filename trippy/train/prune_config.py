"""`point_removal:` and `shade_prune:` config blocks (TRIPS's rule + trippy's audit-aligned one).

Module: trippy.train.prune_config
Purpose: describe, in two YAML-round-trippable dataclasses, when and how a
    training run drops points from its cloud. Kept in its own module
    (dataclasses + `trippy.constants` only, no numpy/torch) for the same
    reason `trippy.hybrid.config_a` is -- `trippy.train.config` is imported
    by `trippy eval --help` and by every checkpoint load, so it must stay
    cheap. The numpy/torch machinery that consumes these lives in
    `trippy.train.prune`.
Invariants:
    - `PointRemovalConfig` is TRIPS's rule and nothing else: drop every
      point whose *effective* confidence `sigmoid(10 * raw_conf)` is below
      `conf_threshold`, on epochs `start_epoch + i*every_epochs`
      (`src/apps/train.cpp:846-851` and `:533-538`,
      `src/lib/models/NeuralTexture.h:42`). `enabled = False` is the
      default, matching TRIPS's own shipped configs, which disable it by
      putting `start_removing_points_epoch = 2000` beyond
      `num_epochs = 600` (`configs/train_normalnet.ini:8,133`).
    - `ShadePruneConfig` is NOT a TRIPS rule and never claims to be. It is
      an audit-aligned heuristic: it removes points that are inside the
      exact region `~/Splats/tools/depthprior_shade_audit.py` measures, are
      dark by that script's own Rec.709 luminance test, AND are below a
      confidence threshold. Because it removes the thing being measured,
      any run using it must be reported next to its held-out shade PSNR --
      if that PSNR drops, the removed points were carrying real signal.
      See docs/EXPERIMENTS.md "EXP-0010" for the wording this must be
      reported with.
    - `min_points` is a trippy addition with no TRIPS counterpart; see
      `trippy.constants.POINT_REMOVAL_DEFAULT_MIN_POINTS` for why trippy
      needs one (its confidence is initialised from the source PLY's
      opacity, not from TRIPS's uniform 0.9933).
Units: `conf_threshold` and `lum_threshold` are dimensionless in [0, 1];
    `znear_frac`/`zfar_frac` are fractions of a shade frame's own median
    observed sparse-point depth (world units cancel).
Related docs: docs/TRIPS_REFERENCE.md Sec. 2 (confidence parametrisation),
    Sec. 7 (schedule); docs/EXPERIMENTS.md "EXP-0010: point removal";
    docs/ARCHITECTURE.md "train/".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trippy.constants import (
    POINT_REMOVAL_DEFAULT_CONF_THRESHOLD,
    POINT_REMOVAL_DEFAULT_EVERY_EPOCHS,
    POINT_REMOVAL_DEFAULT_MIN_POINTS,
    POINT_REMOVAL_DEFAULT_START_EPOCH,
    SHADE_FRAMES_KK,
    SHADE_PRUNE_DEFAULT_CONF_THRESHOLD,
    SHADE_PRUNE_DEFAULT_LUM_THRESHOLD,
    SHADE_PRUNE_DEFAULT_MIN_POINTS,
    SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
)


@dataclass
class PointRemovalConfig:
    """The `point_removal:` block -- TRIPS's confidence-based point removal, ported.

    Attributes:
        enabled: master switch. False (the default) is a hard no-op: the
            trainer's point count never changes, exactly as before this
            feature existed. TRIPS's own shipped configs are likewise
            effectively disabled (`configs/train_normalnet.ini:133`
            schedules the first removal at epoch 2000 of a 600-epoch run).
        start_epoch: first epoch a removal pass runs
            (`PointAddingParams::start_removing_points_epoch`,
            `src/lib/data/Settings.h:403`; code default 200, and negative
            values disable removal entirely in TRIPS -- here use
            `enabled: false` for that).
        every_epochs: removal repeats on `start_epoch + i*every_epochs`
            (`point_removal_epoch_interval`, `Settings.h:406`; code default
            50). Must be >= 1.
        conf_threshold: a point is removed when
            `sigmoid(10*raw_conf) < conf_threshold`
            (`src/apps/train.cpp:846-851`). TRIPS's code default is 0.3
            (`Settings.h:427`); its shipped config uses 0.500000119
            (`train_normalnet.ini:134`).
        min_points: never let the cloud fall below this many points -- a
            trippy addition (TRIPS has no floor). When a pass would breach
            it, the `min_points` highest-confidence points are kept
            instead, so the outcome stays deterministic.
    """

    enabled: bool = False
    start_epoch: int = POINT_REMOVAL_DEFAULT_START_EPOCH
    every_epochs: int = POINT_REMOVAL_DEFAULT_EVERY_EPOCHS
    conf_threshold: float = POINT_REMOVAL_DEFAULT_CONF_THRESHOLD
    min_points: int = POINT_REMOVAL_DEFAULT_MIN_POINTS

    def __post_init__(self) -> None:
        if self.every_epochs < 1:
            raise ValueError(f"point_removal.every_epochs must be >= 1, got {self.every_epochs}")
        if self.start_epoch < 0:
            raise ValueError(f"point_removal.start_epoch must be >= 0, got {self.start_epoch}")
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError(f"point_removal.conf_threshold must be in [0, 1], got {self.conf_threshold}")
        if self.min_points < 0:
            raise ValueError(f"point_removal.min_points must be >= 0, got {self.min_points}")

    def fires_at(self, epoch: int) -> bool:
        """True when `epoch` is one of `start_epoch + i*every_epochs` (i >= 0) and enabled.

        Mirrors TRIPS's own schedule construction
        (`src/apps/train.cpp:533-538`): the start epoch itself always
        fires, and afterwards every epoch whose offset from the start is a
        multiple of the interval.
        """
        if not self.enabled or epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.every_epochs == 0


@dataclass
class ShadePruneConfig:
    """The `shade_prune:` block -- trippy's audit-aligned heuristic (see module docstring).

    Attributes:
        enabled: master switch for the *pruning*. False (default) is a hard
            no-op on the point cloud. It does NOT switch off `log_dark_mass`
            below: measuring the audit statistic is always safe, pruning on
            it is the loaded act.
        log_dark_mass: compute and log the in-region dark-mass fraction at
            every `Trainer.evaluate`, from the current parameters, so the
            number Jordan's complaint is measured by is visible *during*
            training instead of only after an export + Splats-tool run.
            Defaults True; degrades to an `{"error": ...}` field (never an
            exception) when the region cannot be built -- e.g. a scene whose
            `frames` are not registered, which is every scene except
            kk-coherent.
        frames: the shade frames defining the region; defaults to
            `trippy.constants.SHADE_FRAMES_KK`
            (IMG_3828.jpg..IMG_3833.jpg), which is also
            `depthprior_shade_audit.py`'s own `SHADE_FRAMES` default.
        znear_frac, zfar_frac: the depth slab, as fractions of each frame's
            own median observed sparse-point depth (that script's
            `--znear-frac` / `--zfar-frac` defaults, 0.05 / 0.50).
        lum_threshold: Rec.709 luminance below which a point counts as
            "dark" (the script's `--thresholds` entry the leaderboard
            quotes, 0.25).
        conf_threshold: a point is pruned only if it is in-region AND dark
            AND `sigmoid(10*raw_conf) < conf_threshold`.
        start_epoch, every_epochs: same schedule shape as
            `PointRemovalConfig`, kept independent so a run can prune on a
            different cadence from TRIPS's own rule.
        min_points: floor on the surviving point count (see
            `PointRemovalConfig.min_points`).
        scene_txt: COLMAP model directory the region is built from. Empty
            (the default) means `trippy.scene.dataset.resolve_sparse_dir`
            on the run's own `scene_root`, i.e. the same reconstruction the
            run trains against. Set it to a `sparse_txt` path to pin the
            region to exactly the model `depthprior_shade_audit.py --scene`
            is pointed at.
    """

    enabled: bool = False
    log_dark_mass: bool = True
    frames: list[str] = field(default_factory=lambda: list(SHADE_FRAMES_KK))
    znear_frac: float = SHADE_PRUNE_DEFAULT_ZNEAR_FRAC
    zfar_frac: float = SHADE_PRUNE_DEFAULT_ZFAR_FRAC
    lum_threshold: float = SHADE_PRUNE_DEFAULT_LUM_THRESHOLD
    conf_threshold: float = SHADE_PRUNE_DEFAULT_CONF_THRESHOLD
    start_epoch: int = POINT_REMOVAL_DEFAULT_START_EPOCH
    every_epochs: int = POINT_REMOVAL_DEFAULT_EVERY_EPOCHS
    min_points: int = SHADE_PRUNE_DEFAULT_MIN_POINTS
    scene_txt: str = ""

    def __post_init__(self) -> None:
        if self.every_epochs < 1:
            raise ValueError(f"shade_prune.every_epochs must be >= 1, got {self.every_epochs}")
        if self.start_epoch < 0:
            raise ValueError(f"shade_prune.start_epoch must be >= 0, got {self.start_epoch}")
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError(f"shade_prune.conf_threshold must be in [0, 1], got {self.conf_threshold}")
        if not 0.0 <= self.lum_threshold <= 1.0:
            raise ValueError(f"shade_prune.lum_threshold must be in [0, 1], got {self.lum_threshold}")
        if not 0.0 <= self.znear_frac < self.zfar_frac:
            raise ValueError(
                "shade_prune requires 0 <= znear_frac < zfar_frac, got "
                f"{self.znear_frac} / {self.zfar_frac}"
            )
        if self.min_points < 0:
            raise ValueError(f"shade_prune.min_points must be >= 0, got {self.min_points}")
        if self.enabled and not self.frames:
            raise ValueError("shade_prune.enabled needs at least one frame in shade_prune.frames")

    def fires_at(self, epoch: int) -> bool:
        """True when `epoch` is one of `start_epoch + i*every_epochs` (i >= 0) and enabled."""
        if not self.enabled or epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.every_epochs == 0
