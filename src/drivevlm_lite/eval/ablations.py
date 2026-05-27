"""Stage 5 ablation helpers for the Impromptu-format VLA.

The Stage 4 baseline (ADE 0.61 m on the biased 500-sample subset) is produced
with the full Impromptu prompt: a single front-camera image plus a textual
past-ego-status block of the form::

    Provided are the previous ego vehicle status recorded over the last 1.0
    seconds (at 0.5-second intervals). ... [x, y]:.(t-1.0s) [-9.6, 0.51],
    Acceleration: X 0.44, Y 0.44 m/s^2, Velocity: 9.19 m/s, Steering angle:
    0.36 (positive: left turn, negative: right turn), (t-0.5s) ...

Open-loop nuScenes ADE is known to be heavily ego-status-driven (an ego-only
MLP reaches ~0.35 m with no vision at all). Stage 5 measures how much of our
ADE comes from that shortcut versus the front-camera image, by re-running the
*same checkpoint* with controlled corruptions of the input.

This module is intentionally free of ``torch`` / ``PIL`` so the transforms and
analysis can be unit-tested with plain ``python``. The image-side corruptions
(black / image swap) only need a plan flag from :func:`ablation_plan`; the
actual pixel manipulation lives in the eval script.

Ablation rows
-------------
- ``full``                — unchanged baseline (image + full ego status).
- ``no_kinematics``       — keep past positions, drop velocity/acceleration/steering.
- ``no_ego``              — drop the whole past-ego-status block (vision-only).
- ``black_image``         — keep full text, feed an all-zero (black) image
                            (vision-masked + full ego status: the ego-only upper bound).
- ``time_shifted_image``  — keep full text, swap in a same-log image
                            within ±0.7 s (probes robustness to small time shifts;
                            the donor is *almost* the right scene, just off by a frame).
- ``true_mismatch_image`` — keep full text, swap in an image from a different
                            scene (different log, or same log ≥5 s away).
                            The clean "does the model condition on scene content?"
                            probe; replaces the prior ``mismatch_image`` row whose
                            ``rows[(idx+1) % n]`` donor was 80 % same-scene +0.5 s.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence


ABLATIONS = (
    "full",
    "no_kinematics",
    "no_ego",
    "black_image",
    "time_shifted_image",
    "true_mismatch_image",
)


# Ablation names that were removed in the v1-closeout fix and now raise a
# helpful error instead of silently working with broken semantics.
_DEPRECATED_ABLATIONS = {
    "mismatch_image": (
        "removed during the 2026-05-27 v1 closeout. The old donor was "
        "rows[(idx+1) % n], which was the same-scene +0.5 s next keyframe "
        "in ~80% of cases. Use 'true_mismatch_image' for the clean "
        "different-scene probe, or 'time_shifted_image' for the small-time-"
        "shift probe."
    ),
}


# The ego-status block always starts with this sentence and runs to the end of
# the user message.
_EGO_BLOCK_RE = re.compile(r"Provided are the previous ego vehicle status.*$", re.DOTALL)

# One per-timestep kinematics chunk, e.g.
#   ", Acceleration: X 0.44, Y 0.44 m/s^2, Velocity: 9.19 m/s, Steering angle:
#    0.36 (positive: left turn, negative: right turn)"
# Leaves the leading "(t-1.0s) [-9.6, 0.51]" position untouched.
_KINEMATICS_RE = re.compile(
    r",\s*Acceleration:.*?\(positive: left turn, negative: right turn\)",
    re.DOTALL,
)

# nuScenes CAM_FRONT keyframe filename, e.g.
# 'n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439262460.jpg'.
# The log identifier is everything before the camera token; the timestamp is
# in integer microseconds.
_CAM_FRONT_RE = re.compile(
    r"^(?P<log>.+?)__CAM_FRONT__(?P<ts>\d+)\.(?:jpe?g|png)$",
    re.IGNORECASE,
)

# Donor-eligibility thresholds (microseconds).
TIME_SHIFT_MAX_DT_US = 700_000      # ±0.7 s — adjacent keyframes are 0.5 s apart.
SCENE_BOUNDARY_DT_US = 5_000_000    # within-log scene transitions exceed this gap.


@dataclass(frozen=True)
class AblationPlan:
    """How a named ablation modifies the two input channels.

    ``text`` is the text transform key; ``image`` is one of
    ``keep`` / ``black`` / ``time_shifted`` / ``true_mismatch``.
    """

    text: str
    image: str


_PLANS = {
    "full":                AblationPlan(text="full",           image="keep"),
    "no_kinematics":       AblationPlan(text="no_kinematics",  image="keep"),
    "no_ego":              AblationPlan(text="no_ego",         image="keep"),
    "black_image":         AblationPlan(text="full",           image="black"),
    "time_shifted_image":  AblationPlan(text="full",           image="time_shifted"),
    "true_mismatch_image": AblationPlan(text="full",           image="true_mismatch"),
}


def ablation_plan(name: str) -> AblationPlan:
    """Return the :class:`AblationPlan` for ``name``.

    Raises ``ValueError`` for unknown names, with a tailored message when the
    name is a known-deprecated alias (e.g. the old ``mismatch_image``).
    """
    if name in _DEPRECATED_ABLATIONS:
        raise ValueError(
            f"ablation {name!r} was {_DEPRECATED_ABLATIONS[name]}"
        )
    try:
        return _PLANS[name]
    except KeyError:
        raise ValueError(
            f"unknown ablation {name!r}; expected one of {', '.join(ABLATIONS)}"
        ) from None


def strip_kinematics(text: str) -> str:
    """Remove velocity / acceleration / steering, keeping only the past
    positions in the ego-status block."""
    return _KINEMATICS_RE.sub("", text)


def strip_ego_status(text: str) -> str:
    """Remove the entire past-ego-status block, leaving the task instruction
    and image placeholder (the vision-only condition).

    Trailing whitespace before the block is trimmed so the prompt does not end
    on a dangling space.
    """
    return _EGO_BLOCK_RE.sub("", text).rstrip()


def transform_user_text(text: str, ablation: str) -> str:
    """Apply the text-side transform for ``ablation`` to a user message."""
    key = ablation_plan(ablation).text
    if key == "full":
        return text
    if key == "no_kinematics":
        return strip_kinematics(text)
    if key == "no_ego":
        return strip_ego_status(text)
    raise ValueError(f"unhandled text transform {key!r}")


def parse_cam_front_path(path: str) -> tuple[str, int]:
    """Extract ``(log_id, timestamp_us)`` from a CAM_FRONT image path.

    Expected basename pattern: ``<log_id>__CAM_FRONT__<microsecond_ts>.<ext>``,
    e.g. ``n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439262460.jpg``.
    Only the basename is examined, so absolute / relative / mixed-separator
    parent paths are all fine. Raises ``ValueError`` for paths that do not
    match the expected pattern.
    """
    basename = PurePosixPath(path.replace("\\", "/")).name
    match = _CAM_FRONT_RE.match(basename)
    if not match:
        raise ValueError(
            f"cannot parse CAM_FRONT path {path!r}: expected basename "
            "'<log>__CAM_FRONT__<microsecond_ts>.<ext>'"
        )
    return match.group("log"), int(match.group("ts"))


@dataclass(frozen=True)
class DonorIndex:
    """Per-row donor indices for the two image-swap ablations.

    For each subset row index ``i``, ``time_shifted[i]`` is the chosen donor
    index for the ``time_shifted_image`` ablation and ``true_mismatch[i]`` is
    the chosen donor for ``true_mismatch_image``. A value of ``-1`` means no
    eligible candidate exists for that row in the supplied subset (e.g. a
    single-row log offers no time-shifted partner). Callers should either
    skip such rows or fall back to the row's own image and record that fact.
    """

    time_shifted: tuple[int, ...]
    true_mismatch: tuple[int, ...]


def build_donor_index(image_paths: Sequence[str], *, seed: int = 42) -> DonorIndex:
    """Precompute per-row donor indices, deterministic for a given seed.

    For each row index ``i`` with image at ``image_paths[i]``:

    - ``time_shifted[i]`` is a uniformly-random index ``j`` chosen from the
      candidates ``{j : j != i, log(j) == log(i),
      |ts(j) - ts(i)| <= TIME_SHIFT_MAX_DT_US}``. In nuScenes this is
      typically the +0.5 s or -0.5 s adjacent keyframe.
    - ``true_mismatch[i]`` is a uniformly-random index ``j`` from a different
      scene than ``i`` — either a different log entirely, or the same log
      but ``|ts(j) - ts(i)| > SCENE_BOUNDARY_DT_US`` (a within-log scene
      transition).

    Returns ``-1`` in place of any index whose candidate set is empty.

    The seed governs the choice; calling twice with the same paths and seed
    yields identical donor maps. Image paths must each be parseable by
    :func:`parse_cam_front_path`.
    """
    parsed: list[tuple[str, int]] = [parse_cam_front_path(p) for p in image_paths]
    n = len(parsed)

    # Group row indices by log.
    by_log: dict[str, list[int]] = {}
    for i, (log, _ts) in enumerate(parsed):
        by_log.setdefault(log, []).append(i)

    # Precompute, per log, the indices that belong to ANY OTHER log. This is
    # the bulk of the true-mismatch candidate set; small same-log-far-away
    # additions are merged in per-row below.
    cross_log_pool: dict[str, list[int]] = {}
    for log in by_log:
        cross_log_pool[log] = [
            j for other_log, idxs in by_log.items() if other_log != log for j in idxs
        ]

    rng = random.Random(seed)
    time_shifted: list[int] = []
    true_mismatch: list[int] = []

    for i, (log_i, ts_i) in enumerate(parsed):
        # Time-shifted: same log, within ±0.7 s, not self.
        ts_cands = [
            j for j in by_log[log_i]
            if j != i and abs(parsed[j][1] - ts_i) <= TIME_SHIFT_MAX_DT_US
        ]
        time_shifted.append(rng.choice(ts_cands) if ts_cands else -1)

        # True-mismatch: different log, OR same log but >5 s away.
        cross = cross_log_pool[log_i]
        far_same = [
            j for j in by_log[log_i]
            if j != i and abs(parsed[j][1] - ts_i) > SCENE_BOUNDARY_DT_US
        ]
        total = len(cross) + len(far_same)
        if total == 0:
            true_mismatch.append(-1)
        else:
            k = rng.randrange(total)
            true_mismatch.append(cross[k] if k < len(cross) else far_same[k - len(cross)])

    return DonorIndex(
        time_shifted=tuple(time_shifted),
        true_mismatch=tuple(true_mismatch),
    )


def classify_maneuver(
    gt_waypoints: Sequence[tuple[float, float]],
    *,
    stop_displacement_m: float = 2.0,
    turn_lateral_m: float = 4.0,
) -> str:
    """Classify a ground-truth 3 s trajectory into a coarse maneuver bucket.

    Uses the final (t=3 s) waypoint, expressed in the current ego frame where
    +x is forward and +y is left:

    - ``stop``     — total displacement magnitude below ``stop_displacement_m``
                     (the ego is essentially stationary over the horizon).
    - ``left``     — final lateral offset above ``turn_lateral_m``.
    - ``right``    — final lateral offset below ``-turn_lateral_m``.
    - ``straight`` — everything else.
    - ``unknown``  — empty trajectory.

    Thresholds are deliberately conservative and configurable; the goal is a
    legible per-maneuver ADE breakdown, not a precise turn detector.
    """
    if not gt_waypoints:
        return "unknown"
    fx, fy = gt_waypoints[-1]
    if math.hypot(fx, fy) < stop_displacement_m:
        return "stop"
    if fy > turn_lateral_m:
        return "left"
    if fy < -turn_lateral_m:
        return "right"
    return "straight"


def percentiles(
    values: Sequence[float],
    points: Sequence[float] = (25.0, 50.0, 75.0, 95.0),
) -> dict[str, float]:
    """Linear-interpolated percentiles of ``values`` (numpy default method).

    Returns a dict keyed ``p25`` / ``p50`` / ... Empty input yields an empty
    dict.
    """
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return {}
    n = len(clean)
    out: dict[str, float] = {}
    for p in points:
        if n == 1:
            out[f"p{int(p)}"] = clean[0]
            continue
        rank = (p / 100.0) * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        frac = rank - lo
        out[f"p{int(p)}"] = clean[lo] + (clean[hi] - clean[lo]) * frac
    return out
