"""Stage 5 ablation helpers for the Impromptu-format VLA.

The Stage 4 baseline (ADE 0.61 m) is produced with the full Impromptu prompt:
a single front-camera image plus a textual past-ego-status block of the form::

    Provided are the previous ego vehicle status recorded over the last 1.0
    seconds (at 0.5-second intervals). ... [x, y]:.(t-1.0s) [-9.6, 0.51],
    Acceleration: X 0.44, Y 0.44 m/s^2, Velocity: 9.19 m/s, Steering angle:
    0.36 (positive: left turn, negative: right turn), (t-0.5s) ...

Open-loop nuScenes ADE is known to be heavily ego-status-driven (an ego-only
MLP reaches ~0.35 m with no vision at all). Stage 5 measures how much of our
0.61 m comes from that shortcut versus the front-camera image, by re-running
the *same checkpoint* with controlled corruptions of the input.

This module is intentionally free of ``torch`` / ``PIL`` so the transforms and
analysis can be unit-tested with plain ``python``. The image-side corruptions
(black / mismatched image) only need a boolean flag from :func:`ablation_plan`;
the actual pixel manipulation lives in the eval script.

Ablation rows
-------------
- ``full``           — unchanged baseline (image + full ego status).
- ``no_kinematics``  — keep past positions, drop velocity/acceleration/steering.
- ``no_ego``         — drop the whole past-ego-status block (vision-only).
- ``black_image``    — keep full text, feed an all-zero (black) image
                       (vision-masked + full ego status: the ego-only upper bound).
- ``mismatch_image`` — keep full text, swap in a different sample's image
                       (does the model actually read *this* scene?).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


ABLATIONS = (
    "full",
    "no_kinematics",
    "no_ego",
    "black_image",
    "mismatch_image",
)

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


@dataclass(frozen=True)
class AblationPlan:
    """How a named ablation modifies the two input channels.

    ``text`` is the text transform key; ``image`` is one of
    ``keep`` / ``black`` / ``mismatch``.
    """

    text: str
    image: str


_PLANS = {
    "full": AblationPlan(text="full", image="keep"),
    "no_kinematics": AblationPlan(text="no_kinematics", image="keep"),
    "no_ego": AblationPlan(text="no_ego", image="keep"),
    "black_image": AblationPlan(text="full", image="black"),
    "mismatch_image": AblationPlan(text="full", image="mismatch"),
}


def ablation_plan(name: str) -> AblationPlan:
    """Return the :class:`AblationPlan` for ``name``."""
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
