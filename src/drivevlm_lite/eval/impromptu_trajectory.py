"""Parsing and metrics for Impromptu-format trajectory predictions.

The Impromptu assistant answer looks like::

    <PLANNING>Predicted future movement details ... The output is formatted
    as [x, y]: [2.19, 0.04], [4.34, 0.19], [5.97, 0.43], [8.03, 0.89],
    [10.08, 1.56], [12.04, 2.51]</PLANNING>

There are six (x, y) pairs at 0.5 s spacing, totalling a 3 s horizon. The
first pair after ``[x, y]:`` is the t=0.5 s waypoint; the last is t=3.0 s.

This module provides:

- :func:`parse_planning_text` to pull the six numeric pairs out of a raw
  model response (robust to missing ``</PLANNING>`` tag, extra prose, or
  partial outputs).
- :func:`ade`, :func:`fde`, :func:`split_lateral_longitudinal_ade` for
  computing aggregate metrics over a list of waypoints.

The expected horizon length is six but the parser does not enforce it;
callers can request a minimum length and treat shorter outputs as
parse failures.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# Matches a numeric ``[x, y]`` pair. Letters (the literal ``[x, y]`` format
# hint inside the answer) are intentionally not matched.
_PAIR_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_PLANNING_RE = re.compile(r"<PLANNING>(.*?)(?:</PLANNING>|$)", re.DOTALL)


def parse_planning_text(text: str) -> list[tuple[float, float]]:
    """Return all numeric ``[x, y]`` pairs found inside the ``<PLANNING>``
    block of ``text``. Returns an empty list if no block is found.

    Robust to:
    - text outside the PLANNING tag (ignored),
    - missing closing ``</PLANNING>`` tag,
    - the literal ``[x, y]:`` format hint inside the block (skipped because
      it contains letters, not numbers).
    """
    if not text:
        return []
    match = _PLANNING_RE.search(text)
    body = match.group(1) if match else text
    pairs: list[tuple[float, float]] = []
    for x_str, y_str in _PAIR_RE.findall(body):
        try:
            pairs.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return pairs


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def ade(pred: Sequence[tuple[float, float]], gt: Sequence[tuple[float, float]]) -> float:
    """Average per-step Euclidean displacement between ``pred`` and ``gt``.

    The two sequences are compared up to the shorter length. Raises
    ``ValueError`` if either is empty.
    """
    if not pred or not gt:
        raise ValueError("ade requires non-empty pred and gt sequences.")
    n = min(len(pred), len(gt))
    total = sum(_euclid(pred[i], gt[i]) for i in range(n))
    return total / n


def fde(pred: Sequence[tuple[float, float]], gt: Sequence[tuple[float, float]]) -> float:
    """Final-step Euclidean displacement between ``pred`` and ``gt``.

    Compares the elements at index ``min(len(pred), len(gt)) - 1``.
    """
    if not pred or not gt:
        raise ValueError("fde requires non-empty pred and gt sequences.")
    n = min(len(pred), len(gt)) - 1
    return _euclid(pred[n], gt[n])


def split_lateral_longitudinal_ade(
    pred: Sequence[tuple[float, float]],
    gt: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Return (longitudinal_ADE, lateral_ADE) — the mean absolute error in
    the x (forward) and y (lateral) channels respectively.

    Useful for Stage 5 because lateral error is more safety-relevant than
    longitudinal error for ego trajectory prediction.
    """
    if not pred or not gt:
        raise ValueError("split_lateral_longitudinal_ade requires non-empty inputs.")
    n = min(len(pred), len(gt))
    long_sum = sum(abs(pred[i][0] - gt[i][0]) for i in range(n))
    lat_sum = sum(abs(pred[i][1] - gt[i][1]) for i in range(n))
    return long_sum / n, lat_sum / n
