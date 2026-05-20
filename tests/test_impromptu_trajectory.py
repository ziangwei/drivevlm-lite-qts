"""Tests for the Impromptu trajectory parser and metrics.

Run without pytest::

    PYTHONPATH=src python tests/test_impromptu_trajectory.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.impromptu_trajectory import (
    ade,
    fde,
    parse_planning_text,
    split_lateral_longitudinal_ade,
)


def test_parse_clean_planning_block():
    text = (
        "<PLANNING>Predicted future movement details for the next 3 seconds "
        "(sampled at 0.5-second intervals). The output is formatted as "
        "[x, y]: [2.19, 0.04], [4.34, 0.19], [5.97, 0.43], [8.03, 0.89], "
        "[10.08, 1.56], [12.04, 2.51]</PLANNING>"
    )
    pairs = parse_planning_text(text)
    assert len(pairs) == 6
    assert pairs[0] == (2.19, 0.04)
    assert pairs[-1] == (12.04, 2.51)


def test_parse_without_closing_tag():
    text = "<PLANNING>The output is formatted as [x, y]: [1.0, 0.0], [2.0, 0.5]"
    pairs = parse_planning_text(text)
    assert pairs == [(1.0, 0.0), (2.0, 0.5)]


def test_parse_negative_values():
    text = "<PLANNING>... [x, y]: [-1.5, -0.3], [-3.0, -0.7]</PLANNING>"
    pairs = parse_planning_text(text)
    assert pairs == [(-1.5, -0.3), (-3.0, -0.7)]


def test_parse_garbage_returns_empty():
    assert parse_planning_text("no planning tag here") == []
    assert parse_planning_text("") == []


def test_parse_skips_format_hint():
    # The literal "[x, y]" hint contains letters and should not be captured.
    text = "<PLANNING>... format [x, y]: [1.0, 0.5]</PLANNING>"
    assert parse_planning_text(text) == [(1.0, 0.5)]


def test_ade_basic():
    pred = [(0.0, 0.0), (1.0, 0.0)]
    gt = [(0.0, 0.0), (0.0, 0.0)]
    # step 0: 0, step 1: 1 → mean = 0.5
    assert math.isclose(ade(pred, gt), 0.5)


def test_ade_uses_shorter_length():
    pred = [(1.0, 0.0)]
    gt = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    assert math.isclose(ade(pred, gt), 1.0)


def test_fde_basic():
    pred = [(0.0, 0.0), (3.0, 4.0)]
    gt = [(0.0, 0.0), (0.0, 0.0)]
    # last-step distance = sqrt(9 + 16) = 5
    assert math.isclose(fde(pred, gt), 5.0)


def test_split_lateral_longitudinal():
    pred = [(1.0, 0.5), (2.0, 1.0)]
    gt = [(0.0, 0.0), (0.0, 0.0)]
    lon, lat = split_lateral_longitudinal_ade(pred, gt)
    # longitudinal mean abs error: (1 + 2) / 2 = 1.5
    # lateral mean abs error: (0.5 + 1.0) / 2 = 0.75
    assert math.isclose(lon, 1.5)
    assert math.isclose(lat, 0.75)


def test_ade_raises_on_empty():
    try:
        ade([], [(0.0, 0.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def _run_all() -> int:
    fns = [
        test_parse_clean_planning_block,
        test_parse_without_closing_tag,
        test_parse_negative_values,
        test_parse_garbage_returns_empty,
        test_parse_skips_format_hint,
        test_ade_basic,
        test_ade_uses_shorter_length,
        test_fde_basic,
        test_split_lateral_longitudinal,
        test_ade_raises_on_empty,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
    else:
        print(f"\nAll {len(fns)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(_run_all())
