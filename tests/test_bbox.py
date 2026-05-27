"""Tests for the rotated 2-D bbox geometry used by the collision metric.

Run without pytest::

    PYTHONPATH=src python tests/test_bbox.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.bbox import point_in_rotated_bbox, quick_radius_overlap


# Convention reminder: length = local +x (heading), width = local +y, yaw = box-x in global.


def test_axis_aligned_inside() -> None:
    assert point_in_rotated_bbox((0.0, 0.0), (0.0, 0.0), length=4.0, width=2.0, yaw=0.0)
    assert point_in_rotated_bbox((1.9, 0.9), (0.0, 0.0), 4.0, 2.0, 0.0)


def test_axis_aligned_on_edge_is_inside() -> None:
    assert point_in_rotated_bbox((2.0, 1.0), (0.0, 0.0), 4.0, 2.0, 0.0)


def test_axis_aligned_outside() -> None:
    assert not point_in_rotated_bbox((2.01, 0.0), (0.0, 0.0), 4.0, 2.0, 0.0)
    assert not point_in_rotated_bbox((0.0, 1.01), (0.0, 0.0), 4.0, 2.0, 0.0)


def test_translated_box() -> None:
    # Box centred at (10, 5), 4x2, yaw 0. Point (11, 5) is inside (1 m forward).
    assert point_in_rotated_bbox((11.0, 5.0), (10.0, 5.0), 4.0, 2.0, 0.0)
    assert not point_in_rotated_bbox((13.0, 5.0), (10.0, 5.0), 4.0, 2.0, 0.0)


def test_rotated_90_degrees_swaps_axes() -> None:
    # Yaw 90 deg: length (was along +x) now along +y. So 4x2 box rotated by 90
    # is effectively 2 wide × 4 tall. Point (0, 1.9) is inside (was outside at yaw=0).
    yaw = math.pi / 2
    assert point_in_rotated_bbox((0.0, 1.9), (0.0, 0.0), 4.0, 2.0, yaw)
    # Point (1.9, 0) was inside at yaw=0, should now be outside.
    assert not point_in_rotated_bbox((1.9, 0.0), (0.0, 0.0), 4.0, 2.0, yaw)


def test_rotated_45_degrees() -> None:
    # 4x2 box at 45 deg. The 4-long axis points to upper-right (yaw=+45).
    # Point (1.4, 1.4) is along that diagonal at distance ~2; box length/2 = 2, so inside.
    yaw = math.pi / 4
    assert point_in_rotated_bbox((1.4, 1.4), (0.0, 0.0), 4.0, 2.0, yaw)
    # Perpendicular direction (upper-left): width is 2, so width/2 = 1 m allowed.
    # Point (-0.5, 0.5) is in perpendicular direction at distance ~0.71 m, inside.
    assert point_in_rotated_bbox((-0.5, 0.5), (0.0, 0.0), 4.0, 2.0, yaw)
    # Point 1.5 m perpendicular: outside.
    assert not point_in_rotated_bbox((-1.06, 1.06), (0.0, 0.0), 4.0, 2.0, yaw)


def test_quick_radius_early_out() -> None:
    # Far point: definitely outside the inscribed circle.
    assert not quick_radius_overlap((100.0, 100.0), (0.0, 0.0), 4.0, 2.0)
    # Near point: inside circle (used as upper bound, may be inside or outside the rect).
    assert quick_radius_overlap((1.0, 0.5), (0.0, 0.0), 4.0, 2.0)


def _run_all_tests() -> int:
    fns = [
        test_axis_aligned_inside,
        test_axis_aligned_on_edge_is_inside,
        test_axis_aligned_outside,
        test_translated_box,
        test_rotated_90_degrees_swaps_axes,
        test_rotated_45_degrees,
        test_quick_radius_early_out,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
    else:
        print(f"\nAll {len(fns)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all_tests() else 0)
