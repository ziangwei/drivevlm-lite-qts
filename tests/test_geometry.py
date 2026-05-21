"""Tests for the Stage 6 ego->global geometry helpers.

Run without pytest::

    PYTHONPATH=src python tests/test_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.geometry import (
    ego_to_global,
    ego_to_global_path,
    yaw_from_quaternion,
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_yaw_identity_quaternion() -> None:
    assert _close(yaw_from_quaternion(1.0, 0.0, 0.0, 0.0), 0.0)


def test_yaw_90_degrees_about_z() -> None:
    # Quaternion for +90 deg about z: w=cos(45), z=sin(45).
    w = math.cos(math.pi / 4)
    z = math.sin(math.pi / 4)
    assert _close(yaw_from_quaternion(w, 0.0, 0.0, z), math.pi / 2)


def test_ego_to_global_no_rotation_no_translation() -> None:
    gx, gy = ego_to_global((5.0, 2.0), (0.0, 0.0), 0.0)
    assert _close(gx, 5.0) and _close(gy, 2.0)


def test_ego_to_global_translation_only() -> None:
    gx, gy = ego_to_global((5.0, 2.0), (100.0, -50.0), 0.0)
    assert _close(gx, 105.0) and _close(gy, -48.0)


def test_ego_to_global_yaw_90() -> None:
    # Heading north (yaw=+90 deg): forward (+x ego) -> +Y global, left (+y ego) -> -X global.
    gx, gy = ego_to_global((10.0, 0.0), (0.0, 0.0), math.pi / 2)
    assert _close(gx, 0.0) and _close(gy, 10.0)
    gx, gy = ego_to_global((0.0, 3.0), (0.0, 0.0), math.pi / 2)
    assert _close(gx, -3.0) and _close(gy, 0.0)


def test_ego_to_global_path_matches_pointwise() -> None:
    pts = [(1.0, 0.0), (2.0, 1.0)]
    out = ego_to_global_path(pts, (10.0, 20.0), 0.0)
    assert out == [(11.0, 20.0), (12.0, 21.0)]


def _run_all_tests() -> int:
    test_fns = [
        test_yaw_identity_quaternion,
        test_yaw_90_degrees_about_z,
        test_ego_to_global_no_rotation_no_translation,
        test_ego_to_global_translation_only,
        test_ego_to_global_yaw_90,
        test_ego_to_global_path_matches_pointwise,
    ]
    failures = 0
    for fn in test_fns:
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
        print(f"\nAll {len(test_fns)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all_tests() else 0)
