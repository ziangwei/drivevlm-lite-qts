"""Tests for the Stage 5 ablation transforms and analysis helpers.

Run without pytest::

    PYTHONPATH=src python tests/test_ablations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.ablations import (
    ABLATIONS,
    ablation_plan,
    classify_maneuver,
    percentiles,
    strip_ego_status,
    strip_kinematics,
    transform_user_text,
)


# A realistic two-timestep Impromptu user prompt.
_PROMPT = (
    "You are an autonomous driving agent. You have access to a front view "
    "camera image of a vehicle <image>. Your task is to do your best to "
    "predict future waypoints for the vehicle over the next 3 timesteps, "
    "given the vehicle's intent inferred from the images."
    "Provided are the previous ego vehicle status recorded over the last 0.5 "
    "seconds (at 0.5-second intervals). This includes the x and y coordinates "
    "of the ego vehicle. Positive x means forward direction while positive y "
    "means leftwards. The data is presented in the format [x, y]:."
    "(t-0.5s) [-4.83, 0.15], Acceleration: X 0.06, Y 0.56 m/s^2, Velocity: "
    "9.38 m/s, Steering angle: 0.5 (positive: left turn, negative: right turn), "
    "(t-0.0s) [0.0, 0.0], Acceleration: X -0.1, Y 0.96 m/s^2, Velocity: 9.45 "
    "m/s, Steering angle: 0.55 (positive: left turn, negative: right turn)"
)


def test_strip_kinematics_keeps_positions_drops_kinematics() -> None:
    out = strip_kinematics(_PROMPT)
    assert "Acceleration" not in out
    assert "Velocity" not in out
    assert "Steering angle" not in out
    # positions and the timestep markers survive
    assert "(t-0.5s) [-4.83, 0.15]" in out
    assert "(t-0.0s) [0.0, 0.0]" in out
    # the framing sentence survives
    assert "Provided are the previous ego vehicle status" in out


def test_strip_ego_status_removes_block_keeps_instruction() -> None:
    out = strip_ego_status(_PROMPT)
    assert "Provided are the previous ego vehicle status" not in out
    assert "(t-0.0s)" not in out
    assert "Velocity" not in out
    assert out.endswith("inferred from the images.")
    assert "<image>" in out


def test_transform_user_text_dispatch() -> None:
    assert transform_user_text(_PROMPT, "full") == _PROMPT
    assert transform_user_text(_PROMPT, "no_kinematics") == strip_kinematics(_PROMPT)
    assert transform_user_text(_PROMPT, "no_ego") == strip_ego_status(_PROMPT)
    # image-side ablations leave the text untouched
    assert transform_user_text(_PROMPT, "black_image") == _PROMPT
    assert transform_user_text(_PROMPT, "mismatch_image") == _PROMPT


def test_ablation_plan_image_channels() -> None:
    assert ablation_plan("full").image == "keep"
    assert ablation_plan("no_kinematics").image == "keep"
    assert ablation_plan("black_image").image == "black"
    assert ablation_plan("mismatch_image").image == "mismatch"
    assert set(_PLANS_keys()) == set(ABLATIONS)


def _PLANS_keys() -> list[str]:
    return [a for a in ABLATIONS if ablation_plan(a)]


def test_ablation_plan_unknown_raises() -> None:
    try:
        ablation_plan("nope")
    except ValueError as exc:
        assert "unknown ablation" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown ablation")


def test_classify_maneuver_buckets() -> None:
    straight = [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0), (10.0, 0.0), (12.0, 0.1)]
    left = [(2.0, 0.2), (4.0, 0.8), (6.0, 1.8), (8.0, 3.0), (10.0, 4.5), (11.0, 6.0)]
    right = [(2.0, -0.2), (4.0, -0.8), (6.0, -1.8), (8.0, -3.0), (10.0, -4.5), (11.0, -6.0)]
    stop = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0)]
    assert classify_maneuver(straight) == "straight"
    assert classify_maneuver(left) == "left"
    assert classify_maneuver(right) == "right"
    assert classify_maneuver(stop) == "stop"
    assert classify_maneuver([]) == "unknown"


def test_percentiles_linear() -> None:
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    out = percentiles(vals, points=(0, 50, 100))
    assert out["p0"] == 0.0
    assert out["p50"] == 2.0
    assert out["p100"] == 4.0
    # default points present
    assert set(percentiles(vals).keys()) == {"p25", "p50", "p75", "p95"}
    assert percentiles([]) == {}


def _run_all_tests() -> int:
    test_fns = [
        test_strip_kinematics_keeps_positions_drops_kinematics,
        test_strip_ego_status_removes_block_keeps_instruction,
        test_transform_user_text_dispatch,
        test_ablation_plan_image_channels,
        test_ablation_plan_unknown_raises,
        test_classify_maneuver_buckets,
        test_percentiles_linear,
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
