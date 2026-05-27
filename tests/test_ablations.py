"""Tests for the Stage 5 ablation transforms, donor selection, and analysis.

Run without pytest::

    PYTHONPATH=src python tests/test_ablations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.ablations import (
    ABLATIONS,
    SCENE_BOUNDARY_DT_US,
    TIME_SHIFT_MAX_DT_US,
    ablation_plan,
    build_donor_index,
    classify_maneuver,
    parse_cam_front_path,
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


# Realistic CAM_FRONT keyframe paths spanning two logs. Keyframes within a log
# are 0.5 s = 500_000 us apart; the third row in log A is +6 s away to
# simulate a within-log scene transition (true-mismatch eligible).
_LOG_A = "n015-2018-07-11-11-54-16+0800"
_LOG_B = "n008-2018-08-01-15-16-36-0400"
_BASE_TS = 1531281439262460  # microseconds


def _path(log: str, ts_us: int) -> str:
    return f"data/nuscenes/samples/CAM_FRONT/{log}__CAM_FRONT__{ts_us}.jpg"


_FIXTURE_PATHS = [
    _path(_LOG_A, _BASE_TS),                                   # 0  log A   t=0.0s
    _path(_LOG_A, _BASE_TS +     500_000),                     # 1  log A   t=+0.5s
    _path(_LOG_A, _BASE_TS +   1_000_000),                     # 2  log A   t=+1.0s
    _path(_LOG_A, _BASE_TS +   6_000_000),                     # 3  log A   t=+6.0s (different scene within log)
    _path(_LOG_B, _BASE_TS + 200_000_000),                     # 4  log B
    _path(_LOG_B, _BASE_TS + 200_500_000),                     # 5  log B
]


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
    assert transform_user_text(_PROMPT, "time_shifted_image") == _PROMPT
    assert transform_user_text(_PROMPT, "true_mismatch_image") == _PROMPT


def test_ablation_plan_image_channels() -> None:
    assert ablation_plan("full").image == "keep"
    assert ablation_plan("no_kinematics").image == "keep"
    assert ablation_plan("black_image").image == "black"
    assert ablation_plan("time_shifted_image").image == "time_shifted"
    assert ablation_plan("true_mismatch_image").image == "true_mismatch"
    # ABLATIONS and _PLANS stay in sync
    plan_keys = [a for a in ABLATIONS if ablation_plan(a)]
    assert set(plan_keys) == set(ABLATIONS)


def test_ablation_plan_unknown_raises() -> None:
    try:
        ablation_plan("nope")
    except ValueError as exc:
        assert "unknown ablation" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown ablation")


def test_ablation_plan_deprecated_mismatch_image_raises_with_guidance() -> None:
    """The old 'mismatch_image' name now raises with a tailored message."""
    try:
        ablation_plan("mismatch_image")
    except ValueError as exc:
        msg = str(exc)
        assert "true_mismatch_image" in msg
        assert "time_shifted_image" in msg
    else:
        raise AssertionError("expected ValueError for deprecated mismatch_image")


def test_parse_cam_front_path_extracts_log_and_ts() -> None:
    log, ts = parse_cam_front_path(
        "data/nuscenes/samples/CAM_FRONT/"
        "n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439262460.jpg"
    )
    assert log == "n015-2018-07-11-11-54-16+0800"
    assert ts == 1531281439262460


def test_parse_cam_front_path_accepts_mixed_separators_and_png() -> None:
    log, ts = parse_cam_front_path(
        r"D:\data\nuscenes\samples\CAM_FRONT\n008-2018-08-01-15-16-36-0400__CAM_FRONT__123.png"
    )
    assert log == "n008-2018-08-01-15-16-36-0400"
    assert ts == 123


def test_parse_cam_front_path_rejects_bad_basename() -> None:
    try:
        parse_cam_front_path("some/other/CAM_BACK__999.jpg")
    except ValueError as exc:
        assert "CAM_FRONT" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-CAM_FRONT path")


def test_build_donor_index_time_shifted_picks_same_log_within_window() -> None:
    donor = build_donor_index(_FIXTURE_PATHS, seed=42)
    # For each row, the time-shifted donor must be (a) a valid index,
    # (b) same log, (c) within ±0.7 s.
    for i, j in enumerate(donor.time_shifted):
        if j < 0:
            continue
        log_i, ts_i = parse_cam_front_path(_FIXTURE_PATHS[i])
        log_j, ts_j = parse_cam_front_path(_FIXTURE_PATHS[j])
        assert j != i
        assert log_j == log_i, f"row {i}: time_shifted donor {j} crossed logs"
        assert abs(ts_j - ts_i) <= TIME_SHIFT_MAX_DT_US, (
            f"row {i}: donor {j} dt={abs(ts_j - ts_i)} exceeds {TIME_SHIFT_MAX_DT_US}"
        )
    # Rows 0/1/2 are densely packed (0.5 s apart) so all should resolve.
    assert donor.time_shifted[0] >= 0
    assert donor.time_shifted[1] >= 0
    assert donor.time_shifted[2] >= 0
    # Row 3 is +6 s away from {0,1,2} (well outside ±0.7 s) and is alone in
    # its scene; no time-shifted partner exists → sentinel.
    assert donor.time_shifted[3] == -1
    # Rows 4/5 are 0.5 s apart in log B → mutual partners.
    assert donor.time_shifted[4] == 5
    assert donor.time_shifted[5] == 4


def test_build_donor_index_true_mismatch_picks_different_scene() -> None:
    donor = build_donor_index(_FIXTURE_PATHS, seed=42)
    for i, j in enumerate(donor.true_mismatch):
        assert j >= 0, f"row {i}: no true_mismatch donor — fixture should always have one"
        assert j != i
        log_i, ts_i = parse_cam_front_path(_FIXTURE_PATHS[i])
        log_j, ts_j = parse_cam_front_path(_FIXTURE_PATHS[j])
        # Either different log entirely, OR same log >5 s away.
        if log_j == log_i:
            assert abs(ts_j - ts_i) > SCENE_BOUNDARY_DT_US, (
                f"row {i}: same-log donor {j} only {abs(ts_j - ts_i)} us away "
                f"— must exceed {SCENE_BOUNDARY_DT_US} us to count as a true mismatch"
            )


def test_build_donor_index_is_deterministic_for_same_seed() -> None:
    a = build_donor_index(_FIXTURE_PATHS, seed=42)
    b = build_donor_index(_FIXTURE_PATHS, seed=42)
    assert a.time_shifted == b.time_shifted
    assert a.true_mismatch == b.true_mismatch


def test_build_donor_index_seed_changes_choice() -> None:
    """Different seeds should generally produce different donor maps over a
    fixture with multiple eligible candidates per row."""
    a = build_donor_index(_FIXTURE_PATHS, seed=1)
    b = build_donor_index(_FIXTURE_PATHS, seed=2)
    assert (a.time_shifted, a.true_mismatch) != (b.time_shifted, b.true_mismatch)


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
        test_ablation_plan_deprecated_mismatch_image_raises_with_guidance,
        test_parse_cam_front_path_extracts_log_and_ts,
        test_parse_cam_front_path_accepts_mixed_separators_and_png,
        test_parse_cam_front_path_rejects_bad_basename,
        test_build_donor_index_time_shifted_picks_same_log_within_window,
        test_build_donor_index_true_mismatch_picks_different_scene,
        test_build_donor_index_is_deterministic_for_same_seed,
        test_build_donor_index_seed_changes_choice,
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
