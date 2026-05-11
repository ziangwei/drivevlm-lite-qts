from pathlib import Path

from drivevlm_lite.data.autodrive_r2 import convert_record, parse_waypoint_pairs, resolve_image_path


def test_parse_waypoint_pairs_accepts_parenthesized_points():
    assert parse_waypoint_pairs("(1.0, 2.5), (3, -4)") == [(1.0, 2.5), (3.0, -4.0)]


def test_resolve_image_path_maps_nuscenes_samples():
    root = Path("/data/nuscenes")
    resolved = resolve_image_path("other/root/samples/CAM_FRONT/frame.jpg", nuscenes_root=root)
    assert resolved == root / "samples/CAM_FRONT/frame.jpg"


def test_convert_record_builds_cot_vla_row():
    record = {
        "id": "abc",
        "image": "samples/CAM_FRONT/frame.jpg",
        "question": "Predict trajectory.",
        "cot": "The road is clear.",
        "trajectory": [[1, 0], [2, 0.5]],
    }

    row = convert_record(record, nuscenes_root=Path("/data/nuscenes"), require_images=True)

    assert row is not None
    assert row["sample_id"] == "abc"
    assert row["images"] == ["/data/nuscenes/samples/CAM_FRONT/frame.jpg"]
    assert row["trajectory"][-1] == {"t": 1.0, "x": 2.0, "y": 0.5}
    assert "<think>The road is clear.</think>" in row["messages"][1]["content"]
    assert "TRAJ:" in row["messages"][1]["content"]
