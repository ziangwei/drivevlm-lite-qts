import json
from pathlib import Path

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.data.nuscenes_cot import (
    CotFeatures,
    FrontAgent,
    build_vla_cot_ablation_files,
    synthesize_trajectory_cot,
)
from drivevlm_lite.data.nuscenes_trajectory import parse_trajectory_text


def test_synthesize_trajectory_cot_mentions_motion_and_front_agent():
    cot = synthesize_trajectory_cot(
        CotFeatures(
            ego_speed=8.0,
            future_speed=5.0,
            future_heading_rad=0.2,
            front_agent=FrontAgent(category="car", x=12.0, y=1.0, relative_speed=-1.5),
        ),
        [(1.0, 0.1), (2.0, 0.4), (3.0, 0.8)],
    )

    assert "Step 1 (Perception)" in cot
    assert "car" in cot
    assert "curves left" in cot
    assert "Decelerate" in cot


def test_build_vla_cot_ablation_files_writes_paired_direct_and_cot_rows(tmp_path):
    root = tmp_path / "nuscenes"
    version_dir = root / "v1.0-trainval"
    version_dir.mkdir(parents=True)
    _write_json(version_dir / "sample.json", _sample_rows())
    _write_json(version_dir / "sensor.json", [{"token": "sensor_front", "channel": "CAM_FRONT"}])
    _write_json(
        version_dir / "calibrated_sensor.json",
        [{"token": "calib_front", "sensor_token": "sensor_front"}],
    )
    _write_json(version_dir / "sample_data.json", _sample_data_rows())
    _write_json(version_dir / "ego_pose.json", _ego_pose_rows())
    _write_json(version_dir / "sample_annotation.json", _annotation_rows())

    train_input = tmp_path / "train.jsonl"
    val_input = tmp_path / "val.jsonl"
    write_jsonl(train_input, [_vla_row("s1")])
    write_jsonl(val_input, [_vla_row("s2")])

    summary = build_vla_cot_ablation_files(
        train_input=train_input,
        val_input=val_input,
        out_dir=tmp_path / "out",
        nuscenes_root=root,
        train_samples=1,
        val_samples=1,
    )

    assert summary["counts"] == {"direct_train": 1, "direct_val": 1, "cot_train": 1, "cot_val": 1}
    assert summary["feature_coverage"]["features"] == 2
    assert summary["feature_coverage"]["ego_speed"] == 2
    cot_val = read_jsonl(Path(summary["paths"]["cot_val"]))[0]
    direct_val = read_jsonl(Path(summary["paths"]["direct_val"]))[0]
    assert direct_val["task"] == "vla_trajectory"
    assert cot_val["task"] == "vla_trajectory_cot"
    assert "Reasoning:" in cot_val["messages"][1]["content"]
    assert "Trajectory: TRAJ:" in cot_val["messages"][1]["content"]
    assert len(parse_trajectory_text(cot_val["messages"][1]["content"])) == 6


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _sample_rows() -> list[dict]:
    return [
        {"token": "s0", "scene_token": "scene_a", "timestamp": 0, "prev": "", "next": "s1"},
        {"token": "s1", "scene_token": "scene_a", "timestamp": 500000, "prev": "s0", "next": "s2"},
        {"token": "s2", "scene_token": "scene_a", "timestamp": 1000000, "prev": "s1", "next": ""},
    ]


def _sample_data_rows() -> list[dict]:
    return [
        {
            "token": f"sd_{sample}",
            "sample_token": sample,
            "channel": "CAM_FRONT",
            "is_key_frame": True,
            "ego_pose_token": f"pose_{sample}",
            "calibrated_sensor_token": "calib_front",
            "filename": f"samples/CAM_FRONT/{sample}.jpg",
        }
        for sample in ("s0", "s1", "s2")
    ]


def _ego_pose_rows() -> list[dict]:
    return [
        {"token": "pose_s0", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
        {"token": "pose_s1", "translation": [2, 0, 0], "rotation": [1, 0, 0, 0]},
        {"token": "pose_s2", "translation": [5, 0, 0], "rotation": [1, 0, 0, 0]},
    ]


def _annotation_rows() -> list[dict]:
    return [
        {
            "token": "ann1",
            "sample_token": "s1",
            "instance_token": "agent1",
            "translation": [12, 1, 0],
            "category_name": "vehicle.car",
            "next": "ann2",
        },
        {
            "token": "ann2",
            "sample_token": "s2",
            "instance_token": "agent1",
            "translation": [18, 1, 0],
            "category_name": "vehicle.car",
            "next": "",
        },
    ]


def _vla_row(sample_id: str) -> dict:
    trajectory = [
        {"t": 0.5, "x": 1.0, "y": 0.0},
        {"t": 1.0, "x": 2.0, "y": 0.0},
        {"t": 1.5, "x": 3.0, "y": 0.1},
        {"t": 2.0, "x": 4.0, "y": 0.2},
        {"t": 2.5, "x": 5.0, "y": 0.3},
        {"t": 3.0, "x": 6.0, "y": 0.4},
    ]
    return {
        "sample_id": sample_id,
        "task": "vla_trajectory",
        "images": [f"/data/nuscenes/samples/CAM_FRONT/{sample_id}.jpg"],
        "messages": [
            {"role": "user", "content": "Predict trajectory."},
            {"role": "assistant", "content": "TRAJ: <t=0.5,x=1.000,y=0.000>"},
        ],
        "trajectory": trajectory,
        "metadata": {"scene_token": "scene_a"},
    }
