import importlib.util
from pathlib import Path


def _load_prepare_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "21_prepare_drivelmm_o1.py"
    spec = importlib.util.spec_from_file_location("prepare_drivelmm_o1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_bucket_detects_planning_question():
    prepare = _load_prepare_module()
    assert prepare._task_bucket("What is the best way to avoid a collision?") == "drivelmm_o1_planning"


def test_convert_record_resolves_images_without_lidar_input():
    prepare = _load_prepare_module()
    row, missing = prepare._convert_record(
        {
            "idx": "x",
            "image": ["samples/CAM_FRONT/a.jpg"],
            "question": "Describe the scene.",
            "answer": "**Step-by-Step Reasoning**:\n1. Road is clear.",
            "lidar": "frame.pcd.bin",
        },
        Path("/data/nuscenes"),
        allow_missing_images=True,
    )

    assert missing == 1
    assert row is not None
    assert row["task"] == "drivelmm_o1_perception"
    assert Path(row["images"][0]).as_posix().endswith("/data/nuscenes/samples/CAM_FRONT/a.jpg")
    assert row["metadata"]["has_step_by_step_reasoning"] is True
    assert Path(row["metadata"]["lidar"]).as_posix().endswith("/data/nuscenes/samples/LIDAR_TOP/frame.pcd.bin")
