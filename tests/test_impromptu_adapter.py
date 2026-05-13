"""Synthetic tests for the Impromptu adapter.

These tests do not require the real nuScenes keyframe tree. They build a
tiny on-disk fixture, run the adapter, and check the rewritten paths,
existence flags, and JSONL output.

Run without pytest::

    PYTHONPATH=src python tests/test_impromptu_adapter.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.impromptu_adapter import (
    AdapterStats,
    iter_rewritten,
    load_impromptu_records,
    rewrite_image_paths,
    write_records_jsonl,
)


def _make_record(
    sample_id: str,
    image_rel: str,
    user_text: str = "user prompt",
    assistant_text: str = "<PLANNING>traj</PLANNING>",
) -> dict[str, object]:
    return {
        "id": sample_id,
        "images": [image_rel],
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def _write_json(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_rewrite_image_paths_strips_prefix(tmp_path: Path) -> None:
    record = _make_record("a", "nuscenes/samples/CAM_FRONT/x.jpg")
    nuscenes_root = tmp_path / "nuscenes"
    out = rewrite_image_paths(record, nuscenes_root)
    assert out["images"] == [str(nuscenes_root / "samples/CAM_FRONT/x.jpg")]
    assert record["images"] == ["nuscenes/samples/CAM_FRONT/x.jpg"]


def test_rewrite_keeps_unprefixed_paths(tmp_path: Path) -> None:
    record = _make_record("a", "samples/CAM_FRONT/x.jpg")
    out = rewrite_image_paths(record, tmp_path / "nuscenes")
    assert out["images"] == [str(tmp_path / "nuscenes" / "samples/CAM_FRONT/x.jpg")]


def test_iter_rewritten_marks_existing_images(tmp_path: Path) -> None:
    nuscenes_root = tmp_path / "nuscenes"
    real_image = nuscenes_root / "samples/CAM_FRONT/exists.jpg"
    real_image.parent.mkdir(parents=True, exist_ok=True)
    real_image.write_bytes(b"")

    records = [
        _make_record("ok", "nuscenes/samples/CAM_FRONT/exists.jpg"),
        _make_record("missing", "nuscenes/samples/CAM_FRONT/missing.jpg"),
        _make_record("empty", "nuscenes/samples/CAM_FRONT/empty.jpg"),
    ]
    records[2]["images"] = []

    pairs = list(iter_rewritten(records, nuscenes_root, require_image=True))
    assert len(pairs) == 3
    ids = [r["id"] for r, _ in pairs]
    flags = [ok for _, ok in pairs]
    assert ids == ["ok", "missing", "empty"]
    assert flags == [True, False, False]


def test_iter_rewritten_respects_limit(tmp_path: Path) -> None:
    nuscenes_root = tmp_path / "nuscenes"
    records = [
        _make_record(f"r{i}", "nuscenes/samples/CAM_FRONT/x.jpg") for i in range(5)
    ]
    pairs = list(iter_rewritten(records, nuscenes_root, require_image=False, limit=2))
    assert [r["id"] for r, _ in pairs] == ["r0", "r1"]


def test_write_records_jsonl_drops_missing(tmp_path: Path) -> None:
    nuscenes_root = tmp_path / "nuscenes"
    real_image = nuscenes_root / "samples/CAM_FRONT/exists.jpg"
    real_image.parent.mkdir(parents=True, exist_ok=True)
    real_image.write_bytes(b"")

    records = [
        _make_record("ok", "nuscenes/samples/CAM_FRONT/exists.jpg"),
        _make_record("missing", "nuscenes/samples/CAM_FRONT/missing.jpg"),
    ]
    out = tmp_path / "out.jsonl"
    pairs = iter_rewritten(records, nuscenes_root, require_image=True)
    stats = write_records_jsonl(out, pairs, drop_missing=True)
    assert isinstance(stats, AdapterStats)
    assert stats.total == 2
    assert stats.written == 1
    assert stats.missing_images == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["id"] == "ok"
    assert row["images"][0].endswith("exists.jpg")


def test_write_records_jsonl_keeps_missing_when_requested(tmp_path: Path) -> None:
    nuscenes_root = tmp_path / "nuscenes"
    records = [_make_record("missing", "nuscenes/samples/CAM_FRONT/missing.jpg")]
    out = tmp_path / "out.jsonl"
    pairs = iter_rewritten(records, nuscenes_root, require_image=False)
    stats = write_records_jsonl(out, pairs, drop_missing=False)
    assert stats.total == 1
    assert stats.written == 1
    assert stats.missing_images == 1


def test_load_impromptu_records_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "nuscenes_train.json"
    records = [_make_record("a", "nuscenes/samples/CAM_FRONT/x.jpg")]
    _write_json(src, records)
    loaded = load_impromptu_records(src)
    assert loaded == records


def _run_all_tests(tmp_root: Path) -> int:
    test_fns = [
        test_rewrite_image_paths_strips_prefix,
        test_rewrite_keeps_unprefixed_paths,
        test_iter_rewritten_marks_existing_images,
        test_iter_rewritten_respects_limit,
        test_write_records_jsonl_drops_missing,
        test_write_records_jsonl_keeps_missing_when_requested,
        test_load_impromptu_records_roundtrip,
    ]
    failures = 0
    for fn in test_fns:
        sub = tmp_root / fn.__name__
        sub.mkdir(parents=True, exist_ok=True)
        try:
            fn(sub)
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
        print(f"\nAll {len(test_fns)} tests passed.")
    return failures


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        failures = _run_all_tests(Path(tmp))
    sys.exit(1 if failures else 0)
