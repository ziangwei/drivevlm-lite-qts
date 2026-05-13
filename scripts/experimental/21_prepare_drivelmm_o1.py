from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.autodrive_r2 import iter_json_records, resolve_image_path
from drivevlm_lite.data.jsonl import write_jsonl


def _task_bucket(question: str) -> str:
    text = question.lower()
    if any(term in text for term in ("collision", "avoid", "best way", "least likely", "safe", "behavior")):
        return "drivelmm_o1_planning"
    if any(term in text for term in ("future", "predict", "will", "likely", "next")):
        return "drivelmm_o1_prediction"
    if any(term in text for term in ("describe", "object", "scene", "traffic", "visible", "important")):
        return "drivelmm_o1_perception"
    return "drivelmm_o1_reasoning"


def _answer_has_reasoning(answer: str) -> bool:
    text = answer.lower()
    return "step-by-step" in text or "reasoning" in text or bool(re.search(r"\n\s*1[\.\)]", answer))


def _resolve_lidar(path: str, nuscenes_root: Path | None) -> str:
    raw = Path(path)
    if raw.is_absolute() or nuscenes_root is None:
        return str(raw)
    normalized = path.replace("\\", "/")
    if "/" in normalized:
        return str(nuscenes_root / normalized)
    return str(nuscenes_root / "samples" / "LIDAR_TOP" / normalized)


def _convert_record(
    record: dict[str, Any],
    nuscenes_root: Path | None,
    allow_missing_images: bool,
) -> tuple[dict[str, Any] | None, int]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    raw_images = record.get("image", [])
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not question or not answer or not isinstance(raw_images, list) or not raw_images:
        return None, 0

    images = [str(resolve_image_path(str(path), nuscenes_root=nuscenes_root)) for path in raw_images]
    missing_images = sum(1 for path in images if not Path(path).exists())
    if missing_images and not allow_missing_images:
        return None, missing_images

    metadata: dict[str, Any] = {
        "source": "DriveLMM-o1",
        "raw_idx": record.get("idx"),
        "raw_image_count": len(raw_images),
        "has_step_by_step_reasoning": _answer_has_reasoning(answer),
    }
    if record.get("lidar"):
        metadata["lidar"] = _resolve_lidar(str(record["lidar"]), nuscenes_root)

    return (
        {
            "sample_id": str(record.get("idx", "")),
            "task": _task_bucket(question),
            "images": images,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "metadata": metadata,
        },
        missing_images,
    )


def _select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return rows
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:count]


def _convert_file(
    path: Path,
    nuscenes_root: Path | None,
    allow_missing_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    stats = Counter(
        {
            "input_records": 0,
            "converted_records": 0,
            "skipped_records": 0,
            "rows_with_missing_images": 0,
            "missing_images": 0,
        }
    )
    for record in iter_json_records(path):
        stats["input_records"] += 1
        row, missing = _convert_record(record, nuscenes_root, allow_missing_images)
        if missing:
            stats["rows_with_missing_images"] += 1
            stats["missing_images"] += missing
        if row is None:
            stats["skipped_records"] += 1
            continue
        rows.append(row)
        stats["converted_records"] += 1
    return rows, dict(stats)


def _message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(str(row.get("task", "unknown")) for row in rows)
    image_counts = Counter(str(len(row.get("images", []))) for row in rows)
    reasoning_count = sum(1 for row in rows if row.get("metadata", {}).get("has_step_by_step_reasoning"))
    answer_words = [_word_count(_message(row, "assistant")) for row in rows]
    return {
        "rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "image_count_distribution": dict(sorted(image_counts.items(), key=lambda item: int(item[0]))),
        "step_by_step_reasoning_rows": reasoning_count,
        "avg_answer_words": sum(answer_words) / max(1, len(answer_words)),
    }


def _write_summary(
    path: Path,
    summary: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# DriveLMM-o1 Prepare",
        "",
        f"- train_input: {summary['train_input']}",
        f"- val_input: {summary['val_input']}",
        f"- train_output: {summary['train_output']}",
        f"- val_output: {summary['val_output']}",
        f"- train_rows: {summary['train']['rows']}",
        f"- val_rows: {summary['val']['rows']}",
        f"- train_missing_images: {summary['train_stats']['missing_images']}",
        f"- val_missing_images: {summary['val_stats']['missing_images']}",
        "",
        "## Train Tasks",
        "",
        "| task | count |",
        "| --- | ---: |",
    ]
    for task, count in summary["train"]["task_counts"].items():
        lines.append(f"| {task} | {count} |")
    lines.extend(["", "## Image Counts", "", "| split | image count | rows |", "| --- | ---: | ---: |"])
    for split in ("train", "val"):
        for image_count, count in summary[split]["image_count_distribution"].items():
            lines.append(f"| {split} | {image_count} | {count} |")
    lines.extend(["", "## Example", ""])
    example = val_rows[0] if val_rows else train_rows[0] if train_rows else None
    if example:
        lines.extend(
            [
                f"- sample_id: {example.get('sample_id')}",
                f"- task: {example.get('task')}",
                f"- images: {len(example.get('images', []))}",
                f"- first_images: {example.get('images', [])[:2]}",
                f"- question: {_message(example, 'user')}",
                "",
                "```text",
                _message(example, "assistant")[:1200],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--val-input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed_drivelmm_o1"), type=Path)
    parser.add_argument("--nuscenes-root", default=None, type=Path)
    parser.add_argument("--train-samples", default=0, type=int)
    parser.add_argument("--val-samples", default=0, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    train_rows_all, train_stats = _convert_file(args.train_input, args.nuscenes_root, args.allow_missing_images)
    val_rows_all, val_stats = _convert_file(args.val_input, args.nuscenes_root, args.allow_missing_images)
    train_rows = _select(train_rows_all, args.train_samples, args.seed)
    val_rows = _select(val_rows_all, args.val_samples, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_output = args.out_dir / "drivelmm_o1_train.jsonl"
    val_output = args.out_dir / "drivelmm_o1_val.jsonl"
    write_jsonl(train_output, train_rows)
    write_jsonl(val_output, val_rows)

    summary = {
        "train_input": str(args.train_input),
        "val_input": str(args.val_input),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train": _summarize_rows(train_rows),
        "val": _summarize_rows(val_rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary(args.out_dir / "summary.md", summary, train_rows, val_rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
