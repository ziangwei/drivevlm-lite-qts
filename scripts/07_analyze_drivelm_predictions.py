from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.eval.metrics import exact_match


OBJECT_RE = re.compile(r"<c\d+,[^>]+>")
COORD_RE = re.compile(r"<c\d+,[^>]*,\d+(?:\.\d+)?,\d+(?:\.\d+)?>")
CAMERA_RE = re.compile(r"CAM_(?:FRONT|BACK)(?:_(?:LEFT|RIGHT))?")


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _row_exact_match(row: dict[str, Any]) -> float:
    if "exact_match" in row:
        return float(row["exact_match"])
    return exact_match(str(row.get("prediction", "")), str(row.get("answer", "")))


def _features(row: dict[str, Any], long_answer_words: int) -> list[str]:
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    text = f"{question}\n{answer}"
    features = ["all"]
    if OBJECT_RE.search(text):
        features.append("has_object_ids")
    if COORD_RE.search(text):
        features.append("has_coordinates")
    if CAMERA_RE.search(text):
        features.append("has_camera_names")
    if _word_count(answer) <= 8:
        features.append("short_answer")
    if _word_count(answer) >= long_answer_words:
        features.append("long_answer")
    return features


def _format_rate(value: float) -> str:
    return f"{value:.3f}"


def _summary(rows: list[dict[str, Any]], long_answer_words: int) -> dict[str, Any]:
    feature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        for feature in _features(row, long_answer_words):
            feature_rows[feature].append(row)
        task_rows[str(row.get("task", "unknown"))].append(row)

    def make_stats(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        stats = []
        for label, group_rows in sorted(groups.items()):
            matches = [_row_exact_match(row) for row in group_rows]
            stats.append(
                {
                    "bucket": label,
                    "count": len(group_rows),
                    "exact_match": sum(matches) / max(1, len(matches)),
                }
            )
        return stats

    return {
        "count": len(rows),
        "overall_exact_match": sum(_row_exact_match(row) for row in rows) / max(1, len(rows)),
        "by_feature": make_stats(feature_rows),
        "by_task": make_stats(task_rows),
    }


def _failure_examples(rows: list[dict[str, Any]], limit: int, long_answer_words: int) -> list[dict[str, Any]]:
    failures = [row for row in rows if _row_exact_match(row) == 0.0]
    failures.sort(key=lambda row: (_word_count(str(row.get("answer", ""))), str(row.get("sample_id", ""))), reverse=True)
    output = []
    for row in failures[:limit]:
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "task": row.get("task"),
                "features": [feature for feature in _features(row, long_answer_words) if feature != "all"],
                "question": row.get("question"),
                "answer": row.get("answer"),
                "prediction": row.get("prediction"),
            }
        )
    return output


def _write_markdown(path: Path, summary: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# DriveLM Prediction Analysis",
        "",
        f"- count: {summary['count']}",
        f"- overall_exact_match: {_format_rate(float(summary['overall_exact_match']))}",
        "",
        "## By Feature",
        "",
        "| bucket | count | exact_match |",
        "| --- | ---: | ---: |",
    ]
    for item in summary["by_feature"]:
        lines.append(f"| {item['bucket']} | {item['count']} | {_format_rate(float(item['exact_match']))} |")

    lines.extend(["", "## By Task", "", "| task | count | exact_match |", "| --- | ---: | ---: |"])
    for item in summary["by_task"]:
        lines.append(f"| {item['bucket']} | {item['count']} | {_format_rate(float(item['exact_match']))} |")

    lines.extend(["", "## Failure Examples", ""])
    for idx, item in enumerate(failures, start=1):
        lines.extend(
            [
                f"### {idx}. {item.get('sample_id')} ({item.get('task')})",
                "",
                f"- features: {', '.join(item.get('features') or ['none'])}",
                f"- question: {item.get('question')}",
                f"- answer: {item.get('answer')}",
                f"- prediction: {item.get('prediction')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--failure-examples", default=20, type=int)
    parser.add_argument("--long-answer-words", default=30, type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    summary = _summary(rows, args.long_answer_words)
    failures = _failure_examples(rows, args.failure_examples, args.long_answer_words)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        print("\nFailure examples:")
        for item in failures[: min(5, len(failures))]:
            print(f"- {item.get('sample_id')} [{', '.join(item.get('features') or ['none'])}]")
            print(f"  Q: {item.get('question')}")
            print(f"  GT: {item.get('answer')}")
            print(f"  PRED: {item.get('prediction')}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "analysis.json").write_text(
            json.dumps({"summary": summary, "failures": failures}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_markdown(args.out_dir / "analysis.md", summary, failures)
        print(f"Wrote analysis: {args.out_dir}")


if __name__ == "__main__":
    main()
