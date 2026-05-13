from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl


def _message(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/reasoning_sft_check"), type=Path)
    parser.add_argument("--limit", default=100, type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    checked = rows[: args.limit] if args.limit > 0 else rows
    args.out_dir.mkdir(parents=True, exist_ok=True)

    task_counts = Counter(str(row.get("task", "unknown")) for row in rows)
    image_count_distribution = Counter(str(len(row.get("images", []))) for row in rows)
    missing_images = 0
    question_words = []
    answer_words = []
    reasoning_rows = 0
    examples = []

    for row in checked:
        for image in row.get("images", []):
            if not Path(image).exists():
                missing_images += 1
        question = _message(row, "user")
        answer = _message(row, "assistant")
        question_words.append(float(_word_count(question)))
        answer_words.append(float(_word_count(answer)))
        if row.get("metadata", {}).get("has_step_by_step_reasoning"):
            reasoning_rows += 1
        if len(examples) < 5:
            examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "task": row.get("task"),
                    "images": row.get("images", [])[:2],
                    "question": question,
                    "answer_preview": answer[:1000],
                }
            )

    report = {
        "input": str(args.input),
        "total_rows": len(rows),
        "checked_rows": len(checked),
        "missing_images": missing_images,
        "step_by_step_reasoning_rows_checked": reasoning_rows,
        "avg_question_words_checked": _mean(question_words),
        "avg_answer_words_checked": _mean(answer_words),
        "task_counts": dict(sorted(task_counts.items())),
        "image_count_distribution": dict(sorted(image_count_distribution.items(), key=lambda item: int(item[0]))),
        "examples": examples,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Reasoning SFT Data Check",
        "",
        f"- input: {args.input}",
        f"- total_rows: {report['total_rows']}",
        f"- checked_rows: {report['checked_rows']}",
        f"- missing_images: {report['missing_images']}",
        f"- step_by_step_reasoning_rows_checked: {report['step_by_step_reasoning_rows_checked']}",
        f"- avg_question_words_checked: {report['avg_question_words_checked']:.1f}",
        f"- avg_answer_words_checked: {report['avg_answer_words_checked']:.1f}",
        "",
        "## Tasks",
        "",
        "| task | count |",
        "| --- | ---: |",
    ]
    for task, count in report["task_counts"].items():
        lines.append(f"| {task} | {count} |")
    lines.extend(["", "## Image Counts", "", "| image count | rows |", "| ---: | ---: |"])
    for image_count, count in report["image_count_distribution"].items():
        lines.append(f"| {image_count} | {count} |")
    lines.extend(["", "## Examples", ""])
    for item in examples:
        lines.extend(
            [
                f"### {item['sample_id']}",
                "",
                f"- task: {item['task']}",
                f"- first_images: {item['images']}",
                f"- question: {item['question']}",
                "",
                "```text",
                item["answer_preview"],
                "```",
                "",
            ]
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote reasoning SFT check: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
