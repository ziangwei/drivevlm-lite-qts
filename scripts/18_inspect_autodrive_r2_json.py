from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.autodrive_r2 import (
    convert_record,
    iter_json_records,
    summarize_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/autodrive_r2_json_inspect"), type=Path)
    parser.add_argument("--nuscenes-root", default=None, type=Path)
    parser.add_argument("--image-root", default=None, type=Path)
    parser.add_argument("--limit", default=200, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_json_records(args.input))
    summary = summarize_schema(records, limit=args.limit)

    converted_examples = []
    missing_images = 0
    converted_count = 0
    for record in records[: args.limit]:
        row = convert_record(
            record,
            nuscenes_root=args.nuscenes_root,
            image_root=args.image_root,
            answer_mode="cot",
            require_images=False,
        )
        if row is None:
            continue
        converted_count += 1
        for image in row.get("images", []):
            if image and not Path(image).exists():
                missing_images += 1
        if len(converted_examples) < 3:
            converted_examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "image_count": len(row.get("images", [])),
                    "first_images": row.get("images", [])[:2],
                    "question": row["messages"][0]["content"],
                    "answer": row["messages"][1]["content"][:800],
                    "trajectory": row.get("trajectory"),
                }
            )

    report = {
        "input": str(args.input),
        "total_records": len(records),
        "scanned_records": summary["scanned_records"],
        "converted_candidate_rows": converted_count,
        "missing_images_in_converted_candidates": missing_images,
        **summary,
        "converted_examples": converted_examples,
    }

    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# AutoDrive-R2 JSON Inspect",
        "",
        f"- input: {args.input}",
        f"- total_records: {len(records)}",
        f"- scanned_records: {summary['scanned_records']}",
        f"- image_candidate_rows: {summary['image_candidate_rows']}",
        f"- trajectory_candidate_rows: {summary['trajectory_candidate_rows']}",
        f"- cot_candidate_rows: {summary['cot_candidate_rows']}",
        f"- question_candidate_rows: {summary['question_candidate_rows']}",
        f"- answer_candidate_rows: {summary['answer_candidate_rows']}",
        f"- converted_candidate_rows: {converted_count}",
        f"- missing_images_in_converted_candidates: {missing_images}",
        "",
        "## Top Keys",
        "",
    ]
    for key, count in summary["top_keys"].items():
        lines.append(f"- `{key}`: {count}")

    lines.extend(["", "## Converted Examples", ""])
    for example in converted_examples:
        lines.extend(
            [
                f"### {example['sample_id']}",
                "",
                f"- image_count: {example['image_count']}",
                f"- first_images: {example['first_images']}",
                f"- question: {example['question']}",
                f"- answer_preview: {example['answer']}",
                "",
            ]
        )

    lines.extend(["", "## Raw Examples", ""])
    for idx, example in enumerate(summary["examples"], start=1):
        lines.extend(
            [
                f"### raw_{idx}",
                "",
                "```json",
                json.dumps(example, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )

    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "total_records",
                    "converted_candidate_rows",
                    "missing_images_in_converted_candidates",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote inspect report: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
