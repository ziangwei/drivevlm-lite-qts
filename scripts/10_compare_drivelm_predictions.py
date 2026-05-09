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


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id")
        key = str(sample_id) if sample_id is not None else str(idx)
        indexed[key] = row
    return indexed


def _bucket_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    count = len(pairs)
    baseline_correct = 0
    candidate_correct = 0
    both_correct = 0
    both_wrong = 0
    baseline_only = 0
    candidate_only = 0
    for baseline, candidate in pairs:
        b_ok = _row_exact_match(baseline) == 1.0
        c_ok = _row_exact_match(candidate) == 1.0
        baseline_correct += int(b_ok)
        candidate_correct += int(c_ok)
        both_correct += int(b_ok and c_ok)
        both_wrong += int((not b_ok) and (not c_ok))
        baseline_only += int(b_ok and not c_ok)
        candidate_only += int(c_ok and not b_ok)
    return {
        "count": count,
        "baseline_em": baseline_correct / max(1, count),
        "candidate_em": candidate_correct / max(1, count),
        "delta": (candidate_correct - baseline_correct) / max(1, count),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
    }


def _summaries(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    long_answer_words: int,
) -> dict[str, list[dict[str, Any]]]:
    feature_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    task_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for baseline, candidate in pairs:
        for feature in _features(baseline, long_answer_words):
            feature_buckets[feature].append((baseline, candidate))
        task_buckets[str(baseline.get("task", "unknown"))].append((baseline, candidate))

    def make(groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> list[dict[str, Any]]:
        rows = []
        for bucket, bucket_pairs in sorted(groups.items()):
            item = _bucket_summary(bucket_pairs)
            item["bucket"] = bucket
            rows.append(item)
        return rows

    return {"by_feature": make(feature_buckets), "by_task": make(task_buckets)}


def _examples(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    limit: int,
    wanted: str,
) -> list[dict[str, Any]]:
    output = []
    for baseline, candidate in pairs:
        b_ok = _row_exact_match(baseline) == 1.0
        c_ok = _row_exact_match(candidate) == 1.0
        if wanted == "candidate_only" and not (c_ok and not b_ok):
            continue
        if wanted == "baseline_only" and not (b_ok and not c_ok):
            continue
        output.append(
            {
                "sample_id": baseline.get("sample_id"),
                "task": baseline.get("task"),
                "question": baseline.get("question"),
                "answer": baseline.get("answer"),
                "baseline_prediction": baseline.get("prediction"),
                "candidate_prediction": candidate.get("prediction"),
            }
        )
        if len(output) >= limit:
            break
    return output


def _rate(value: float) -> str:
    return f"{value:.3f}"


def _write_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| bucket | count | baseline EM | candidate EM | delta | baseline only | candidate only | both correct | both wrong |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {bucket} | {count} | {b} | {c} | {d} | {bo} | {co} | {bc} | {bw} |".format(
                bucket=row["bucket"],
                count=row["count"],
                b=_rate(float(row["baseline_em"])),
                c=_rate(float(row["candidate_em"])),
                d=_rate(float(row["delta"])),
                bo=row["baseline_only"],
                co=row["candidate_only"],
                bc=row["both_correct"],
                bw=row["both_wrong"],
            )
        )
    lines.append("")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DriveLM Prediction Comparison",
        "",
        f"- baseline: {report['baseline_name']}",
        f"- candidate: {report['candidate_name']}",
        f"- matched_count: {report['matched_count']}",
        f"- baseline_em: {_rate(float(report['overall']['baseline_em']))}",
        f"- candidate_em: {_rate(float(report['overall']['candidate_em']))}",
        f"- delta: {_rate(float(report['overall']['delta']))}",
        "",
    ]
    _write_table(lines, "By Feature", report["by_feature"])
    _write_table(lines, "By Task", report["by_task"])

    lines.extend(["## Candidate Only Correct Examples", ""])
    for item in report["candidate_only_examples"]:
        lines.extend(
            [
                f"### {item.get('sample_id')} ({item.get('task')})",
                "",
                f"- question: {item.get('question')}",
                f"- answer: {item.get('answer')}",
                f"- baseline: {item.get('baseline_prediction')}",
                f"- candidate: {item.get('candidate_prediction')}",
                "",
            ]
        )

    lines.extend(["## Baseline Only Correct Examples", ""])
    for item in report["baseline_only_examples"]:
        lines.extend(
            [
                f"### {item.get('sample_id')} ({item.get('task')})",
                "",
                f"- question: {item.get('question')}",
                f"- answer: {item.get('answer')}",
                f"- baseline: {item.get('baseline_prediction')}",
                f"- candidate: {item.get('candidate_prediction')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--examples", default=10, type=int)
    parser.add_argument("--long-answer-words", default=30, type=int)
    args = parser.parse_args()

    baseline_rows = _index_rows(read_jsonl(args.baseline))
    candidate_rows = _index_rows(read_jsonl(args.candidate))
    keys = [key for key in baseline_rows if key in candidate_rows]
    pairs = [(baseline_rows[key], candidate_rows[key]) for key in keys]

    bucket_report = _summaries(pairs, args.long_answer_words)
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "matched_count": len(pairs),
        "overall": _bucket_summary(pairs),
        "by_feature": bucket_report["by_feature"],
        "by_task": bucket_report["by_task"],
        "candidate_only_examples": _examples(pairs, args.examples, "candidate_only"),
        "baseline_only_examples": _examples(pairs, args.examples, "baseline_only"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.out_dir / "comparison.md", report)
    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote comparison: {args.out_dir}")


if __name__ == "__main__":
    main()
