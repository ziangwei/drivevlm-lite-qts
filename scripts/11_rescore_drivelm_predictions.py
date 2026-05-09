from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.eval.metrics import exact_match, relaxed_exact_match, token_f1, yes_no_match


def _question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question", "")), str(row.get("answer", ""))


def _parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --prediction.")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Prediction name cannot be empty.")
    return name, Path(path)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_values: list[float] = []
    relaxed_values: list[float] = []
    f1_values: list[float] = []
    yes_no_values: list[float] = []
    latencies: list[float] = []
    input_tokens: list[float] = []
    grid_tokens: list[float] = []
    image_counts: list[float] = []

    for row in rows:
        _, answer = _question_and_answer(row)
        prediction = str(row.get("prediction", ""))
        strict_values.append(exact_match(prediction, answer))
        relaxed_values.append(relaxed_exact_match(prediction, answer))
        f1_values.append(token_f1(prediction, answer))
        yn = yes_no_match(prediction, answer)
        if yn is not None:
            yes_no_values.append(yn)
        if row.get("latency_s") is not None:
            latencies.append(float(row["latency_s"]))
        if row.get("input_tokens") is not None:
            input_tokens.append(float(row["input_tokens"]))
        if row.get("image_grid_tokens") is not None:
            grid_tokens.append(float(row["image_grid_tokens"]))
        if row.get("selected_images") is not None:
            image_counts.append(float(len(row["selected_images"])))
        elif row.get("images") is not None:
            image_counts.append(float(len(row["images"])))

    return {
        "count": len(rows),
        "strict_em": _avg(strict_values) or 0.0,
        "relaxed_em": _avg(relaxed_values) or 0.0,
        "token_f1": _avg(f1_values) or 0.0,
        "yes_no_accuracy": _avg(yes_no_values),
        "yes_no_count": len(yes_no_values),
        "avg_latency_s": _avg(latencies),
        "avg_input_tokens": _avg(input_tokens),
        "avg_image_grid_tokens": _avg(grid_tokens),
        "avg_images": _avg(image_counts),
    }


def _score_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("task", "unknown"))].append(row)
    output = []
    for task, task_rows in sorted(buckets.items()):
        item = _score_rows(task_rows)
        item["task"] = task
        output.append(item)
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DriveLM Prediction Rescore",
        "",
        "## Overall",
        "",
        "| run | count | strict EM | relaxed EM | token F1 | yes/no acc | yes/no n | latency s | input tokens | grid tokens | images |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        metrics = run["overall"]
        lines.append(
            "| {name} | {count} | {strict} | {relaxed} | {f1} | {yn} | {ynn} | {lat} | {inp} | {grid} | {imgs} |".format(
                name=run["name"],
                count=metrics["count"],
                strict=_fmt(metrics["strict_em"]),
                relaxed=_fmt(metrics["relaxed_em"]),
                f1=_fmt(metrics["token_f1"]),
                yn=_fmt(metrics["yes_no_accuracy"]),
                ynn=metrics["yes_no_count"],
                lat=_fmt(metrics["avg_latency_s"]),
                inp=_fmt(metrics["avg_input_tokens"], 1),
                grid=_fmt(metrics["avg_image_grid_tokens"], 1),
                imgs=_fmt(metrics["avg_images"], 2),
            )
        )

    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| run | task | count | strict EM | relaxed EM | token F1 | yes/no acc | yes/no n |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        for metrics in run["by_task"]:
            lines.append(
                "| {name} | {task} | {count} | {strict} | {relaxed} | {f1} | {yn} | {ynn} |".format(
                    name=run["name"],
                    task=metrics["task"],
                    count=metrics["count"],
                    strict=_fmt(metrics["strict_em"]),
                    relaxed=_fmt(metrics["relaxed_em"]),
                    f1=_fmt(metrics["token_f1"]),
                    yn=_fmt(metrics["yes_no_accuracy"]),
                    ynn=metrics["yes_no_count"],
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, type=_parse_prediction_arg)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    runs = []
    for name, path in args.prediction:
        rows = read_jsonl(path)
        runs.append(
            {
                "name": name,
                "path": str(path),
                "overall": _score_rows(rows),
                "by_task": _score_by_task(rows),
            }
        )

    report = {"runs": runs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(args.out_dir / "summary.md", report)
    print(json.dumps(report, indent=2))
    print(f"Wrote rescore summary: {args.out_dir}")


if __name__ == "__main__":
    main()
