from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.data.nuscenes_trajectory import ade, fde, parse_trajectory_text


def _answer(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _target(row: dict[str, Any]) -> list[tuple[float, float]]:
    if row.get("trajectory"):
        return [(float(item["x"]), float(item["y"])) for item in row["trajectory"]]
    return parse_trajectory_text(_answer(row))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trajectory_prior(rows: list[dict[str, Any]], mode: str, steps: int) -> list[tuple[float, float]]:
    if mode == "zero":
        return [(0.0, 0.0) for _ in range(steps)]

    targets = [_target(row) for row in rows]
    targets = [target for target in targets if len(target) >= steps]
    if not targets:
        raise ValueError("No valid train trajectories for prior baseline.")

    out = []
    for idx in range(steps):
        xs = [target[idx][0] for target in targets]
        ys = [target[idx][1] for target in targets]
        if mode == "train_mean":
            out.append((_mean(xs), _mean(ys)))
        elif mode == "train_median":
            out.append((float(statistics.median(xs)), float(statistics.median(ys))))
        elif mode == "train_mean_straight":
            out.append((_mean(xs), 0.0))
        else:
            raise ValueError(f"Unknown prior mode: {mode}")
    return out


def _evaluate_prior(
    rows: list[dict[str, Any]],
    prior: list[tuple[float, float]],
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ades = []
    fdes = []
    predictions = []
    for row in rows:
        target = _target(row)
        row_ade = ade(prior, target)
        row_fde = fde(prior, target)
        if row_ade is not None:
            ades.append(row_ade)
        if row_fde is not None:
            fdes.append(row_fde)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "mode": mode,
                "target": target,
                "prediction": prior,
                "ade": row_ade,
                "fde": row_fde,
            }
        )

    metrics = {
        "mode": mode,
        "count": len(rows),
        "valid_ade_count": len(ades),
        "ade": _mean(ades),
        "fde": _mean(fdes),
        "prior": prior,
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/vla_prior_baselines"), type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument(
        "--modes",
        default="zero,train_mean,train_median,train_mean_straight",
        help="Comma-separated: zero,train_mean,train_median,train_mean_straight",
    )
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.input)
    if args.limit > 0:
        eval_rows = eval_rows[: args.limit]
    first_target = _target(eval_rows[0]) if eval_rows else []
    steps = len(first_target)
    if steps == 0:
        raise ValueError("Evaluation rows do not contain parseable trajectories.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    all_metrics = []
    for mode in modes:
        prior = _trajectory_prior(train_rows, mode, steps)
        metrics, predictions = _evaluate_prior(eval_rows, prior, mode)
        all_metrics.append(metrics)
        write_jsonl(args.out_dir / f"{mode}_predictions.jsonl", predictions)

    (args.out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    lines = [
        "# VLA Prior Baselines",
        "",
        f"- train: {args.train}",
        f"- input: {args.input}",
        "",
        "| mode | count | ADE m | FDE m | valid ADE n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in all_metrics:
        lines.append(
            f"| {item['mode']} | {item['count']} | {item['ade']:.3f} | {item['fde']:.3f} | "
            f"{item['valid_ade_count']} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))
    print(f"Wrote prior baseline summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
