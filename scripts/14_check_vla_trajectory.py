from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.data.nuscenes_trajectory import ade, fde, parse_trajectory_text


def _answer(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _target(row: dict) -> list[tuple[float, float]]:
    if row.get("trajectory"):
        return [(float(item["x"]), float(item["y"])) for item in row["trajectory"]]
    return parse_trajectory_text(_answer(row))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/vla_data_check"), type=Path)
    parser.add_argument("--limit", default=20, type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    sample_rows = rows[: args.limit] if args.limit > 0 else rows
    args.out_dir.mkdir(parents=True, exist_ok=True)

    valid_parse = 0
    missing_images = 0
    final_distances = []
    roundtrip_ades = []
    roundtrip_fdes = []
    examples = []

    for row in sample_rows:
        target = _target(row)
        parsed = parse_trajectory_text(_answer(row))
        if parsed:
            valid_parse += 1
        rt_ade = ade(parsed, target)
        rt_fde = fde(parsed, target)
        if rt_ade is not None:
            roundtrip_ades.append(rt_ade)
        if rt_fde is not None:
            roundtrip_fdes.append(rt_fde)
        if target:
            x, y = target[-1]
            final_distances.append((x * x + y * y) ** 0.5)
        for image in row.get("images", []):
            if not Path(image).exists():
                missing_images += 1
        if len(examples) < 5:
            examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "answer": _answer(row),
                    "trajectory": row.get("trajectory"),
                    "images": row.get("images", [])[:2],
                }
            )

    report = {
        "input": str(args.input),
        "total_rows": len(rows),
        "checked_rows": len(sample_rows),
        "valid_parse": valid_parse,
        "missing_images": missing_images,
        "roundtrip_ade": _mean(roundtrip_ades),
        "roundtrip_fde": _mean(roundtrip_fdes),
        "mean_final_distance_m": _mean(final_distances),
        "median_final_distance_m": statistics.median(final_distances) if final_distances else 0.0,
        "examples": examples,
    }

    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# VLA Trajectory Data Check",
        "",
        f"- input: {args.input}",
        f"- total_rows: {report['total_rows']}",
        f"- checked_rows: {report['checked_rows']}",
        f"- valid_parse: {report['valid_parse']}",
        f"- missing_images: {report['missing_images']}",
        f"- roundtrip_ade: {report['roundtrip_ade']:.6f}",
        f"- roundtrip_fde: {report['roundtrip_fde']:.6f}",
        f"- mean_final_distance_m: {report['mean_final_distance_m']:.3f}",
        f"- median_final_distance_m: {report['median_final_distance_m']:.3f}",
        "",
        "## Examples",
        "",
    ]
    for item in examples:
        lines.extend(
            [
                f"### {item['sample_id']}",
                "",
                f"- answer: {item['answer']}",
                f"- first_images: {item['images']}",
                "",
            ]
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote VLA data check: {args.out_dir}")


if __name__ == "__main__":
    main()
