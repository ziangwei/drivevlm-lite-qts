from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _vla_row(label: str, metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    metrics = _read_json(metrics_path)
    return {
        "run": label,
        "kind": "vlm",
        "count": metrics.get("count"),
        "parse_rate": metrics.get("parse_rate"),
        "usable_points": metrics.get("usable_point_count_rate"),
        "ade": metrics.get("ade"),
        "fde": metrics.get("fde"),
        "valid_ade_count": metrics.get("valid_ade_count"),
        "latency": metrics.get("avg_latency_s"),
        "images": metrics.get("avg_images"),
        "image_mode": metrics.get("image_mode"),
    }


def _prior_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows = []
    for item in _read_json(metrics_path):
        rows.append(
            {
                "run": f"prior:{item.get('mode')}",
                "kind": "prior",
                "count": item.get("count"),
                "parse_rate": None,
                "usable_points": None,
                "ade": item.get("ade"),
                "fde": item.get("fde"),
                "valid_ade_count": item.get("valid_ade_count"),
                "latency": None,
                "images": 0.0,
                "image_mode": "none",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    suite_dir = args.suite_dir
    rows: list[dict[str, Any]] = []
    rows.extend(_prior_rows(suite_dir / "priors" / "metrics.json"))

    for label, dirname in [
        ("zero-shot all", "zeroshot_all"),
        ("lora all", "lora_all"),
        ("lora front3", "lora_front3"),
        ("lora mismatch all", "lora_mismatch_all"),
        ("lora mismatch front3", "lora_mismatch_front3"),
    ]:
        row = _vla_row(label, suite_dir / dirname / "metrics.json")
        if row is not None:
            rows.append(row)

    lines = [
        "# VLA Final Suite",
        "",
        f"- suite_dir: {suite_dir}",
        "",
        "| run | kind | count | parse | usable 6pt | ADE m | FDE m | valid ADE n | latency s | images | image mode |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {kind} | {count} | {parse} | {usable} | {ade} | {fde} | {valid} | {latency} | {images} | {mode} |".format(
                run=row["run"],
                kind=row["kind"],
                count=_fmt(row["count"], 0),
                parse=_fmt(row["parse_rate"]),
                usable=_fmt(row["usable_points"]),
                ade=_fmt(row["ade"]),
                fde=_fmt(row["fde"]),
                valid=_fmt(row["valid_ade_count"], 0),
                latency=_fmt(row["latency"]),
                images=_fmt(row["images"], 2),
                mode=row["image_mode"],
            )
        )

    lora_all = next((row for row in rows if row["run"] == "lora all"), None)
    lora_front3 = next((row for row in rows if row["run"] == "lora front3"), None)
    mismatch = next((row for row in rows if row["run"] == "lora mismatch all"), None)
    prior_mean = next((row for row in rows if row["run"] == "prior:train_mean"), None)
    if lora_all and prior_mean and mismatch:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                (
                    f"- lora all vs train_mean prior ADE: {_fmt(lora_all['ade'])} vs {_fmt(prior_mean['ade'])}; "
                    f"lower is better."
                ),
                (
                    f"- mismatched images ADE: {_fmt(mismatch['ade'])}; if this is near the prior and worse than lora all, "
                    "the model is using current-scene visual input."
                ),
            ]
        )
    if lora_all and lora_front3:
        lines.append(
            f"- front3 ADE: {_fmt(lora_front3['ade'])} with {_fmt(lora_front3['images'], 2)} images, compared with lora all ADE {_fmt(lora_all['ade'])}."
        )

    out_path = args.out or suite_dir / "final_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote VLA suite summary: {out_path}")


if __name__ == "__main__":
    main()
