"""Stage 5 — post-hoc analysis over finished ablation runs (no GPU needed).

Reads every ``<root>/<ablation>/{metrics.json,predictions.jsonl}`` produced by
``run_ablation_matrix.sh`` (and/or the plain Stage 4 ``eval_vla`` output) and
emits:

1. ``<root>/ablation_matrix.csv`` — one row per ablation with parse rate,
   ADE / FDE, lateral / longitudinal split, ADE percentiles, and latency.
2. ``<root>/maneuver_breakdown.csv`` — per-maneuver ADE for the ``full`` row
   (straight / left / right / stop), classified from the ground-truth
   trajectory.
3. ``<root>/ablation_summary.md`` — both tables rendered for the report.

Run::

    PYTHONPATH=src python scripts/eval/analyze_ablations.py \
        --root reports/ablation_matrix_v1_500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.eval.ablations import ABLATIONS, classify_maneuver, percentiles

_MATRIX_FIELDS = [
    "ablation",
    "count",
    "parse_rate",
    "ade_mean",
    "fde_mean",
    "lon_ade_mean",
    "lat_ade_mean",
    "p25",
    "p50",
    "p75",
    "p95",
    "latency_mean_s",
]
_MANEUVERS = ("straight", "left", "right", "stop", "unknown")


def _ade_values(predictions: list[dict[str, Any]]) -> list[float]:
    return [r["ade"] for r in predictions if r.get("ade") is not None]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _discover_runs(root: Path) -> list[tuple[str, Path]]:
    """Return (label, dir) for every subdir containing a metrics.json, ordered
    by the canonical ablation order then alphabetically."""
    runs: list[tuple[str, Path]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if (sub / "metrics.json").is_file():
            runs.append((sub.name, sub))
    # Also accept a metrics.json directly under root (a single Stage 4 run).
    if (root / "metrics.json").is_file():
        runs.append((root.name, root))

    def order(item: tuple[str, Path]) -> tuple[int, str]:
        label = item[0]
        rank = ABLATIONS.index(label) if label in ABLATIONS else len(ABLATIONS)
        return rank, label

    return sorted(runs, key=order)


def _matrix_row(label: str, run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    pred_path = run_dir / "predictions.jsonl"
    pct: dict[str, float] = {}
    if pred_path.is_file():
        pct = percentiles(_ade_values(read_jsonl(pred_path)))
    return {
        "ablation": metrics.get("ablation", label),
        "count": metrics.get("count"),
        "parse_rate": metrics.get("parse_rate"),
        "ade_mean": metrics.get("ade_mean"),
        "fde_mean": metrics.get("fde_mean"),
        "lon_ade_mean": metrics.get("lon_ade_mean"),
        "lat_ade_mean": metrics.get("lat_ade_mean"),
        "p25": pct.get("p25"),
        "p50": pct.get("p50"),
        "p75": pct.get("p75"),
        "p95": pct.get("p95"),
        "latency_mean_s": metrics.get("latency_mean_s"),
    }


def _maneuver_breakdown(run_dir: Path) -> list[dict[str, Any]]:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.is_file():
        return []
    buckets: dict[str, list[float]] = {m: [] for m in _MANEUVERS}
    for row in read_jsonl(pred_path):
        gt = row.get("gt_waypoints") or []
        gt_pairs = [(float(x), float(y)) for x, y in gt]
        bucket = classify_maneuver(gt_pairs)
        if row.get("ade") is not None:
            buckets[bucket].append(row["ade"])
    out: list[dict[str, Any]] = []
    for m in _MANEUVERS:
        vals = buckets[m]
        if not vals:
            continue
        out.append({
            "maneuver": m,
            "n": len(vals),
            "ade_mean": sum(vals) / len(vals),
        })
    return out


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path,
        help="Directory holding per-ablation subdirs (or a single eval run).")
    parser.add_argument("--maneuver-from", default="full",
        help="Which ablation's predictions to use for the maneuver breakdown.")
    args = parser.parse_args()

    runs = _discover_runs(args.root)
    if not runs:
        raise SystemExit(f"no metrics.json found under {args.root}")

    matrix = [_matrix_row(label, run_dir) for label, run_dir in runs]
    _write_csv(args.root / "ablation_matrix.csv", _MATRIX_FIELDS, matrix)

    run_lookup = {label: run_dir for label, run_dir in runs}
    maneuver_dir = run_lookup.get(args.maneuver_from) or runs[0][1]
    maneuver = _maneuver_breakdown(maneuver_dir)
    if maneuver:
        _write_csv(args.root / "maneuver_breakdown.csv",
                   ["maneuver", "n", "ade_mean"], maneuver)

    matrix_md = _md_table(
        ["ablation", "n", "parse", "ADE", "FDE", "lonADE", "latADE", "p50", "p95", "lat_s"],
        [[
            _fmt(r["ablation"]), _fmt(r["count"]), _fmt(r["parse_rate"], 3),
            _fmt(r["ade_mean"]), _fmt(r["fde_mean"]),
            _fmt(r["lon_ade_mean"]), _fmt(r["lat_ade_mean"]),
            _fmt(r["p50"]), _fmt(r["p95"]), _fmt(r["latency_mean_s"], 2),
        ] for r in matrix],
    )
    parts = ["# Stage 5 ablation matrix", "", matrix_md, ""]
    if maneuver:
        parts += [
            f"## Maneuver breakdown ({args.maneuver_from} row)", "",
            _md_table(
                ["maneuver", "n", "ADE"],
                [[m["maneuver"], str(m["n"]), _fmt(m["ade_mean"])] for m in maneuver],
            ),
            "",
        ]
    (args.root / "ablation_summary.md").write_text("\n".join(parts), encoding="utf-8")

    print("\n".join(parts))
    print(f"\nWrote {args.root / 'ablation_matrix.csv'}")
    if maneuver:
        print(f"Wrote {args.root / 'maneuver_breakdown.csv'}")
    print(f"Wrote {args.root / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
