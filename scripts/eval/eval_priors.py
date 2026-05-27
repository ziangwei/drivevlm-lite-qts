"""Three-tier prior baselines: zero / train-mean / train-median.

A locked v1 spec target — gives a clean lower bound for "how much does the
model actually beat a no-model baseline?" Each prior is a fixed 6-waypoint
trajectory in the ego frame, scored against val GT with the same ADE / FDE /
lon-ADE / lat-ADE pipeline as Stage 4. No GPU, no images: this is pure
JSONL arithmetic.

The three priors:

- ``zero``         all six waypoints are ``(0, 0)`` (the ego does not move).
- ``train_mean``   per-timestep mean of train.jsonl GT waypoints.
- ``train_median`` per-timestep median.

``train_mean`` and ``train_median`` are essentially the "always predict the
average / median future of the training distribution" baseline. Most nuScenes
train samples are straight driving, so this will look like "go forward at the
training-average speed" — a reasonable but weak straw man.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.eval.impromptu_trajectory import (
    ade,
    fde,
    parse_planning_text,
    split_lateral_longitudinal_ade,
)


HORIZON = 6


def _gt_from_row(row: dict[str, Any]) -> list[tuple[float, float]]:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            pairs = parse_planning_text(str(msg.get("content", "")))
            if len(pairs) >= HORIZON:
                return pairs[:HORIZON]
            return pairs
    return []


def _collect_gt(jsonl_path: Path) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for row in read_jsonl(jsonl_path):
        gt = _gt_from_row(row)
        if len(gt) == HORIZON:
            out.append(gt)
    return out


def _mean_trajectory(trajs: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(HORIZON):
        xs = [t[i][0] for t in trajs]
        ys = [t[i][1] for t in trajs]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def _median_trajectory(trajs: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(HORIZON):
        xs = sorted(t[i][0] for t in trajs)
        ys = sorted(t[i][1] for t in trajs)
        out.append((statistics.median(xs), statistics.median(ys)))
    return out


def _score(prior: list[tuple[float, float]], val_gts: list[list[tuple[float, float]]]) -> dict[str, float]:
    ades, fdes, lons, lats = [], [], [], []
    for gt in val_gts:
        if len(gt) < HORIZON:
            continue
        ades.append(ade(prior, gt))
        fdes.append(fde(prior, gt))
        lon, lat = split_lateral_longitudinal_ade(prior, gt)
        lons.append(lon)
        lats.append(lat)
    return {
        "count": len(ades),
        "ade_mean": sum(ades) / len(ades),
        "fde_mean": sum(fdes) / len(fdes),
        "lon_ade_mean": sum(lons) / len(lons),
        "lat_ade_mean": sum(lats) / len(lats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-file", required=True, type=Path,
        help="Training JSONL (used to compute the mean / median priors).")
    parser.add_argument("--val-file", required=True, type=Path,
        help="Validation JSONL to score priors against.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit-val", default=0, type=int,
        help="Optional: score only the first N val rows (for matched comparison).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading train GT from {args.train_file}")
    train_gts = _collect_gt(args.train_file)
    print(f"  {len(train_gts)} train trajectories with full {HORIZON} waypoints")
    if not train_gts:
        sys.exit("no train GTs — bad input?")

    print(f"loading val GT from {args.val_file}")
    val_gts = _collect_gt(args.val_file)
    if args.limit_val > 0:
        val_gts = val_gts[: args.limit_val]
    print(f"  scoring against {len(val_gts)} val trajectories")

    priors = {
        "zero": [(0.0, 0.0)] * HORIZON,
        "train_mean": _mean_trajectory(train_gts),
        "train_median": _median_trajectory(train_gts),
    }

    results = {name: _score(traj, val_gts) | {"prior": traj}
               for name, traj in priors.items()}
    payload = {
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "horizon": HORIZON,
        "n_train_trajectories": len(train_gts),
        "n_val_trajectories": len(val_gts),
        "priors": results,
    }
    (args.out_dir / "prior_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "prior"}
         for k, v in results.items()},
        indent=2,
    ))
    print(f"\nWrote {args.out_dir / 'prior_metrics.json'}")


if __name__ == "__main__":
    main()
