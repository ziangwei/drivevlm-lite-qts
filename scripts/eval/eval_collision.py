"""Stage 7 — open-loop collision rate against nuScenes GT agent boxes.

Driving-credibility metric complementary to Stage 6 off-road: off-road asks
"is the path on the road?", collision asks "does the path drive through any
other agent?". Together they cover both static (road) and dynamic (other
vehicles, pedestrians) constraints.

Per row of a predictions.jsonl:

1. Look up the CAM_FRONT basename in the pose index → ego global pose at t=0.
2. Look up the same basename in the collision index → future-keyframe agent
   bounding boxes for t = 0.5 s ... 3.0 s.
3. Lift each predicted (and GT) waypoint into the global frame.
4. At each future timestep i, test the waypoint against every agent's 2-D
   rotated bbox; record a hit if any contains it.

Reports the standard open-loop collision rate per waypoint and per trajectory,
for both prediction and GT. GT collision rate is reported as the sanity floor
(should be ~0 % — the logged ego trajectory by definition does not collide).

Limitation (standard caveat): the future agents follow their *logged* GT
trajectories, not their reaction to the ego's predicted path. That is the
"open-loop" qualifier in the metric name; closed-loop would require a
simulator with reactive agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.bbox import point_in_rotated_bbox, quick_radius_overlap
from drivevlm_lite.eval.geometry import ego_to_global_path, yaw_from_quaternion


def _preflight(pose_index: Path, collision_index: Path) -> list[str]:
    problems: list[str] = []
    if not pose_index.is_file():
        problems.append(f"pose index missing: {pose_index}. Build with build_pose_index.py.")
    if not collision_index.is_file():
        problems.append(
            f"collision index missing: {collision_index}. "
            "Build with build_collision_index.py.")
    return problems


def _as_pairs(raw: Any) -> list[tuple[float, float]]:
    if not raw:
        return []
    return [(float(p[0]), float(p[1])) for p in raw]


def _waypoint_hits_any_agent(point, agents) -> bool:
    px, py = point
    for agent in agents:
        cx, cy = agent["x"], agent["y"]
        l, w = agent["l"], agent["w"]
        if not quick_radius_overlap((px, py), (cx, cy), l, w):
            continue
        if point_in_rotated_bbox((px, py), (cx, cy), l, w, agent["yaw"]):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True, type=Path,
        help="A Stage 4/5 predictions.jsonl.")
    parser.add_argument("--pose-index", required=True, type=Path)
    parser.add_argument("--collision-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    problems = _preflight(args.pose_index, args.collision_index)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    print("preflight ok: pose index + collision index present")
    if args.check_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.pose_index.open("r", encoding="utf-8") as h:
        pose_idx: dict[str, dict] = json.load(h)
    with args.collision_index.open("r", encoding="utf-8") as h:
        col_idx: dict[str, dict] = json.load(h)

    rows = read_jsonl(args.predictions)
    if args.limit > 0:
        rows = rows[: args.limit]

    per_sample: list[dict[str, Any]] = []
    pred_wp_hit = pred_wp_total = 0
    gt_wp_hit = gt_wp_total = 0
    pred_traj_hit = gt_traj_hit = 0
    scored = unresolved = 0

    for row in rows:
        image = row.get("image")
        if not image:
            unresolved += 1
            continue
        key = f"samples/CAM_FRONT/{Path(image).name}"
        pose = pose_idx.get(key)
        col = col_idx.get(key)
        if pose is None or col is None:
            unresolved += 1
            continue

        tx, ty = float(pose["tx"]), float(pose["ty"])
        qw, qx, qy, qz = (float(v) for v in pose["rotation"])
        yaw = yaw_from_quaternion(qw, qx, qy, qz)

        pred_pairs = _as_pairs(row.get("pred_waypoints"))
        gt_pairs = _as_pairs(row.get("gt_waypoints"))
        if not pred_pairs or not gt_pairs:
            unresolved += 1
            continue

        pred_global = ego_to_global_path(pred_pairs, (tx, ty), yaw)
        gt_global = ego_to_global_path(gt_pairs, (tx, ty), yaw)
        futures = col["future_samples"]

        pred_flags: list[bool] = []
        gt_flags: list[bool] = []
        n_steps = min(len(pred_global), len(gt_global), len(futures))
        for i in range(n_steps):
            agents = futures[i]["agents"]
            pred_flags.append(_waypoint_hits_any_agent(pred_global[i], agents))
            gt_flags.append(_waypoint_hits_any_agent(gt_global[i], agents))

        pred_wp_hit += sum(pred_flags)
        pred_wp_total += len(pred_flags)
        gt_wp_hit += sum(gt_flags)
        gt_wp_total += len(gt_flags)
        pred_traj_hit += 1 if any(pred_flags) else 0
        gt_traj_hit += 1 if any(gt_flags) else 0
        scored += 1

        per_sample.append({
            "id": row.get("id"),
            "n_future_steps": n_steps,
            "pred_collisions": sum(pred_flags),
            "gt_collisions": sum(gt_flags),
            "pred_any_collision": any(pred_flags),
            "gt_any_collision": any(gt_flags),
        })

    write_jsonl(args.out_dir / "collision_per_sample.jsonl", per_sample)
    metrics = {
        "predictions": str(args.predictions),
        "pose_index": str(args.pose_index),
        "collision_index": str(args.collision_index),
        "scored": scored,
        "unresolved": unresolved,
        "pred_collision_rate_waypoint": (pred_wp_hit / pred_wp_total) if pred_wp_total else None,
        "pred_collision_rate_trajectory": (pred_traj_hit / scored) if scored else None,
        "gt_collision_rate_waypoint": (gt_wp_hit / gt_wp_total) if gt_wp_total else None,
        "gt_collision_rate_trajectory": (gt_traj_hit / scored) if scored else None,
    }
    (args.out_dir / "collision_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {args.out_dir / 'collision_metrics.json'}")
    print(f"Wrote {args.out_dir / 'collision_per_sample.jsonl'}")


if __name__ == "__main__":
    main()
