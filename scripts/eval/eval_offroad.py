"""Stage 6 — off-road / drivable-area rate via the nuScenes HD map.

Adds a driving-semantic metric on top of ADE/FDE: what fraction of the
*predicted* trajectory leaves the map's drivable area? ADE rewards being close
to the logged path; off-road rate asks the orthogonal question of whether the
predicted path is even on the road, which the Stage 5 ego-status shortcut has
no reason to get right.

Memory note: this script intentionally does **not** use ``NuScenes(version=...)``.
That class deserialises every trainval table into Python dicts (~8-15 GB) and
gets OOM-killed on small CPU nodes. Instead, run ``build_pose_index.py`` once
to produce a small cache (~few MB), and this script reads only that cache and
the per-location ``NuScenesMap`` JSON files.

Pipeline, per row of a Stage 4/5 ``predictions.jsonl``:

1. Look up the CAM_FRONT image basename in the pose cache → global translation,
   rotation quaternion, map location.
2. Lift the ego-frame predicted (and GT) waypoints into the global frame.
3. Query ``NuScenesMap.layers_on_point`` for ``drivable_area``; a point is
   off-road when no drivable-area polygon contains it.

Outputs ``<out-dir>/offroad_metrics.json`` and ``<out-dir>/offroad_per_sample.jsonl``.
GT off-road rate is reported alongside the prediction's as a sanity floor
(logged ego trajectories should be ~0 % off-road).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.geometry import ego_to_global_path, yaw_from_quaternion


def _preflight(nuscenes_root: Path, pose_index: Path) -> list[str]:
    """Return a list of human-readable problems; empty means good to go."""
    problems: list[str] = []
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"nuscenes-devkit not importable: {exc}. "
                        "Install: pip install nuscenes-devkit --break-system-packages")
    expansion = nuscenes_root / "maps" / "expansion"
    if not expansion.is_dir() or not any(expansion.glob("*.json")):
        problems.append(
            f"map-expansion pack missing under {expansion}. "
            "Need maps/expansion/*.json (nuScenes-map-expansion-v1.3).")
    if not pose_index.is_file():
        problems.append(
            f"pose index missing: {pose_index}. "
            "Build it once with scripts/eval/build_pose_index.py.")
    return problems


def _as_pairs(raw: Any) -> list[tuple[float, float]]:
    if not raw:
        return []
    return [(float(p[0]), float(p[1])) for p in raw]


def _offroad_flags(global_pts, nmap) -> list[bool]:
    """True where a global point is *off* the drivable area."""
    flags: list[bool] = []
    for gx, gy in global_pts:
        layers = nmap.layers_on_point(gx, gy, layer_names=["drivable_area"])
        token = layers.get("drivable_area", "")
        flags.append(not token)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True, type=Path,
        help="A Stage 4/5 predictions.jsonl (use the 'full' ablation row).")
    parser.add_argument("--pose-index", required=True, type=Path,
        help="Output of scripts/eval/build_pose_index.py.")
    parser.add_argument("--nuscenes-root", required=True, type=Path,
        help="Only used to point NuScenesMap at maps/expansion/<location>.json.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--check-only", action="store_true",
        help="Run the preflight (devkit + map-expansion + pose index) and exit.")
    args = parser.parse_args()

    problems = _preflight(args.nuscenes_root, args.pose_index)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    print("preflight ok: devkit + map-expansion + pose index present")
    if args.check_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from nuscenes.map_expansion.map_api import NuScenesMap

    print(f"loading pose index from {args.pose_index}")
    with args.pose_index.open("r", encoding="utf-8") as handle:
        pose_idx: dict[str, dict] = json.load(handle)

    map_cache: dict[str, Any] = {}

    def get_map(location: str):
        if location not in map_cache:
            print(f"loading NuScenesMap '{location}'")
            map_cache[location] = NuScenesMap(
                dataroot=str(args.nuscenes_root), map_name=location)
        return map_cache[location]

    rows = read_jsonl(args.predictions)
    if args.limit > 0:
        rows = rows[: args.limit]

    per_sample: list[dict[str, Any]] = []
    pred_wp_off = pred_wp_total = 0
    gt_wp_off = gt_wp_total = 0
    pred_traj_off = gt_traj_off = 0
    scored = 0
    unresolved = 0

    for row in rows:
        image = row.get("image")
        if not image:
            unresolved += 1
            continue
        key = f"samples/CAM_FRONT/{Path(image).name}"
        entry = pose_idx.get(key)
        if entry is None:
            unresolved += 1
            continue

        tx, ty = float(entry["tx"]), float(entry["ty"])
        qw, qx, qy, qz = (float(v) for v in entry["rotation"])
        yaw = yaw_from_quaternion(qw, qx, qy, qz)
        nmap = get_map(entry["location"])

        pred_pairs = _as_pairs(row.get("pred_waypoints"))
        gt_pairs = _as_pairs(row.get("gt_waypoints"))
        if not pred_pairs or not gt_pairs:
            unresolved += 1
            continue

        pred_flags = _offroad_flags(ego_to_global_path(pred_pairs, (tx, ty), yaw), nmap)
        gt_flags = _offroad_flags(ego_to_global_path(gt_pairs, (tx, ty), yaw), nmap)

        pred_wp_off += sum(pred_flags)
        pred_wp_total += len(pred_flags)
        gt_wp_off += sum(gt_flags)
        gt_wp_total += len(gt_flags)
        pred_traj_off += 1 if any(pred_flags) else 0
        gt_traj_off += 1 if any(gt_flags) else 0
        scored += 1

        per_sample.append({
            "id": row.get("id"),
            "location": entry["location"],
            "pred_offroad_waypoints": sum(pred_flags),
            "pred_n_waypoints": len(pred_flags),
            "gt_offroad_waypoints": sum(gt_flags),
            "pred_any_offroad": any(pred_flags),
            "gt_any_offroad": any(gt_flags),
        })

    write_jsonl(args.out_dir / "offroad_per_sample.jsonl", per_sample)
    metrics = {
        "predictions": str(args.predictions),
        "pose_index": str(args.pose_index),
        "scored": scored,
        "unresolved": unresolved,
        "pred_offroad_rate_waypoint": (pred_wp_off / pred_wp_total) if pred_wp_total else None,
        "pred_offroad_rate_trajectory": (pred_traj_off / scored) if scored else None,
        "gt_offroad_rate_waypoint": (gt_wp_off / gt_wp_total) if gt_wp_total else None,
        "gt_offroad_rate_trajectory": (gt_traj_off / scored) if scored else None,
    }
    (args.out_dir / "offroad_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {args.out_dir / 'offroad_metrics.json'}")
    print(f"Wrote {args.out_dir / 'offroad_per_sample.jsonl'}")


if __name__ == "__main__":
    main()
