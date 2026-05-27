"""Build a per-sample collision index for the open-loop collision metric.

For every CAM_FRONT keyframe present in the input val/predictions file:

- Resolve the nuScenes ``sample_token`` (current keyframe at t=0).
- Walk ``sample.next`` 6 times to collect the future keyframes at
  t = 0.5 s, 1.0 s, ..., 3.0 s (nuScenes keyframes are at 2 Hz, matching the
  trajectory's 6-waypoint horizon).
- For each future keyframe, list the GT bounding boxes of every other agent
  in the safety-relevant categories ``vehicle.*`` and ``human.*``: BEV centre
  ``(x, y)``, ``(length, width)`` in metres, and yaw in radians.

The result is a small JSON file consumed by ``eval_collision.py``. The
streaming approach (``ijson`` on ``sample_data.json`` and ``sample_annotation.json``)
keeps peak RAM under ~500 MB on the trainval tables — the same reason the
Stage 6 pose index was built this way rather than via the ``NuScenes`` class.

Install ``ijson`` once::

    pip install ijson --break-system-packages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import ijson
except ImportError:
    sys.exit("ijson missing. Install: pip install ijson --break-system-packages")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl
from drivevlm_lite.eval.geometry import yaw_from_quaternion


SAFETY_PREFIXES = ("vehicle.", "human.")
HORIZON_STEPS = 6           # 6 future keyframes => t = 0.5 .. 3.0 s
STEP_SECONDS = 0.5


def _key(image_path: str) -> str:
    return f"samples/CAM_FRONT/{Path(image_path).name}"


def _collect_needed_filenames(val_file: Path) -> set[str]:
    rows = read_jsonl(val_file)
    out: set[str] = set()
    for row in rows:
        # Works for val.jsonl (uses 'images' list) and predictions.jsonl (uses 'image').
        if "image" in row and row["image"]:
            out.add(_key(row["image"]))
        for p in row.get("images", []) or []:
            out.add(_key(p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-file", required=True, type=Path,
        help="JSONL with CAM_FRONT image paths (val.jsonl or any predictions.jsonl).")
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    meta = args.nuscenes_root / args.version
    for name in (
        "sample_data.json", "sample.json", "sample_annotation.json",
        "category.json", "instance.json",
    ):
        if not (meta / name).is_file():
            sys.exit(f"missing metadata table: {meta / name}")

    needed_fnames = _collect_needed_filenames(args.val_file)
    print(f"need {len(needed_fnames)} CAM_FRONT keyframes from {args.val_file}")

    # Pass 1 — stream sample_data.json, build fname -> sample_token.
    print(f"streaming {meta/'sample_data.json'} ...")
    fname_to_sample: dict[str, str] = {}
    with (meta / "sample_data.json").open("rb") as handle:
        for rec in ijson.items(handle, "item"):
            fn = rec.get("filename")
            if fn in needed_fnames:
                fname_to_sample[fn] = rec["sample_token"]
    print(f"  resolved {len(fname_to_sample)} fname -> sample_token")
    if not fname_to_sample:
        sys.exit("no CAM_FRONT filenames matched in sample_data.json; bad val file?")

    # Pass 2 — load sample.json (small). Build sample_token -> next.
    print("loading sample.json ...")
    sample_next: dict[str, str] = {}
    with (meta / "sample.json").open("r", encoding="utf-8") as handle:
        for rec in json.load(handle):
            sample_next[rec["token"]] = rec.get("next") or ""

    # Walk the next chain for each needed sample, collect all sample_tokens we
    # need annotations for (the present keyframe itself + 6 future ones).
    fname_to_future: dict[str, list[str]] = {}
    needed_samples: set[str] = set()
    for fn, sm in fname_to_sample.items():
        chain: list[str] = []
        cur = sm
        for _ in range(HORIZON_STEPS):
            nxt = sample_next.get(cur, "")
            if not nxt:
                break
            chain.append(nxt)
            cur = nxt
        fname_to_future[fn] = chain
        needed_samples.update(chain)
    print(f"  needing annotations for {len(needed_samples)} future samples")

    # Pass 3 — load category.json + instance.json.
    # IMPORTANT (nuScenes schema): sample_annotation has NO category_token; it
    # has instance_token. The category lives on the instance. So we must take
    # the extra hop sample_annotation.instance_token -> instance.category_token
    # -> category.name. An earlier version of this script went straight from
    # sample_annotation to category and filtered out every record.
    with (meta / "category.json").open("r", encoding="utf-8") as handle:
        cat_name = {rec["token"]: rec["name"] for rec in json.load(handle)}
    with (meta / "instance.json").open("r", encoding="utf-8") as handle:
        instance_to_cat = {rec["token"]: rec["category_token"] for rec in json.load(handle)}
    print(f"loaded {len(cat_name)} categories, {len(instance_to_cat)} instances")

    # Pass 4 — stream sample_annotation.json. Keep only annotations in our
    # needed sample set whose (instance -> category) is safety-relevant.
    print(f"streaming {meta/'sample_annotation.json'} (large) ...")
    by_sample: dict[str, list[dict[str, Any]]] = {s: [] for s in needed_samples}
    seen = matched_sample = kept = 0
    with (meta / "sample_annotation.json").open("rb") as handle:
        for rec in ijson.items(handle, "item"):
            seen += 1
            stoken = rec.get("sample_token")
            if stoken not in by_sample:
                continue
            matched_sample += 1
            cat_tok = instance_to_cat.get(rec.get("instance_token", ""), "")
            cname = cat_name.get(cat_tok, "")
            if not cname.startswith(SAFETY_PREFIXES):
                continue
            tx, ty = float(rec["translation"][0]), float(rec["translation"][1])
            sw, sl = float(rec["size"][0]), float(rec["size"][1])
            qw, qx, qy, qz = (float(v) for v in rec["rotation"])
            yaw = yaw_from_quaternion(qw, qx, qy, qz)
            by_sample[stoken].append({
                "cat": cname, "x": tx, "y": ty,
                "l": sl, "w": sw, "yaw": yaw,
            })
            kept += 1
    print(f"  streamed {seen} annotations total, "
          f"{matched_sample} in needed samples, "
          f"{kept} safety-relevant kept")

    # Build final per-fname structure.
    out: dict[str, dict[str, Any]] = {}
    for fn, future_samples in fname_to_future.items():
        out[fn] = {
            "sample_token": fname_to_sample[fn],
            "future_samples": [
                {"dt": (i + 1) * STEP_SECONDS, "agents": by_sample.get(sm, [])}
                for i, sm in enumerate(future_samples)
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(out, handle)
    print(f"wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
