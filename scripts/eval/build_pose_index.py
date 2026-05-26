"""Build a tiny CAM_FRONT pose index for Stage 6.

The first cut of ``eval_offroad.py`` instantiated ``NuScenes(version='v1.0-trainval',
...)`` which deserialises every metadata table into Python dicts and needs
~8-15 GB of RAM — fine on a fat node, OOM-killed on a typical CPU node. For
Stage 6 we only need a tiny subset:

    filename(CAM_FRONT keyframe) -> (tx, ty, [w,x,y,z], location)

So this script streams the two big tables (``sample_data.json`` and
``ego_pose.json``) with ``ijson`` and loads the small ones (sample / scene /
log) with stdlib ``json``. Peak RAM stays under ~500 MB; the resulting cache is
a few MB. Run once; ``eval_offroad.py`` then reads only the cache.

Install ``ijson`` first if missing::

    pip install ijson --break-system-packages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import ijson
except ImportError:
    sys.exit("ijson missing. Install: pip install ijson --break-system-packages")


def _as_float_list(seq) -> list[float]:
    return [float(x) for x in seq]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out", required=True, type=Path,
        help="Where to write the cache JSON (under data/ — gitignored).")
    args = parser.parse_args()

    meta = args.nuscenes_root / args.version
    for name in ("sample_data.json", "ego_pose.json", "sample.json", "scene.json", "log.json"):
        if not (meta / name).is_file():
            sys.exit(f"missing metadata table: {meta / name}")

    # Pass 1 — stream sample_data.json, keep only CAM_FRONT keyframes.
    print(f"streaming {meta/'sample_data.json'} (large) ...")
    fname_to_tokens: dict[str, tuple[str, str]] = {}
    needed_eps: set[str] = set()
    needed_samples: set[str] = set()
    with (meta / "sample_data.json").open("rb") as handle:
        for rec in ijson.items(handle, "item"):
            fn = rec["filename"]
            if not fn.startswith("samples/CAM_FRONT/"):
                continue
            ep_tok = rec["ego_pose_token"]
            sm_tok = rec["sample_token"]
            fname_to_tokens[fn] = (ep_tok, sm_tok)
            needed_eps.add(ep_tok)
            needed_samples.add(sm_tok)
    print(f"  {len(fname_to_tokens)} CAM_FRONT keyframes")

    # Pass 2 — stream ego_pose.json, keep only what we need.
    print(f"streaming {meta/'ego_pose.json'} (large) ...")
    ep_to_pose: dict[str, dict] = {}
    with (meta / "ego_pose.json").open("rb") as handle:
        for rec in ijson.items(handle, "item"):
            tok = rec["token"]
            if tok in needed_eps:
                ep_to_pose[tok] = {
                    "translation": _as_float_list(rec["translation"]),
                    "rotation": _as_float_list(rec["rotation"]),
                }
    print(f"  {len(ep_to_pose)} ego poses kept")

    # Smaller tables — plain json.load.
    print("loading sample/scene/log tables (small) ...")
    with (meta / "sample.json").open("r", encoding="utf-8") as handle:
        sample_to_scene = {
            rec["token"]: rec["scene_token"]
            for rec in json.load(handle)
            if rec["token"] in needed_samples
        }
    with (meta / "scene.json").open("r", encoding="utf-8") as handle:
        scene_to_log = {rec["token"]: rec["log_token"] for rec in json.load(handle)}
    with (meta / "log.json").open("r", encoding="utf-8") as handle:
        log_to_location = {rec["token"]: rec["location"] for rec in json.load(handle)}

    # Join → final index.
    print("building combined index ...")
    out: dict[str, dict] = {}
    for fname, (ep_tok, sm_tok) in fname_to_tokens.items():
        pose = ep_to_pose.get(ep_tok)
        scene_tok = sample_to_scene.get(sm_tok)
        if pose is None or scene_tok is None:
            continue
        log_tok = scene_to_log.get(scene_tok)
        location = log_to_location.get(log_tok) if log_tok else None
        if location is None:
            continue
        out[fname] = {
            "tx": pose["translation"][0],
            "ty": pose["translation"][1],
            "rotation": pose["rotation"],  # [w, x, y, z]
            "location": location,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(out, handle)
    print(f"wrote {len(out)} entries to {args.out}")


if __name__ == "__main__":
    main()
