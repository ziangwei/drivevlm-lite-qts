"""Convert Impromptu-VLA's nuScenes JSON files into project-format JSONL.

The script rewrites image paths so they point at the local nuScenes
keyframe tree, optionally checks every image exists, and writes the
result to ``data/processed_vla_impromptu/{train,val}.jsonl`` (or wherever
``--out-dir`` points).

CPU-only job. Pure JSON manipulation plus a filesystem stat per image; on
a single core the full 28 130 + 6 020 sample conversion takes about a
minute including image-existence checks.

The ``--num-gpus`` flag is accepted but ignored; it exists so the family
of project scripts has a uniform interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.impromptu_adapter import (
    iter_rewritten,
    load_impromptu_records,
    write_records_jsonl,
)


def _convert_one(
    impromptu_json,
    out_jsonl,
    nuscenes_root,
    *,
    check_images,
    drop_missing,
    limit,
):
    records = load_impromptu_records(impromptu_json)
    pairs = iter_rewritten(
        records,
        nuscenes_root,
        require_image=check_images,
        check_existence=check_images,
        limit=limit,
    )
    stats = write_records_jsonl(out_jsonl, pairs, drop_missing=drop_missing)
    return {
        "input": str(impromptu_json),
        "output": str(out_jsonl),
        "total": stats.total,
        "written": stats.written,
        "missing_images": stats.missing_images,
        "skipped_no_image": stats.skipped_no_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impromptu-root", default=Path("data/external/impromptu_vla"), type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path,
        help="Local path that contains the 'samples/' subdirectory of nuScenes keyframes.")
    parser.add_argument("--out-dir", default=Path("data/processed_vla_impromptu"), type=Path)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--keep-missing", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1,
        help="Accepted for interface uniformity; this script is CPU-only.")
    args = parser.parse_args()

    if args.num_gpus != 1:
        print(f"note: --num-gpus={args.num_gpus} ignored; this script is CPU-only.")

    summary = {
        "nuscenes_root": str(args.nuscenes_root),
        "out_dir": str(args.out_dir),
        "image_check": not args.skip_image_check,
        "drop_missing": not args.keep_missing,
        "splits": {},
    }

    plan = [
        ("train", args.impromptu_root / "nuscenes_train.json", args.out_dir / "train.jsonl", args.limit_train),
        ("val", args.impromptu_root / "nuscenes_test.json", args.out_dir / "val.jsonl", args.limit_val),
    ]
    for name, in_path, out_path, limit in plan:
        if not in_path.is_file():
            print(f"WARNING: missing input {in_path}; skipping {name} split")
            continue
        result = _convert_one(
            impromptu_json=in_path,
            out_jsonl=out_path,
            nuscenes_root=args.nuscenes_root,
            check_images=not args.skip_image_check,
            drop_missing=not args.keep_missing,
            limit=limit,
        )
        summary["splits"][name] = result
        print(f"[{name}] {result}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote summary: {args.out_dir / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
