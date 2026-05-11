from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.nuscenes_cot import build_vla_cot_ablation_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired direct-vs-synthetic-CoT VLA trajectory JSONL files."
    )
    parser.add_argument("--train-input", default=Path("data/processed_vla_scene/nuscenes_vla_train.jsonl"), type=Path)
    parser.add_argument("--val-input", default=Path("data/processed_vla_scene/nuscenes_vla_val.jsonl"), type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed_vla_cot_ablation_500"), type=Path)
    parser.add_argument(
        "--nuscenes-root",
        default=Path("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes"),
        type=Path,
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--train-samples", default=500, type=int)
    parser.add_argument("--val-samples", default=100, type=int)
    parser.add_argument("--step-seconds", default=0.5, type=float)
    args = parser.parse_args()

    summary = build_vla_cot_ablation_files(
        train_input=args.train_input,
        val_input=args.val_input,
        out_dir=args.out_dir,
        nuscenes_root=args.nuscenes_root,
        version=args.version,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        step_seconds=args.step_seconds,
    )
    print(json.dumps(summary, indent=2))
    print(f"summary_md={args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
