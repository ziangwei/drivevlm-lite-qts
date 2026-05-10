from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import write_jsonl
from drivevlm_lite.data.nuscenes_trajectory import DEFAULT_CAMERAS, build_trajectory_samples_with_stats, to_sft_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out-dir", default=Path("data/processed_vla"), type=Path)
    parser.add_argument("--train-samples", default=1000, type=int)
    parser.add_argument("--val-samples", default=100, type=int)
    parser.add_argument("--future-steps", default=6, type=int)
    parser.add_argument("--step-seconds", default=0.5, type=float)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--max-missing-images", default=0, type=int)
    parser.add_argument("--candidate-multiplier", default=20, type=int)
    args = parser.parse_args()

    total = args.train_samples + args.val_samples
    candidate_limit = total * max(1, args.candidate_multiplier) if total > 0 else 0
    result = build_trajectory_samples_with_stats(
        args.nuscenes_root,
        version=args.version,
        future_steps=args.future_steps,
        cameras=DEFAULT_CAMERAS,
        max_missing_images=args.max_missing_images,
        candidate_limit=candidate_limit,
        seed=args.seed,
    )
    samples = result.samples

    selected = samples[:total] if total > 0 else samples
    val = selected[: args.val_samples]
    train = selected[args.val_samples :]

    train_rows = [to_sft_row(sample, step_seconds=args.step_seconds) for sample in train]
    val_rows = [to_sft_row(sample, step_seconds=args.step_seconds) for sample in val]

    train_path = args.out_dir / "nuscenes_vla_train.jsonl"
    val_path = args.out_dir / "nuscenes_vla_val.jsonl"
    n_train = write_jsonl(train_path, train_rows)
    n_val = write_jsonl(val_path, val_rows)

    print(f"nuscenes_root={args.nuscenes_root}")
    print(f"version={args.version}")
    print(f"loaded_valid_trajectory_samples={len(samples)}")
    print(f"candidate_multiplier={args.candidate_multiplier}")
    print(f"candidate_limit={candidate_limit}")
    for key, value in sorted(result.stats.items()):
        print(f"stat_{key}={value}")
    print(f"future_steps={args.future_steps}")
    print(f"cameras={','.join(DEFAULT_CAMERAS)}")
    print(f"wrote_train={n_train} -> {train_path}")
    print(f"wrote_val={n_val} -> {val_path}")
    if val_rows:
        print("example_answer=" + val_rows[0]["messages"][1]["content"])
        print("example_images=" + str(len(val_rows[0]["images"])))


if __name__ == "__main__":
    main()
