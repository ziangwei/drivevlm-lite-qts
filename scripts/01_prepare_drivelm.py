from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.drivelm import iter_drivelm_samples
from drivevlm_lite.data.jsonl import write_jsonl
from drivevlm_lite.data.schema import to_sft_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--train-samples", default=10000, type=int)
    parser.add_argument("--val-samples", default=1000, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    samples = list(iter_drivelm_samples(args.qa_file, args.image_root))
    random.Random(args.seed).shuffle(samples)

    total = args.train_samples + args.val_samples
    selected = samples[:total] if total > 0 else samples
    val = selected[: args.val_samples]
    train = selected[args.val_samples :]

    train_rows = [dataclasses.asdict(to_sft_record(sample)) for sample in train]
    val_rows = [dataclasses.asdict(to_sft_record(sample)) for sample in val]

    train_path = args.out_dir / "drivelm_sft_train.jsonl"
    val_path = args.out_dir / "drivelm_sft_val.jsonl"
    n_train = write_jsonl(train_path, train_rows)
    n_val = write_jsonl(val_path, val_rows)

    print(f"Loaded DriveLM samples: {len(samples)}")
    print(f"Wrote train: {n_train} -> {train_path}")
    print(f"Wrote val: {n_val} -> {val_path}")


if __name__ == "__main__":
    main()
