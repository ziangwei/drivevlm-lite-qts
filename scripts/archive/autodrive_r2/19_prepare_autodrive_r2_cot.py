from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.autodrive_r2 import convert_record, iter_json_records
from drivevlm_lite.data.jsonl import write_jsonl


def _split_rows(rows: list[dict], train_count: int, val_count: int, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val = shuffled[:val_count]
    train = shuffled[val_count : val_count + train_count]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed_vla_cot"), type=Path)
    parser.add_argument("--nuscenes-root", default=None, type=Path)
    parser.add_argument("--image-root", default=None, type=Path)
    parser.add_argument("--train-samples", default=1000, type=int)
    parser.add_argument("--val-samples", default=100, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--step-seconds", default=0.5, type=float)
    parser.add_argument("--answer-mode", choices=("cot", "direct", "original"), default="cot")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-no-images", action="store_true")
    args = parser.parse_args()

    converted: list[dict] = []
    skipped_no_trajectory = 0
    skipped_no_images = 0
    rows_with_missing_images = 0
    missing_image_count = 0

    for record in iter_json_records(args.input):
        row = convert_record(
            record,
            nuscenes_root=args.nuscenes_root,
            image_root=args.image_root,
            answer_mode=args.answer_mode,
            step_seconds=args.step_seconds,
            require_images=not args.allow_no_images,
        )
        if row is None:
            probe = convert_record(
                record,
                nuscenes_root=args.nuscenes_root,
                image_root=args.image_root,
                answer_mode=args.answer_mode,
                step_seconds=args.step_seconds,
                require_images=False,
            )
            if probe is None:
                skipped_no_trajectory += 1
            else:
                skipped_no_images += 1
            continue

        missing_for_row = [image for image in row.get("images", []) if not Path(image).exists()]
        if missing_for_row:
            rows_with_missing_images += 1
            missing_image_count += len(missing_for_row)
            if not args.allow_missing_images:
                continue
        converted.append(row)

    train_rows, val_rows = _split_rows(converted, args.train_samples, args.val_samples, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "autodrive_r2_vla_cot_train.jsonl"
    val_path = args.out_dir / "autodrive_r2_vla_cot_val.jsonl"
    n_train = write_jsonl(train_path, train_rows)
    n_val = write_jsonl(val_path, val_rows)

    summary = {
        "input": str(args.input),
        "out_dir": str(args.out_dir),
        "converted_rows": len(converted),
        "wrote_train": n_train,
        "wrote_val": n_val,
        "skipped_no_trajectory": skipped_no_trajectory,
        "skipped_no_images": skipped_no_images,
        "rows_with_missing_images": rows_with_missing_images,
        "missing_image_count": missing_image_count,
        "answer_mode": args.answer_mode,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# AutoDrive-R2 CoT Prepare",
        "",
        f"- input: {args.input}",
        f"- out_dir: {args.out_dir}",
        f"- converted_rows: {len(converted)}",
        f"- wrote_train: {n_train} -> {train_path}",
        f"- wrote_val: {n_val} -> {val_path}",
        f"- skipped_no_trajectory: {skipped_no_trajectory}",
        f"- skipped_no_images: {skipped_no_images}",
        f"- rows_with_missing_images: {rows_with_missing_images}",
        f"- missing_image_count: {missing_image_count}",
        f"- answer_mode: {args.answer_mode}",
        "",
    ]
    if val_rows:
        lines.extend(
            [
                "## Example",
                "",
                f"- sample_id: {val_rows[0].get('sample_id')}",
                f"- images: {val_rows[0].get('images', [])[:2]}",
                f"- answer: {val_rows[0]['messages'][1]['content'][:1000]}",
                "",
            ]
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote train: {train_path}")
    print(f"Wrote val: {val_path}")
    print(f"Wrote summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
