"""Evaluate the Impromptu-format VLA on nuScenes val (open-loop ADE / FDE).

For each row of the val JSONL the script:

1. Builds the user message (text + single front-camera image) from the
   row's existing ``messages`` field.
2. Generates an assistant response via ``model.generate``.
3. Parses the six ``[x, y]`` waypoints from the generated PLANNING block.
4. Parses the six ground-truth waypoints from the JSONL row's assistant
   message.
5. Computes ADE / FDE and a few breakdowns.

Outputs land under ``--out-dir``:

- ``predictions.jsonl`` — one row per sample with raw output, parsed
  waypoints, GT waypoints, and per-sample metrics.
- ``metrics.json``     — aggregate ADE / FDE / parse_rate / lat / long.

Subset selection (``--limit > 0``) is controlled by ``--sample-mode``:

- ``random``     — shuffle all rows with ``Random(seed)`` then take ``limit``.
                   Default; reproducible across runs for the same seed.
- ``prefix``     — legacy ``rows[:limit]``. val.jsonl is sorted by log+time,
                   so this collapses to one or two scenes — kept for back-compat
                   but not the recommended mode.
- ``stratified`` — group rows by log_id (parsed from CAM_FRONT filename) and
                   take ``limit / n_logs`` from each, shuffled per-log with
                   ``Random(seed)``. Useful for log-balanced debug runs.

Single-GPU by default. Multi-GPU is launched separately via the shell
wrapper (passes ``--num-gpus 2`` and lets the Trainer / accelerate
inferences set ``CUDA_VISIBLE_DEVICES``); the python script itself runs
on whatever device PyTorch sees.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.ablations import (
    ABLATIONS,
    ablation_plan,
    build_donor_index,
    parse_cam_front_path,
    transform_user_text,
)
from drivevlm_lite.eval.impromptu_trajectory import (
    ade,
    fde,
    parse_planning_text,
    split_lateral_longitudinal_ade,
)


SAMPLE_MODES = ("prefix", "random", "stratified")


def _user_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _assistant_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _row_first_image(row: dict[str, Any]) -> str:
    paths = row.get("images") or []
    return str(paths[0]) if paths else ""


def _select_subset(
    rows: list[dict[str, Any]],
    mode: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pick a subset of ``rows`` according to ``mode`` and ``limit``.

    ``limit <= 0`` or ``limit >= len(rows)`` returns ``rows`` unchanged.
    Returned rows are kept in their original val.jsonl order so the
    per-sample predictions remain easy to cross-reference with the file.
    """
    n = len(rows)
    if limit <= 0 or limit >= n:
        return list(rows)
    if mode == "prefix":
        return rows[:limit]
    rng = random.Random(seed)
    if mode == "random":
        idxs = list(range(n))
        rng.shuffle(idxs)
        picked = sorted(idxs[:limit])
        return [rows[i] for i in picked]
    if mode == "stratified":
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            path = _row_first_image(row)
            try:
                log, _ = parse_cam_front_path(path)
            except ValueError:
                log = "_unknown"
            groups[log].append(i)
        n_groups = max(1, len(groups))
        per_group = max(1, limit // n_groups)
        picked: list[int] = []
        for log in sorted(groups):
            group_idxs = list(groups[log])
            rng.shuffle(group_idxs)
            picked.extend(group_idxs[:per_group])
        # Top up rounding shortfall (rare) with a stable random tail.
        if len(picked) < limit:
            taken = set(picked)
            remaining = [i for i in range(n) if i not in taken]
            rng.shuffle(remaining)
            picked.extend(remaining[: limit - len(picked)])
        picked = sorted(picked[:limit])
        return [rows[i] for i in picked]
    raise ValueError(f"unknown sample-mode {mode!r}; expected one of {SAMPLE_MODES}")


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def _blacken(images: list[Image.Image]) -> list[Image.Image]:
    """Return all-zero (black) images of matching size — the vision-masked
    condition. Text (full ego status) is left intact by the caller."""
    return [Image.new("RGB", img.size, (0, 0, 0)) for img in images]


def _build_prompt(processor, question: str, images: list[Image.Image]) -> str:
    user_content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Path or name of the Qwen3-VL base model.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter checkpoint dir.")
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int, help="Max samples to eval (0 = all).")
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--ablation", default="full", choices=ABLATIONS,
        help="Stage 5 input corruption applied at inference time (default: full baseline).")
    parser.add_argument("--sample-mode", default="random", choices=SAMPLE_MODES,
        help="How to pick a subset when --limit > 0. random+seed=42 is the default "
             "for reproducibility; prefix is the legacy rows[:limit] behaviour.")
    parser.add_argument("--seed", default=42, type=int,
        help="RNG seed for --sample-mode and the image-swap donor selection.")
    parser.add_argument("--num-gpus", default=1, type=int,
        help="Accepted for interface uniformity; this script uses one GPU.")
    args = parser.parse_args()

    plan = ablation_plan(args.ablation)
    print(
        f"ablation={args.ablation}  text={plan.text}  image={plan.image}  "
        f"sample_mode={args.sample_mode}  seed={args.seed}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"loading model from {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel
        print(f"loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device)
    model.eval()

    all_rows = read_jsonl(args.val_file)
    available_rows = len(all_rows)
    rows = _select_subset(all_rows, args.sample_mode, args.limit, args.seed)
    print(
        f"val_file={args.val_file}  available={available_rows}  "
        f"evaluating={len(rows)}  mode={args.sample_mode}"
    )

    # Donor index for the two image-swap ablations is precomputed from the
    # post-subset row list so indices line up with the loop below.
    donor = None
    if plan.image in ("time_shifted", "true_mismatch"):
        donor_paths = [_row_first_image(r) for r in rows]
        donor = build_donor_index(donor_paths, seed=args.seed)
        n_missing = sum(1 for j in getattr(donor, plan.image) if j < 0)
        if n_missing:
            print(
                f"WARNING: {n_missing}/{len(rows)} rows have no eligible "
                f"{plan.image} donor in this subset; falling back to the "
                "row's own image and flagging donor_missing=true in "
                "predictions.jsonl."
            )

    predictions: list[dict[str, Any]] = []
    parse_ok = 0
    parse_full6 = 0
    ade_values: list[float] = []
    fde_values: list[float] = []
    lon_values: list[float] = []
    lat_values: list[float] = []
    latencies: list[float] = []

    for idx, row in enumerate(tqdm(rows, desc="VLA eval")):
        question = transform_user_text(_user_text(row), args.ablation)
        gt_text = _assistant_text(row)
        gt_waypoints = parse_planning_text(gt_text)

        image_paths = [str(p) for p in row.get("images", [])]
        donor_image: str | None = None
        donor_missing = False
        if plan.image == "time_shifted":
            j = donor.time_shifted[idx]
            if j < 0:
                load_paths = image_paths
                donor_missing = True
            else:
                load_paths = [str(p) for p in rows[j].get("images", [])]
                donor_image = load_paths[0] if load_paths else None
        elif plan.image == "true_mismatch":
            j = donor.true_mismatch[idx]
            if j < 0:
                load_paths = image_paths
                donor_missing = True
            else:
                load_paths = [str(p) for p in rows[j].get("images", [])]
                donor_image = load_paths[0] if load_paths else None
        else:
            load_paths = image_paths
        images = _load_images(load_paths)
        if plan.image == "black":
            images = _blacken(images)
        prompt_text = _build_prompt(processor, question, images)
        inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        inputs = _to_device(inputs, device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        latency = time.perf_counter() - start

        new_tokens = generated[:, input_len:]
        prediction = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        pred_waypoints = parse_planning_text(prediction)
        per_sample: dict[str, Any] = {
            "id": row.get("id", row.get("sample_id")),
            "image": image_paths[0] if image_paths else None,
            "prediction": prediction,
            "pred_waypoints": pred_waypoints,
            "gt_waypoints": gt_waypoints,
            "latency_s": latency,
        }
        if donor_image is not None:
            per_sample["donor_image"] = donor_image
        if donor_missing:
            per_sample["donor_missing"] = True

        if pred_waypoints and gt_waypoints:
            parse_ok += 1
            if len(pred_waypoints) >= 6:
                parse_full6 += 1
            sample_ade = ade(pred_waypoints, gt_waypoints)
            sample_fde = fde(pred_waypoints, gt_waypoints)
            sample_lon, sample_lat = split_lateral_longitudinal_ade(pred_waypoints, gt_waypoints)
            ade_values.append(sample_ade)
            fde_values.append(sample_fde)
            lon_values.append(sample_lon)
            lat_values.append(sample_lat)
            per_sample.update({
                "ade": sample_ade,
                "fde": sample_fde,
                "lon_ade": sample_lon,
                "lat_ade": sample_lat,
            })
        else:
            per_sample.update({"ade": None, "fde": None, "lon_ade": None, "lat_ade": None})

        latencies.append(latency)
        predictions.append(per_sample)

    write_jsonl(args.out_dir / "predictions.jsonl", predictions)

    n = len(predictions)
    metrics = {
        "val_file": str(args.val_file),
        "model": args.model,
        "adapter": args.adapter,
        "ablation": args.ablation,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "count": n,
        "available_rows": available_rows,
        "parse_rate": parse_ok / max(1, n),
        "parse_full6_rate": parse_full6 / max(1, n),
        "ade_mean": (sum(ade_values) / len(ade_values)) if ade_values else None,
        "fde_mean": (sum(fde_values) / len(fde_values)) if fde_values else None,
        "lon_ade_mean": (sum(lon_values) / len(lon_values)) if lon_values else None,
        "lat_ade_mean": (sum(lat_values) / len(lat_values)) if lat_values else None,
        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else None,
        "max_new_tokens": args.max_new_tokens,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote predictions to {args.out_dir / 'predictions.jsonl'}")
    print(f"Wrote metrics to {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
