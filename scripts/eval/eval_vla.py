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

Single-GPU by default. Multi-GPU is launched separately via the shell
wrapper (passes ``--num-gpus 2`` and lets the Trainer / accelerate
inferences set ``CUDA_VISIBLE_DEVICES``); the python script itself runs
on whatever device PyTorch sees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.ablations import ABLATIONS, ablation_plan, transform_user_text
from drivevlm_lite.eval.impromptu_trajectory import (
    ade,
    fde,
    parse_planning_text,
    split_lateral_longitudinal_ade,
)


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
    parser.add_argument("--num-gpus", default=1, type=int,
        help="Accepted for interface uniformity; this script uses one GPU.")
    args = parser.parse_args()

    plan = ablation_plan(args.ablation)
    print(f"ablation={args.ablation}  text={plan.text}  image={plan.image}")

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

    rows = read_jsonl(args.val_file)
    available_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"val_file={args.val_file}  available={available_rows}  evaluating={len(rows)}")

    predictions: list[dict[str, Any]] = []
    parse_ok = 0
    parse_full6 = 0
    ade_values: list[float] = []
    fde_values: list[float] = []
    lon_values: list[float] = []
    lat_values: list[float] = []
    latencies: list[float] = []

    n_rows = len(rows)
    for idx, row in enumerate(tqdm(rows, desc="VLA eval")):
        question = transform_user_text(_user_text(row), args.ablation)
        gt_text = _assistant_text(row)
        gt_waypoints = parse_planning_text(gt_text)

        image_paths = [str(p) for p in row.get("images", [])]
        if plan.image == "mismatch":
            # Pair this row's text with a different scene's image so we can
            # tell whether the model reads the current frame at all.
            donor = rows[(idx + 1) % n_rows]
            load_paths = [str(p) for p in donor.get("images", [])]
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
        "count": n,
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
