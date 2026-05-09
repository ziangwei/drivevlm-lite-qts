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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.metrics import accuracy, exact_match


PIXELS_PER_VISUAL_TOKEN = 28 * 28


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def _question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    for message in row.get("messages", []):
        if message.get("role") == "user":
            question = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            answer = str(message.get("content", ""))
    return question, answer


def _messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _load_processor(model_name_or_path: str, min_pixels: int | None, max_pixels: int | None):
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if min_pixels is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = min_pixels
        if max_pixels is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels
    return processor


def _to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _grid_token_count(inputs: dict[str, Any]) -> int | None:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    grid_cpu = grid.detach().cpu()
    total = 0
    for row in grid_cpu:
        values = [int(value) for value in row.tolist()]
        if len(values) == 3:
            total += values[0] * values[1] * values[2]
    return total


def _eval_one_budget(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    out_dir: Path,
    max_new_tokens: int,
    label: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []
    input_token_counts: list[int] = []
    grid_token_counts: list[int] = []
    latencies: list[float] = []

    for row in tqdm(rows, desc=f"DriveLM visual budget {label}"):
        question, answer = _question_and_answer(row)
        images = _load_images(row.get("images", []))
        messages = _messages(question, images)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=images, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        grid_tokens = _grid_token_count(inputs)
        inputs = _to_device(inputs, device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_s = time.perf_counter() - start

        generated = generated[:, input_len:]
        prediction = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        pred_texts.append(prediction)
        gold_texts.append(answer)
        input_token_counts.append(input_len)
        if grid_tokens is not None:
            grid_token_counts.append(grid_tokens)
        latencies.append(latency_s)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "task": row.get("task"),
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "exact_match": exact_match(prediction, answer),
                "latency_s": latency_s,
                "input_tokens": input_len,
                "image_grid_tokens": grid_tokens,
                "images": row.get("images", []),
            }
        )

    write_jsonl(out_dir / "predictions.jsonl", predictions)
    metrics = {
        "label": label,
        "count": len(predictions),
        "exact_match": accuracy(pred_texts, gold_texts),
        "avg_latency_s": sum(latencies) / max(1, len(latencies)),
        "avg_input_tokens": sum(input_token_counts) / max(1, len(input_token_counts)),
        "avg_image_grid_tokens": (
            sum(grid_token_counts) / len(grid_token_counts) if grid_token_counts else None
        ),
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_summary_markdown(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# DriveLM Visual Budget Evaluation",
        "",
        "| label | count | EM | avg latency s | avg input tokens | avg image grid tokens | max pixels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        grid_tokens = item["avg_image_grid_tokens"]
        lines.append(
            "| {label} | {count} | {em:.3f} | {lat:.3f} | {inp:.1f} | {grid} | {max_pixels} |".format(
                label=item["label"],
                count=item["count"],
                em=float(item["exact_match"]),
                lat=float(item["avg_latency_s"]),
                inp=float(item["avg_input_tokens"]),
                grid="n/a" if grid_tokens is None else f"{float(grid_tokens):.1f}",
                max_pixels="default" if item["max_pixels"] is None else item["max_pixels"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", default="data/processed/drivelm_sft_val.jsonl", type=Path)
    parser.add_argument("--out-root", default="reports/e2_visual_budget", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--visual-token-budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--min-visual-tokens", default=64, type=int)
    parser.add_argument("--include-default", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    budget_specs: list[tuple[str, int | None, int | None]] = []
    if args.include_default:
        budget_specs.append(("default", None, None))
    for budget in args.visual_token_budgets:
        if budget <= 0:
            raise ValueError("visual token budgets must be positive.")
        min_tokens = min(args.min_visual_tokens, budget)
        budget_specs.append(
            (
                f"vtok_{budget}",
                min_tokens * PIXELS_PER_VISUAL_TOKEN,
                budget * PIXELS_PER_VISUAL_TOKEN,
            )
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for label, min_pixels, max_pixels in budget_specs:
        processor = _load_processor(args.model, min_pixels, max_pixels)
        metrics = _eval_one_budget(
            model=model,
            processor=processor,
            rows=rows,
            device=device,
            out_dir=args.out_root / label,
            max_new_tokens=args.max_new_tokens,
            label=label,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "input": str(args.input),
        "limit": args.limit,
        "metrics": all_metrics,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(args.out_root / "summary.md", all_metrics)
    print(f"Wrote visual budget summary: {args.out_root}")


if __name__ == "__main__":
    main()
