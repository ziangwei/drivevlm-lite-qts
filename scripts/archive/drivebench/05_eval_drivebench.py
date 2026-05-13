from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.image_loading import ImageLoader
from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.eval.metrics import exact_match, relaxed_exact_match, token_f1, yes_no_match


def _question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    for message in row.get("messages", []):
        if message.get("role") == "user":
            question = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            answer = str(message.get("content", ""))
    question = question or str(row.get("question", ""))
    answer = answer or str(row.get("answer", ""))
    return question, answer


def _messages(question: str, images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _load_processor(model_name_or_path: str, visual_token_budget: int | None):
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if visual_token_budget and visual_token_budget > 0:
        kwargs["min_pixels"] = min(64, visual_token_budget) * 28 * 28
        kwargs["max_pixels"] = visual_token_budget * 28 * 28

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and visual_token_budget and visual_token_budget > 0:
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = min(64, visual_token_budget) * 28 * 28
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = visual_token_budget * 28 * 28
    return processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", default=Path("data/processed/drivebench_eval.jsonl"), type=Path)
    parser.add_argument("--image-zip", default=None, type=Path)
    parser.add_argument("--zip-condition", default=None)
    parser.add_argument("--out", default=Path("reports/drivebench_eval"), type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--visual-token-budget", default=128, type=int)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText

    args.out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    available_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.limit > 0 and len(rows) < args.limit:
        print(
            "WARNING: requested limit is larger than the input JSONL. "
            f"requested_limit={args.limit} available_rows={available_rows}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = _load_processor(args.model, args.visual_token_budget)
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

    predictions: list[dict[str, Any]] = []
    strict_scores: list[float] = []
    relaxed_scores: list[float] = []
    f1_scores: list[float] = []
    yes_no_scores: list[float] = []
    latencies: list[float] = []

    with ImageLoader(args.image_zip, zip_condition=args.zip_condition) as image_loader:
        for row in tqdm(rows, desc="DriveBench eval"):
            question, answer = _question_and_answer(row)
            image_paths = [str(path) for path in row.get("images", [])]
            images = image_loader.load_many(image_paths)
            messages = _messages(question, images)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt")
            input_len = int(inputs["input_ids"].shape[1])
            inputs = _to_device(inputs, device)

            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            latency_s = time.perf_counter() - start

            generated = generated[:, input_len:]
            prediction = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            strict = exact_match(prediction, answer) if answer else None
            relaxed = relaxed_exact_match(prediction, answer) if answer else None
            f1 = token_f1(prediction, answer) if answer else None
            yes_no = yes_no_match(prediction, answer) if answer else None
            if strict is not None:
                strict_scores.append(strict)
            if relaxed is not None:
                relaxed_scores.append(relaxed)
            if f1 is not None:
                f1_scores.append(f1)
            if yes_no is not None:
                yes_no_scores.append(yes_no)
            latencies.append(latency_s)

            predictions.append(
                {
                    "sample_id": row.get("sample_id"),
                    "task": row.get("task"),
                    "question": question,
                    "answer": answer,
                    "prediction": prediction,
                    "strict_em": strict,
                    "relaxed_em": relaxed,
                    "token_f1": f1,
                    "yes_no_match": yes_no,
                    "latency_s": latency_s,
                    "images": image_paths,
                }
            )

    write_jsonl(args.out / "predictions.jsonl", predictions)
    metrics = {
        "count": len(predictions),
        "available_rows": available_rows,
        "model": args.model,
        "adapter": args.adapter,
        "input": str(args.input),
        "image_zip": str(args.image_zip) if args.image_zip else None,
        "zip_condition": args.zip_condition,
        "visual_token_budget": args.visual_token_budget,
        "strict_em": _mean(strict_scores),
        "relaxed_em": _mean(relaxed_scores),
        "token_f1": _mean(f1_scores),
        "yes_no_accuracy": _mean(yes_no_scores),
        "yes_no_count": len(yes_no_scores),
        "avg_latency_s": _mean(latencies),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = [
        "# DriveBench Evaluation",
        "",
        "| count | strict EM | relaxed EM | token F1 | yes/no acc | yes/no n | avg latency s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {count} | {strict:.3f} | {relaxed:.3f} | {f1:.3f} | {yn} | {ynn} | {lat:.3f} |".format(
            count=metrics["count"],
            strict=float(metrics["strict_em"] or 0.0),
            relaxed=float(metrics["relaxed_em"] or 0.0),
            f1=float(metrics["token_f1"] or 0.0),
            yn="n/a" if metrics["yes_no_accuracy"] is None else f"{float(metrics['yes_no_accuracy']):.3f}",
            ynn=metrics["yes_no_count"],
            lat=float(metrics["avg_latency_s"] or 0.0),
        ),
    ]
    (args.out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions: {args.out / 'predictions.jsonl'}")
    print(f"Wrote metrics: {args.out / 'metrics.json'}")
    print(f"Wrote summary: {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
