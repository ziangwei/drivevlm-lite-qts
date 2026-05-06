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


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def _question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages", [])
    question = ""
    answer = ""
    for message in messages:
        if message.get("role") == "user":
            question = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            answer = str(message.get("content", ""))
    return question, answer


def _messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--input", default="data/processed/drivelm_sft_val.jsonl", type=Path)
    parser.add_argument("--out", default="reports/e0_drivelm_zero_shot", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    predictions: list[dict[str, Any]] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []

    args.out.mkdir(parents=True, exist_ok=True)
    pred_path = args.out / "predictions.jsonl"
    metrics_path = args.out / "metrics.json"

    for row in tqdm(rows, desc="DriveLM zero-shot"):
        question, answer = _question_and_answer(row)
        images = _load_images(row.get("images", []))
        messages = _messages(question, images)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=images, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        latency_s = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        prediction = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        pred_texts.append(prediction)
        gold_texts.append(answer)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "task": row.get("task"),
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "exact_match": exact_match(prediction, answer),
                "latency_s": latency_s,
                "images": row.get("images", []),
            }
        )

    write_jsonl(pred_path, predictions)
    metrics = {
        "count": len(predictions),
        "exact_match": accuracy(pred_texts, gold_texts),
        "avg_latency_s": sum(item["latency_s"] for item in predictions) / max(1, len(predictions)),
        "model": args.model,
        "input": str(args.input),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
