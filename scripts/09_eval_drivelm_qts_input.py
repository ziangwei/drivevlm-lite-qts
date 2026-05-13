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
from drivevlm_lite.camera_selection import camera_name_from_path, select_images_by_query


def _pixels_from_visual_tokens(tokens: int | None) -> int | None:
    if tokens is None or tokens <= 0:
        return None
    return tokens * 28 * 28


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


def _messages(question: str, images: list[Image.Image], cameras: list[str], label_images: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for idx, image in enumerate(images):
        if label_images:
            camera = cameras[idx] if idx < len(cameras) else f"IMAGE_{idx}"
            content.append({"type": "text", "text": f"{camera} view:"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _load_processor(model_name_or_path: str, visual_token_budget: int | None):
    from transformers import AutoProcessor

    min_pixels = _pixels_from_visual_tokens(min(64, visual_token_budget) if visual_token_budget else None)
    max_pixels = _pixels_from_visual_tokens(visual_token_budget)
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
    total = 0
    for row in grid.detach().cpu():
        values = [int(value) for value in row.tolist()]
        if len(values) == 3:
            total += values[0] * values[1] * values[2]
    return total


def _eval_strategy(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    out_dir: Path,
    strategy: str,
    fallback: str,
    max_selected_images: int,
    max_new_tokens: int,
    label_images: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []
    latencies: list[float] = []
    input_token_counts: list[int] = []
    grid_token_counts: list[int] = []
    image_counts: list[int] = []
    query_selected_count = 0

    for row in tqdm(rows, desc=f"DriveLM QTS input {strategy}"):
        question, answer = _question_and_answer(row)
        all_paths = [str(path) for path in row.get("images", [])]
        selection = select_images_by_query(
            all_paths,
            question,
            strategy=strategy,
            max_images=max_selected_images,
            fallback=fallback,
        )
        if selection.reason == "query":
            query_selected_count += 1
        images = _load_images(selection.paths)
        cameras = selection.cameras or [camera_name_from_path(path) or "UNKNOWN" for path in selection.paths]
        messages = _messages(question, images, cameras, label_images=label_images)
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
        latencies.append(latency_s)
        input_token_counts.append(input_len)
        image_counts.append(len(selection.paths))
        if grid_tokens is not None:
            grid_token_counts.append(grid_tokens)
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
                "all_images": all_paths,
                "selected_images": selection.paths,
                "selected_cameras": selection.cameras,
                "selection_reason": selection.reason,
            }
        )

    write_jsonl(out_dir / "predictions.jsonl", predictions)
    metrics = {
        "strategy": strategy,
        "count": len(predictions),
        "exact_match": accuracy(pred_texts, gold_texts),
        "avg_latency_s": sum(latencies) / max(1, len(latencies)),
        "avg_input_tokens": sum(input_token_counts) / max(1, len(input_token_counts)),
        "avg_image_grid_tokens": (
            sum(grid_token_counts) / len(grid_token_counts) if grid_token_counts else None
        ),
        "avg_images": sum(image_counts) / max(1, len(image_counts)),
        "query_selected_count": query_selected_count,
        "fallback_count": len(predictions) - query_selected_count,
        "max_selected_images": max_selected_images,
        "fallback": fallback,
        "label_images": label_images,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_summary_markdown(path: Path, metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# DriveLM QTS Input Selection Evaluation",
        "",
        "| strategy | count | EM | avg latency s | avg input tokens | avg image grid tokens | avg images | query selected | fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        grid_tokens = item["avg_image_grid_tokens"]
        lines.append(
            "| {strategy} | {count} | {em:.3f} | {lat:.3f} | {inp:.1f} | {grid} | {imgs:.2f} | {q} | {fb} |".format(
                strategy=item["strategy"],
                count=item["count"],
                em=float(item["exact_match"]),
                lat=float(item["avg_latency_s"]),
                inp=float(item["avg_input_tokens"]),
                grid="n/a" if grid_tokens is None else f"{float(grid_tokens):.1f}",
                imgs=float(item["avg_images"]),
                q=item["query_selected_count"],
                fb=item["fallback_count"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", default="data/processed/drivelm_sft_val.jsonl", type=Path)
    parser.add_argument("--out-root", default="reports/e2_qts_input", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--visual-token-budget", default=128, type=int)
    parser.add_argument("--strategies", nargs="+", default=["all", "qts_rule", "qts_rule_front", "front_only"])
    parser.add_argument("--max-selected-images", default=3, type=int)
    parser.add_argument("--fallback", choices=["all", "front"], default="all")
    parser.add_argument("--label-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText

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

    rows = read_jsonl(args.input)
    available_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.limit > 0 and len(rows) < args.limit:
        print(
            "WARNING: requested limit is larger than the input JSONL. "
            f"requested_limit={args.limit} available_rows={available_rows}"
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for strategy in args.strategies:
        metrics = _eval_strategy(
            model=model,
            processor=processor,
            rows=rows,
            device=device,
            out_dir=args.out_root / strategy,
            strategy=strategy,
            fallback=args.fallback,
            max_selected_images=args.max_selected_images,
            max_new_tokens=args.max_new_tokens,
            label_images=args.label_images,
        )
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "input": str(args.input),
        "requested_limit": args.limit,
        "available_rows": available_rows,
        "evaluated_rows": len(rows),
        "visual_token_budget": args.visual_token_budget,
        "strategies": args.strategies,
        "metrics": all_metrics,
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(args.out_root / "summary.md", all_metrics)
    print(f"Wrote QTS input selection summary: {args.out_root}")


if __name__ == "__main__":
    main()
