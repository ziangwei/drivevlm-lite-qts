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
from drivevlm_lite.data.nuscenes_trajectory import ade, fde, parse_trajectory_text
from drivevlm_lite.qts import camera_name_from_path


IMAGE_MODES = ("all", "front3", "front", "none", "mismatch_all", "mismatch_front3", "mismatch_front")


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


def _target(row: dict[str, Any], answer: str) -> list[tuple[float, float]]:
    if row.get("trajectory"):
        return [(float(item["x"]), float(item["y"])) for item in row["trajectory"]]
    return parse_trajectory_text(answer)


def _messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _select_image_paths(paths: list[str], image_mode: str) -> list[str]:
    if image_mode == "all":
        return list(paths)
    if image_mode == "none":
        return []

    camera_to_path: dict[str, str] = {}
    for path in paths:
        camera = camera_name_from_path(path)
        if camera:
            camera_to_path.setdefault(camera, path)

    if image_mode == "front":
        return [camera_to_path["CAM_FRONT"]] if "CAM_FRONT" in camera_to_path else []
    if image_mode == "front3":
        return [
            camera_to_path[camera]
            for camera in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
            if camera in camera_to_path
        ]
    raise ValueError(f"Unknown image mode: {image_mode}")


def _image_source(
    rows: list[dict[str, Any]],
    row_idx: int,
    image_mode: str,
    mismatch_offset: int,
) -> tuple[dict[str, Any], str]:
    if not image_mode.startswith("mismatch_"):
        return rows[row_idx], image_mode
    if len(rows) <= 1:
        return rows[row_idx], image_mode.removeprefix("mismatch_")

    offset = mismatch_offset % len(rows)
    if offset == 0:
        offset = 1
    return rows[(row_idx + offset) % len(rows)], image_mode.removeprefix("mismatch_")


def _processor_inputs(processor: Any, text: str, images: list[Image.Image]) -> dict[str, Any]:
    if images:
        return processor(text=[text], images=images, return_tensors="pt")
    return processor(text=[text], return_tensors="pt")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fixed_horizon_prediction(
    parsed: list[tuple[float, float]],
    target: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    if not target or len(parsed) < len(target):
        return None
    return parsed[: len(target)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", default="data/processed_vla/nuscenes_vla_val.jsonl", type=Path)
    parser.add_argument("--out", default="reports/vla_eval", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--max-new-tokens", default=192, type=int)
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default="all")
    parser.add_argument("--mismatch-offset", default=17, type=int)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
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

    predictions: list[dict[str, Any]] = []
    ades: list[float] = []
    fdes: list[float] = []
    parse_ok = 0
    exact_points = 0
    usable_points = 0
    underfull_points = 0
    overfull_points = 0
    parsed_point_counts = []
    latencies = []

    args.out.mkdir(parents=True, exist_ok=True)

    for row_idx, row in enumerate(tqdm(rows, desc="VLA trajectory eval")):
        question, answer = _question_and_answer(row)
        target = _target(row, answer)
        image_source_row, select_mode = _image_source(rows, row_idx, args.image_mode, args.mismatch_offset)
        selected_image_paths = _select_image_paths(image_source_row.get("images", []), select_mode)
        images = _load_images(selected_image_paths)
        messages = _messages(question, images)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _processor_inputs(processor, text, images)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        latency_s = time.perf_counter() - start
        latencies.append(latency_s)

        input_len = inputs["input_ids"].shape[1]
        prediction = processor.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed = parse_trajectory_text(prediction)
        fixed_prediction = _fixed_horizon_prediction(parsed, target)
        row_ade = ade(fixed_prediction, target) if fixed_prediction is not None else None
        row_fde = fde(fixed_prediction, target) if fixed_prediction is not None else None
        parsed_point_counts.append(float(len(parsed)))
        if parsed:
            parse_ok += 1
        if len(parsed) == len(target):
            exact_points += 1
        if fixed_prediction is not None:
            usable_points += 1
        if len(parsed) < len(target):
            underfull_points += 1
        if len(parsed) > len(target):
            overfull_points += 1
        if row_ade is not None:
            ades.append(row_ade)
        if row_fde is not None:
            fdes.append(row_fde)

        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "target": target,
                "parsed": parsed,
                "fixed_prediction": fixed_prediction,
                "parsed_point_count": len(parsed),
                "target_point_count": len(target),
                "ade": row_ade,
                "fde": row_fde,
                "latency_s": latency_s,
                "images": row.get("images", []),
                "selected_images": selected_image_paths,
                "image_source_sample_id": image_source_row.get("sample_id"),
                "image_mode": args.image_mode,
            }
        )

    metrics = {
        "count": len(predictions),
        "parse_rate": parse_ok / max(1, len(predictions)),
        "exact_point_count_rate": exact_points / max(1, len(predictions)),
        "usable_point_count_rate": usable_points / max(1, len(predictions)),
        "underfull_point_count": underfull_points,
        "overfull_point_count": overfull_points,
        "avg_parsed_points": _mean(parsed_point_counts),
        "ade": _mean(ades),
        "fde": _mean(fdes),
        "valid_ade_count": len(ades),
        "avg_latency_s": _mean(latencies),
        "avg_images": _mean([float(len(item["selected_images"])) for item in predictions]),
        "image_mode": args.image_mode,
        "mismatch_offset": args.mismatch_offset,
        "model": args.model,
        "adapter": args.adapter,
        "input": str(args.input),
    }

    write_jsonl(args.out / "predictions.jsonl", predictions)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# VLA Trajectory Evaluation",
        "",
        "| count | parse rate | exact points | usable points | ADE m | FDE m | valid ADE n | avg latency s | avg images |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {metrics['count']} | {metrics['parse_rate']:.3f} | "
            f"{metrics['exact_point_count_rate']:.3f} | {metrics['usable_point_count_rate']:.3f} | "
            f"{metrics['ade']:.3f} | "
            f"{metrics['fde']:.3f} | {metrics['valid_ade_count']} | {metrics['avg_latency_s']:.3f} | "
            f"{metrics['avg_images']:.2f} |"
        ),
        "",
        f"- model: {args.model}",
        f"- adapter: {args.adapter or 'none'}",
        f"- input: {args.input}",
        f"- image_mode: {args.image_mode}",
        f"- mismatch_offset: {args.mismatch_offset}",
        f"- avg_parsed_points: {metrics['avg_parsed_points']:.2f}",
        f"- underfull_point_count: {metrics['underfull_point_count']}",
        f"- overfull_point_count: {metrics['overfull_point_count']}",
    ]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote predictions: {args.out / 'predictions.jsonl'}")
    print(f"Wrote metrics: {args.out / 'metrics.json'}")
    print(f"Wrote summary: {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
