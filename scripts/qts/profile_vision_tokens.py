"""Probe the visual-token layout and the prefill/decode latency split.

This is the v2/QTS day-1 de-risk step, run *before* any token-pruning code:

1. **Token structure** — for a real val sample, how many visual tokens does
   Qwen3-VL actually emit (``input_ids == image_token_id``), what fraction of
   the prompt are they, and what is the image grid (``image_grid_thw``)? This
   tells us how much there is to prune and where the tokens sit in the sequence.
2. **Latency split** — of the ~8 s/sample end-to-end cost, how much is the
   one-shot *prefill* (processing the prompt, which is where visual tokens live)
   versus the autoregressive *decode* of the trajectory text? Pruning visual
   tokens only helps the prefill (and the per-step KV attention); if decode
   dominates wall-clock, the efficiency story has to be reframed.

Nothing is corrupted or pruned here — this profiles the unmodified ``full``
baseline. Reuses the model/adapter loading convention from
``scripts/eval/eval_vla.py``.

Server command (single GPU, pinned):

    bash scripts/qts/run_profile.sh

Output: a JSON summary to stdout and, if ``--out-dir`` is given,
``<out-dir>/profile.json``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl


def _user_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def _build_prompt(processor, question: str, images: list[Image.Image]) -> str:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _to_device(inputs, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _resolve_image_token_id(model, processor) -> int | None:
    """Find the placeholder token id that marks visual positions in input_ids."""
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(model.config, attr, None)
        if val is not None:
            return int(val)
    tok = getattr(processor, "image_token", None) or "<|image_pad|>"
    try:
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0:
            return int(tid)
    except Exception:
        pass
    return None


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time_generate(model, inputs, max_new_tokens: int, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    _sync(device)
    return time.perf_counter() - t0


def _time_prefill(model, inputs, device: str) -> float:
    """One forward pass over the full prompt (prefill / time-to-first-logits)."""
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, use_cache=True)
    _sync(device)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Path or name of the Qwen3-VL base model.")
    parser.add_argument("--adapter", default=None, help="LoRA adapter checkpoint dir.")
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--limit", default=5, type=int, help="Number of val samples to profile.")
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--warmup", default=2, type=int,
        help="Warmup generate() calls before timing (kernel autotune / cache).")
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"loading model from {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa",
    )
    if args.adapter:
        from peft import PeftModel
        print(f"loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device)
    model.eval()

    image_token_id = _resolve_image_token_id(model, processor)
    print(f"image_token_id={image_token_id}")

    rows = read_jsonl(args.val_file)[: args.limit]
    print(f"profiling {len(rows)} samples from {args.val_file}\n")

    # ----- Warmup (kernels, autotune) ----------------------------------------
    if rows and args.warmup > 0:
        r0 = rows[0]
        imgs = _load_images([str(p) for p in r0.get("images", [])])
        prompt = _build_prompt(processor, _user_text(r0), imgs)
        warm_inputs = _to_device(
            processor(text=[prompt], images=imgs, return_tensors="pt"), device
        )
        for _ in range(args.warmup):
            _time_generate(model, warm_inputs, 8, device)

    per_sample = []
    for i, row in enumerate(rows):
        image_paths = [str(p) for p in row.get("images", [])]
        images = _load_images(image_paths)
        prompt = _build_prompt(processor, _user_text(row), images)
        enc = processor(text=[prompt], images=images, return_tensors="pt")
        input_ids = enc["input_ids"]
        input_len = int(input_ids.shape[1])
        if image_token_id is not None:
            visual_tokens = int((input_ids == image_token_id).sum().item())
        else:
            visual_tokens = -1
        grid = enc.get("image_grid_thw")
        grid_list = grid.tolist() if hasattr(grid, "tolist") else grid

        inputs = _to_device(enc, device)

        prefill_s = _time_prefill(model, inputs, device)
        t1 = _time_generate(model, inputs, 1, device)
        tn = _time_generate(model, inputs, args.max_new_tokens, device)
        n = max(2, args.max_new_tokens)
        decode_per_token = (tn - t1) / (n - 1)
        prefill_gen = max(0.0, t1 - decode_per_token)
        decode_total = tn - prefill_gen
        decode_share = decode_total / tn if tn > 0 else float("nan")

        rec = {
            "idx": i,
            "input_len": input_len,
            "visual_tokens": visual_tokens,
            "text_tokens": input_len - visual_tokens if visual_tokens >= 0 else None,
            "visual_fraction": (visual_tokens / input_len) if visual_tokens >= 0 else None,
            "image_grid_thw": grid_list,
            "prefill_forward_s": round(prefill_s, 4),
            "prefill_gen_s": round(prefill_gen, 4),
            "decode_per_token_s": round(decode_per_token, 5),
            "decode_total_s": round(decode_total, 4),
            "full_generate_s": round(tn, 4),
            "decode_share": round(decode_share, 4),
        }
        per_sample.append(rec)
        print(json.dumps(rec))

    def _mean(key):
        vals = [r[key] for r in per_sample if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_samples": len(per_sample),
        "max_new_tokens": args.max_new_tokens,
        "image_token_id": image_token_id,
        "mean_input_len": _mean("input_len"),
        "mean_visual_tokens": _mean("visual_tokens"),
        "mean_visual_fraction": _mean("visual_fraction"),
        "mean_prefill_forward_s": _mean("prefill_forward_s"),
        "mean_prefill_gen_s": _mean("prefill_gen_s"),
        "mean_decode_per_token_s": _mean("decode_per_token_s"),
        "mean_decode_total_s": _mean("decode_total_s"),
        "mean_full_generate_s": _mean("full_generate_s"),
        "mean_decode_share": _mean("decode_share"),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(
        "\nReading: decode_share is the fraction of wall-clock spent in "
        "autoregressive decode (NOT prefill). Visual-token pruning helps "
        "prefill + per-step KV attention; if decode_share is high, the "
        "headline latency win from pruning is bounded by (1 - decode_share)."
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = {"summary": summary, "per_sample": per_sample}
        (args.out_dir / "profile.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out_dir / 'profile.json'}")


if __name__ == "__main__":
    main()
