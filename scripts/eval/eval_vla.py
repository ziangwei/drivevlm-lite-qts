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
  waypoints, GT waypoints, and per-sample metrics. Written in line-buffered
  append mode so a mid-run crash never loses prior samples.
- ``run_meta.json``    — sidecar capturing the run's args (val_file,
  ablation, sample_mode, seed, limit, max_new_tokens). Resume refuses to
  proceed if these drift, so two incompatible runs cannot quietly merge
  into the same predictions file.
- ``metrics.json``     — aggregate ADE / FDE / parse_rate / lat / long,
  recomputed at the end from the full predictions.jsonl.

Resume behaviour (default on):
  - If ``predictions.jsonl`` already exists in ``--out-dir`` and
    ``run_meta.json`` matches the current args, samples whose ids are
    already present are skipped. A partial trailing line from the previous
    crash is truncated automatically.
  - Pass ``--no-resume`` to wipe and start over.

Subset selection (``--limit > 0``) is controlled by ``--sample-mode``:

- ``random``     — shuffle all rows with ``Random(seed)`` then take ``limit``.
                   Default; reproducible across runs for the same seed.
- ``prefix``     — legacy ``rows[:limit]``. val.jsonl is sorted by log+time,
                   so this collapses to one or two scenes — kept for back-compat
                   but not the recommended mode.
- ``stratified`` — group rows by log_id (parsed from CAM_FRONT filename) and
                   take ``limit / n_logs`` from each, shuffled per-log with
                   ``Random(seed)``. Useful for log-balanced debug runs.
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

from drivevlm_lite.data.jsonl import read_jsonl
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
from drivevlm_lite.eval.resume import (
    check_meta_compatible,
    read_jsonl_robust,
    truncate_to_last_newline,
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


def _row_key(row: dict[str, Any], idx: int) -> str:
    """Stable key used to skip already-done rows on resume.

    Prefers the dataset's own ``id`` / ``sample_id``; falls back to the
    subset position (which is itself deterministic for a given sample-mode
    + seed + val.jsonl, so it survives a resume).
    """
    rid = row.get("id") or row.get("sample_id")
    return str(rid) if rid is not None else f"__idx_{idx}"


def _select_subset(rows, mode, limit, seed):
    """Pick a subset of ``rows`` according to ``mode`` and ``limit``."""
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
        if len(picked) < limit:
            taken = set(picked)
            remaining = [i for i in range(n) if i not in taken]
            rng.shuffle(remaining)
            picked.extend(remaining[: limit - len(picked)])
        picked = sorted(picked[:limit])
        return [rows[i] for i in picked]
    raise ValueError(f"unknown sample-mode {mode!r}; expected one of {SAMPLE_MODES}")


def _load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def _blacken(images):
    """All-zero (black) images of matching size — the vision-masked condition."""
    return [Image.new("RGB", img.size, (0, 0, 0)) for img in images]


def _build_prompt(processor, question, images):
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _to_device(inputs, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _write_metrics(pred_path: Path, args: argparse.Namespace, available_rows: int) -> None:
    """Recompute aggregate metrics from the full predictions.jsonl on disk.

    Reading at the end (rather than accumulating in memory during the loop)
    means the numbers stay correct after a resume that merged old + new rows.
    """
    all_preds, _ = read_jsonl_robust(pred_path)
    n = len(all_preds)
    ade_v = [r["ade"] for r in all_preds if r.get("ade") is not None]
    fde_v = [r["fde"] for r in all_preds if r.get("fde") is not None]
    lon_v = [r["lon_ade"] for r in all_preds if r.get("lon_ade") is not None]
    lat_v = [r["lat_ade"] for r in all_preds if r.get("lat_ade") is not None]
    lat_s = [r["latency_s"] for r in all_preds if r.get("latency_s") is not None]
    parse_ok = sum(
        1 for r in all_preds if r.get("pred_waypoints") and r.get("gt_waypoints")
    )
    parse_full6 = sum(
        1 for r in all_preds
        if r.get("pred_waypoints")
        and len(r["pred_waypoints"]) >= 6
        and r.get("gt_waypoints")
    )
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
        "ade_mean": (sum(ade_v) / len(ade_v)) if ade_v else None,
        "fde_mean": (sum(fde_v) / len(fde_v)) if fde_v else None,
        "lon_ade_mean": (sum(lon_v) / len(lon_v)) if lon_v else None,
        "lat_ade_mean": (sum(lat_v) / len(lat_v)) if lat_v else None,
        "latency_mean_s": (sum(lat_s) / len(lat_s)) if lat_s else None,
        "max_new_tokens": args.max_new_tokens,
    }
    out_dir = pred_path.parent
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote predictions to {pred_path}")
    print(f"Wrote metrics to {out_dir / 'metrics.json'}")


def _count_donor_missing(donor, image_kind: str) -> int:
    """Count -1 sentinels in the relevant donor list."""
    series = donor.time_shifted if image_kind == "time_shifted" else donor.true_mismatch
    return sum(1 for j in series if j < 0)


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
    parser.add_argument("--no-resume", action="store_true",
        help="Wipe any existing predictions.jsonl + run_meta.json in --out-dir "
             "and start fresh. Default behaviour resumes from prior progress.")
    parser.add_argument("--num-gpus", default=1, type=int,
        help="Accepted for interface uniformity; this script uses one GPU.")
    args = parser.parse_args()

    plan = ablation_plan(args.ablation)
    print(
        f"ablation={args.ablation}  text={plan.text}  image={plan.image}  "
        f"sample_mode={args.sample_mode}  seed={args.seed}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "predictions.jsonl"
    meta_path = args.out_dir / "run_meta.json"

    # ----- Resume handshake ---------------------------------------------------
    current_meta = {
        "val_file": str(args.val_file),
        "ablation": args.ablation,
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.no_resume:
        for p in (pred_path, meta_path):
            if p.exists():
                p.unlink()
        print("--no-resume: wiped predictions.jsonl and run_meta.json (if any)")
    else:
        # Refuse to silently merge into a predictions.jsonl from a different
        # run (e.g. seed change, ablation change). Force the user to choose.
        if pred_path.exists() and pred_path.stat().st_size > 0 and not meta_path.exists():
            raise SystemExit(
                f"{pred_path} exists but {meta_path} does not -- cannot verify "
                "the existing file was produced by a compatible run. Pass "
                "--no-resume to wipe and restart, or change --out-dir."
            )
        mismatch = check_meta_compatible(meta_path, current_meta)
        if mismatch:
            raise SystemExit(
                mismatch
                + "\nPass --no-resume to wipe, or use a different --out-dir."
            )

    n_truncated = truncate_to_last_newline(pred_path)
    if n_truncated:
        print(
            f"resume: truncated {n_truncated} bytes of partial trailing line "
            f"in {pred_path}"
        )
    existing, n_bad = read_jsonl_robust(pred_path)
    if n_bad:
        print(f"resume: skipped {n_bad} malformed line(s) in {pred_path}")
    done_keys: set[str] = set()
    for r in existing:
        rid = r.get("id") or r.get("sample_id")
        if rid is not None:
            done_keys.add(str(rid))
    if existing:
        print(f"resume: {len(existing)} samples already present, will skip those ids")

    meta_path.write_text(json.dumps(current_meta, indent=2), encoding="utf-8")

    # ----- Load val + select subset ------------------------------------------
    all_rows = read_jsonl(args.val_file)
    available_rows = len(all_rows)
    rows = _select_subset(all_rows, args.sample_mode, args.limit, args.seed)
    print(
        f"val_file={args.val_file}  available={available_rows}  "
        f"in_subset={len(rows)}  mode={args.sample_mode}  seed={args.seed}"
    )

    # Donor index for image-swap ablations. Built deterministically from the
    # post-subset row list, so it's identical on resume.
    donor = None
    if plan.image in ("time_shifted", "true_mismatch"):
        donor_paths = [_row_first_image(r) for r in rows]
        donor = build_donor_index(donor_paths, seed=args.seed)
        n_missing = _count_donor_missing(donor, plan.image)
        if n_missing:
            print(
                f"WARNING: {n_missing}/{len(rows)} rows have no eligible "
                f"{plan.image} donor in this subset; falling back to the "
                "row's own image and flagging donor_missing=true in "
                "predictions.jsonl."
            )

    # ----- Decide what's left + short-circuit if everything done -------------
    n_done_in_subset = sum(
        1 for idx, row in enumerate(rows) if _row_key(row, idx) in done_keys
    )
    n_remaining = len(rows) - n_done_in_subset
    print(f"resume status: {n_done_in_subset} done / {n_remaining} remaining")

    if n_remaining == 0:
        print("nothing to do -- only writing metrics.json")
        _write_metrics(pred_path, args, available_rows)
        return

    # ----- Model load (only when there's actual work) -------------------------
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

    # ----- Eval loop (append-only, line-buffered) ----------------------------
    with pred_path.open("a", encoding="utf-8", buffering=1) as out_handle:
        for idx, row in enumerate(tqdm(rows, desc="VLA eval")):
            row_key = _row_key(row, idx)
            if row_key in done_keys:
                continue

            question = transform_user_text(_user_text(row), args.ablation)
            gt_text = _assistant_text(row)
            gt_waypoints = parse_planning_text(gt_text)

            image_paths = [str(p) for p in row.get("images", [])]
            donor_image = None
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
                "id": row_key,
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
                sample_ade = ade(pred_waypoints, gt_waypoints)
                sample_fde = fde(pred_waypoints, gt_waypoints)
                sample_lon, sample_lat = split_lateral_longitudinal_ade(
                    pred_waypoints, gt_waypoints
                )
                per_sample.update({
                    "ade": sample_ade,
                    "fde": sample_fde,
                    "lon_ade": sample_lon,
                    "lat_ade": sample_lat,
                })
            else:
                per_sample.update({
                    "ade": None, "fde": None, "lon_ade": None, "lat_ade": None,
                })

            out_handle.write(json.dumps(per_sample, ensure_ascii=False) + "\n")
            out_handle.flush()
            done_keys.add(row_key)

    _write_metrics(pred_path, args, available_rows)


if __name__ == "__main__":
    main()
