from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from drivevlm_lite.data.nuscenes_trajectory import trajectory_answer, trajectory_question


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TRAJECTORY_KEYS = (
    "trajectory",
    "future_trajectory",
    "gt_trajectory",
    "waypoints",
    "future_waypoints",
    "traj",
    "planning",
)
COT_KEYS = ("cot", "chain_of_thought", "reasoning", "think", "thinking", "rationale", "reflection")
ANSWER_KEYS = ("answer", "output", "response", "completion", "assistant")
QUESTION_KEYS = ("question", "prompt", "instruction", "query", "input")


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, a JSON array, or a dict containing a data split."""
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "{":
            data = json.load(handle)
            yield from _records_from_loaded(data)
            return
        if first == "[":
            data = json.load(handle)
            yield from _records_from_loaded(data)
            return
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def _records_from_loaded(data: Any) -> Iterator[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(data, dict):
        return
    for key in ("data", "train", "val", "test", "samples", "annotations", "records", "items"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
    yield data


def summarize_schema(records: Iterable[dict[str, Any]], limit: int = 200) -> dict[str, Any]:
    rows = []
    top_keys: Counter[str] = Counter()
    image_candidates = 0
    trajectory_candidates = 0
    cot_candidates = 0
    question_candidates = 0
    answer_candidates = 0

    for idx, record in enumerate(records):
        if idx >= limit:
            break
        rows.append(record)
        top_keys.update(record.keys())
        if find_image_paths(record):
            image_candidates += 1
        if extract_waypoints(record) or parse_waypoint_pairs(extract_text(record, ANSWER_KEYS)):
            trajectory_candidates += 1
        if extract_cot(record):
            cot_candidates += 1
        if extract_question(record):
            question_candidates += 1
        if extract_answer_text(record):
            answer_candidates += 1

    examples = [_small_record_preview(row) for row in rows[:3]]
    return {
        "scanned_records": len(rows),
        "top_keys": dict(top_keys.most_common(50)),
        "image_candidate_rows": image_candidates,
        "trajectory_candidate_rows": trajectory_candidates,
        "cot_candidate_rows": cot_candidates,
        "question_candidate_rows": question_candidates,
        "answer_candidate_rows": answer_candidates,
        "examples": examples,
    }


def convert_record(
    record: dict[str, Any],
    nuscenes_root: Path | None = None,
    image_root: Path | None = None,
    answer_mode: str = "cot",
    step_seconds: float = 0.5,
    require_images: bool = True,
) -> dict[str, Any] | None:
    raw_images = find_image_paths(record)
    images = [str(resolve_image_path(path, nuscenes_root=nuscenes_root, image_root=image_root)) for path in raw_images]
    if require_images and not images:
        return None

    waypoints = extract_waypoints(record)
    if not waypoints:
        waypoints = parse_waypoint_pairs(extract_answer_text(record))
    if not waypoints:
        return None

    question = extract_question(record)
    if not question:
        question = trajectory_question(step_seconds * len(waypoints), len(waypoints))

    traj_answer = trajectory_answer(waypoints, step_seconds=step_seconds)
    cot = extract_cot(record)
    if answer_mode == "direct" or not cot:
        answer = traj_answer
    elif answer_mode == "original":
        answer = extract_answer_text(record) or traj_answer
    else:
        answer = f"<think>{cot}</think>\n<answer>{traj_answer}</answer>"

    sample_id = extract_sample_id(record)
    metadata = {
        "source": "autodrive_r2",
        "raw_sample_id": sample_id,
        "has_cot": bool(cot),
        "raw_image_count": len(raw_images),
    }
    for key in ("scene_token", "scene_id", "token", "timestamp", "frame_id", "sample_token"):
        if key in record and isinstance(record[key], str | int | float):
            metadata[key] = record[key]

    return {
        "sample_id": sample_id,
        "task": "vla_trajectory_cot" if cot and answer_mode == "cot" else "vla_trajectory",
        "images": images,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "trajectory": [
            {"t": round((idx + 1) * step_seconds, 3), "x": x, "y": y}
            for idx, (x, y) in enumerate(waypoints)
        ],
        "metadata": metadata,
    }


def find_image_paths(value: Any) -> list[str]:
    out: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            if _looks_like_image(item):
                out.append(item)
            return
        if isinstance(item, list | tuple):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    return _dedupe(out)


def resolve_image_path(path: str, nuscenes_root: Path | None = None, image_root: Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if image_root is not None:
        candidate = image_root / path
        if candidate.exists():
            return candidate
    normalized = path.replace("\\", "/")
    if nuscenes_root is not None:
        for marker in ("samples/", "sweeps/"):
            idx = normalized.find(marker)
            if idx >= 0:
                return nuscenes_root / normalized[idx:]
        for prefix in ("data/nuscenes/", "nuscenes/"):
            if normalized.startswith(prefix):
                return nuscenes_root / normalized[len(prefix) :]
    return raw


def extract_waypoints(record: dict[str, Any]) -> list[tuple[float, float]]:
    for key in TRAJECTORY_KEYS:
        if key in record:
            waypoints = _coerce_waypoints(record[key])
            if waypoints:
                return waypoints
    for key, value in record.items():
        key_lower = str(key).lower()
        if "trajectory" in key_lower or "waypoint" in key_lower:
            waypoints = _coerce_waypoints(value)
            if waypoints:
                return waypoints
    return []


def _coerce_waypoints(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        return parse_waypoint_pairs(value)
    if isinstance(value, dict):
        for key in ("points", "waypoints", "trajectory", "future"):
            if key in value:
                parsed = _coerce_waypoints(value[key])
                if parsed:
                    return parsed
        if "x" in value and "y" in value:
            return [(round(float(value["x"]), 3), round(float(value["y"]), 3))]
        return []
    if not isinstance(value, list | tuple):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and "x" in item and "y" in item:
            out.append((round(float(item["x"]), 3), round(float(item["y"]), 3)))
        elif isinstance(item, list | tuple) and len(item) >= 2 and _is_number(item[0]) and _is_number(item[1]):
            out.append((round(float(item[0]), 3), round(float(item[1]), 3)))
    return out


def parse_waypoint_pairs(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return []
    patterns = [
        re.compile(
            r"<\s*t\s*=\s*[-+]?\d+(?:\.\d+)?\s*,\s*x\s*="
            r"([-+]?\d+(?:\.\d+)?)\s*,\s*y\s*=([-+]?\d+(?:\.\d+)?)\s*>"
        ),
        re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)"),
        re.compile(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]"),
    ]
    for pattern in patterns:
        matches = [(round(float(x), 3), round(float(y), 3)) for x, y in pattern.findall(text)]
        if matches:
            return matches
    return []


def extract_cot(record: dict[str, Any]) -> str:
    cot = extract_text(record, COT_KEYS)
    if cot:
        return _strip_wrappers(cot)
    answer = extract_answer_text(record)
    match = re.search(r"<think>(.*?)</think>", answer or "", flags=re.DOTALL | re.IGNORECASE)
    if match:
        return _strip_wrappers(match.group(1))
    return ""


def extract_question(record: dict[str, Any]) -> str:
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role in {"user", "human"}:
                return _message_text(message)
    return extract_text(record, QUESTION_KEYS)


def extract_answer_text(record: dict[str, Any]) -> str:
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role in {"assistant", "gpt", "model"}:
                return _message_text(message)
    return extract_text(record, ANSWER_KEYS)


def extract_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        text = _text_value(value)
        if text:
            return text
    return ""


def extract_sample_id(record: dict[str, Any]) -> str:
    for key in ("sample_id", "sample_token", "token", "id", "frame_id", "qa_id"):
        value = record.get(key)
        if isinstance(value, str | int):
            return str(value)
    return str(abs(hash(json.dumps(_small_record_preview(record), sort_keys=True, ensure_ascii=False))))


def _message_text(message: dict[str, Any]) -> str:
    value = message.get("content") or message.get("value") or message.get("text")
    return _text_value(value)


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


def _small_record_preview(record: dict[str, Any]) -> dict[str, Any]:
    preview = {}
    for key, value in list(record.items())[:20]:
        preview[key] = _preview_value(value)
    return preview


def _preview_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _preview_value(item) for key, item in list(value.items())[:8]}
    if isinstance(value, list):
        return [_preview_value(item) for item in value[:3]]
    if isinstance(value, str):
        return value[:300]
    return value


def _strip_wrappers(text: str) -> str:
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _looks_like_image(text: str) -> bool:
    lowered = text.lower()
    return any(lowered.endswith(suffix) or suffix + "?" in lowered for suffix in IMAGE_SUFFIXES)


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
