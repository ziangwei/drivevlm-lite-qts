from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from drivevlm_lite.data.schema import DrivingSample


def _images_from_record(record: dict[str, Any], root: Path) -> list[Path]:
    image_path = record.get("image_path") or record.get("images")
    if isinstance(image_path, dict):
        values = [image_path[k] for k in sorted(image_path)]
    elif isinstance(image_path, list):
        values = image_path
    elif isinstance(image_path, str):
        values = [image_path]
    else:
        values = []

    out: list[Path] = []
    for value in values:
        path = Path(value)
        if not path.is_absolute() and path.parts[:1] == ("data",):
            path = Path(*path.parts[1:])
        out.append(path if path.is_absolute() else root / path)
    return out


def iter_drivebench_json(path: Path, image_root: Path) -> Iterator[DrivingSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("data") or data.get("records") or list(data.values())
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported DriveBench JSON structure in {path}")

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not question:
            continue
        sample_id = str(row.get("id") or row.get("frame_token") or idx)
        yield DrivingSample(
            sample_id=sample_id,
            images=_images_from_record(row, image_root),
            question=str(question),
            answer=str(answer) if answer is not None else None,
            task=str(row.get("question_type") or row.get("tag") or ""),
            metadata={k: v for k, v in row.items() if k not in {"question", "answer", "image_path"}},
        )
