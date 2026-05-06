from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from drivevlm_lite.data.schema import DrivingSample


TASK_KEYS = ("perception", "prediction", "planning", "behavior", "motion")


def _resolve_image_paths(image_paths: dict[str, str], image_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for _, raw_path in sorted(image_paths.items()):
        path = Path(raw_path)
        if not path.is_absolute():
            parts = path.parts
            if "samples" in parts:
                samples_idx = parts.index("samples")
                path = image_root / Path(*parts[samples_idx + 1 :])
            else:
                path = image_root / path
        resolved.append(path)
    return resolved


def _iter_qa_items(frame: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    qa = frame.get("QA", {})
    if not isinstance(qa, dict):
        return
    for task in TASK_KEYS:
        items = qa.get(task, [])
        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                yield task, item


def iter_drivelm_samples(qa_file: Path, image_root: Path) -> Iterator[DrivingSample]:
    """Yield unified samples from the DriveLM-nuScenes JSON structure."""
    data = json.loads(qa_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected DriveLM JSON root to be a dict, got {type(data)!r}")

    for scene_token, scene in data.items():
        key_frames = scene.get("key_frames", {}) if isinstance(scene, dict) else {}
        if not isinstance(key_frames, dict):
            continue
        for frame_token, frame in key_frames.items():
            image_paths = frame.get("image_paths", {})
            if not isinstance(image_paths, dict):
                continue
            images = _resolve_image_paths(image_paths, image_root=image_root)
            for task, item in _iter_qa_items(frame):
                question = item.get("Q") or item.get("question")
                answer = item.get("A") or item.get("answer")
                if not question or answer is None:
                    continue
                qa_id = item.get("id") or item.get("qid") or abs(hash((scene_token, frame_token, task, question)))
                yield DrivingSample(
                    sample_id=f"{scene_token}:{frame_token}:{task}:{qa_id}",
                    images=images,
                    question=str(question),
                    answer=str(answer),
                    task=task,
                    metadata={"scene_token": scene_token, "frame_token": frame_token},
                )
