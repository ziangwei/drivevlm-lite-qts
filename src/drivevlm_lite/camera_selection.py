"""Rule-based camera selection utilities.

Used by VQA visual-budget experiments and Mini-VLA evaluation scripts to
pick a subset of the six nuScenes camera views before inference. This is
a pre-encoder input prune; it does not modify Qwen3-VL internals.

The learned token-selector variant once lived in this module under the
name ``QueryAwareTokenSelector`` but was never trained or integrated. It
has been moved to ``drivevlm_lite.experimental.qts_neural`` and is not
part of the v1 pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CAMERA_RE = re.compile(r"CAM_(?:FRONT_RIGHT|FRONT_LEFT|BACK_RIGHT|BACK_LEFT|FRONT|BACK)")


@dataclass(frozen=True)
class ImageSelection:
    paths: list[str]
    cameras: list[str]
    reason: str


def camera_name_from_path(path: str | Path) -> str | None:
    match = CAMERA_RE.search(str(path).upper())
    return match.group(0) if match else None


def infer_query_cameras(question: str) -> list[str]:
    """Infer likely useful camera views from a DriveLM question."""
    selected: list[str] = []

    def add(*cameras: str) -> None:
        for camera in cameras:
            if camera not in selected:
                selected.append(camera)

    text = question.lower()
    for match in CAMERA_RE.finditer(question.upper()):
        add(match.group(0))

    if "front right" in text or "front-right" in text:
        add("CAM_FRONT_RIGHT", "CAM_FRONT")
    if "front left" in text or "front-left" in text:
        add("CAM_FRONT_LEFT", "CAM_FRONT")
    if "back right" in text or "back-right" in text or "rear right" in text or "rear-right" in text:
        add("CAM_BACK_RIGHT", "CAM_BACK")
    if "back left" in text or "back-left" in text or "rear left" in text or "rear-left" in text:
        add("CAM_BACK_LEFT", "CAM_BACK")

    if "front" in text:
        add("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
    if "back" in text or "behind" in text or "rear" in text:
        add("CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    if "left" in text:
        add("CAM_FRONT_LEFT", "CAM_BACK_LEFT")
    if "right" in text:
        add("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT")

    return selected


def select_images_by_query(
    image_paths: list[str],
    question: str,
    strategy: str = "qts_rule",
    max_images: int = 3,
    fallback: str = "all",
) -> ImageSelection:
    """Select camera images before VLM inference using a query-aware rule."""
    camera_to_path: dict[str, str] = {}
    path_cameras: list[str] = []
    for raw_path in image_paths:
        camera = camera_name_from_path(raw_path) or "UNKNOWN"
        path_cameras.append(camera)
        camera_to_path.setdefault(camera, raw_path)

    if strategy == "all":
        return ImageSelection(paths=list(image_paths), cameras=path_cameras, reason="all")

    if strategy == "front_only":
        path = camera_to_path.get("CAM_FRONT", image_paths[0] if image_paths else "")
        camera = camera_name_from_path(path) or "UNKNOWN"
        return ImageSelection(paths=[path] if path else [], cameras=[camera] if path else [], reason="front_only")

    if strategy not in {"qts_rule", "qts_rule_front"}:
        raise ValueError(f"Unknown image selection strategy: {strategy}")

    query_cameras = infer_query_cameras(question)
    if strategy == "qts_rule_front" and "CAM_FRONT" not in query_cameras:
        query_cameras.append("CAM_FRONT")

    selected_cameras = [camera for camera in query_cameras if camera in camera_to_path]
    if max_images > 0:
        selected_cameras = selected_cameras[:max_images]

    if selected_cameras:
        return ImageSelection(
            paths=[camera_to_path[camera] for camera in selected_cameras],
            cameras=selected_cameras,
            reason="query",
        )

    if fallback == "front":
        path = camera_to_path.get("CAM_FRONT", image_paths[0] if image_paths else "")
        camera = camera_name_from_path(path) or "UNKNOWN"
        return ImageSelection(paths=[path] if path else [], cameras=[camera] if path else [], reason="fallback_front")
    if fallback == "all":
        return ImageSelection(paths=list(image_paths), cameras=path_cameras, reason="fallback_all")
    raise ValueError(f"Unknown fallback: {fallback}")
