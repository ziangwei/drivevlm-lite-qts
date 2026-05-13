"""Adapter that converts Impromptu-VLA's nuScenes JSON files into the JSONL
format used by this project's training and evaluation scripts.

The Impromptu repository ships two ready-made files:

- ``nuscenes_train.json`` — 28 130 samples (full nuScenes train, 700 scenes)
- ``nuscenes_test.json`` —  6 020 samples (full nuScenes val, 150 scenes)

Each item is ``{"id", "images", "messages"}`` where ``images`` is a single
``"nuscenes/samples/CAM_FRONT/<file>.jpg"`` path and ``messages`` contains
the user prompt (with past ego status) and the assistant answer (a
``<PLANNING>...</PLANNING>`` block of six 2-D waypoints).

This adapter does three things:

1. Rewrites the image path so it points at the actual nuScenes keyframe
   tree on the server.
2. Optionally verifies every rewritten path exists on disk.
3. Writes the result to JSONL one row per line, matching what
   ``scripts/04_train_sft.py`` already consumes.

The schema is preserved verbatim; no resampling, no prompt rewriting. v1
trains and evaluates on Impromptu's exact split for direct comparability.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMPROMPTU_IMAGE_PREFIX = "nuscenes/"


@dataclass(frozen=True)
class AdapterStats:
    total: int
    written: int
    missing_images: int
    skipped_no_image: int


def load_impromptu_records(path: Path) -> list[dict[str, Any]]:
    """Load Impromptu's nuScenes JSON. Returns a list of records as-is."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}")
    return data


def rewrite_image_paths(record: dict[str, Any], nuscenes_root: Path) -> dict[str, Any]:
    """Return a copy of ``record`` with each image path made absolute under
    ``nuscenes_root``.

    Impromptu's paths look like ``nuscenes/samples/CAM_FRONT/<file>.jpg``.
    ``nuscenes_root`` should point at the directory that *contains*
    ``samples/`` (i.e. the equivalent of Impromptu's ``nuscenes/`` prefix).
    """
    new_record = dict(record)
    new_images: list[str] = []
    for raw in record.get("images", []) or []:
        rel = str(raw)
        if rel.startswith(IMPROMPTU_IMAGE_PREFIX):
            rel = rel[len(IMPROMPTU_IMAGE_PREFIX):]
        absolute = nuscenes_root / rel
        new_images.append(str(absolute))
    new_record["images"] = new_images
    return new_record


def iter_rewritten(
    records,
    nuscenes_root: Path,
    *,
    require_image: bool = True,
    check_existence: bool = True,
    limit = None,
):
    """Yield ``(rewritten_record, image_ok)`` pairs."""
    count = 0
    for record in records:
        if limit is not None and count >= limit:
            return
        rewritten = rewrite_image_paths(record, nuscenes_root)
        paths = rewritten.get("images", [])
        if not paths:
            yield rewritten, False
            count += 1
            continue
        if check_existence:
            all_exist = all(Path(p).is_file() for p in paths)
        else:
            all_exist = True
        if require_image and not all_exist:
            yield rewritten, False
            count += 1
            continue
        yield rewritten, all_exist
        count += 1


def write_records_jsonl(
    out_path: Path,
    pairs,
    *,
    drop_missing: bool,
) -> AdapterStats:
    """Write the rewritten records to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    missing = 0
    skipped_no_image = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for rewritten, image_ok in pairs:
            total += 1
            has_paths = bool(rewritten.get("images"))
            if not has_paths:
                skipped_no_image += 1
                if drop_missing:
                    continue
            elif not image_ok:
                missing += 1
                if drop_missing:
                    continue
            handle.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
            written += 1
    return AdapterStats(
        total=total,
        written=written,
        missing_images=missing,
        skipped_no_image=skipped_no_image,
    )
