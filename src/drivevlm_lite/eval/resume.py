"""Streaming-append + crash-resume helpers for the eval scripts.

The eval script writes ``predictions.jsonl`` in line-buffered append mode so
that progress survives a mid-run crash (OOM, wall-time kill, ssh drop). On
restart, :func:`truncate_to_last_newline` first cleans up any partial line
the last process left behind, :func:`read_jsonl_robust` then loads the
surviving rows, and :func:`check_meta_compatible` refuses to resume if the
caller's args (val_file / ablation / seed / sample_mode / limit /
max_new_tokens) drift from the previous run that wrote those rows.

Kept torch-free so it can be unit-tested with plain python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def truncate_to_last_newline(path: Path) -> int:
    """Ensure ``path`` ends with a complete line, returning bytes removed.

    Line-buffered writes mean a crash mid-line leaves a partial JSON record
    after the last ``\\n``. Resuming with a plain append would concatenate
    new content onto that fragment and corrupt both. This helper truncates
    any trailing partial line so the next append begins on a fresh line.

    Idempotent. Returns 0 for missing / empty / already-clean files.
    """
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data:
        return 0
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        # No newline anywhere → entire content is one partial line.
        path.write_bytes(b"")
        return len(data)
    if last_nl == len(data) - 1:
        return 0
    new_data = data[: last_nl + 1]
    path.write_bytes(new_data)
    return len(data) - len(new_data)


def read_jsonl_robust(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL file, skipping any malformed lines.

    Returns ``(rows, n_skipped)``. Missing / empty files yield ``([], 0)``.
    Use this on resume so a single corrupt line (very rare even with the
    truncate helper above) does not abort the whole eval.
    """
    if not path.exists() or path.stat().st_size == 0:
        return [], 0
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def check_meta_compatible(
    meta_path: Path,
    current_meta: dict[str, Any],
) -> str | None:
    """Compare a saved ``run_meta.json`` to the caller's current args.

    Returns ``None`` when the saved meta is missing (treat as first run) or
    when every key in ``current_meta`` matches the saved meta. Returns a
    human-readable diff string otherwise so the caller can ``raise
    SystemExit(msg)`` and tell the user exactly which knob disagrees.

    Keys present only in the saved meta are ignored (allows the schema to
    grow without breaking resumes).
    """
    if not meta_path.exists():
        return None
    try:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"existing {meta_path} is malformed JSON: {exc}"
    differences = [
        f"  {k}: prev={prev.get(k)!r}  curr={current_meta[k]!r}"
        for k in current_meta
        if prev.get(k) != current_meta[k]
    ]
    if differences:
        return (
            f"resume guard: {meta_path} disagrees with current args:\n"
            + "\n".join(differences)
        )
    return None
