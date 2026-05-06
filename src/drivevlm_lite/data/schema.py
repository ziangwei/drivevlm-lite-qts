from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DrivingSample:
    sample_id: str
    images: list[Path]
    question: str
    answer: str | None = None
    task: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SFTRecord:
    images: list[str]
    messages: list[dict[str, str]]
    sample_id: str
    task: str | None = None


def to_sft_record(sample: DrivingSample) -> SFTRecord:
    if sample.answer is None:
        raise ValueError(f"Sample {sample.sample_id} has no answer and cannot become SFT data.")

    return SFTRecord(
        images=[str(path) for path in sample.images],
        messages=[
            {"role": "user", "content": sample.question},
            {"role": "assistant", "content": sample.answer},
        ],
        sample_id=sample.sample_id,
        task=sample.task,
    )
