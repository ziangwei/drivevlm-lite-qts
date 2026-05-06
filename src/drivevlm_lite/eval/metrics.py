from __future__ import annotations

import re
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 .,_-]", "", text)
    return text


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length.")
    if not predictions:
        return 0.0
    return sum(exact_match(p, a) for p, a in zip(predictions, answers)) / len(predictions)
