from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 .,_-]", "", text)
    return text


def normalize_answer_relaxed(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def relaxed_exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer_relaxed(prediction) == normalize_answer_relaxed(answer))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer_relaxed(prediction).split()
    answer_tokens = normalize_answer_relaxed(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(answer_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0
    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def yes_no_answer(text: str) -> str | None:
    normalized = normalize_answer_relaxed(text)
    if not normalized:
        return None
    first = normalized.split()[0]
    if first in {"yes", "no"}:
        return first
    return None


def yes_no_match(prediction: str, answer: str) -> float | None:
    gold = yes_no_answer(answer)
    if gold is None:
        return None
    pred = yes_no_answer(prediction)
    return float(pred == gold)


def accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    if len(predictions) != len(answers):
        raise ValueError("predictions and answers must have the same length.")
    if not predictions:
        return 0.0
    return sum(exact_match(p, a) for p, a in zip(predictions, answers)) / len(predictions)
