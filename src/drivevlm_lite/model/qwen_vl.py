from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def load_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def build_messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def load_qwen_vl(model_name_or_path: str, **kwargs: Any):
    """Lazy-load Qwen3-VL so importing the package does not require transformers/torch."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return model, processor
