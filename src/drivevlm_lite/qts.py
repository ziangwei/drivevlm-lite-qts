from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


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
    """Select camera images before VLM inference using a query-aware rule.

    This is a practical QTS-lite input selector. It does not modify Qwen3-VL internals;
    it first tests whether query-conditioned visual input pruning is useful enough to
    justify deeper visual-token selector integration.
    """
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


class QueryAwareTokenSelector(nn.Module):
    """Select visual tokens conditioned on a text/query embedding.

    This is the QTS-lite module. It is intentionally isolated so the first integration
    can happen at the visual-token-to-LLM boundary without rewriting Qwen3-VL internals.
    """

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.25, heads: int = 4, gamma: float = 0.5):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        self.keep_ratio = keep_ratio
        self.gamma = gamma
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(self, visual_tokens: torch.Tensor, query_embedding: torch.Tensor):
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must be [batch, tokens, hidden].")
        if query_embedding.ndim != 2:
            raise ValueError("query_embedding must be [batch, hidden].")

        batch, token_count, hidden = visual_tokens.shape
        query = query_embedding.unsqueeze(1).expand(-1, token_count, -1)
        scores = self.score_mlp(torch.cat([visual_tokens, visual_tokens * query], dim=-1)).squeeze(-1)

        keep_count = max(1, int(token_count * self.keep_ratio))
        topk = scores.topk(keep_count, dim=1).indices
        gather_idx = topk.unsqueeze(-1).expand(-1, -1, hidden)
        kept = visual_tokens.gather(1, gather_idx)

        keep_mask = torch.zeros(batch, token_count, dtype=torch.bool, device=visual_tokens.device)
        keep_mask.scatter_(1, topk, True)

        # Pad dropped tokens per batch so cross-attention can stay batched.
        dropped_rows = []
        max_dropped = token_count - keep_count
        for b_idx in range(batch):
            dropped = visual_tokens[b_idx, ~keep_mask[b_idx]]
            if dropped.shape[0] < max_dropped:
                pad = dropped.new_zeros(max_dropped - dropped.shape[0], hidden)
                dropped = torch.cat([dropped, pad], dim=0)
            dropped_rows.append(dropped)
        dropped_tokens = torch.stack(dropped_rows, dim=0)

        if dropped_tokens.shape[1] > 0:
            attn_out, _ = self.cross_attn(kept, dropped_tokens, dropped_tokens)
            kept = kept + self.gamma * attn_out

        return kept, topk, scores
