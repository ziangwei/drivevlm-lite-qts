from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class Pose:
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrajectorySample:
    sample_id: str
    scene_token: str
    timestamp: int
    images: list[str]
    waypoints: list[tuple[float, float]]


def load_nuscenes_tables(nuscenes_root: Path, version: str) -> dict[str, Any]:
    version_dir = nuscenes_root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"nuScenes version directory not found: {version_dir}")

    def load_table(name: str) -> list[dict[str, Any]]:
        path = version_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"nuScenes table not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    samples = load_table("sample")
    sample_data = load_table("sample_data")
    ego_poses = load_table("ego_pose")
    scenes = load_table("scene")

    return {
        "samples": {row["token"]: row for row in samples},
        "sample_data": {row["token"]: row for row in sample_data},
        "ego_poses": {row["token"]: row for row in ego_poses},
        "scenes": {row["token"]: row for row in scenes},
    }


def build_trajectory_samples(
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
    future_steps: int = 6,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    max_missing_images: int = 0,
) -> list[TrajectorySample]:
    tables = load_nuscenes_tables(nuscenes_root, version)
    samples: dict[str, dict[str, Any]] = tables["samples"]
    sample_data: dict[str, dict[str, Any]] = tables["sample_data"]
    ego_poses: dict[str, dict[str, Any]] = tables["ego_poses"]

    out: list[TrajectorySample] = []
    for sample in samples.values():
        future_tokens = _future_sample_tokens(sample, samples, future_steps)
        if len(future_tokens) < future_steps:
            continue

        data = sample.get("data", {})
        if not all(camera in data for camera in cameras):
            continue

        image_paths = []
        missing_images = 0
        for camera in cameras:
            sd = sample_data[data[camera]]
            image_path = nuscenes_root / sd["filename"]
            if not image_path.exists():
                missing_images += 1
            image_paths.append(str(image_path))
        if missing_images > max_missing_images:
            continue

        current_pose = _pose_for_sample(sample, sample_data, ego_poses)
        future_waypoints = []
        for token in future_tokens:
            future_pose = _pose_for_sample(samples[token], sample_data, ego_poses)
            x, y, _ = _global_to_ego(future_pose.translation, current_pose)
            future_waypoints.append((round(x, 3), round(y, 3)))

        out.append(
            TrajectorySample(
                sample_id=sample["token"],
                scene_token=sample["scene_token"],
                timestamp=int(sample["timestamp"]),
                images=image_paths,
                waypoints=future_waypoints,
            )
        )
    out.sort(key=lambda item: (item.scene_token, item.timestamp))
    return out


def trajectory_answer(waypoints: list[tuple[float, float]], step_seconds: float = 0.5) -> str:
    parts = []
    for idx, (x, y) in enumerate(waypoints, start=1):
        t = idx * step_seconds
        parts.append(f"<t={t:.1f},x={x:.3f},y={y:.3f}>")
    return "TRAJ: " + " ".join(parts)


def trajectory_question(future_seconds: float = 3.0, future_steps: int = 6) -> str:
    return (
        "Predict the ego vehicle trajectory for the next "
        f"{future_seconds:.1f} seconds as {future_steps} future waypoints in the current ego frame. "
        "Return only: TRAJ: <t=0.5,x=...,y=...> ..."
    )


def to_sft_row(sample: TrajectorySample, step_seconds: float = 0.5) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "task": "vla_trajectory",
        "images": sample.images,
        "messages": [
            {"role": "user", "content": trajectory_question(step_seconds * len(sample.waypoints), len(sample.waypoints))},
            {"role": "assistant", "content": trajectory_answer(sample.waypoints, step_seconds)},
        ],
        "trajectory": [{"t": round((idx + 1) * step_seconds, 3), "x": x, "y": y} for idx, (x, y) in enumerate(sample.waypoints)],
        "metadata": {
            "scene_token": sample.scene_token,
            "timestamp": sample.timestamp,
        },
    }


def parse_trajectory_text(text: str) -> list[tuple[float, float]]:
    import re

    pattern = re.compile(
        r"<\s*t\s*=\s*[-+]?\d+(?:\.\d+)?\s*,\s*x\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([-+]?\d+(?:\.\d+)?)\s*>"
    )
    return [(float(match.group(1)), float(match.group(2))) for match in pattern.finditer(text)]


def ade(predicted: list[tuple[float, float]], target: list[tuple[float, float]]) -> float | None:
    count = min(len(predicted), len(target))
    if count == 0:
        return None
    return sum(_distance(predicted[idx], target[idx]) for idx in range(count)) / count


def fde(predicted: list[tuple[float, float]], target: list[tuple[float, float]]) -> float | None:
    count = min(len(predicted), len(target))
    if count == 0:
        return None
    return _distance(predicted[count - 1], target[count - 1])


def _future_sample_tokens(sample: dict[str, Any], samples: dict[str, dict[str, Any]], future_steps: int) -> list[str]:
    tokens = []
    next_token = sample.get("next") or ""
    while next_token and len(tokens) < future_steps:
        if next_token not in samples:
            break
        tokens.append(next_token)
        next_token = samples[next_token].get("next") or ""
    return tokens


def _pose_for_sample(
    sample: dict[str, Any],
    sample_data: dict[str, dict[str, Any]],
    ego_poses: dict[str, dict[str, Any]],
) -> Pose:
    data = sample.get("data", {})
    sd_token = data.get("CAM_FRONT") or next(iter(data.values()))
    sd = sample_data[sd_token]
    pose_row = ego_poses[sd["ego_pose_token"]]
    return Pose(
        translation=tuple(float(value) for value in pose_row["translation"]),
        rotation=tuple(float(value) for value in pose_row["rotation"]),
    )


def _global_to_ego(global_xyz: tuple[float, float, float], current_pose: Pose) -> tuple[float, float, float]:
    delta = tuple(global_xyz[idx] - current_pose.translation[idx] for idx in range(3))
    return _rotate_vector(_quat_inverse(current_pose.rotation), delta)


def _quat_inverse(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = quat
    norm = w * w + x * x + y * y + z * z
    return (w / norm, -x / norm, -y / norm, -z / norm)


def _rotate_vector(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vec
    # q * v * q^-1, expanded to avoid extra dependencies.
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
