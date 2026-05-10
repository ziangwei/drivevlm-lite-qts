from __future__ import annotations

import json
import math
import random
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


@dataclass(frozen=True)
class TrajectoryBuildResult:
    samples: list[TrajectorySample]
    stats: dict[str, int]


def _iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk and not buffer.strip():
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"Expected top-level JSON array in {path}")
                    buffer = buffer[1:]
                    started = True
                    continue
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    return
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    item, end_idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield item
                buffer = buffer[end_idx:]
            if not chunk:
                break


def _table_path(nuscenes_root: Path, version: str, name: str) -> Path:
    version_dir = nuscenes_root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"nuScenes version directory not found: {version_dir}")
    path = version_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"nuScenes table not found: {path}")
    return path


def load_nuscenes_tables(
    nuscenes_root: Path,
    version: str,
    future_steps: int,
    cameras: tuple[str, ...],
    candidate_limit: int = 0,
    seed: int | None = None,
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sample")):
        samples[row["token"]] = {
            "token": row["token"],
            "scene_token": row["scene_token"],
            "timestamp": row["timestamp"],
            "next": row.get("next") or "",
            # Official nuScenes sample.json does not store the camera data mapping.
            # It is reconstructed below from sample_data -> calibrated_sensor -> sensor.
            "data": {},
        }

    calibrated_sensor_channels = _load_calibrated_sensor_channels(nuscenes_root, version)
    sample_data: dict[str, dict[str, Any]] = {}
    keyframe_camera_records = 0
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sample_data")):
        sample_token = row.get("sample_token")
        if sample_token not in samples:
            continue
        channel = row.get("channel") or calibrated_sensor_channels.get(row.get("calibrated_sensor_token", ""))
        if channel not in cameras:
            continue
        if row.get("is_key_frame") is False:
            continue
        token = row["token"]
        compact = {
            "token": token,
            "sample_token": sample_token,
            "channel": channel,
            "filename": row["filename"],
            "ego_pose_token": row["ego_pose_token"],
        }
        sample_data[token] = compact
        samples[sample_token]["data"][channel] = token
        keyframe_camera_records += 1

    candidate_tokens = []
    needed_sample_tokens: set[str] = set()
    for sample in samples.values():
        if not all(camera in sample.get("data", {}) for camera in cameras):
            continue
        future_tokens = _future_sample_tokens(sample, samples, future_steps)
        if len(future_tokens) < future_steps:
            continue
        candidate_tokens.append(sample["token"])

    if seed is not None:
        random.Random(seed).shuffle(candidate_tokens)
    total_candidate_tokens = len(candidate_tokens)
    if candidate_limit > 0:
        candidate_tokens = candidate_tokens[:candidate_limit]

    for token in candidate_tokens:
        needed_sample_tokens.add(token)
        needed_sample_tokens.update(_future_sample_tokens(samples[token], samples, future_steps))

    needed_sample_data_tokens: set[str] = set()
    for token in needed_sample_tokens:
        data = samples[token].get("data", {})
        needed_sample_data_tokens.update(str(value) for value in data.values())

    needed_pose_tokens: set[str] = set()
    sample_data = {token: row for token, row in sample_data.items() if token in needed_sample_data_tokens}
    for row in sample_data.values():
        needed_pose_tokens.add(row["ego_pose_token"])

    ego_poses: dict[str, dict[str, Any]] = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "ego_pose")):
        token = row["token"]
        if token not in needed_pose_tokens:
            continue
        ego_poses[token] = {
            "token": token,
            "translation": row["translation"],
            "rotation": row["rotation"],
        }

    return {
        "samples": samples,
        "candidate_tokens": candidate_tokens,
        "sample_data": sample_data,
        "ego_poses": ego_poses,
        "stats": {
            "total_samples": len(samples),
            "total_candidate_tokens": total_candidate_tokens,
            "indexed_candidate_tokens": len(candidate_tokens),
            "indexed_keyframe_camera_records": keyframe_camera_records,
            "needed_sample_data_tokens": len(needed_sample_data_tokens),
            "indexed_sample_data_tokens": len(sample_data),
            "needed_pose_tokens": len(needed_pose_tokens),
            "indexed_pose_tokens": len(ego_poses),
        },
    }


def _load_calibrated_sensor_channels(nuscenes_root: Path, version: str) -> dict[str, str]:
    sensor_channels = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sensor")):
        sensor_channels[row["token"]] = row.get("channel", "")

    calibrated_channels = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "calibrated_sensor")):
        channel = sensor_channels.get(row.get("sensor_token", ""), "")
        if channel:
            calibrated_channels[row["token"]] = channel
    return calibrated_channels


def build_trajectory_samples_with_stats(
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
    future_steps: int = 6,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    max_missing_images: int = 0,
    candidate_limit: int = 0,
    seed: int | None = None,
) -> TrajectoryBuildResult:
    tables = load_nuscenes_tables(
        nuscenes_root,
        version,
        future_steps=future_steps,
        cameras=cameras,
        candidate_limit=candidate_limit,
        seed=seed,
    )
    samples: dict[str, dict[str, Any]] = tables["samples"]
    candidate_tokens: list[str] = tables["candidate_tokens"]
    sample_data: dict[str, dict[str, Any]] = tables["sample_data"]
    ego_poses: dict[str, dict[str, Any]] = tables["ego_poses"]
    stats = dict(tables["stats"])
    stats.update(
        {
            "dropped_missing_camera_keys": 0,
            "dropped_missing_images": 0,
            "dropped_missing_current_pose": 0,
            "dropped_missing_future_pose": 0,
            "valid_trajectory_samples": 0,
        }
    )

    out: list[TrajectorySample] = []
    for token in candidate_tokens:
        sample = samples[token]
        future_tokens = _future_sample_tokens(sample, samples, future_steps)
        if len(future_tokens) < future_steps:
            continue

        data = sample.get("data", {})
        if not all(camera in data for camera in cameras):
            stats["dropped_missing_camera_keys"] += 1
            continue

        image_paths = []
        missing_images = 0
        for camera in cameras:
            sd = sample_data.get(data[camera])
            if sd is None:
                missing_images += 1
                image_paths.append(str(nuscenes_root / f"MISSING/{camera}/{data[camera]}"))
                continue
            image_path = nuscenes_root / sd["filename"]
            if not image_path.exists():
                missing_images += 1
            image_paths.append(str(image_path))
        if missing_images > max_missing_images:
            stats["dropped_missing_images"] += 1
            continue

        current_pose = _pose_for_sample(sample, sample_data, ego_poses)
        if current_pose is None:
            stats["dropped_missing_current_pose"] += 1
            continue
        future_waypoints = []
        for token in future_tokens:
            future_pose = _pose_for_sample(samples[token], sample_data, ego_poses)
            if future_pose is None:
                future_waypoints = []
                break
            x, y, _ = _global_to_ego(future_pose.translation, current_pose)
            future_waypoints.append((round(x, 3), round(y, 3)))
        if len(future_waypoints) < future_steps:
            stats["dropped_missing_future_pose"] += 1
            continue

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
    stats["valid_trajectory_samples"] = len(out)
    return TrajectoryBuildResult(samples=out, stats=stats)


def build_trajectory_samples(
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
    future_steps: int = 6,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    max_missing_images: int = 0,
    candidate_limit: int = 0,
    seed: int | None = None,
) -> list[TrajectorySample]:
    return build_trajectory_samples_with_stats(
        nuscenes_root,
        version=version,
        future_steps=future_steps,
        cameras=cameras,
        max_missing_images=max_missing_images,
        candidate_limit=candidate_limit,
        seed=seed,
    ).samples


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
) -> Pose | None:
    data = sample.get("data", {})
    sd_token = data.get("CAM_FRONT") or next(iter(data.values()))
    sd = sample_data.get(sd_token)
    if sd is None:
        return None
    pose_row = ego_poses.get(sd["ego_pose_token"])
    if pose_row is None:
        return None
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
