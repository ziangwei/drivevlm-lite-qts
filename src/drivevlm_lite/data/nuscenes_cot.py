from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl
from drivevlm_lite.data.nuscenes_trajectory import parse_trajectory_text, trajectory_answer


FRONT_AGENT_MAX_X_M = 30.0
FRONT_AGENT_MAX_ABS_Y_M = 6.0


@dataclass(frozen=True)
class EgoPose:
    token: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrontAgent:
    category: str
    x: float
    y: float
    relative_speed: float | None


@dataclass(frozen=True)
class CotFeatures:
    ego_speed: float | None
    future_speed: float | None
    future_heading_rad: float
    front_agent: FrontAgent | None


def build_vla_cot_ablation_files(
    train_input: Path,
    val_input: Path,
    out_dir: Path,
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
    train_samples: int = 500,
    val_samples: int = 100,
    step_seconds: float = 0.5,
) -> dict[str, Any]:
    train_rows = _limit_rows(read_jsonl(train_input), train_samples)
    val_rows = _limit_rows(read_jsonl(val_input), val_samples)
    selected_rows = train_rows + val_rows
    features = build_cot_feature_index(selected_rows, nuscenes_root=nuscenes_root, version=version)

    direct_train = [_direct_row(row) for row in train_rows]
    direct_val = [_direct_row(row) for row in val_rows]
    cot_train = [_cot_row(row, features.get(str(row.get("sample_id"))), step_seconds) for row in train_rows]
    cot_val = [_cot_row(row, features.get(str(row.get("sample_id"))), step_seconds) for row in val_rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "direct_train": out_dir / "nuscenes_vla_direct_train.jsonl",
        "direct_val": out_dir / "nuscenes_vla_direct_val.jsonl",
        "cot_train": out_dir / "nuscenes_vla_cot_train.jsonl",
        "cot_val": out_dir / "nuscenes_vla_cot_val.jsonl",
    }
    counts = {
        "direct_train": write_jsonl(paths["direct_train"], direct_train),
        "direct_val": write_jsonl(paths["direct_val"], direct_val),
        "cot_train": write_jsonl(paths["cot_train"], cot_train),
        "cot_val": write_jsonl(paths["cot_val"], cot_val),
    }
    summary = {
        "train_input": str(train_input),
        "val_input": str(val_input),
        "out_dir": str(out_dir),
        "version": version,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "paths": {key: str(value) for key, value in paths.items()},
        "counts": counts,
        "feature_coverage": _feature_coverage(features, selected_rows),
        "example": _example_summary(cot_val[0] if cot_val else cot_train[0] if cot_train else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_md(out_dir / "summary.md", summary)
    return summary


def build_cot_feature_index(
    rows: list[dict[str, Any]],
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
) -> dict[str, CotFeatures]:
    sample_tokens = {str(row.get("sample_id")) for row in rows if row.get("sample_id")}
    sample_table = _load_sample_table(nuscenes_root, version)
    pose_by_sample = _load_pose_by_sample(nuscenes_root, version, sample_table, sample_tokens)
    annotations = _load_front_annotation_candidates(nuscenes_root, version, sample_tokens)
    next_annotation_tokens = {ann["next"] for anns in annotations.values() for ann in anns if ann.get("next")}
    next_annotations = _load_annotations_by_token(nuscenes_root, version, next_annotation_tokens)

    out: dict[str, CotFeatures] = {}
    for row in rows:
        token = str(row.get("sample_id"))
        points = _target(row)
        current_sample = sample_table.get(token)
        current_pose = pose_by_sample.get(token)
        ego_speed = _ego_speed(token, sample_table, pose_by_sample)
        future_speed = _future_speed(points)
        future_heading = _future_heading(points)
        front_agent = None
        if current_sample and current_pose:
            front_agent = _nearest_front_agent(
                annotations.get(token, []),
                next_annotations,
                current_sample,
                sample_table,
                current_pose,
                pose_by_sample,
            )
        out[token] = CotFeatures(
            ego_speed=ego_speed,
            future_speed=future_speed,
            future_heading_rad=future_heading,
            front_agent=front_agent,
        )
    return out


def synthesize_trajectory_cot(features: CotFeatures | None, points: list[tuple[float, float]]) -> str:
    if features is None:
        features = CotFeatures(
            ego_speed=None,
            future_speed=_future_speed(points),
            future_heading_rad=_future_heading(points),
            front_agent=None,
        )

    perception = _perception_step(features)
    prediction = _prediction_step(features)
    planning = _planning_step(features)
    return "\n".join(
        [
            "Reasoning:",
            f"Step 1 (Perception): {perception}",
            f"Step 2 (Prediction): {prediction}",
            f"Step 3 (Planning): {planning}",
        ]
    )


def _direct_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["task"] = "vla_trajectory"
    return out


def _cot_row(row: dict[str, Any], features: CotFeatures | None, step_seconds: float) -> dict[str, Any]:
    points = _target(row)
    answer = trajectory_answer(points, step_seconds=step_seconds)
    cot = synthesize_trajectory_cot(features, points)
    out = dict(row)
    out["task"] = "vla_trajectory_cot"
    out["messages"] = [
        {
            "role": "user",
            "content": (
                "Predict the ego vehicle trajectory for the next "
                f"{step_seconds * len(points):.1f} seconds as {len(points)} future waypoints "
                "in the current ego frame. First provide brief reasoning, then provide the trajectory."
            ),
        },
        {"role": "assistant", "content": f"{cot}\nTrajectory: {answer}"},
    ]
    metadata = dict(row.get("metadata", {}))
    metadata["cot_source"] = "nuscenes_ego_pose_and_sample_annotation"
    if features:
        metadata["cot_features"] = _features_to_json(features)
    out["metadata"] = metadata
    return out


def _load_sample_table(nuscenes_root: Path, version: str) -> dict[str, dict[str, Any]]:
    return {
        row["token"]: {
            "token": row["token"],
            "scene_token": row["scene_token"],
            "timestamp": int(row["timestamp"]),
            "prev": row.get("prev") or "",
            "next": row.get("next") or "",
        }
        for row in _iter_json_array(_table_path(nuscenes_root, version, "sample"))
    }


def _load_pose_by_sample(
    nuscenes_root: Path,
    version: str,
    sample_table: dict[str, dict[str, Any]],
    sample_tokens: set[str],
) -> dict[str, EgoPose]:
    needed_sample_tokens = set(sample_tokens)
    for token in list(sample_tokens):
        sample = sample_table.get(token)
        if not sample:
            continue
        if sample.get("prev"):
            needed_sample_tokens.add(sample["prev"])
        if sample.get("next"):
            needed_sample_tokens.add(sample["next"])

    calibrated_sensor_channels = _load_calibrated_sensor_channels(nuscenes_root, version)
    pose_token_by_sample: dict[str, str] = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sample_data")):
        sample_token = row.get("sample_token")
        if sample_token not in needed_sample_tokens:
            continue
        channel = row.get("channel") or calibrated_sensor_channels.get(row.get("calibrated_sensor_token", ""))
        if channel != "CAM_FRONT":
            continue
        if row.get("is_key_frame") is False:
            continue
        pose_token_by_sample[str(sample_token)] = str(row["ego_pose_token"])

    needed_pose_tokens = set(pose_token_by_sample.values())
    pose_rows: dict[str, EgoPose] = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "ego_pose")):
        token = str(row["token"])
        if token not in needed_pose_tokens:
            continue
        pose_rows[token] = EgoPose(
            token=token,
            translation=tuple(float(value) for value in row["translation"]),
            rotation=tuple(float(value) for value in row["rotation"]),
        )
    return {
        sample_token: pose_rows[pose_token]
        for sample_token, pose_token in pose_token_by_sample.items()
        if pose_token in pose_rows
    }


def _load_front_annotation_candidates(
    nuscenes_root: Path,
    version: str,
    sample_tokens: set[str],
) -> dict[str, list[dict[str, Any]]]:
    instance_categories = _load_instance_categories(nuscenes_root, version)
    annotations: dict[str, list[dict[str, Any]]] = {token: [] for token in sample_tokens}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sample_annotation")):
        sample_token = str(row.get("sample_token", ""))
        if sample_token not in sample_tokens:
            continue
        category = str(row.get("category_name") or instance_categories.get(str(row.get("instance_token", ""))) or "object")
        if not _is_relevant_agent(category):
            continue
        annotations[sample_token].append(
            {
                "token": str(row["token"]),
                "sample_token": sample_token,
                "instance_token": str(row.get("instance_token", "")),
                "translation": tuple(float(value) for value in row["translation"]),
                "category_name": category,
                "next": str(row.get("next") or ""),
            }
        )
    return annotations


def _load_instance_categories(nuscenes_root: Path, version: str) -> dict[str, str]:
    category_names = {
        str(row["token"]): str(row.get("name", "object"))
        for row in _iter_json_array(_table_path(nuscenes_root, version, "category"))
    }
    return {
        str(row["token"]): category_names.get(str(row.get("category_token", "")), "object")
        for row in _iter_json_array(_table_path(nuscenes_root, version, "instance"))
    }


def _load_annotations_by_token(
    nuscenes_root: Path,
    version: str,
    tokens: set[str],
) -> dict[str, dict[str, Any]]:
    if not tokens:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _iter_json_array(_table_path(nuscenes_root, version, "sample_annotation")):
        token = str(row["token"])
        if token not in tokens:
            continue
        out[token] = {
            "token": token,
            "sample_token": str(row.get("sample_token", "")),
            "translation": tuple(float(value) for value in row["translation"]),
            "category_name": str(row.get("category_name", "object")),
        }
    return out


def _nearest_front_agent(
    annotations: list[dict[str, Any]],
    next_annotations: dict[str, dict[str, Any]],
    current_sample: dict[str, Any],
    sample_table: dict[str, dict[str, Any]],
    current_pose: EgoPose,
    pose_by_sample: dict[str, EgoPose],
) -> FrontAgent | None:
    candidates = []
    for ann in annotations:
        rel = _global_to_ego(ann["translation"], current_pose)
        x, y = rel[0], rel[1]
        if x <= 0.0 or x > FRONT_AGENT_MAX_X_M or abs(y) > FRONT_AGENT_MAX_ABS_Y_M:
            continue
        rel_speed = _annotation_relative_speed(
            ann,
            next_annotations,
            current_sample,
            sample_table,
            current_pose,
            pose_by_sample,
            current_rel_x=x,
        )
        candidates.append(
            FrontAgent(
                category=_short_category(str(ann["category_name"])),
                x=round(x, 2),
                y=round(y, 2),
                relative_speed=round(rel_speed, 2) if rel_speed is not None else None,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.x, abs(item.y)))


def _annotation_relative_speed(
    ann: dict[str, Any],
    next_annotations: dict[str, dict[str, Any]],
    current_sample: dict[str, Any],
    sample_table: dict[str, dict[str, Any]],
    current_pose: EgoPose,
    pose_by_sample: dict[str, EgoPose],
    current_rel_x: float,
) -> float | None:
    next_token = ann.get("next")
    next_ann = next_annotations.get(next_token) if next_token else None
    next_sample_token = current_sample.get("next") or ""
    next_sample = sample_table.get(next_sample_token)
    next_pose = pose_by_sample.get(next_sample_token)
    if not next_ann or not next_sample or not next_pose:
        return None
    if next_ann.get("sample_token") != next_sample_token:
        return None
    dt = (int(next_sample["timestamp"]) - int(current_sample["timestamp"])) / 1_000_000.0
    if dt <= 0:
        return None
    next_rel_x = _global_to_ego(next_ann["translation"], next_pose)[0]
    return (next_rel_x - current_rel_x) / dt


def _ego_speed(
    token: str,
    sample_table: dict[str, dict[str, Any]],
    pose_by_sample: dict[str, EgoPose],
) -> float | None:
    current = sample_table.get(token)
    current_pose = pose_by_sample.get(token)
    if not current or not current_pose or not current.get("prev"):
        return None
    previous = sample_table.get(current["prev"])
    previous_pose = pose_by_sample.get(current["prev"])
    if not previous or not previous_pose:
        return None
    dt = (int(current["timestamp"]) - int(previous["timestamp"])) / 1_000_000.0
    if dt <= 0:
        return None
    return _distance_3d(current_pose.translation, previous_pose.translation) / dt


def _target(row: dict[str, Any]) -> list[tuple[float, float]]:
    if row.get("trajectory"):
        return [(float(item["x"]), float(item["y"])) for item in row["trajectory"]]
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return parse_trajectory_text(str(message.get("content", "")))
    return []


def _future_speed(points: list[tuple[float, float]], step_seconds: float = 0.5) -> float | None:
    if not points:
        return None
    previous = (0.0, 0.0)
    speeds = []
    for point in points:
        speeds.append(math.hypot(point[0] - previous[0], point[1] - previous[1]) / step_seconds)
        previous = point
    return sum(speeds[-2:]) / min(2, len(speeds))


def _future_heading(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    x, y = points[-1]
    return math.atan2(y, max(abs(x), 1e-6))


def _perception_step(features: CotFeatures) -> str:
    speed = "unknown speed" if features.ego_speed is None else f"{features.ego_speed:.1f} m/s"
    if features.front_agent is None:
        return f"The ego vehicle is moving at {speed}. No tracked agent is within 30 m in the forward corridor."
    agent = features.front_agent
    rel = "unknown relative speed" if agent.relative_speed is None else f"relative speed {agent.relative_speed:.1f} m/s"
    return (
        f"The ego vehicle is moving at {speed}. The nearest front agent is a {agent.category} "
        f"about {agent.x:.1f} m ahead and {abs(agent.y):.1f} m lateral offset, with {rel}."
    )


def _prediction_step(features: CotFeatures) -> str:
    heading_deg = math.degrees(features.future_heading_rad)
    if abs(features.future_heading_rad) < 0.1:
        path = "The future path remains mostly straight."
    elif features.future_heading_rad > 0:
        path = f"The future path curves left by about {abs(heading_deg):.0f} degrees."
    else:
        path = f"The future path curves right by about {abs(heading_deg):.0f} degrees."

    if features.ego_speed is None or features.future_speed is None:
        return path
    if features.future_speed < max(1.0, features.ego_speed * 0.7):
        trend = "The ego vehicle is expected to slow down."
    elif features.future_speed > max(1.0, features.ego_speed * 1.2):
        trend = "The ego vehicle is expected to accelerate."
    else:
        trend = "The ego vehicle is expected to keep a similar speed."
    return f"{path} {trend}"


def _planning_step(features: CotFeatures) -> str:
    if features.ego_speed is None or features.future_speed is None:
        action = "Follow the predicted safe path."
    elif features.future_speed < 0.7:
        action = "Come close to a stop."
    elif features.future_speed < max(1.0, features.ego_speed * 0.7):
        action = "Decelerate."
    elif features.future_speed > max(1.0, features.ego_speed * 1.2):
        action = "Accelerate moderately."
    else:
        action = "Maintain speed."
    if features.front_agent and features.front_agent.x < 15.0:
        return f"{action} Keep extra margin because a front agent is close."
    return action


def _feature_coverage(features: dict[str, CotFeatures], rows: list[dict[str, Any]]) -> dict[str, int]:
    tokens = [str(row.get("sample_id")) for row in rows]
    values = [features[token] for token in tokens if token in features]
    return {
        "rows": len(rows),
        "features": len(values),
        "ego_speed": sum(1 for item in values if item.ego_speed is not None),
        "future_speed": sum(1 for item in values if item.future_speed is not None),
        "front_agent": sum(1 for item in values if item.front_agent is not None),
    }


def _features_to_json(features: CotFeatures) -> dict[str, Any]:
    return {
        "ego_speed": features.ego_speed,
        "future_speed": features.future_speed,
        "future_heading_rad": features.future_heading_rad,
        "front_agent": None
        if features.front_agent is None
        else {
            "category": features.front_agent.category,
            "x": features.front_agent.x,
            "y": features.front_agent.y,
            "relative_speed": features.front_agent.relative_speed,
        },
    }


def _example_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    answer = ""
    question = ""
    for message in row.get("messages", []):
        if message.get("role") == "user":
            question = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            answer = str(message.get("content", ""))
    return {
        "sample_id": row.get("sample_id"),
        "question": question,
        "answer": answer[:1200],
        "images": row.get("images", [])[:2],
        "metadata": row.get("metadata", {}),
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# VLA CoT Ablation Data",
        "",
        f"- train_input: {summary['train_input']}",
        f"- val_input: {summary['val_input']}",
        f"- out_dir: {summary['out_dir']}",
        f"- train_samples: {summary['train_samples']}",
        f"- val_samples: {summary['val_samples']}",
        "",
        "## Outputs",
        "",
        "| split | rows | path |",
        "| --- | ---: | --- |",
    ]
    for key, output_path in summary["paths"].items():
        lines.append(f"| {key} | {summary['counts'][key]} | {output_path} |")

    lines.extend(["", "## Feature Coverage", "", "| feature | count |", "| --- | ---: |"])
    for key, value in summary["feature_coverage"].items():
        lines.append(f"| {key} | {value} |")

    if summary.get("example"):
        example = summary["example"]
        lines.extend(
            [
                "",
                "## Example",
                "",
                f"- sample_id: {example['sample_id']}",
                f"- first_images: {example['images']}",
                f"- question: {example['question']}",
                "",
                "```text",
                example["answer"],
                "```",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _limit_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return rows[:count] if count > 0 else rows


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
    path = nuscenes_root / version / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"nuScenes table not found: {path}")
    return path


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


def _global_to_ego(global_xyz: tuple[float, float, float], current_pose: EgoPose) -> tuple[float, float, float]:
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
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


def _distance_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _is_relevant_agent(category: str) -> bool:
    return category.startswith("vehicle") or category.startswith("human.pedestrian")


def _short_category(category: str) -> str:
    if category.startswith("vehicle."):
        return category.removeprefix("vehicle.")
    if category.startswith("human.pedestrian"):
        return "pedestrian"
    return category
