from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .emotion_prepare import EMOTION_VECTOR_ORDER, validate_emotion_items
from .utils import read_json, write_json


def load_emotion_plan(path: str | Path) -> list[dict[str, Any]]:
    items = read_json(path)
    errors = validate_emotion_plan(items)
    if errors:
        raise ValueError("Invalid emotion JSON:\n" + "\n".join(f"- {error}" for error in errors))
    return [normalize_line_item(item) for item in items]


def validate_emotion_plan(items: Any) -> list[str]:
    errors = validate_emotion_items(items)
    if not isinstance(items, list):
        return errors

    seen_ids: set[int] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        label = f"item[{index}]"
        try:
            line_id = int(item.get("id"))
        except (TypeError, ValueError):
            errors.append(f"{label}: id must be an integer")
            continue
        if line_id in seen_ids:
            errors.append(f"{label}: duplicate id {line_id}")
        seen_ids.add(line_id)
    return errors


def normalize_line_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["id"] = int(normalized["id"])
    if "start" in normalized:
        normalized["start"] = float(normalized["start"])
    if "end" in normalized:
        normalized["end"] = float(normalized["end"])
    if "target_duration" not in normalized or normalized["target_duration"] in {None, ""}:
        start = float(normalized.get("start", 0.0))
        end = float(normalized.get("end", start))
        normalized["target_duration"] = max(0.0, end - start)
    else:
        normalized["target_duration"] = float(normalized["target_duration"])
    return normalized


def get_line_by_id(items: list[dict[str, Any]], line_id: int) -> dict[str, Any]:
    for item in items:
        if int(item["id"]) == int(line_id):
            return normalize_line_item(item)
    raise KeyError(f"Emotion JSON does not contain line id {line_id}")


def parse_emo_vector(value: str | list[Any]) -> list[float]:
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, list) or len(raw) != len(EMOTION_VECTOR_ORDER):
        raise ValueError(
            f"emo_vector must be an {len(EMOTION_VECTOR_ORDER)}-number list in order {EMOTION_VECTOR_ORDER}"
        )
    vector: list[float] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (int, float)):
            raise ValueError(f"emo_vector[{index}] must be a number")
        value_float = float(item)
        if not 0.0 <= value_float <= 1.0:
            raise ValueError(f"emo_vector[{index}] must be between 0.0 and 1.0")
        vector.append(value_float)
    return vector


def load_emo_vector_file(path: str | Path) -> list[float]:
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("emo_vector")
    return parse_emo_vector(data)


def apply_line_override(
    items: list[dict[str, Any]],
    line_id: int,
    emo_alpha: float | None = None,
    emo_vector: list[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(items)
    override_notes: dict[str, Any] = {}
    changed_line: dict[str, Any] | None = None

    for item in updated:
        if int(item["id"]) != int(line_id):
            continue
        if emo_alpha is not None:
            if not 0.0 <= float(emo_alpha) <= 1.0:
                raise ValueError("--emo-alpha must be between 0.0 and 1.0")
            item["emo_alpha"] = float(emo_alpha)
            override_notes["emo_alpha"] = float(emo_alpha)
        if emo_vector is not None:
            item["emotion_method"] = "emo_vector"
            item["emo_vector"] = parse_emo_vector(emo_vector)
            override_notes["emotion_method"] = "emo_vector"
            override_notes["emo_vector"] = item["emo_vector"]
        changed_line = normalize_line_item(item)
        break

    if changed_line is None:
        raise KeyError(f"Emotion JSON does not contain line id {line_id}")
    return updated, changed_line, override_notes


def write_updated_emotion_plan(items: list[dict[str, Any]], output_dir: str | Path) -> Path:
    return write_json(Path(output_dir) / "emotion_plan.updated.json", items)
