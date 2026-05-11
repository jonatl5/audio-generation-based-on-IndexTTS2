from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .emotion_plan import normalize_line_item


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class SentenceGroup:
    group_id: str
    line_ids: list[int]
    speaker: str
    start: float
    end: float
    target_duration: float
    text: str
    emotion: dict[str, Any]
    lines: list[dict[str, Any]]
    combined: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["line_count"] = len(self.line_ids)
        return data


def first_word(text: str) -> str:
    match = _WORD_RE.search(text.strip())
    return match.group(0) if match else ""


def starts_with_lowercase_word(text: str) -> bool:
    word = first_word(text)
    return bool(word) and word[0].islower()


def should_join_sentence(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    try:
        contiguous = int(current["id"]) == int(previous["id"]) + 1
    except (KeyError, TypeError, ValueError):
        contiguous = False
    return (
        contiguous
        and str(current.get("speaker", "")) == str(previous.get("speaker", ""))
        and starts_with_lowercase_word(str(current.get("text", "")))
    )


def normalize_text_for_join(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned


def join_group_text(lines: list[dict[str, Any]]) -> str:
    return " ".join(
        normalize_text_for_join(line.get("text", ""))
        for line in lines
        if normalize_text_for_join(line.get("text", ""))
    )


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values) if values else 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _vectors_equal(vectors: list[list[float] | None]) -> bool:
    if not vectors:
        return True
    first = vectors[0]
    return all(vector == first for vector in vectors)


def blend_group_emotion(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines:
        return {}

    weights = [max(0.0, float(line.get("target_duration", 0.0))) for line in lines]
    alphas = [float(line.get("emo_alpha", 0.55)) for line in lines]
    vectors = [line.get("emo_vector") for line in lines]
    unique_texts: list[str] = []
    for line in lines:
        text = str(line.get("emo_text") or "").strip()
        if text and text not in unique_texts:
            unique_texts.append(text)

    blended: dict[str, Any] = {
        "emotion_method": "emo_vector" if any(vector is not None for vector in vectors) else lines[0].get("emotion_method", "emo_text"),
        "emo_text": "; ".join(unique_texts) or lines[0].get("emo_text"),
        "emo_alpha": round(_weighted_average(alphas, weights), 6),
        "use_random": any(bool(line.get("use_random", False)) for line in lines),
    }

    if any(vector is not None for vector in vectors):
        valid_vectors = [list(map(float, vector)) for vector in vectors if isinstance(vector, list)]
        if len(valid_vectors) == len(lines) and _vectors_equal(valid_vectors):
            blended["emo_vector"] = valid_vectors[0]
            blended["emotion_blend"] = "identical"
            blended["emotion_source"] = "copied"
        elif len(valid_vectors) == len(lines):
            blended["emo_vector"] = [
                round(_weighted_average([vector[index] for vector in valid_vectors], weights), 6)
                for index in range(len(valid_vectors[0]))
            ]
            blended["emotion_blend"] = "duration_weighted_by_target_duration"
            blended["emotion_source"] = "blended"
        else:
            blended["emo_vector"] = lines[0].get("emo_vector")
            blended["emotion_blend"] = "fallback_first_vector"
            blended["emotion_source"] = "copied"
    else:
        blended["emo_vector"] = None
        blended["emotion_blend"] = "emo_text"
        blended["emotion_source"] = "copied"
    return blended


def _effective_line(line: dict[str, Any], emotion: dict[str, Any] | None) -> dict[str, Any]:
    emotion = emotion or {}
    merged = dict(line)
    for key, value in emotion.items():
        if value is not None and value != "":
            merged[key] = value
    return normalize_line_item(merged)


def build_sentence_groups(
    lines: list[dict[str, Any]],
    emotions: list[dict[str, Any]],
) -> list[SentenceGroup]:
    emotion_by_id = {int(item["id"]): normalize_line_item(item) for item in emotions}
    effective_lines = [_effective_line(line, emotion_by_id.get(int(line["id"]))) for line in lines]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for line in effective_lines:
        if not current:
            current = [line]
            continue
        if should_join_sentence(current[-1], line):
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    if current:
        groups.append(current)

    sentence_groups: list[SentenceGroup] = []
    for group_lines in groups:
        line_ids = [int(line["id"]) for line in group_lines]
        group_id = f"group_{line_ids[0]:04d}_{line_ids[-1]:04d}"
        target_duration = sum(float(line.get("target_duration", 0.0)) for line in group_lines)
        sentence_groups.append(
            SentenceGroup(
                group_id=group_id,
                line_ids=line_ids,
                speaker=str(group_lines[0].get("speaker", "")),
                start=float(group_lines[0].get("start", 0.0)),
                end=float(group_lines[-1].get("end", group_lines[0].get("start", 0.0))),
                target_duration=target_duration,
                text=join_group_text(group_lines) if len(group_lines) > 1 else str(group_lines[0].get("text", "")),
                emotion=blend_group_emotion(group_lines),
                lines=group_lines,
                combined=len(group_lines) > 1,
            )
        )
    return sentence_groups
