from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import DubbingConfig
from .utils import read_json, write_json


EMOTION_VECTOR_ORDER = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]


def _contains_any(text: str, words: set[str]) -> bool:
    lower = text.lower()
    return any(word in lower for word in words)


def suggest_emotion_text(text: str, speaker: str = "", config: DubbingConfig | None = None) -> tuple[str, float]:
    config = config or DubbingConfig()
    lower = text.lower()
    alpha = config.default_emo_alpha

    danger_words = {"danger", "run", "hide", "attack", "kill", "dead", "death", "help", "monster"}
    confrontation_words = {
        "why",
        "lie",
        "dare",
        "coward",
        "fool",
        "trash",
        "worthless",
        "shut",
        "fight",
    }
    apology_words = {"sorry", "apologize", "forgive", "please", "mistake"}
    sad_words = {"sad", "cry", "alone", "lost", "pain", "hurt", "miss", "regret", "goodbye"}
    villain_words = {"kneel", "obey", "pathetic", "weak", "punish", "submit", "serve"}

    if "?" in text or _contains_any(lower, confrontation_words):
        alpha = min(config.max_emo_alpha, 0.68)
        return "angry, questioning, tense, slightly fast pace", alpha
    if "!" in text and _contains_any(lower, danger_words):
        alpha = min(config.max_emo_alpha, 0.74)
        return "surprised, urgent, fearful tension, quick breath", alpha
    if "!" in text:
        alpha = min(config.max_emo_alpha, 0.66)
        return "surprised, energetic, clear emphasis", alpha
    if _contains_any(lower, apology_words):
        alpha = min(config.max_emo_alpha, 0.62)
        return "apologetic, nervous, low voice, gentle pace", alpha
    if _contains_any(lower, sad_words):
        alpha = min(config.max_emo_alpha, 0.68)
        return "sad, melancholic, restrained, soft voice", alpha
    if _contains_any(lower, villain_words):
        alpha = min(config.max_emo_alpha, 0.65)
        return "arrogant, cold, confident, controlled pace", alpha
    if re.search(r"\b(i|we|he|she|they)\b", lower):
        return "calm narration, steady pace, clear and grounded", alpha
    if speaker.lower() in {"manager", "announcer"}:
        return "formal announcement, clear projection, steady pace", alpha
    return "calm narration, slightly mysterious, steady pace", alpha


def build_emotion_items(lines: list[dict[str, Any]], config: DubbingConfig) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in lines:
        emo_text, alpha = suggest_emotion_text(line["text"], line.get("speaker", ""), config)
        items.append(
            {
                "id": line["id"],
                "speaker": line["speaker"],
                "start": line["start"],
                "end": line["end"],
                "target_duration": line["target_duration"],
                "text": line["text"],
                "emotion_method": "emo_text",
                "emo_text": emo_text,
                "emo_alpha": round(alpha, 3),
                "use_random": bool(config.use_random),
                "emo_vector": None,
            }
        )
    return items


def write_emotion_file(lines: list[dict[str, Any]], path: str | Path, config: DubbingConfig) -> Path:
    return write_json(path, build_emotion_items(lines, config))


def validate_emotion_items(items: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list):
        return ["Emotion file root must be a JSON array."]
    required = {"id", "speaker", "start", "end", "text"}
    for index, item in enumerate(items):
        label = f"item[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}: expected object")
            continue
        for field in required:
            if field not in item:
                errors.append(f"{label}: missing required field {field!r}")
        alpha = item.get("emo_alpha", 0.55)
        if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
            errors.append(f"{label}: emo_alpha must be a number from 0.0 to 1.0")
        vector = item.get("emo_vector")
        if vector is not None:
            if not isinstance(vector, list) or len(vector) != 8:
                errors.append(
                    f"{label}: emo_vector must be null or an 8-number list in order {EMOTION_VECTOR_ORDER}"
                )
            else:
                for vector_index, value in enumerate(vector):
                    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        errors.append(
                            f"{label}: emo_vector[{vector_index}] must be a number from 0.0 to 1.0"
                        )
        method = item.get("emotion_method", "emo_text")
        if method not in {"emo_text", "emo_vector"}:
            errors.append(f"{label}: emotion_method must be 'emo_text' or 'emo_vector'")
        if method == "emo_text" and not item.get("emo_text"):
            errors.append(f"{label}: emo_text is required when emotion_method is 'emo_text'")
    return errors


def load_validated_emotions(path: str | Path) -> list[dict[str, Any]]:
    items = read_json(path)
    errors = validate_emotion_items(items)
    if errors:
        raise ValueError("Invalid emotion file:\n" + "\n".join(f"- {error}" for error in errors))
    return items


def wait_for_emotion_edits(path: str | Path) -> list[dict[str, Any]]:
    emotion_path = Path(path)
    print(
        f"The emotion file has been generated at: {emotion_path}. "
        "Please edit `emo_text`, `emo_alpha`, or `emo_vector`, save the file, "
        "then press Enter to continue."
    )
    while True:
        input()
        items = read_json(emotion_path)
        errors = validate_emotion_items(items)
        if not errors:
            return items
        print("The emotion file has validation errors:")
        for error in errors:
            print(f"- {error}")
        print("Please fix the file, save it, then press Enter to continue.")

