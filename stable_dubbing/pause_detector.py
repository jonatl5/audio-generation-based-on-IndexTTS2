from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .utils import audio_duration


ALLOWED_PAUSE_PUNCTUATION = set(".,?!;:…—\n\r")
EXTRA_ALLOWED_PUNCTUATION = set("，。？！；：、–")
_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_ASR_MODEL: Any = None
_ASR_ENGINE: str | None = None
_ASR_ERROR: str | None = None


def _require_pydub():
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except ImportError as exc:
        raise RuntimeError("pydub is required for pause detection. Install project dependencies.") from exc
    return AudioSegment, detect_silence


def detect_silences(
    audio_path: str | Path,
    min_silence_len: int = 350,
    silence_db_offset: float = 25.0,
) -> dict[str, Any]:
    AudioSegment, detect_silence = _require_pydub()
    with Path(audio_path).open("rb") as handle:
        audio = AudioSegment.from_file(handle)
    average_dbfs = float(audio.dBFS)
    if math.isinf(average_dbfs):
        silence_thresh = -80.0
    else:
        silence_thresh = average_dbfs - float(silence_db_offset)

    ranges_ms = detect_silence(
        audio,
        min_silence_len=max(1, int(min_silence_len)),
        silence_thresh=silence_thresh,
    )
    return {
        "audio_length_ms": len(audio),
        "duration_sec": len(audio) / 1000.0,
        "average_dbfs": average_dbfs,
        "silence_thresh": silence_thresh,
        "silences": [
            {
                "start_ms": int(start),
                "end_ms": int(end),
                "duration_ms": int(end - start),
            }
            for start, end in ranges_ms
        ],
    }


def _tokenize(text: str) -> list[dict[str, Any]]:
    return [
        {"word": match.group(0), "start": match.start(), "end": match.end()}
        for match in _WORD_RE.finditer(text)
    ]


def _has_allowed_punctuation(text: str) -> bool:
    return (
        "..." in text
        or any(char in ALLOWED_PAUSE_PUNCTUATION for char in text)
        or any(char in EXTRA_ALLOWED_PUNCTUATION for char in text)
    )


def _nearest_boundary_from_position(text: str, char_position: float) -> dict[str, Any]:
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return {
            "allowed": False,
            "reason": "not enough words to classify pause boundary",
            "after_word": "",
            "before_word": "",
            "word_before_pause": "",
            "word_after_pause": "",
        }

    best: tuple[float, int] | None = None
    for index in range(len(tokens) - 1):
        boundary_center = (tokens[index]["end"] + tokens[index + 1]["start"]) / 2.0
        distance = abs(boundary_center - char_position)
        if best is None or distance < best[0]:
            best = (distance, index)

    assert best is not None
    index = best[1]
    before = tokens[index]
    after = tokens[index + 1]
    gap = text[before["end"] : after["start"]]
    window_start = max(0, before["end"] - 2)
    window_end = min(len(text), after["start"] + 2)
    allowed = _has_allowed_punctuation(gap) or _has_allowed_punctuation(text[window_start:window_end])
    if allowed:
        reason = "long internal pause near allowed punctuation"
    else:
        reason = "long internal pause without nearby punctuation"
    return {
        "allowed": allowed,
        "reason": reason,
        "after_word": before["word"],
        "before_word": after["word"],
        "word_before_pause": before["word"],
        "word_after_pause": after["word"],
    }


def _load_asr_model() -> tuple[Any, str] | None:
    global _ASR_MODEL, _ASR_ENGINE, _ASR_ERROR
    if _ASR_MODEL is not None and _ASR_ENGINE is not None:
        return _ASR_MODEL, _ASR_ENGINE
    if _ASR_ERROR is not None:
        return None

    try:
        from faster_whisper import WhisperModel

        _ASR_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        _ASR_ENGINE = "faster-whisper"
        return _ASR_MODEL, _ASR_ENGINE
    except Exception as first_error:
        try:
            import whisper

            _ASR_MODEL = whisper.load_model("base")
            _ASR_ENGINE = "openai-whisper"
            return _ASR_MODEL, _ASR_ENGINE
        except Exception as second_error:
            _ASR_ERROR = f"faster-whisper failed ({first_error}); openai-whisper failed ({second_error})"
            return None


def _word_timestamps_from_asr(audio_path: str | Path) -> tuple[list[dict[str, Any]], str | None]:
    loaded = _load_asr_model()
    if loaded is None:
        return [], _ASR_ERROR or "ASR unavailable"
    model, engine = loaded
    words: list[dict[str, Any]] = []
    try:
        if engine == "faster-whisper":
            segments, _info = model.transcribe(str(audio_path), word_timestamps=True)
            for segment in segments:
                for word in segment.words or []:
                    words.append({"word": word.word.strip(), "start": float(word.start), "end": float(word.end)})
        else:
            result = model.transcribe(str(audio_path), word_timestamps=True)
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    words.append(
                        {
                            "word": str(word.get("word", "")).strip(),
                            "start": float(word.get("start", 0.0)),
                            "end": float(word.get("end", 0.0)),
                        }
                    )
    except Exception as exc:
        return [], f"ASR word alignment failed: {exc}"
    return [word for word in words if word["word"]], None


def _boundary_from_asr_words(
    text: str,
    pause_start_sec: float,
    pause_end_sec: float,
    duration_sec: float,
    words: list[dict[str, Any]],
) -> dict[str, Any] | None:
    before_words = [word for word in words if float(word["end"]) <= pause_start_sec + 0.05]
    after_words = [word for word in words if float(word["start"]) >= pause_end_sec - 0.05]
    if not before_words or not after_words:
        return None

    before = before_words[-1]["word"].strip(".,?!;:…—，。？！；：、")
    after = after_words[0]["word"].strip(".,?!;:…—，。？！；：、")
    text_tokens = _tokenize(text)
    normalized_before = before.lower()
    normalized_after = after.lower()
    for index in range(len(text_tokens) - 1):
        left = text_tokens[index]
        right = text_tokens[index + 1]
        if left["word"].lower() == normalized_before and right["word"].lower() == normalized_after:
            gap = text[left["end"] : right["start"]]
            allowed = _has_allowed_punctuation(gap)
            return {
                "allowed": allowed,
                "reason": (
                    "long internal pause near allowed punctuation"
                    if allowed
                    else "long internal pause without nearby punctuation"
                ),
                "after_word": left["word"],
                "before_word": right["word"],
                "word_before_pause": left["word"],
                "word_after_pause": right["word"],
                "alignment_method": "asr",
            }

    char_position = (pause_start_sec / max(duration_sec, 0.001)) * len(text)
    result = _nearest_boundary_from_position(text, char_position)
    result["alignment_method"] = "asr_fallback_text_position"
    return result


def classify_pause_allowed(
    text: str,
    pause_start_sec: float,
    pause_end_sec: float,
    duration_sec: float,
    word_timestamps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if word_timestamps:
        asr_result = _boundary_from_asr_words(
            text,
            pause_start_sec,
            pause_end_sec,
            duration_sec,
            word_timestamps,
        )
        if asr_result is not None:
            return asr_result

    char_position = (pause_start_sec / max(duration_sec, 0.001)) * len(text)
    result = _nearest_boundary_from_position(text, char_position)
    result["alignment_method"] = "text_position"
    return result


def analyze_abnormal_pauses(
    audio_path: str,
    text: str,
    min_pause_ms: int = 350,
    silence_db_offset: float = 25.0,
    use_asr_alignment: bool = True,
) -> dict[str, Any]:
    silence_result = detect_silences(
        audio_path,
        min_silence_len=min_pause_ms,
        silence_db_offset=silence_db_offset,
    )
    duration_sec = float(silence_result["duration_sec"])
    total_ms = int(silence_result["audio_length_ms"])
    word_timestamps: list[dict[str, Any]] = []
    asr_note = None
    if use_asr_alignment:
        word_timestamps, asr_note = _word_timestamps_from_asr(audio_path)

    pauses: list[dict[str, Any]] = []
    abnormal_count = 0
    total_abnormal_sec = 0.0
    edge_margin_ms = max(80, min_pause_ms // 2)
    extreme_edge_ms = max(2000, min_pause_ms * 4)

    for silence in silence_result["silences"]:
        start_ms = int(silence["start_ms"])
        end_ms = int(silence["end_ms"])
        pause_duration_ms = int(silence["duration_ms"])
        touches_start = start_ms <= edge_margin_ms
        touches_end = end_ms >= total_ms - edge_margin_ms

        if touches_start or touches_end:
            if pause_duration_ms < extreme_edge_ms:
                continue
            allowed = False
            reason = "extremely long leading silence" if touches_start else "extremely long trailing silence"
            classification = {
                "allowed": allowed,
                "reason": reason,
                "after_word": "",
                "before_word": "",
                "word_before_pause": "",
                "word_after_pause": "",
                "alignment_method": "edge",
            }
        else:
            classification = classify_pause_allowed(
                text,
                start_ms / 1000.0,
                end_ms / 1000.0,
                duration_sec,
                word_timestamps=word_timestamps,
            )
            allowed = bool(classification["allowed"])

        pause = {
            "start_sec": round(start_ms / 1000.0, 3),
            "end_sec": round(end_ms / 1000.0, 3),
            "pause_start": round(start_ms / 1000.0, 3),
            "pause_end": round(end_ms / 1000.0, 3),
            "duration_ms": pause_duration_ms,
            **classification,
        }
        if not pause["allowed"]:
            abnormal_count += 1
            total_abnormal_sec += pause_duration_ms / 1000.0
        pauses.append(pause)

    return {
        "audio_path": str(audio_path),
        "duration_sec": round(duration_sec, 3),
        "average_dbfs": silence_result["average_dbfs"],
        "silence_thresh": silence_result["silence_thresh"],
        "min_pause_ms": int(min_pause_ms),
        "silence_db_offset": float(silence_db_offset),
        "alignment_method": "asr" if word_timestamps else "text_position",
        "asr_note": asr_note,
        "has_abnormal_pause": abnormal_count > 0,
        "abnormal_pause_count": abnormal_count,
        "total_abnormal_pause_sec": round(total_abnormal_sec, 3),
        "pauses": pauses,
    }


def score_pause_analysis(analysis: dict[str, Any], target_duration: float | None = None) -> float:
    score = 0.0
    score += 10.0 * int(analysis.get("abnormal_pause_count", 0))
    score += 5.0 * float(analysis.get("total_abnormal_pause_sec", 0.0))
    if target_duration is not None and target_duration > 0 and analysis.get("duration_sec") is not None:
        score += 3.0 * abs(float(analysis["duration_sec"]) - float(target_duration))
    return round(score, 4)
