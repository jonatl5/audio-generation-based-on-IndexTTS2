from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pause_detector import _word_timestamps_from_asr, detect_silences
from .utils import audio_duration, ensure_dir


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class BoundaryCut:
    before_line_id: int
    after_line_id: int
    cut_sec: float
    method: str
    previous_word: str = ""
    next_word: str = ""
    previous_word_end_sec: float | None = None
    next_word_start_sec: float | None = None
    selected_silence: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroupSplitResult:
    audio_path: str
    duration_sec: float
    word_alignment_method: str
    word_alignment_note: str | None
    silence_detection: dict[str, Any]
    cuts: list[BoundaryCut]
    piece_paths: dict[int, str]
    fallback_used: bool
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cuts"] = [cut.to_dict() for cut in self.cuts]
        return data


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(value).lower())


def line_word_tokens(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for line in lines:
        line_id = int(line["id"])
        for match in _WORD_RE.finditer(str(line.get("text", ""))):
            tokens.append(
                {
                    "line_id": line_id,
                    "word": match.group(0),
                    "normalized": normalize_word(match.group(0)),
                    "timestamp": None,
                }
            )
    return tokens


def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word", "")).strip()


def align_text_tokens_to_word_timestamps(
    lines: list[dict[str, Any]],
    word_timestamps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokens = line_word_tokens(lines)
    words = [
        {**word, "normalized": normalize_word(_word_text(word))}
        for word in word_timestamps
        if normalize_word(_word_text(word))
    ]
    word_index = 0
    for token in tokens:
        if word_index >= len(words):
            break
        found_index: int | None = None
        lookahead_end = min(len(words), word_index + 8)
        for candidate_index in range(word_index, lookahead_end):
            if words[candidate_index]["normalized"] == token["normalized"]:
                found_index = candidate_index
                break
        if found_index is None:
            found_index = word_index
            token["alignment_warning"] = "word timestamp matched by order, not text"
        token["timestamp"] = words[found_index]
        word_index = found_index + 1
    return tokens


def _line_boundary_tokens(tokens: list[dict[str, Any]], before_line_id: int, after_line_id: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    before = [token for token in tokens if int(token["line_id"]) == int(before_line_id) and token.get("timestamp")]
    after = [token for token in tokens if int(token["line_id"]) == int(after_line_id) and token.get("timestamp")]
    return (before[-1] if before else None, after[0] if after else None)


def _select_silence_between(
    silences: list[dict[str, Any]],
    previous_end_ms: float,
    next_start_ms: float,
) -> dict[str, Any] | None:
    lower = min(previous_end_ms, next_start_ms) - 120.0
    upper = max(previous_end_ms, next_start_ms) + 120.0
    candidates: list[dict[str, Any]] = []
    for silence in silences:
        start = float(silence["start_ms"])
        end = float(silence["end_ms"])
        midpoint = (start + end) / 2.0
        overlaps = end >= lower and start <= upper
        midpoint_inside = lower <= midpoint <= upper
        if overlaps or midpoint_inside:
            candidates.append(silence)
    if not candidates:
        return None
    target = (previous_end_ms + next_start_ms) / 2.0
    return min(
        candidates,
        key=lambda silence: abs(((float(silence["start_ms"]) + float(silence["end_ms"])) / 2.0) - target),
    )


def align_words_whisperx(
    audio_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    batch_size: int = 16,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    try:
        import whisperx
    except Exception as exc:
        words, fallback_note = _word_timestamps_from_asr(audio_path)
        note = f"WhisperX unavailable: {exc}"
        if fallback_note:
            note = f"{note}; ASR fallback: {fallback_note}"
        return words, "asr_fallback" if words else "unavailable", note

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        audio = whisperx.load_audio(str(audio_path))
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        result = model.transcribe(audio, batch_size=batch_size)
        language_code = language or result.get("language") or "en"
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        aligned = whisperx.align(result.get("segments", []), model_a, metadata, audio, device, return_char_alignments=False)
        raw_words = aligned.get("word_segments") or []
        if not raw_words:
            for segment in aligned.get("segments", []):
                raw_words.extend(segment.get("words", []) or [])
        words = [
            {
                "word": str(word.get("word", "")).strip(),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            }
            for word in raw_words
            if str(word.get("word", "")).strip() and word.get("start") is not None and word.get("end") is not None
        ]
        return words, "whisperx", None
    except Exception as exc:
        if logger:
            logger.exception("WhisperX alignment failed for %s", audio_path)
        words, fallback_note = _word_timestamps_from_asr(audio_path)
        note = f"WhisperX alignment failed: {exc}"
        if fallback_note:
            note = f"{note}; ASR fallback: {fallback_note}"
        return words, "asr_fallback" if words else "unavailable", note


def build_boundary_cuts(
    audio_path: str | Path,
    lines: list[dict[str, Any]],
    word_timestamps: list[dict[str, Any]] | None = None,
    silence_result: dict[str, Any] | None = None,
    manual_cut_points_sec: list[float] | None = None,
    min_pause_ms: int = 180,
    silence_db_offset: float = 25.0,
    min_piece_ms: int = 120,
) -> tuple[list[BoundaryCut], dict[str, Any], bool, list[str]]:
    duration_sec = audio_duration(audio_path)
    duration_ms = duration_sec * 1000.0
    silence_result = silence_result or detect_silences(audio_path, min_silence_len=min_pause_ms, silence_db_offset=silence_db_offset)
    silences = list(silence_result.get("silences", []))
    words = word_timestamps or []
    tokens = align_text_tokens_to_word_timestamps(lines, words) if words else line_word_tokens(lines)
    cuts: list[BoundaryCut] = []
    flags: list[str] = []
    fallback_used = False
    cumulative_target = 0.0
    total_target = sum(max(0.0, float(line.get("target_duration", 0.0))) for line in lines)

    for index in range(len(lines) - 1):
        before_line = lines[index]
        after_line = lines[index + 1]
        before_id = int(before_line["id"])
        after_id = int(after_line["id"])
        cumulative_target += max(0.0, float(before_line.get("target_duration", 0.0)))

        if manual_cut_points_sec and index < len(manual_cut_points_sec):
            cut = BoundaryCut(
                before_line_id=before_id,
                after_line_id=after_id,
                cut_sec=float(manual_cut_points_sec[index]),
                method="manual",
            )
            cuts.append(cut)
            continue

        before_token, after_token = _line_boundary_tokens(tokens, before_id, after_id)
        if before_token and after_token:
            previous_ts = before_token["timestamp"]
            next_ts = after_token["timestamp"]
            previous_end_ms = float(previous_ts["end"]) * 1000.0
            next_start_ms = float(next_ts["start"]) * 1000.0
            selected_silence = _select_silence_between(silences, previous_end_ms, next_start_ms)
            if selected_silence:
                cut_ms = (float(selected_silence["start_ms"]) + float(selected_silence["end_ms"])) / 2.0
                method = "silence_midpoint"
            else:
                cut_ms = (previous_end_ms + next_start_ms) / 2.0
                method = "word_gap_midpoint"
                fallback_used = True
                flags.append("word_gap_midpoint_fallback")
            cuts.append(
                BoundaryCut(
                    before_line_id=before_id,
                    after_line_id=after_id,
                    cut_sec=round(cut_ms / 1000.0, 6),
                    method=method,
                    previous_word=str(before_token["word"]),
                    next_word=str(after_token["word"]),
                    previous_word_end_sec=round(previous_end_ms / 1000.0, 6),
                    next_word_start_sec=round(next_start_ms / 1000.0, 6),
                    selected_silence=selected_silence,
                    warnings=[str(before_token.get("alignment_warning"))] if before_token.get("alignment_warning") else [],
                )
            )
            continue

        ratio = cumulative_target / total_target if total_target > 0 else (index + 1) / len(lines)
        cut_ms = duration_ms * ratio
        fallback_used = True
        flags.append("proportional_duration_fallback")
        cuts.append(
            BoundaryCut(
                before_line_id=before_id,
                after_line_id=after_id,
                cut_sec=round(cut_ms / 1000.0, 6),
                method="proportional_duration",
                warnings=["word alignment unavailable for this boundary"],
            )
        )

    adjusted: list[BoundaryCut] = []
    last_ms = 0.0
    for remaining_index, cut in enumerate(cuts):
        remaining_cuts = len(cuts) - remaining_index - 1
        min_cut = last_ms + float(min_piece_ms)
        max_cut = duration_ms - (remaining_cuts + 1) * float(min_piece_ms)
        cut_ms = max(min_cut, min(max_cut, cut.cut_sec * 1000.0))
        if abs(cut_ms - cut.cut_sec * 1000.0) > 1.0:
            cut.warnings.append("cut clamped to preserve minimum piece duration")
            flags.append("wrong_pause_manual_review")
            fallback_used = True
        cut.cut_sec = round(cut_ms / 1000.0, 6)
        adjusted.append(cut)
        last_ms = cut_ms
    return adjusted, silence_result, fallback_used, sorted(set(flags))


def export_line_pieces(
    audio_path: str | Path,
    lines: list[dict[str, Any]],
    cuts: list[BoundaryCut],
    raw_dir: str | Path,
    suffix: str = "_raw",
) -> dict[int, str]:
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required to split combined sentence audio") from exc

    source = Path(audio_path)
    output_dir = ensure_dir(raw_dir)
    audio = AudioSegment.from_file(source)
    cut_points_ms = [0] + [max(0, int(round(cut.cut_sec * 1000.0))) for cut in cuts] + [len(audio)]
    paths: dict[int, str] = {}
    for index, line in enumerate(lines):
        start_ms = cut_points_ms[index]
        end_ms = cut_points_ms[index + 1]
        line_id = int(line["id"])
        target = output_dir / f"line_{line_id:04d}{suffix}.wav"
        exported = audio[start_ms:end_ms].export(target, format="wav")
        exported.close()
        paths[line_id] = str(target)
    return paths


def split_combined_audio_to_lines(
    audio_path: str | Path,
    lines: list[dict[str, Any]],
    raw_dir: str | Path,
    word_timestamps: list[dict[str, Any]] | None = None,
    word_alignment_method: str = "provided",
    word_alignment_note: str | None = None,
    silence_result: dict[str, Any] | None = None,
    manual_cut_points_sec: list[float] | None = None,
    min_pause_ms: int = 180,
    silence_db_offset: float = 25.0,
    min_piece_ms: int = 120,
) -> GroupSplitResult:
    cuts, detected_silence, fallback_used, flags = build_boundary_cuts(
        audio_path,
        lines,
        word_timestamps=word_timestamps,
        silence_result=silence_result,
        manual_cut_points_sec=manual_cut_points_sec,
        min_pause_ms=min_pause_ms,
        silence_db_offset=silence_db_offset,
        min_piece_ms=min_piece_ms,
    )
    piece_paths = export_line_pieces(audio_path, lines, cuts, raw_dir, suffix="_raw")
    return GroupSplitResult(
        audio_path=str(audio_path),
        duration_sec=round(audio_duration(audio_path), 6),
        word_alignment_method=word_alignment_method,
        word_alignment_note=word_alignment_note,
        silence_detection=detected_silence,
        cuts=cuts,
        piece_paths=piece_paths,
        fallback_used=fallback_used,
        flags=flags,
    )
