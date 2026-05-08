from __future__ import annotations

import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SilenceCleanupConfig
from .utils import audio_duration, ensure_dir, run_command


_SENTENCE_PUNCTUATION = set(".!?。！？")
_CLAUSE_PUNCTUATION = set(";；:：")
_COMMA_PUNCTUATION = set(",，")


@dataclass
class SilenceCleanupResult:
    input_path: str
    output_path: str
    input_duration: float
    output_duration: float
    detected_silences: int
    protected_pauses: int
    applied: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _seconds_to_ms(value: float) -> int:
    return max(0, int(round(value * 1000)))


def _internal_pause_keeps_ms(text: str, config: SilenceCleanupConfig) -> list[int]:
    keeps: list[int] = []
    for match in re.finditer(r"[,，;；:：.!?。！？]+|(?<!\w)[-–—]+(?!\w)", text):
        if not text[: match.start()].strip() or not text[match.end() :].strip():
            continue
        token = match.group(0)
        if any(char in _SENTENCE_PUNCTUATION for char in token):
            keeps.append(_seconds_to_ms(config.keep_sentence_silence))
        elif any(char in _CLAUSE_PUNCTUATION for char in token) or any(char in "-–—" for char in token):
            keeps.append(_seconds_to_ms(config.keep_clause_silence))
        elif any(char in _COMMA_PUNCTUATION for char in token):
            keeps.append(_seconds_to_ms(config.keep_comma_silence))
    return keeps


def _classify_protected_silences(
    silence_ranges: list[list[int]],
    audio_length_ms: int,
    pause_keeps_ms: list[int],
) -> dict[int, int]:
    internal = [
        (index, end - start)
        for index, (start, end) in enumerate(silence_ranges)
        if start > 0 and end < audio_length_ms
    ]
    if not internal or not pause_keeps_ms:
        return {}

    selected = sorted(internal, key=lambda item: item[1], reverse=True)[: len(pause_keeps_ms)]
    selected_indices = [index for index, _duration in selected]
    selected_indices.sort()
    return {index: keep for index, keep in zip(selected_indices, pause_keeps_ms)}


def _ffmpeg_tighten_silence(
    input_path: Path,
    output_path: Path,
    config: SilenceCleanupConfig,
    logger: logging.Logger | None,
) -> None:
    filter_graph = (
        "silenceremove="
        "stop_periods=-1:"
        f"stop_duration={config.min_silence_duration:.3f}:"
        f"stop_threshold={config.silence_threshold_db:.1f}dB:"
        f"stop_silence={config.keep_internal_silence:.3f}"
    )
    run_command(["ffmpeg", "-y", "-i", str(input_path), "-af", filter_graph, str(output_path)], logger=logger)


def tighten_silences(
    input_path: str | Path,
    output_path: str | Path,
    text: str,
    config: SilenceCleanupConfig,
    logger: logging.Logger | None = None,
) -> SilenceCleanupResult:
    source = Path(input_path)
    target = Path(output_path)
    ensure_dir(target.parent)

    input_duration = audio_duration(source)
    if not config.enabled:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return SilenceCleanupResult(
            input_path=str(source),
            output_path=str(target),
            input_duration=input_duration,
            output_duration=audio_duration(target),
            detected_silences=0,
            protected_pauses=0,
            applied=False,
        )

    warnings: list[str] = []
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence

        audio = AudioSegment.from_file(source)
        silence_ranges = detect_silence(
            audio,
            min_silence_len=max(1, _seconds_to_ms(config.min_silence_duration)),
            silence_thresh=float(config.silence_threshold_db),
        )
        if not silence_ranges:
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            return SilenceCleanupResult(
                input_path=str(source),
                output_path=str(target),
                input_duration=input_duration,
                output_duration=audio_duration(target),
                detected_silences=0,
                protected_pauses=0,
                applied=False,
            )

        pause_keeps_ms = (
            _internal_pause_keeps_ms(text, config) if config.preserve_punctuation_pauses else []
        )
        protected = _classify_protected_silences(silence_ranges, len(audio), pause_keeps_ms)
        internal_keep_ms = _seconds_to_ms(config.keep_internal_silence)
        edge_keep_ms = _seconds_to_ms(config.keep_edge_silence)

        tightened = AudioSegment.empty()
        cursor = 0
        for index, (start, end) in enumerate(silence_ranges):
            if start > cursor:
                tightened += audio[cursor:start]

            original_silence_ms = end - start
            if start <= 0 or end >= len(audio):
                keep_ms = min(edge_keep_ms, original_silence_ms)
            elif index in protected:
                keep_ms = min(protected[index], original_silence_ms)
            else:
                keep_ms = min(internal_keep_ms, original_silence_ms)

            if keep_ms > 0:
                tightened += AudioSegment.silent(duration=keep_ms, frame_rate=audio.frame_rate)
            cursor = end

        if cursor < len(audio):
            tightened += audio[cursor:]
        tightened.export(target, format=target.suffix.lstrip(".") or "wav")
    except Exception as exc:
        warnings.append(f"pydub silence cleanup failed; used ffmpeg fallback: {exc}")
        _ffmpeg_tighten_silence(source, target, config, logger)
        return SilenceCleanupResult(
            input_path=str(source),
            output_path=str(target),
            input_duration=input_duration,
            output_duration=audio_duration(target),
            detected_silences=0,
            protected_pauses=0,
            applied=True,
            warnings=warnings,
        )

    output_duration = audio_duration(target)
    if logger:
        logger.info(
            "Silence cleanup %s -> %s duration %.3f -> %.3f detected=%s protected=%s",
            source,
            target,
            input_duration,
            output_duration,
            len(silence_ranges),
            len(protected),
        )
    return SilenceCleanupResult(
        input_path=str(source),
        output_path=str(target),
        input_duration=input_duration,
        output_duration=output_duration,
        detected_silences=len(silence_ranges),
        protected_pauses=len(protected),
        applied=output_duration < input_duration,
        warnings=warnings,
    )
