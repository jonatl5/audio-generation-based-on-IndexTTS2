from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def _require_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required for audio assembly. Install project dependencies.") from exc
    return AudioSegment


def concatenate_raw_lines(
    line_audio_paths: list[str | Path],
    output_path: str | Path,
    gap_ms: int = 120,
    logger: logging.Logger | None = None,
) -> list[str]:
    AudioSegment = _require_pydub()
    warnings: list[str] = []
    combined = AudioSegment.silent(duration=0)
    gap = AudioSegment.silent(duration=gap_ms)
    for index, path in enumerate(line_audio_paths):
        audio_path = Path(path)
        if not audio_path.exists():
            warnings.append(f"missing raw line audio: {audio_path}")
            continue
        if index > 0:
            combined += gap
        combined += AudioSegment.from_file(audio_path)
    target = Path(output_path)
    ensure_dir(target.parent)
    combined.export(target, format=target.suffix.lstrip(".") or "wav")
    if logger:
        logger.info("Raw concatenated audio written to %s", target)
    return warnings


def assemble_aligned_track(
    lines: list[dict[str, Any]],
    line_audio_by_id: dict[int, str | Path],
    video_duration: float,
    output_path: str | Path,
    sample_rate: int = 24000,
    logger: logging.Logger | None = None,
) -> list[str]:
    AudioSegment = _require_pydub()
    warnings: list[str] = []
    duration_ms = max(1, int(round(video_duration * 1000)))
    track = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    previous_end = -1.0

    for line in lines:
        line_id = int(line["id"])
        mapped_path = line_audio_by_id.get(line_id)
        if not mapped_path:
            warnings.append(f"line {line_id}: aligned audio missing")
            continue
        path = Path(mapped_path)
        if not path.exists():
            warnings.append(f"line {line_id}: aligned audio missing: {path}")
            continue
        if path.is_dir():
            warnings.append(f"line {line_id}: aligned audio path is a directory: {path}")
            continue
        if line["start"] < previous_end:
            warnings.append(f"line {line_id}: subtitle overlaps previous line")
        previous_end = max(previous_end, float(line["end"]))
        audio = AudioSegment.from_file(path)
        track = track.overlay(audio, position=int(round(float(line["start"]) * 1000)))

    target = Path(output_path)
    ensure_dir(target.parent)
    track.export(target, format=target.suffix.lstrip(".") or "wav")
    if logger:
        logger.info("Timestamp-aligned audio written to %s", target)
    return warnings
