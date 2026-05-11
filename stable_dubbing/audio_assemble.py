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


def _unit_audio_path(unit: dict[str, Any]) -> str | Path | None:
    return (
        unit.get("aligned_audio_path")
        or unit.get("aligned_output")
        or unit.get("public_output")
        or unit.get("raw_audio_path")
        or unit.get("raw_output")
    )


def assemble_generation_units_track(
    generation_units: list[dict[str, Any]],
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

    units = sorted(
        generation_units,
        key=lambda unit: (float(unit.get("start", 0.0)), str(unit.get("unit_id", ""))),
    )
    for unit in units:
        unit_id = str(unit.get("unit_id") or unit.get("id") or "")
        mapped_path = _unit_audio_path(unit)
        if not mapped_path:
            warnings.append(f"unit {unit_id}: aligned audio missing")
            continue
        path = Path(mapped_path)
        if not path.exists():
            warnings.append(f"unit {unit_id}: aligned audio missing: {path}")
            continue
        if path.is_dir():
            warnings.append(f"unit {unit_id}: aligned audio path is a directory: {path}")
            continue

        start = float(unit.get("start", 0.0))
        end = float(unit.get("end", start))
        if start < previous_end:
            warnings.append(f"unit {unit_id}: timeline overlaps previous unit")
        previous_end = max(previous_end, end)

        audio = AudioSegment.from_file(path)
        target_ms = max(0, int(round(float(unit.get("span_target_duration", max(0.0, end - start))) * 1000)))
        if target_ms and abs(len(audio) - target_ms) > 80:
            warnings.append(
                f"unit {unit_id}: aligned audio duration differs from unit span by {len(audio) - target_ms} ms"
            )
        track = track.overlay(audio, position=int(round(start * 1000)))

    target = Path(output_path)
    ensure_dir(target.parent)
    track.export(target, format=target.suffix.lstrip(".") or "wav")
    if logger:
        logger.info("Generation-unit aligned audio written to %s", target)
    return warnings


def assemble_generation_units_manifest(
    generation_units_path: str | Path,
    video_duration: float,
    output_path: str | Path,
    sample_rate: int = 24000,
    logger: logging.Logger | None = None,
) -> list[str]:
    import json

    with Path(generation_units_path).open("r", encoding="utf-8") as handle:
        units = json.load(handle)
    if not isinstance(units, list):
        raise ValueError(f"generation units manifest must contain a list: {generation_units_path}")
    return assemble_generation_units_track(
        units,
        video_duration,
        output_path,
        sample_rate=sample_rate,
        logger=logger,
    )
