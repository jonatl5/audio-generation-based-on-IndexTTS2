from __future__ import annotations

from pathlib import Path
from typing import Any

from .emotion_plan import normalize_line_item
from .utils import ensure_dir, read_json, write_json


def _require_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required for recombination. Install project dependencies.") from exc
    return AudioSegment


def find_line_audio(lines_dir: str | Path, line_id: int) -> Path | None:
    directory = Path(lines_dir)
    candidates = [
        directory / f"line_{line_id:04d}.wav",
        directory / f"line_{line_id:03d}.wav",
        directory / f"line_{line_id}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_generation_units_manifest(lines_dir: str | Path) -> Path | None:
    directory = Path(lines_dir)
    candidates = [
        directory / "generation_units.json",
        directory.parent / "work" / "generation_units.json",
        directory.parent / "generation_units.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _should_use_timestamps(lines: list[dict[str, Any]], use_timestamps: bool | None) -> bool:
    if use_timestamps is not None:
        return bool(use_timestamps)
    return all("start" in line and "end" in line for line in lines)


def _unit_audio_path(unit: dict[str, Any], lines_dir: str | Path) -> Path | None:
    raw_path = (
        unit.get("aligned_audio_path")
        or unit.get("aligned_output")
        or unit.get("public_output")
        or unit.get("raw_audio_path")
        or unit.get("raw_output")
    )
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
        if not path.is_absolute():
            candidate = Path(lines_dir) / path
            if candidate.exists():
                return candidate
    unit_id = str(unit.get("unit_id") or "")
    if unit_id:
        candidate = Path(lines_dir) / f"{unit_id}.wav"
        if candidate.exists():
            return candidate
    return None


def recombine_generation_units(
    units: list[dict[str, Any]],
    lines_dir: str | Path,
    output_path: str | Path,
    gap_ms: int = 100,
    use_timestamps: bool | None = None,
    crossfade_ms: int = 0,
) -> dict[str, Any]:
    AudioSegment = _require_pydub()
    sorted_units = sorted(units, key=lambda unit: (float(unit.get("start", 0.0)), str(unit.get("unit_id", ""))))
    use_timing = _should_use_timestamps(sorted_units, use_timestamps)
    warnings: list[str] = []
    loaded: list[dict[str, Any]] = []

    for unit in sorted_units:
        unit_id = str(unit.get("unit_id") or "")
        path = _unit_audio_path(unit, lines_dir)
        if path is None:
            warnings.append(f"unit {unit_id}: missing unit audio in {lines_dir}")
            continue
        with path.open("rb") as handle:
            audio = AudioSegment.from_file(handle)
        loaded.append({"unit": unit, "path": path, "audio": audio})

    target = Path(output_path)
    ensure_dir(target.parent)

    if use_timing:
        total_ms = 1
        for item in loaded:
            unit = item["unit"]
            audio = item["audio"]
            start_ms = max(0, int(round(float(unit.get("start", 0.0)) * 1000)))
            end_ms = int(round(float(unit.get("end", unit.get("start", 0.0))) * 1000))
            total_ms = max(total_ms, end_ms, start_ms + len(audio))
            slot_ms = max(0, end_ms - start_ms)
            if slot_ms and len(audio) > slot_ms + 80:
                warnings.append(
                    f"unit {unit.get('unit_id')}: regenerated audio is {len(audio) - slot_ms} ms longer than its timestamp slot"
                )
        combined = AudioSegment.silent(duration=total_ms)
        for item in loaded:
            start_ms = max(0, int(round(float(item["unit"].get("start", 0.0)) * 1000)))
            combined = combined.overlay(item["audio"], position=start_ms)
    else:
        combined = AudioSegment.silent(duration=0)
        for index, item in enumerate(loaded):
            audio = item["audio"]
            if index == 0:
                combined += audio
                continue
            if gap_ms > 0:
                combined += AudioSegment.silent(duration=gap_ms)
                combined += audio
            else:
                fade = min(max(0, int(crossfade_ms)), len(combined), len(audio))
                combined = combined.append(audio, crossfade=fade)

    exported = combined.export(target, format=target.suffix.lstrip(".") or "wav")
    exported.close()
    report = {
        "output": str(target),
        "lines_dir": str(lines_dir),
        "unit_count": len(sorted_units),
        "combined_unit_count": len(loaded),
        "combined_line_count": len(loaded),
        "gap_ms": int(gap_ms),
        "use_timestamps": use_timing,
        "crossfade_ms": int(crossfade_ms),
        "warnings": warnings,
        "units": [
            {
                "unit_id": item["unit"].get("unit_id"),
                "source_line_indices": item["unit"].get("source_line_indices") or item["unit"].get("line_ids", []),
                "audio": str(item["path"]),
                "duration_ms": len(item["audio"]),
                "start": item["unit"].get("start"),
                "end": item["unit"].get("end"),
            }
            for item in loaded
        ],
    }
    write_json(target.with_suffix(target.suffix + ".report.json"), report)
    return report


def recombine_lines(
    emotion_plan: list[dict[str, Any]],
    lines_dir: str | Path,
    output_path: str | Path,
    gap_ms: int = 100,
    use_timestamps: bool | None = None,
    crossfade_ms: int = 0,
) -> dict[str, Any]:
    units_manifest = find_generation_units_manifest(lines_dir)
    if units_manifest is not None:
        return recombine_generation_units(
            read_json(units_manifest),
            lines_dir,
            output_path,
            gap_ms=gap_ms,
            use_timestamps=use_timestamps,
            crossfade_ms=crossfade_ms,
        )

    AudioSegment = _require_pydub()
    lines = sorted((normalize_line_item(item) for item in emotion_plan), key=lambda item: (float(item.get("start", 0.0)), int(item["id"])))
    use_timing = _should_use_timestamps(lines, use_timestamps)
    warnings: list[str] = []
    loaded: list[dict[str, Any]] = []

    for line in lines:
        line_id = int(line["id"])
        path = find_line_audio(lines_dir, line_id)
        if path is None:
            warnings.append(f"line {line_id}: missing line audio in {lines_dir}")
            continue
        with path.open("rb") as handle:
            audio = AudioSegment.from_file(handle)
        loaded.append({"line": line, "path": path, "audio": audio})

    target = Path(output_path)
    ensure_dir(target.parent)

    if use_timing:
        total_ms = 1
        for item in loaded:
            line = item["line"]
            audio = item["audio"]
            start_ms = max(0, int(round(float(line.get("start", 0.0)) * 1000)))
            end_ms = int(round(float(line.get("end", line.get("start", 0.0))) * 1000))
            total_ms = max(total_ms, end_ms, start_ms + len(audio))
            slot_ms = max(0, end_ms - start_ms)
            if slot_ms and len(audio) > slot_ms:
                warnings.append(
                    f"line {line['id']}: regenerated audio is {len(audio) - slot_ms} ms longer than its timestamp slot"
                )
        combined = AudioSegment.silent(duration=total_ms)
        for item in loaded:
            start_ms = max(0, int(round(float(item["line"].get("start", 0.0)) * 1000)))
            combined = combined.overlay(item["audio"], position=start_ms)
    else:
        combined = AudioSegment.silent(duration=0)
        for index, item in enumerate(loaded):
            audio = item["audio"]
            if index == 0:
                combined += audio
                continue
            if gap_ms > 0:
                combined += AudioSegment.silent(duration=gap_ms)
                combined += audio
            else:
                fade = min(max(0, int(crossfade_ms)), len(combined), len(audio))
                combined = combined.append(audio, crossfade=fade)

    exported = combined.export(target, format=target.suffix.lstrip(".") or "wav")
    exported.close()
    report = {
        "output": str(target),
        "lines_dir": str(lines_dir),
        "line_count": len(lines),
        "combined_line_count": len(loaded),
        "gap_ms": int(gap_ms),
        "use_timestamps": use_timing,
        "crossfade_ms": int(crossfade_ms),
        "warnings": warnings,
        "lines": [
            {
                "id": int(item["line"]["id"]),
                "audio": str(item["path"]),
                "duration_ms": len(item["audio"]),
                "start": item["line"].get("start"),
                "end": item["line"].get("end"),
            }
            for item in loaded
        ],
    }
    write_json(target.with_suffix(target.suffix + ".report.json"), report)
    return report
