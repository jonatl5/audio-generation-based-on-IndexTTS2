from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .utils import write_json


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


class SpeakerMapError(RuntimeError):
    pass


def normalize_speaker_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"\([^()]*\)", " ", stem)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip().casefold()


def _line_get(line: Any, key: str, default: Any = None) -> Any:
    if isinstance(line, dict):
        return line.get(key, default)
    return getattr(line, key, default)


def scan_reference_audio(refs_dir: str | Path) -> dict[str, Path]:
    directory = Path(refs_dir)
    if not directory.exists():
        return {}
    refs: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            refs[normalize_speaker_name(path.name)] = path.resolve()
    return refs


def unique_speakers(lines: list[Any]) -> list[str]:
    seen: set[str] = set()
    speakers: list[str] = []
    for line in lines:
        speaker = str(_line_get(line, "speaker", "")).strip()
        if not speaker or speaker == "MULTI":
            continue
        if speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers


def build_speaker_map(
    lines: list[Any],
    refs_dir: str | Path,
    output_path: str | Path,
    strict: bool = True,
    interactive: bool = True,
) -> tuple[dict[str, str], list[str]]:
    refs = scan_reference_audio(refs_dir)
    warnings: list[str] = []
    mapping: dict[str, str] = {}
    missing: list[str] = []
    narrator = refs.get(normalize_speaker_name("Narrator"))
    first_ref = next(iter(refs.values()), None)

    for speaker in unique_speakers(lines):
        normalized = normalize_speaker_name(speaker)
        matched = refs.get(normalized)
        if matched:
            mapping[speaker] = str(matched)
            continue
        missing.append(speaker)

    if missing and strict:
        raise SpeakerMapError(
            "Missing speaker reference audio for: "
            + ", ".join(missing)
            + f"\nLooked in: {Path(refs_dir).resolve()}"
        )

    for speaker in missing:
        assigned: Path | None = None
        if interactive:
            print(f"Reference audio missing for speaker {speaker!r}.")
            response = input(
                "Enter a path to assign, or press Enter to use Narrator/first available ref: "
            ).strip()
            if response:
                candidate = Path(response).expanduser()
                if candidate.exists():
                    assigned = candidate.resolve()
                else:
                    warnings.append(f"{speaker}: user-provided reference does not exist: {response}")
        if assigned is None:
            assigned = narrator or first_ref
        if assigned is None:
            warnings.append(f"{speaker}: no reference audio available")
            continue
        mapping[speaker] = str(assigned)
        warnings.append(f"{speaker}: assigned fallback reference {assigned}")

    write_json(output_path, mapping)
    return mapping, warnings

