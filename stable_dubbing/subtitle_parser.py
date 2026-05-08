from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DubbingConfig


@dataclass
class ParseResult:
    lines: list["SubtitleLine"]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    unresolved_speakers: list[str] = field(default_factory=list)


@dataclass
class SubtitleLine:
    id: int
    speaker_raw: str
    speaker: str
    gender: str | None
    start: float
    end: float
    target_duration: float
    style: str
    text: str
    source: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = round(self.start, 3)
        data["end"] = round(self.end, 3)
        data["target_duration"] = round(self.target_duration, 3)
        return data


def ass_timestamp_to_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"ASS timestamp must be H:MM:SS.xx, got {value!r}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_timestamp(value: str) -> float:
    cleaned = value.strip().strip("[]")
    parts = cleaned.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(cleaned)


def clean_ass_text(text: str) -> str:
    cleaned = re.sub(r"\{[^}]*\}", "", text)
    cleaned = (
        cleaned.replace(r"\\N", " ")
        .replace(r"\\n", " ")
        .replace(r"\N", " ")
        .replace(r"\n", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_speaker_name(raw_name: str | None) -> tuple[str, str | None]:
    raw = (raw_name or "").strip()
    if not raw:
        return "Unknown", None
    match = re.match(r"^(?P<speaker>.*?)(?:\((?P<gender>[^()]*)\))?\s*$", raw)
    if not match:
        return raw, None
    speaker = re.sub(r"\s+", " ", match.group("speaker").strip())
    gender = match.group("gender")
    return speaker or "Unknown", gender


def _split_multi_speaker_names(raw_name: str) -> list[str]:
    return [part.strip() for part in raw_name.split("&") if part.strip()]


def _split_dash_utterances(text: str) -> list[str]:
    stripped = text.strip()
    if not re.match(r"^[-–—]\s+", stripped):
        return []
    without_first_dash = re.sub(r"^[-–—]\s+", "", stripped)
    parts = [part.strip() for part in re.split(r"\s+[-–—]\s+", without_first_dash)]
    return [part for part in parts if part]


def _line_from_fields(
    line_id: int,
    speaker_raw: str,
    start: float,
    end: float,
    style: str,
    text: str,
    source_file: str,
    source_line_index: int,
    warnings: list[str] | None = None,
) -> SubtitleLine:
    speaker, gender = parse_speaker_name(speaker_raw)
    line_warnings = warnings or []
    if speaker == "Unknown":
        line_warnings = [*line_warnings, "empty speaker name"]
    return SubtitleLine(
        id=line_id,
        speaker_raw=speaker_raw,
        speaker=speaker,
        gender=gender,
        start=start,
        end=end,
        target_duration=max(0.0, end - start),
        style=style,
        text=text,
        source={"file": source_file, "line_index": source_line_index},
        warnings=line_warnings,
    )


def _should_keep_style(style: str, config: DubbingConfig) -> bool:
    if style in set(config.skip_styles):
        return False
    return style in set(config.dialogue_styles)


def parse_ass(path: str | Path, config: DubbingConfig | None = None) -> ParseResult:
    config = config or DubbingConfig()
    source = Path(path)
    warnings: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    lines: list[SubtitleLine] = []
    in_events = False
    format_fields: list[str] | None = None

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_events = stripped.lower() == "[events]"
                continue
            if not in_events:
                continue
            if stripped.startswith("Format:"):
                format_fields = [part.strip() for part in stripped[len("Format:") :].split(",")]
                continue
            if not stripped.startswith("Dialogue:"):
                continue
            if not format_fields:
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "warning": "Dialogue line appeared before Events Format line",
                    }
                )
                continue

            payload = stripped[len("Dialogue:") :].lstrip()
            parts = payload.split(",", maxsplit=len(format_fields) - 1)
            if len(parts) != len(format_fields):
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "warning": f"Expected {len(format_fields)} ASS fields, got {len(parts)}",
                    }
                )
                continue
            row = {field: value for field, value in zip(format_fields, parts)}
            style = row.get("Style", "").strip()
            if not _should_keep_style(style, config):
                continue

            try:
                start = ass_timestamp_to_seconds(row["Start"])
                end = ass_timestamp_to_seconds(row["End"])
            except (KeyError, ValueError) as exc:
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "warning": f"Invalid ASS timestamp: {exc}",
                    }
                )
                continue
            text = clean_ass_text(row.get("Text", ""))
            if not text:
                continue

            speaker_raw = row.get("Name", "").strip()
            raw_speakers = _split_multi_speaker_names(speaker_raw)
            if len(raw_speakers) > 1:
                segments = _split_dash_utterances(text)
                if len(segments) == len(raw_speakers):
                    lengths = [max(len(segment), 1) for segment in segments]
                    total_length = sum(lengths)
                    cursor = start
                    for idx, (speaker_part, segment, length) in enumerate(
                        zip(raw_speakers, segments, lengths)
                    ):
                        segment_end = (
                            end
                            if idx == len(segments) - 1
                            else cursor + (end - start) * (length / total_length)
                        )
                        line_obj = _line_from_fields(
                            len(lines) + 1,
                            speaker_part,
                            cursor,
                            segment_end,
                            style,
                            segment,
                            source.name,
                            line_index,
                            ["split from multi-speaker subtitle"],
                        )
                        if line_obj.speaker == "Unknown":
                            unresolved.add(line_obj.speaker_raw or "Unknown")
                        lines.append(line_obj)
                        cursor = segment_end
                    continue

                warning = (
                    "multi-speaker subtitle could not be safely split; speaker set to MULTI"
                )
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "speaker_raw": speaker_raw,
                        "text": text,
                        "warning": warning,
                    }
                )
                lines.append(
                    SubtitleLine(
                        id=len(lines) + 1,
                        speaker_raw=speaker_raw,
                        speaker="MULTI",
                        gender=None,
                        start=start,
                        end=end,
                        target_duration=max(0.0, end - start),
                        style=style,
                        text=text,
                        source={"file": source.name, "line_index": line_index},
                        warnings=[warning],
                    )
                )
                continue

            line_obj = _line_from_fields(
                len(lines) + 1,
                speaker_raw,
                start,
                end,
                style,
                text,
                source.name,
                line_index,
            )
            if line_obj.speaker == "Unknown":
                unresolved.add(line_obj.speaker_raw or "Unknown")
            lines.append(line_obj)

    return ParseResult(lines=lines, warnings=warnings, unresolved_speakers=sorted(unresolved))


_TXT_BRACKET_RE = re.compile(
    r"^\[(?P<start>.+?)\s*-->\s*(?P<end>.+?)\]\s*(?P<speaker>[^:|]+)\s*:\s*(?P<text>.+)$"
)
_TXT_PIPE_RE = re.compile(
    r"^(?P<start>.+?)\s*-->\s*(?P<end>.+?)\s*\|\s*(?P<speaker>.*?)\s*\|\s*(?P<text>.+)$"
)


def parse_txt(path: str | Path, config: DubbingConfig | None = None) -> ParseResult:
    _ = config or DubbingConfig()
    source = Path(path)
    warnings: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    lines: list[SubtitleLine] = []

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _TXT_BRACKET_RE.match(stripped) or _TXT_PIPE_RE.match(stripped)
            if not match:
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "warning": "Unsupported TXT subtitle line format",
                        "line": stripped,
                    }
                )
                continue
            try:
                start = parse_timestamp(match.group("start"))
                end = parse_timestamp(match.group("end"))
            except ValueError as exc:
                warnings.append(
                    {
                        "source_line_index": line_index,
                        "warning": f"Invalid TXT timestamp: {exc}",
                    }
                )
                continue
            line_obj = _line_from_fields(
                len(lines) + 1,
                match.group("speaker").strip(),
                start,
                end,
                "Default",
                match.group("text").strip(),
                source.name,
                line_index,
            )
            if line_obj.speaker == "Unknown":
                unresolved.add(line_obj.speaker_raw or "Unknown")
            lines.append(line_obj)
    return ParseResult(lines=lines, warnings=warnings, unresolved_speakers=sorted(unresolved))


def parse_subtitle(path: str | Path, config: DubbingConfig | None = None) -> ParseResult:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".ass":
        return parse_ass(source, config)
    if suffix == ".txt":
        return parse_txt(source, config)
    raise ValueError(f"Unsupported subtitle format: {source.suffix}. Use .ass or .txt")
