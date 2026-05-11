from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .emotion_plan import normalize_line_item
from .sentence_groups import blend_group_emotion, build_sentence_groups
from .utils import read_json, write_json


_ENDS_WITH_PUNCTUATION_RE = re.compile(r"[.!?,;:，。！？；：]$")


@dataclass
class GenerationUnit:
    unit_id: str
    source_line_indices: list[int]
    speaker: str
    text: str
    start: float
    end: float
    span_target_duration: float
    summed_line_target_duration: float
    emotion: dict[str, Any]
    is_group: bool
    source_grouping_method: str
    lines: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["line_count"] = len(self.source_line_indices)
        return data


def _unit_id_for(line_ids: list[int]) -> str:
    if len(line_ids) == 1:
        return f"u_{line_ids[0]:04d}"
    return f"u_{line_ids[0]:04d}_{line_ids[-1]:04d}"


def _effective_line(line: dict[str, Any], emotion: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(line)
    for key, value in (emotion or {}).items():
        if value is not None and value != "":
            merged[key] = value
    return normalize_line_item(merged)


def effective_lines_by_id(
    lines: list[dict[str, Any]],
    emotions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    emotion_by_id = {int(item["id"]): normalize_line_item(item) for item in emotions}
    return {
        int(line["id"]): _effective_line(line, emotion_by_id.get(int(line["id"])))
        for line in lines
    }


def normalize_text_for_unit_join(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def join_unit_text(lines: list[dict[str, Any]]) -> str:
    pieces = [normalize_text_for_unit_join(line.get("text", "")) for line in lines]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return ""

    combined = pieces[0]
    for piece in pieces[1:]:
        if not combined:
            combined = piece
        elif _ENDS_WITH_PUNCTUATION_RE.search(combined.rstrip()):
            combined = f"{combined.rstrip()} {piece}"
        else:
            combined = f"{combined.rstrip()} {piece}"
    return combined


def load_manual_groups(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("manual groups JSON must contain an object")
    groups = data.get("groups")
    if not isinstance(groups, list):
        raise ValueError("manual groups JSON must contain a 'groups' list")
    return data


def _manual_group_line_ids(group: dict[str, Any], index: int) -> list[int]:
    if not isinstance(group, dict):
        raise ValueError(f"manual group[{index}] must be an object")
    raw_lines = group.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"manual group[{index}] must include a non-empty lines list")
    line_ids: list[int] = []
    for raw in raw_lines:
        try:
            line_ids.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manual group[{index}] contains a non-integer line number: {raw!r}") from exc
    if len(set(line_ids)) != len(line_ids):
        raise ValueError(f"manual group[{index}] contains duplicate line numbers")
    return line_ids


def _validate_and_sort_manual_groups(
    manual_groups: dict[str, Any],
    effective_by_id: dict[int, dict[str, Any]],
    source_order: list[int],
) -> list[dict[str, Any]]:
    groups = manual_groups.get("groups", [])
    order_index = {line_id: position for position, line_id in enumerate(source_order)}
    covered: set[int] = set()
    normalized_groups: list[dict[str, Any]] = []
    allow_cross_speaker_global = bool(manual_groups.get("allow_cross_speaker", False))

    for index, group in enumerate(groups):
        line_ids = _manual_group_line_ids(group, index)
        missing = [line_id for line_id in line_ids if line_id not in effective_by_id]
        if missing:
            raise ValueError(f"manual group[{index}] references missing source line(s): {missing}")
        overlap = sorted(set(line_ids).intersection(covered))
        if overlap:
            raise ValueError(f"manual group[{index}] overlaps already grouped line(s): {overlap}")

        sorted_line_ids = sorted(line_ids, key=lambda line_id: order_index[line_id])
        positions = [order_index[line_id] for line_id in sorted_line_ids]
        allow_non_contiguous = bool(group.get("allow_non_contiguous", manual_groups.get("allow_non_contiguous", False)))
        if not allow_non_contiguous:
            for previous, current in zip(positions, positions[1:]):
                if current != previous + 1:
                    raise ValueError(
                        f"manual group[{index}] lines must be contiguous in source order: {line_ids}"
                    )

        group_lines = [effective_by_id[line_id] for line_id in sorted_line_ids]
        speakers = {str(line.get("speaker", "")) for line in group_lines}
        allow_cross_speaker = bool(group.get("allow_cross_speaker", allow_cross_speaker_global))
        if len(speakers) > 1 and not allow_cross_speaker:
            raise ValueError(
                f"manual group[{index}] crosses speakers {sorted(speakers)}; "
                "set allow_cross_speaker=true on the group to permit this"
            )

        covered.update(sorted_line_ids)
        normalized = dict(group)
        normalized["lines"] = sorted_line_ids
        normalized["allow_cross_speaker"] = allow_cross_speaker
        normalized_groups.append(normalized)

    return sorted(normalized_groups, key=lambda group: order_index[int(group["lines"][0])])


def _unit_from_lines(
    group_lines: list[dict[str, Any]],
    unit_id: str | None = None,
    text_override: str | None = None,
    source_grouping_method: str = "single",
    warnings: list[str] | None = None,
) -> GenerationUnit:
    if not group_lines:
        raise ValueError("generation unit must include at least one source line")
    line_ids = [int(line["id"]) for line in group_lines]
    start = float(group_lines[0].get("start", 0.0))
    end = float(group_lines[-1].get("end", start))
    summed_target = sum(float(line.get("target_duration", 0.0)) for line in group_lines)
    span_target = max(0.0, end - start)
    speakers = [str(line.get("speaker", "")) for line in group_lines]
    unit_warnings = list(warnings or [])
    if len(set(speakers)) > 1 and "cross_speaker_group" not in unit_warnings:
        unit_warnings.append("cross_speaker_group")
    text = str(text_override).strip() if text_override is not None and str(text_override).strip() else join_unit_text(group_lines)
    return GenerationUnit(
        unit_id=unit_id or _unit_id_for(line_ids),
        source_line_indices=line_ids,
        speaker=speakers[0] if speakers else "",
        text=text,
        start=start,
        end=end,
        span_target_duration=span_target,
        summed_line_target_duration=summed_target,
        emotion=blend_group_emotion(group_lines),
        is_group=len(group_lines) > 1,
        source_grouping_method=source_grouping_method,
        lines=group_lines,
        warnings=unit_warnings,
    )


def build_generation_units(
    lines: list[dict[str, Any]],
    emotions: list[dict[str, Any]],
    manual_groups_path: str | Path | None = None,
    manual_groups_data: dict[str, Any] | None = None,
    enable_auto_groups: bool = True,
) -> list[GenerationUnit]:
    effective_by_id = effective_lines_by_id(lines, emotions)
    source_order = [int(line["id"]) for line in lines]

    if manual_groups_path or manual_groups_data is not None:
        manual_groups = manual_groups_data if manual_groups_data is not None else load_manual_groups(manual_groups_path)  # type: ignore[arg-type]
        if not isinstance(manual_groups, dict):
            raise ValueError("manual groups must be a JSON object")
        normalized_groups = _validate_and_sort_manual_groups(manual_groups, effective_by_id, source_order)
        group_by_first_line = {int(group["lines"][0]): group for group in normalized_groups}
        covered = {int(line_id) for group in normalized_groups for line_id in group["lines"]}
        units: list[GenerationUnit] = []
        for line_id in source_order:
            if line_id in group_by_first_line:
                group = group_by_first_line[line_id]
                group_line_ids = [int(item) for item in group["lines"]]
                group_lines = [effective_by_id[item] for item in group_line_ids]
                unit_id = str(group.get("id") or _unit_id_for(group_line_ids))
                warnings = ["cross_speaker_group"] if len({line["speaker"] for line in group_lines}) > 1 else []
                units.append(
                    _unit_from_lines(
                        group_lines,
                        unit_id=unit_id,
                        text_override=group.get("text"),
                        source_grouping_method="manual",
                        warnings=warnings,
                    )
                )
            elif line_id not in covered:
                units.append(_unit_from_lines([effective_by_id[line_id]], source_grouping_method="single"))
        return units

    if enable_auto_groups:
        units = []
        for group in build_sentence_groups(lines, emotions):
            group_lines = [effective_by_id[int(line_id)] for line_id in group.line_ids]
            units.append(
                _unit_from_lines(
                    group_lines,
                    unit_id=_unit_id_for(group.line_ids),
                    text_override=group.text,
                    source_grouping_method="auto" if group.combined else "single",
                )
            )
        return units

    return [_unit_from_lines([effective_by_id[line_id]], source_grouping_method="single") for line_id in source_order]


def write_generation_units(path: str | Path, units: list[GenerationUnit | dict[str, Any]]) -> Path:
    rows = [unit.to_dict() if isinstance(unit, GenerationUnit) else unit for unit in units]
    return write_json(path, rows)
