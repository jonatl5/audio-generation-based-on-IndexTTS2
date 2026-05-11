from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_json, write_json


def line_report_from_metadata(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("generation_status") or ("failed_generation" if row.get("error") else "passed")
    failed_or_flagged = (
        str(status).startswith("flagged_after_")
        or status in {"failed_generation", "failed_pause_detection", "wrong_pause_manual_review"}
    )
    repair = row.get("pause_repair") or {}
    repair_metadata = repair.get("repair_metadata") if isinstance(repair, dict) else None
    return {
        "id": row.get("line_id"),
        "unit_id": row.get("unit_id"),
        "source_line_indices": row.get("source_line_indices") or row.get("line_ids") or row.get("sentence_group_line_ids", []),
        "speaker": row.get("speaker"),
        "text": row.get("text"),
        "target_duration": row.get("target_duration"),
        "span_target_duration": row.get("span_target_duration", row.get("target_duration")),
        "summed_line_target_duration": row.get("summed_line_target_duration"),
        "start": row.get("start"),
        "end": row.get("end"),
        "status": status,
        "accepted_attempt": row.get("accepted_attempt"),
        "final_audio": row.get("aligned_audio_path") or row.get("public_output") or row.get("aligned_output") or row.get("raw_output"),
        "raw_audio": row.get("raw_audio_path") or row.get("raw_output"),
        "raw_duration": row.get("raw_duration"),
        "final_duration": row.get("final_duration"),
        "duration_scale": row.get("duration_scale"),
        "model_duration_control_used": row.get("model_duration_control_used", False),
        "final_ffmpeg_correction_used": row.get("final_ffmpeg_correction_used", False),
        "pause_retry_count": row.get("pause_retry_count", row.get("attempts", 0)),
        "abnormal_pause_detected": row.get("abnormal_pause_detected", False),
        "final_abnormal_pause_detected": row.get("final_abnormal_pause_detected", row.get("abnormal_pause_detected", False)),
        "passed": not bool(row.get("error")) and not failed_or_flagged,
        "number_of_attempts": row.get("attempts", 0),
        "attempts": row.get("pause_attempts", []),
        "final_score": row.get("final_score"),
        "sentence_group_id": row.get("sentence_group_id"),
        "sentence_group_line_ids": row.get("sentence_group_line_ids", []),
        "combined_text": row.get("combined_text"),
        "combined_audio_uncut": row.get("combined_audio_uncut"),
        "combined_audio_selected": row.get("combined_audio_selected"),
        "split_boundary_cuts": row.get("split_boundary_cuts", []),
        "quality_flags": row.get("quality_flags", []),
        "pause_repair_attempted": row.get("pause_repair_attempted", bool(repair.get("attempted", False))),
        "pause_repair_status": row.get("pause_repair_status", repair.get("status", "skipped")),
        "repaired_audio": row.get("repaired_audio") or repair.get("output_path"),
        "repair_metadata": repair_metadata,
        "pause_detection_before_repair": row.get("pause_detection_before_repair") or repair.get("pause_detection_before_repair"),
        "pause_detection_after_repair": row.get("pause_detection_after_repair") or repair.get("pause_detection_after_repair"),
        "override": row.get("override"),
        "notes": row.get("warnings", []),
        "error": row.get("error"),
    }


def summarize_report_lines(lines: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for item in lines:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "line_count": len(lines),
        "unit_count": len(lines),
        "status_counts": status_counts,
        "flagged_lines": [
            item.get("unit_id") or item.get("id")
            for item in lines
            if str(item.get("status")).startswith("flagged_after_")
            or item.get("status") in {"failed_pause_detection", "failed_generation", "wrong_pause_manual_review"}
            or "duration_speedup_gt_1_5" in item.get("quality_flags", [])
            or "duration_short_lt_0_5" in item.get("quality_flags", [])
        ],
    }


def build_generation_report(
    metadata_rows: list[dict[str, Any]],
    command: str = "",
    emotion_json: str = "",
) -> dict[str, Any]:
    lines = [line_report_from_metadata(row) for row in metadata_rows]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "emotion_json": emotion_json,
        "summary": summarize_report_lines(lines),
        "units": lines,
        "lines": lines,
    }


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    summary = report.get("summary", {})
    lines_md = [
        "# Generation Report",
        "",
        f"- Emotion JSON: {report.get('emotion_json', '')}",
        f"- Units: {summary.get('unit_count', summary.get('line_count', 0))}",
        f"- Status counts: {summary.get('status_counts', {})}",
        "",
        "| Unit | Lines | Speaker | Status | Scale | Raw | Final | Flags | Final audio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.get("lines", []):
        lines_md.append(
            "| {unit} | {lines} | {speaker} | {status} | {scale} | {raw} | {final} | {flags} | {audio} |".format(
                unit=item.get("unit_id") or item.get("id", ""),
                lines=", ".join(str(value) for value in (item.get("source_line_indices") or [item.get("id", "")])),
                speaker=item.get("speaker", ""),
                status=item.get("status", ""),
                scale=item.get("duration_scale", ""),
                raw=item.get("raw_duration", ""),
                final=item.get("final_duration", ""),
                flags=", ".join(item.get("quality_flags", []) or []),
                audio=item.get("final_audio", ""),
            )
        )
        if item.get("sentence_group_id"):
            lines_md.append(
                f"  - group {item.get('sentence_group_id')}: lines {item.get('sentence_group_line_ids')} "
                f"uncut {item.get('combined_audio_uncut', '')} selected {item.get('combined_audio_selected', '')}"
            )
            for cut in item.get("split_boundary_cuts", []):
                lines_md.append(
                    "    - cut {before}->{after} at {cut:.3f}s via {method}".format(
                        before=cut.get("before_line_id", ""),
                        after=cut.get("after_line_id", ""),
                        cut=float(cut.get("cut_sec", 0.0)),
                        method=cut.get("method", ""),
                    )
                )
        if item.get("pause_repair_attempted"):
            repair_metadata = item.get("repair_metadata") or {}
            lines_md.append(
                f"  - repair {item.get('pause_repair_status')}: "
                f"{repair_metadata.get('repaired_pause_count', 0)} pause(s), "
                f"removed {repair_metadata.get('duration_removed_sec', 0.0)}s, "
                f"audio {item.get('repaired_audio', '')}"
            )
        abnormal_attempts = [
            attempt
            for attempt in item.get("attempts", [])
            if int(attempt.get("abnormal_pause_count", 0)) > 0
        ]
        for attempt in abnormal_attempts:
            lines_md.append(
                f"  - line {item.get('id')} attempt {attempt.get('attempt')}: "
                f"{attempt.get('abnormal_pause_count')} abnormal pause(s), score {attempt.get('score')}"
            )
            for pause in attempt.get("pauses", []):
                if pause.get("allowed"):
                    continue
                lines_md.append(
                    "    - {start:.3f}-{end:.3f}s ({duration} ms): {reason}".format(
                        start=float(pause.get("start_sec", pause.get("pause_start", 0.0))),
                        end=float(pause.get("end_sec", pause.get("pause_end", 0.0))),
                        duration=int(pause.get("duration_ms", 0)),
                        reason=pause.get("reason", ""),
                    )
                )
    target.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    return target


def write_json_report(path: str | Path, report: dict[str, Any]) -> Path:
    return write_json(path, report)


def write_generation_report(
    output_dir: str | Path,
    metadata_rows: list[dict[str, Any]],
    command: str = "",
    emotion_json: str = "",
) -> tuple[Path, Path]:
    output = ensure_dir(output_dir)
    report = build_generation_report(metadata_rows, command=command, emotion_json=emotion_json)
    json_path = write_json_report(output / "generation_report.json", report)
    md_path = write_markdown_report(output / "generation_report.md", report)
    return json_path, md_path


def replace_metadata_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    line_id = int(new_row["line_id"])
    replaced = False
    updated: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("line_id", -1)) == line_id:
            updated.append(new_row)
            replaced = True
        else:
            updated.append(row)
    if not replaced:
        updated.append(new_row)
    return sorted(updated, key=lambda item: int(item.get("line_id", 0)))


def update_line_report(output_dir: str | Path, metadata_row: dict[str, Any]) -> tuple[Path, Path]:
    output = ensure_dir(output_dir)
    json_path = output / "generation_report.json"
    existing = read_json(json_path) if json_path.exists() else {"lines": []}
    new_item = line_report_from_metadata(metadata_row)
    line_id = int(new_item["id"])
    lines = [
        item for item in existing.get("lines", [])
        if int(item.get("id", -1)) != line_id
    ]
    lines.append(new_item)
    lines.sort(key=lambda item: int(item.get("id", 0)))
    existing["generated_at"] = datetime.now(timezone.utc).isoformat()
    existing["summary"] = summarize_report_lines(lines)
    existing["lines"] = lines
    write_json_report(json_path, existing)
    md_path = write_markdown_report(output / "generation_report.md", existing)
    return json_path, md_path
