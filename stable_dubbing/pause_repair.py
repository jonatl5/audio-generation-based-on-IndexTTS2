from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .pause_detector import analyze_abnormal_pauses, score_pause_analysis
from .utils import audio_duration, ensure_dir


def _require_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required for pause repair. Install project dependencies.") from exc
    return AudioSegment


def _pause_time_ms(pause: dict[str, Any], primary: str, fallback: str) -> int:
    value = pause.get(primary, pause.get(fallback))
    if value is None:
        raise ValueError(f"pause is missing {primary!r}/{fallback!r}")
    return int(round(float(value) * 1000.0))


def _matching_abnormal_pauses(pause_analysis: dict[str, Any], total_ms: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    threshold_ms = int(pause_analysis.get("min_pause_ms") or 350)
    edge_margin_ms = max(80, threshold_ms // 2)
    matches: list[dict[str, Any]] = []
    for index, pause in enumerate(pause_analysis.get("pauses", []) or []):
        try:
            start_ms = _pause_time_ms(pause, "start_sec", "pause_start")
            end_ms = _pause_time_ms(pause, "end_sec", "pause_end")
        except (TypeError, ValueError) as exc:
            warnings.append(f"pause {index} skipped: {exc}")
            continue
        duration_ms = int(pause.get("duration_ms") or max(0, end_ms - start_ms))
        reason = str(pause.get("reason") or "").lower()
        if bool(pause.get("allowed", False)):
            continue
        if duration_ms < threshold_ms:
            continue
        if start_ms <= edge_margin_ms or end_ms >= total_ms - edge_margin_ms:
            continue
        if "leading" in reason or "trailing" in reason:
            continue
        if start_ms < 0 or end_ms > total_ms or end_ms <= start_ms:
            warnings.append(f"pause {index} skipped: invalid range {start_ms}-{end_ms} ms")
            continue
        matches.append({"index": index, "start_ms": start_ms, "end_ms": end_ms, "duration_ms": duration_ms, "pause": pause})
    matches.sort(key=lambda item: int(item["start_ms"]))
    return matches, warnings


def _clamp_keep_ms(target_keep_ms: int, min_keep_ms: int, max_keep_ms: int, original_ms: int) -> int:
    lower = max(1, int(min_keep_ms))
    upper = max(lower, int(max_keep_ms))
    keep = max(lower, min(upper, int(target_keep_ms)))
    return max(1, min(keep, int(original_ms)))


def repair_abnormal_pauses(
    audio_path: str | Path,
    output_path: str | Path,
    pause_analysis: dict[str, Any],
    target_keep_ms: int = 140,
    min_keep_ms: int = 100,
    max_keep_ms: int = 180,
    fade_ms: int = 8,
    crossfade_ms: int = 8,
) -> dict[str, Any]:
    AudioSegment = _require_pydub()
    source = Path(audio_path)
    target = Path(output_path)
    ensure_dir(target.parent)
    audio = AudioSegment.from_file(source)
    original_sec = audio_duration(source)
    result: dict[str, Any] = {
        "input_path": str(source),
        "output_path": str(target),
        "attempted": False,
        "repaired": False,
        "repaired_pause_count": 0,
        "original_duration_sec": round(original_sec, 6),
        "repaired_duration_sec": round(original_sec, 6),
        "duration_removed_sec": 0.0,
        "repaired_pauses": [],
        "warnings": [],
    }
    pauses, warnings = _matching_abnormal_pauses(pause_analysis, len(audio))
    result["warnings"].extend(warnings)
    if not pauses:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return result

    rebuilt = AudioSegment.empty()
    cursor = 0
    removed_ms = 0
    for pause in pauses:
        start_ms = int(pause["start_ms"])
        end_ms = int(pause["end_ms"])
        if start_ms < cursor:
            result["warnings"].append(f"pause {pause['index']} skipped: overlaps earlier repaired pause")
            continue
        original_pause_ms = end_ms - start_ms
        keep_ms = _clamp_keep_ms(target_keep_ms, min_keep_ms, max_keep_ms, original_pause_ms)
        speech = audio[cursor:start_ms]
        if fade_ms > 0 and len(speech) > 0:
            speech = speech.fade_out(min(int(fade_ms), len(speech)))
        rebuilt += speech
        silence = AudioSegment.silent(duration=keep_ms + max(0, int(crossfade_ms)), frame_rate=audio.frame_rate)
        silence = silence.set_channels(audio.channels).set_sample_width(audio.sample_width)
        rebuilt += silence
        new_start_ms = len(rebuilt) - len(silence)
        cursor = end_ms
        removed_ms += max(0, original_pause_ms - keep_ms)
        result["repaired_pauses"].append(
            {
                "index": pause["index"],
                "old_start_sec": round(start_ms / 1000.0, 6),
                "old_end_sec": round(end_ms / 1000.0, 6),
                "old_duration_ms": original_pause_ms,
                "new_start_sec": round(new_start_ms / 1000.0, 6),
                "new_end_sec": round((new_start_ms + keep_ms) / 1000.0, 6),
                "new_duration_ms": keep_ms,
                "word_before_pause": pause["pause"].get("word_before_pause", ""),
                "word_after_pause": pause["pause"].get("word_after_pause", ""),
                "reason": pause["pause"].get("reason", ""),
            }
        )

    tail = audio[cursor:]
    if fade_ms > 0 and result["repaired_pauses"] and len(tail) > 0:
        tail = tail.fade_in(min(int(fade_ms), len(tail)))
    rebuilt += tail
    if not result["repaired_pauses"]:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return result

    exported = rebuilt.export(target, format="wav")
    exported.close()
    repaired_sec = audio_duration(target)
    result.update(
        {
            "attempted": True,
            "repaired": True,
            "repaired_pause_count": len(result["repaired_pauses"]),
            "repaired_duration_sec": round(repaired_sec, 6),
            "duration_removed_sec": round(original_sec - repaired_sec, 6),
            "nominal_duration_removed_sec": round(removed_ms / 1000.0, 6),
        }
    )
    return result


def repair_selected_attempt(
    selected_record: dict[str, Any],
    output_path: str | Path,
    text: str,
    target_duration: float,
    pause_config: Any,
    repair_config: Any,
    analyze_func: Callable[..., dict[str, Any]] = analyze_abnormal_pauses,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": False,
        "status": "skipped",
        "reason": "",
        "selected_attempt": selected_record.get("attempt"),
        "input_path": selected_record.get("audio"),
        "output_path": str(output_path),
        "repair_metadata": None,
        "pause_detection_before_repair": selected_record.get("analysis"),
        "pause_detection_after_repair": None,
        "score_before_repair": selected_record.get("score"),
        "score_after_repair": None,
        "warnings": [],
    }
    audio_path = selected_record.get("audio")
    pause_analysis = selected_record.get("analysis")
    if not audio_path:
        result["reason"] = "selected attempt has no generated audio"
        return result
    if not pause_analysis:
        result["reason"] = "selected attempt has no pause analysis"
        return result
    if not bool(getattr(repair_config, "enabled", True)):
        result["reason"] = "pause repair disabled"
        return result
    if not bool(pause_analysis.get("has_abnormal_pause", False)):
        result["reason"] = "selected attempt has no abnormal pauses"
        return result

    result["attempted"] = True
    try:
        repair_metadata = repair_abnormal_pauses(
            audio_path,
            output_path,
            pause_analysis,
            target_keep_ms=int(getattr(repair_config, "target_keep_ms", 140)),
            min_keep_ms=int(getattr(repair_config, "min_keep_ms", 100)),
            max_keep_ms=int(getattr(repair_config, "max_keep_ms", 180)),
            fade_ms=int(getattr(repair_config, "fade_ms", 8)),
            crossfade_ms=int(getattr(repair_config, "crossfade_ms", 8)),
        )
        result["repair_metadata"] = repair_metadata
        result["warnings"].extend(repair_metadata.get("warnings", []))
        if not repair_metadata.get("repaired"):
            result.update({"status": "skipped", "reason": "no repairable abnormal internal pauses"})
            return result
        after = analyze_func(
            str(output_path),
            text,
            min_pause_ms=int(getattr(pause_config, "min_pause_ms", 350)),
            silence_db_offset=float(getattr(pause_config, "silence_db_offset", 25.0)),
            use_asr_alignment=bool(getattr(pause_config, "use_asr_alignment", False)),
        )
        score_after = score_pause_analysis(after, target_duration)
        result["pause_detection_after_repair"] = after
        result["score_after_repair"] = score_after
        if bool(after.get("has_abnormal_pause", False)):
            result.update({"status": "repaired_still_flagged", "reason": "repair completed but detector still flags pauses"})
        else:
            result.update({"status": "repaired_passed", "reason": "repair completed and pause detection passed"})
        return result
    except Exception as exc:
        result.update({"status": "failed", "reason": str(exc)})
        result["warnings"].append(str(exc))
        return result
