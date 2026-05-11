from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from .boundary_split import align_words_whisperx, split_combined_audio_to_lines
from .config import DubbingConfig
from .duration_align import choose_alignment_action, render_aligned_audio
from .generation_units import GenerationUnit, build_generation_units, write_generation_units
from .pause_detector import analyze_abnormal_pauses, score_pause_analysis
from .pause_repair import repair_abnormal_pauses, repair_selected_attempt
from .sentence_groups import SentenceGroup, build_sentence_groups
from .silence_cleanup import tighten_silences
from .utils import audio_duration, ensure_dir, write_json, write_jsonl


def split_text_by_punctuation(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?。！？；;])\s+", text) if piece.strip()]
    if len(pieces) <= 1:
        pieces = [piece.strip() for piece in re.split(r"\s*,\s*", text) if piece.strip()]
    return pieces if len(pieces) > 1 else []


def punctuation_gap_after_segment_ms(text: str, config: DubbingConfig) -> int:
    stripped = text.rstrip()
    if stripped.endswith(tuple(".!?。！？")):
        seconds = config.silence_cleanup.keep_sentence_silence
    elif stripped.endswith(tuple(";；:：-–—")):
        seconds = config.silence_cleanup.keep_clause_silence
    else:
        seconds = config.silence_cleanup.keep_comma_silence
    return max(0, int(round(seconds * 1000)))


class IndexTTS2Wrapper:
    def __init__(self, config: DubbingConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("stable_dubbing.tts")
        self.tts: Any = None
        self.repo_path = Path(config.indextts_repo_path).resolve()
        self.cfg_path = (
            Path(config.indextts_cfg_path)
            if config.indextts_cfg_path
            else self.repo_path / "checkpoints" / "config.yaml"
        ).resolve()
        self.model_dir = (
            Path(config.indextts_model_dir)
            if config.indextts_model_dir
            else self.repo_path / "checkpoints"
        ).resolve()

    def initialize(self) -> None:
        if self.tts is not None:
            return
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"IndexTTS2 repo not found at {self.repo_path}. Run setup_env.sh or set indextts_repo_path."
            )
        if not self.cfg_path.exists():
            raise FileNotFoundError(
                f"IndexTTS2 config not found at {self.cfg_path}. Download IndexTeam/IndexTTS-2 checkpoints."
            )
        expected_emo_model = self.model_dir / "qwen0.6bemo4-merge"
        if not expected_emo_model.exists():
            raise FileNotFoundError(
                "IndexTTS2 checkpoints appear incomplete. Missing emotion model directory: "
                f"{expected_emo_model}. Download the full IndexTeam/IndexTTS-2 checkpoint set."
            )
        sys.path.insert(0, str(self.repo_path.resolve()))
        try:
            from indextts.infer_v2 import IndexTTS2
        except ModuleNotFoundError as exc:
            if exc.name == "torchaudio":
                raise RuntimeError(
                    "IndexTTS2 could not import torchaudio in the Python environment "
                    f"currently running this pipeline: {sys.executable}. Install a torchaudio "
                    "build that matches your installed torch/CUDA version, then rerun."
                ) from exc
            raise

        self.logger.info(
            "Initializing IndexTTS2 cfg_path=%s model_dir=%s use_fp16=%s use_cuda_kernel=%s use_deepspeed=%s",
            self.cfg_path,
            self.model_dir,
            self.config.use_fp16,
            self.config.use_cuda_kernel,
            self.config.use_deepspeed,
        )
        self.tts = IndexTTS2(
            cfg_path=str(self.cfg_path),
            model_dir=str(self.model_dir),
            use_fp16=self.config.use_fp16,
            use_cuda_kernel=self.config.use_cuda_kernel,
            use_deepspeed=self.config.use_deepspeed,
        )

    def infer_line(
        self,
        spk_audio_prompt: str,
        text: str,
        output_path: str | Path,
        emotion: dict[str, Any],
        emo_alpha_override: float | None = None,
        use_random_override: bool | None = None,
        interval_silence: int | None = None,
        duration_scale: float | None = None,
    ) -> None:
        self.initialize()
        assert self.tts is not None
        method = emotion.get("emotion_method", "emo_text")
        emo_alpha = float(emo_alpha_override if emo_alpha_override is not None else emotion.get("emo_alpha", 0.55))
        use_random = bool(use_random_override if use_random_override is not None else emotion.get("use_random", False))
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": spk_audio_prompt,
            "text": text,
            "output_path": str(output_path),
            "emo_alpha": emo_alpha,
            "use_random": use_random,
            "verbose": True,
        }
        if interval_silence is not None:
            kwargs["interval_silence"] = int(interval_silence)
        if duration_scale is not None:
            kwargs["duration_scale"] = float(duration_scale)
        if method == "emo_vector" and emotion.get("emo_vector") is not None:
            kwargs["emo_vector"] = emotion["emo_vector"]
            kwargs["use_emo_text"] = False
        else:
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emotion.get("emo_text") or text
        self.logger.info("IndexTTS2 infer line text=%r params=%s", text, {k: v for k, v in kwargs.items() if k != "text"})
        self.tts.infer(**kwargs)


def _exceeds_stretch_cap(raw_duration: float, target_duration: float, config: DubbingConfig) -> bool:
    if target_duration <= 0 or config.alignment.max_stretch_speed_factor <= 0:
        return False
    return raw_duration / target_duration > config.alignment.max_stretch_speed_factor


def _effective_line_value(line: dict[str, Any], emotion: dict[str, Any], key: str, default: Any = None) -> Any:
    value = emotion.get(key)
    if value not in {None, ""}:
        return value
    return line.get(key, default)


def _effective_line(line: dict[str, Any], emotion: dict[str, Any]) -> dict[str, Any]:
    start = float(_effective_line_value(line, emotion, "start", 0.0))
    end = float(_effective_line_value(line, emotion, "end", start))
    target_duration = _effective_line_value(line, emotion, "target_duration", None)
    if target_duration in {None, ""}:
        target_duration = max(0.0, end - start)
    return {
        **line,
        "id": int(_effective_line_value(line, emotion, "id", line.get("id"))),
        "speaker": str(_effective_line_value(line, emotion, "speaker", line.get("speaker", ""))),
        "text": str(_effective_line_value(line, emotion, "text", line.get("text", ""))),
        "start": start,
        "end": end,
        "target_duration": float(target_duration),
    }


def _disabled_pause_analysis(audio_path: str | Path, target_duration: float | None = None) -> dict[str, Any]:
    duration = audio_duration(audio_path)
    analysis = {
        "audio_path": str(audio_path),
        "duration_sec": round(duration, 3),
        "has_abnormal_pause": False,
        "abnormal_pause_count": 0,
        "total_abnormal_pause_sec": 0.0,
        "pauses": [],
        "skipped": "pause detection disabled",
    }
    analysis["score"] = score_pause_analysis(analysis, target_duration)
    return analysis


def _attempt_prefix(item_id: int | str) -> str:
    if isinstance(item_id, int):
        return f"line_{item_id:04d}"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item_id)).strip("._-")
    return text or "unit"


def generate_with_pause_retries(
    line_id: int | str,
    text: str,
    target_duration: float,
    attempt_dir: str | Path,
    config: DubbingConfig,
    infer_attempt: Callable[[Path, int], None],
    logger: logging.Logger | None = None,
    analyze_func: Callable[..., dict[str, Any]] = analyze_abnormal_pauses,
) -> dict[str, Any]:
    pause_config = config.pause_detection
    attempts_dir = ensure_dir(attempt_dir)
    max_attempts = max(1, int(pause_config.max_retries))
    attempt_records: list[dict[str, Any]] = []
    selected_record: dict[str, Any] | None = None
    repair_result: dict[str, Any] | None = None
    status = "failed_generation"
    prefix = _attempt_prefix(line_id)

    for attempt_number in range(1, max_attempts + 1):
        attempt_path = attempts_dir / f"{prefix}_attempt_{attempt_number}.wav"
        record: dict[str, Any] = {
            "attempt": attempt_number,
            "audio": str(attempt_path),
            "generated": False,
            "abnormal_pause_count": 0,
            "score": None,
            "pauses": [],
        }
        try:
            infer_attempt(attempt_path, attempt_number)
        except Exception as exc:
            record.update({"error": str(exc), "score": 999999.0})
            attempt_records.append(record)
            if logger:
                logger.exception("Line %s attempt %s generation failed", line_id, attempt_number)
            continue
        record["generated"] = True

        try:
            if pause_config.enabled:
                analysis = analyze_func(
                    str(attempt_path),
                    text,
                    min_pause_ms=int(pause_config.min_pause_ms),
                    silence_db_offset=float(pause_config.silence_db_offset),
                    use_asr_alignment=bool(pause_config.use_asr_alignment),
                )
                score = score_pause_analysis(analysis, target_duration)
            else:
                analysis = _disabled_pause_analysis(attempt_path, target_duration)
                score = float(analysis["score"])
        except Exception as exc:
            record.update(
                {
                    "error": f"pause detection failed: {exc}",
                    "score": 999998.0,
                    "pause_detection_error": str(exc),
                }
            )
            attempt_records.append(record)
            selected_record = record
            status = "failed_pause_detection"
            if logger:
                logger.exception("Line %s pause detection failed after attempt %s", line_id, attempt_number)
            break

        record.update(
            {
                "duration_sec": analysis.get("duration_sec"),
                "abnormal_pause_count": int(analysis.get("abnormal_pause_count", 0)),
                "has_abnormal_pause": bool(analysis.get("has_abnormal_pause", False)),
                "total_abnormal_pause_sec": analysis.get("total_abnormal_pause_sec", 0.0),
                "score": score,
                "pauses": analysis.get("pauses", []),
                "analysis": analysis,
            }
        )
        attempt_records.append(record)

        if not record["has_abnormal_pause"]:
            selected_record = record
            status = "passed" if attempt_number == 1 else "regenerated_then_passed"
            break

        if logger:
            logger.info(
                "Line %s attempt %s abnormal pauses=%s score=%.4f; retrying if attempts remain",
                line_id,
                attempt_number,
                record["abnormal_pause_count"],
                score,
            )

    if selected_record is None:
        generated_records = [record for record in attempt_records if record.get("generated")]
        if generated_records:
            selected_record = min(generated_records, key=lambda item: float(item.get("score") or 999999.0))
            status = f"flagged_after_{max_attempts}_attempts"
            if pause_config.enabled and int(selected_record.get("abnormal_pause_count", 0)) > 0:
                repair_path = attempts_dir / f"{prefix}_attempt_{selected_record.get('attempt')}_repaired.wav"
                repair_result = repair_selected_attempt(
                    selected_record,
                    repair_path,
                    text,
                    target_duration,
                    pause_config,
                    config.pause_repair,
                    analyze_func=analyze_func,
                )
                if repair_result.get("status") == "repaired_passed":
                    status = "wrong_pause_auto_cut"
                elif repair_result.get("status") in {"repaired_still_flagged", "failed"}:
                    status = "wrong_pause_manual_review"
                    if logger:
                        logger.info(
                            "Line %s pause repair status=%s reason=%s",
                            line_id,
                            repair_result.get("status"),
                            repair_result.get("reason"),
                        )
        elif attempt_records:
            selected_record = attempt_records[-1]

    selected_audio = selected_record.get("audio") if selected_record and selected_record.get("generated") else None
    final_score = selected_record.get("score") if selected_record else None
    final_abnormal_pause_detected = bool(
        selected_record and int(selected_record.get("abnormal_pause_count", 0)) > 0
    )
    if repair_result and repair_result.get("status") in {"repaired_passed", "repaired_still_flagged"}:
        selected_audio = repair_result.get("output_path") or selected_audio
        if repair_result.get("score_after_repair") is not None:
            final_score = repair_result.get("score_after_repair")
        after_repair = repair_result.get("pause_detection_after_repair") or {}
        final_abnormal_pause_detected = bool(after_repair.get("has_abnormal_pause", False))
    accepted_attempt = selected_record.get("attempt") if selected_record else None
    return {
        "status": status,
        "selected_audio": selected_audio,
        "selected_record": selected_record,
        "accepted_attempt": accepted_attempt,
        "attempts": attempt_records,
        "attempt_count": len(attempt_records),
        "final_score": final_score,
        "final_abnormal_pause_detected": final_abnormal_pause_detected,
        "abnormal_pause_detected": bool(
            selected_record and int(selected_record.get("abnormal_pause_count", 0)) > 0
        ),
        "pause_repair": repair_result,
        "pause_detection_before_repair": repair_result.get("pause_detection_before_repair") if repair_result else None,
        "pause_detection_after_repair": repair_result.get("pause_detection_after_repair") if repair_result else None,
        "error": None if selected_audio else "all generation attempts failed",
    }


def duration_quality_flags(raw_duration: float, target_duration: float, config: DubbingConfig) -> list[str]:
    if target_duration <= 0:
        return []
    ratio = raw_duration / target_duration
    flags: list[str] = []
    if ratio > config.alignment.max_stretch_speed_factor:
        flags.append("duration_speedup_gt_1_5")
    if ratio < 0.5:
        flags.append("duration_short_lt_0_5")
    return flags


def align_raw_line_audio(
    raw_output: str | Path,
    aligned_output: str | Path,
    public_output: str | Path,
    target_duration: float,
    config: DubbingConfig,
    logger: logging.Logger | None = None,
    action_prefix: str = "",
) -> tuple[float, dict[str, Any]]:
    raw_duration = audio_duration(raw_output)
    final_decision = choose_alignment_action(raw_duration, target_duration, config.alignment)
    action_label = final_decision.action
    if action_prefix:
        if raw_duration > target_duration:
            action_label = f"{action_prefix}_time_stretch"
        else:
            action_label = f"{action_prefix}_pad_silence_end"
    result = render_aligned_audio(
        raw_output,
        aligned_output,
        target_duration,
        config.alignment,
        logger=logger,
        force=final_decision.should_regenerate,
        action_label=action_label,
    )
    shutil.copy2(aligned_output, public_output)
    metadata = {
        "raw_duration": raw_duration,
        "final_duration": result.final_duration,
        "alignment_action": result.alignment_action,
        "alignment_warnings": result.warnings,
        "quality_flags": duration_quality_flags(raw_duration, target_duration, config),
    }
    return raw_duration, metadata


def repair_raw_piece_if_needed(
    raw_output: Path,
    line_id: int,
    text: str,
    target_duration: float,
    config: DubbingConfig,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if not config.pause_detection.enabled or not config.pause_repair.enabled:
        return {"status": "skipped", "attempted": False, "flags": []}
    try:
        analysis = analyze_abnormal_pauses(
            str(raw_output),
            text,
            min_pause_ms=int(config.pause_detection.min_pause_ms),
            silence_db_offset=float(config.pause_detection.silence_db_offset),
            use_asr_alignment=bool(config.pause_detection.use_asr_alignment),
        )
    except Exception as exc:
        return {"status": "failed_detection", "attempted": False, "reason": str(exc), "flags": ["wrong_pause_manual_review"]}
    if not analysis.get("has_abnormal_pause"):
        return {"status": "passed", "attempted": False, "pause_detection_before_repair": analysis, "flags": []}

    uncut_path = raw_output.with_name(f"line_{line_id:04d}_raw_uncut.wav")
    shutil.copy2(raw_output, uncut_path)
    repaired_path = raw_output.with_name(f"line_{line_id:04d}_raw_repaired.wav")
    repair = repair_abnormal_pauses(
        uncut_path,
        repaired_path,
        analysis,
        target_keep_ms=int(config.pause_repair.target_keep_ms),
        min_keep_ms=int(config.pause_repair.min_keep_ms),
        max_keep_ms=int(config.pause_repair.max_keep_ms),
        fade_ms=int(config.pause_repair.fade_ms),
        crossfade_ms=int(config.pause_repair.crossfade_ms),
    )
    result: dict[str, Any] = {
        "status": "wrong_pause_manual_review",
        "attempted": True,
        "input_path": str(uncut_path),
        "output_path": str(repaired_path),
        "pause_detection_before_repair": analysis,
        "repair_metadata": repair,
        "flags": ["wrong_pause_manual_review"],
    }
    if repair.get("repaired"):
        try:
            after = analyze_abnormal_pauses(
                str(repaired_path),
                text,
                min_pause_ms=int(config.pause_detection.min_pause_ms),
                silence_db_offset=float(config.pause_detection.silence_db_offset),
                use_asr_alignment=bool(config.pause_detection.use_asr_alignment),
            )
            result["pause_detection_after_repair"] = after
            result["score_after_repair"] = score_pause_analysis(after, target_duration)
            shutil.copy2(repaired_path, raw_output)
            if after.get("has_abnormal_pause"):
                result["status"] = "wrong_pause_manual_review"
            else:
                result["status"] = "wrong_pause_auto_cut"
                result["flags"] = ["wrong_pause_auto_cut"]
        except Exception as exc:
            result["reason"] = str(exc)
            if logger:
                logger.exception("Line %s repaired pause recheck failed", line_id)
    return result


def synthesize_line(
    line: dict[str, Any],
    emotion: dict[str, Any],
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
    wrapper: IndexTTS2Wrapper | None = None,
    logger: logging.Logger | None = None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger("stable_dubbing.tts")
    output = Path(output_dir)
    raw_dir = ensure_dir(output / "lines" / "raw")
    aligned_dir = ensure_dir(output / "lines" / "aligned")
    public_lines_dir = ensure_dir(output / "lines")
    attempts_dir = ensure_dir(
        output / "attempts" if config.pause_detection.save_attempts else output / "work" / "attempts"
    )
    effective_line = _effective_line(line, emotion)
    line_id = int(effective_line["id"])
    speaker = effective_line["speaker"]
    text = effective_line["text"]
    target_duration = float(effective_line["target_duration"])
    raw_output = raw_dir / f"line_{line_id:04d}.wav"
    aligned_output = aligned_dir / f"line_{line_id:04d}.wav"
    public_output = public_lines_dir / f"line_{line_id:04d}.wav"
    warnings: list[str] = []
    speaker_reference = speaker_map.get(speaker)
    generation_status = "failed_generation"

    metadata = {
        "line_id": line_id,
        "speaker": speaker,
        "text": text,
        "start": effective_line["start"],
        "end": effective_line["end"],
        "target_duration": target_duration,
        "method": emotion.get("emotion_method", "emo_text"),
        "emo_text": emotion.get("emo_text"),
        "emo_vector": emotion.get("emo_vector"),
        "emo_alpha": emotion.get("emo_alpha"),
        "use_random": emotion.get("use_random", False),
        "speaker_reference": speaker_reference,
        "raw_output": str(raw_output),
        "aligned_output": str(aligned_output),
        "public_output": str(public_output),
        "raw_duration": None,
        "final_duration": None,
        "alignment_action": None,
        "attempts": 0,
        "accepted_attempt": None,
        "pause_attempts": [],
        "abnormal_pause_detected": False,
        "final_abnormal_pause_detected": False,
        "final_score": None,
        "generation_status": generation_status,
        "pause_repair_attempted": False,
        "pause_repair_status": "skipped",
        "pause_repair": None,
        "pause_detection_before_repair": None,
        "pause_detection_after_repair": None,
        "silence_cleanup": None,
        "quality_flags": [],
        "warnings": warnings,
        "override": override or {},
    }

    try:
        if not emotion:
            raise RuntimeError(f"No emotion JSON item found for line {line_id}")
        if not speaker_reference:
            raise RuntimeError(f"No speaker reference mapped for {speaker!r}")
        if wrapper is None:
            wrapper = IndexTTS2Wrapper(config, logger=logger)
        wrapper.initialize()

        def infer_attempt(path: Path, _attempt_number: int) -> None:
            wrapper.infer_line(speaker_reference, text, path, emotion)

        retry_result = generate_with_pause_retries(
            line_id,
            text,
            target_duration,
            attempts_dir,
            config,
            infer_attempt,
            logger=logger,
        )
        generation_status = str(retry_result["status"])
        metadata.update(
            {
                "attempts": retry_result["attempt_count"],
                "accepted_attempt": retry_result["accepted_attempt"],
                "pause_attempts": retry_result["attempts"],
                "abnormal_pause_detected": retry_result["abnormal_pause_detected"],
                "final_abnormal_pause_detected": retry_result["final_abnormal_pause_detected"],
                "final_score": retry_result["final_score"],
                "generation_status": generation_status,
                "pause_repair": retry_result.get("pause_repair"),
                "pause_repair_attempted": bool((retry_result.get("pause_repair") or {}).get("attempted", False)),
                "pause_repair_status": (retry_result.get("pause_repair") or {}).get("status", "skipped"),
                "pause_detection_before_repair": retry_result.get("pause_detection_before_repair"),
                "pause_detection_after_repair": retry_result.get("pause_detection_after_repair"),
            }
        )
        if not retry_result["selected_audio"]:
            raise RuntimeError(str(retry_result.get("error") or "all generation attempts failed"))

        selected_candidate = Path(str(retry_result["selected_audio"]))
        cleanup = tighten_silences(
            selected_candidate,
            raw_output,
            text,
            config.silence_cleanup,
            logger=logger,
        )
        raw_duration = cleanup.output_duration
        metadata["silence_cleanup"] = cleanup.to_dict()
        warnings.extend(cleanup.warnings)
        decision = choose_alignment_action(raw_duration, target_duration, config.alignment)
        logger.info(
            "Line %s selected attempt=%s raw_duration=%.3f target=%.3f action=%s pause_status=%s",
            line_id,
            retry_result["accepted_attempt"],
            raw_duration,
            target_duration,
            decision.action,
            generation_status,
        )

        split_done = False
        if _exceeds_stretch_cap(raw_duration, target_duration, config):
            speed_factor = raw_duration / target_duration
            segments = split_text_by_punctuation(text)
            if segments:
                try:
                    from pydub import AudioSegment

                    segment_audio = AudioSegment.silent(duration=0)
                    for segment_index, segment in enumerate(segments, start=1):
                        segment_path = raw_dir / f"line_{line_id:04d}_segment_{segment_index}.wav"
                        segment_tight_path = raw_dir / f"line_{line_id:04d}_segment_{segment_index}_tight.wav"
                        wrapper.infer_line(speaker_reference, segment, segment_path, emotion)
                        tighten_silences(
                            segment_path,
                            segment_tight_path,
                            segment,
                            config.silence_cleanup,
                            logger=logger,
                        )
                        if segment_index > 1:
                            gap = AudioSegment.silent(
                                duration=punctuation_gap_after_segment_ms(segments[segment_index - 2], config)
                            )
                            segment_audio += gap
                        segment_audio += AudioSegment.from_file(segment_tight_path)
                    segment_audio.export(raw_output, format="wav")
                    raw_duration = audio_duration(raw_output)
                    split_done = True
                    warnings.append(
                        f"speed factor {speed_factor:.3f} exceeds cap "
                        f"{config.alignment.max_stretch_speed_factor:.3f}; regenerated by punctuation split"
                    )
                except Exception as exc:
                    warnings.append(f"punctuation split fallback failed: {exc}")
            else:
                warnings.append(
                    f"speed factor {speed_factor:.3f} exceeds cap "
                    f"{config.alignment.max_stretch_speed_factor:.3f}; no punctuation split available"
                )

        final_decision = choose_alignment_action(raw_duration, target_duration, config.alignment)
        if split_done:
            action_label = (
                "split_then_time_stretch"
                if raw_duration > target_duration
                else "split_then_pad_silence_end"
            )
        elif final_decision.should_regenerate:
            action_label = "time_stretch"
        else:
            action_label = final_decision.action
        result = render_aligned_audio(
            raw_output,
            aligned_output,
            target_duration,
            config.alignment,
            logger=logger,
            force=final_decision.should_regenerate,
            action_label=action_label,
        )

        shutil.copy2(aligned_output, public_output)
        warnings.extend(result.warnings)
        quality_flags = duration_quality_flags(raw_duration, target_duration, config)
        metadata.update(
            {
                "raw_duration": raw_duration,
                "final_duration": result.final_duration,
                "alignment_action": result.alignment_action,
                "quality_flags": quality_flags,
                "warnings": warnings,
                "generation_status": generation_status,
            }
        )
    except Exception as exc:
        logger.exception("Failed to synthesize line %s", line_id)
        metadata.update(
            {
                "error": str(exc),
                "attempts": metadata.get("attempts", 0),
                "warnings": warnings,
                "generation_status": "failed_generation" if generation_status != "failed_pause_detection" else generation_status,
            }
        )
    return metadata


def synthesize_sentence_group(
    group: SentenceGroup,
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
    wrapper: IndexTTS2Wrapper,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger = logger or logging.getLogger("stable_dubbing.tts")
    output = Path(output_dir)
    raw_dir = ensure_dir(output / "lines" / "raw")
    aligned_dir = ensure_dir(output / "lines" / "aligned")
    public_lines_dir = ensure_dir(output / "lines")
    groups_dir = ensure_dir(output / "groups" / group.group_id)
    attempts_dir = ensure_dir(
        (output / "attempts" if config.pause_detection.save_attempts else output / "work" / "attempts") / group.group_id
    )
    speaker_reference = speaker_map.get(group.speaker)
    warnings: list[str] = []
    group_status = "failed_generation"
    group_metadata: dict[str, Any] = {
        "group_id": group.group_id,
        "line_ids": group.line_ids,
        "speaker": group.speaker,
        "text": group.text,
        "target_duration": group.target_duration,
        "start": group.start,
        "end": group.end,
        "emotion": group.emotion,
        "speaker_reference": speaker_reference,
        "attempts": [],
        "accepted_attempt": None,
        "selected_audio": None,
        "uncut_audio": None,
        "repaired_audio": None,
        "split": None,
        "status": group_status,
        "quality_flags": [],
        "warnings": warnings,
    }

    rows: list[dict[str, Any]] = []
    try:
        if not speaker_reference:
            raise RuntimeError(f"No speaker reference mapped for {group.speaker!r}")

        def infer_attempt(path: Path, _attempt_number: int) -> None:
            wrapper.infer_line(speaker_reference, group.text, path, group.emotion)

        retry_result = generate_with_pause_retries(
            group.line_ids[0],
            group.text,
            group.target_duration,
            attempts_dir,
            config,
            infer_attempt,
            logger=logger,
        )
        group_status = str(retry_result["status"])
        selected_record = retry_result.get("selected_record") or {}
        selected_audio = Path(str(retry_result["selected_audio"])) if retry_result.get("selected_audio") else None
        if selected_audio is None:
            raise RuntimeError(str(retry_result.get("error") or "all group generation attempts failed"))

        selected_copy = groups_dir / f"{group.group_id}_selected.wav"
        shutil.copy2(selected_audio, selected_copy)
        uncut_audio = selected_record.get("audio")
        repair = retry_result.get("pause_repair") or {}
        group_metadata.update(
            {
                "attempts": retry_result["attempts"],
                "accepted_attempt": retry_result["accepted_attempt"],
                "selected_audio": str(selected_copy),
                "uncut_audio": uncut_audio,
                "repaired_audio": repair.get("output_path"),
                "status": group_status,
                "pause_repair": retry_result.get("pause_repair"),
                "abnormal_pause_detected": retry_result.get("abnormal_pause_detected", False),
                "final_abnormal_pause_detected": retry_result.get("final_abnormal_pause_detected", False),
                "final_score": retry_result.get("final_score"),
            }
        )

        grouping_config = config.sentence_grouping
        words, word_method, word_note = align_words_whisperx(
            selected_copy,
            model_name=str(grouping_config.whisperx_model),
            language=str(grouping_config.whisperx_language or "") or None,
            device=str(grouping_config.whisperx_device or "") or None,
            compute_type=str(grouping_config.whisperx_compute_type or "") or None,
            batch_size=int(grouping_config.whisperx_batch_size),
            logger=logger,
        )
        split = split_combined_audio_to_lines(
            selected_copy,
            group.lines,
            raw_dir,
            word_timestamps=words,
            word_alignment_method=word_method,
            word_alignment_note=word_note,
            min_pause_ms=int(grouping_config.boundary_min_silence_ms),
            silence_db_offset=float(config.pause_detection.silence_db_offset),
            min_piece_ms=int(config.pause_repair.min_piece_ms),
        )
        group_flags = list(split.flags)
        if split.fallback_used:
            group_flags.append("wrong_pause_manual_review")
        group_metadata["split"] = split.to_dict()
        group_metadata["quality_flags"] = sorted(set(group_flags))

        for line in group.lines:
            line_id = int(line["id"])
            raw_output = Path(split.piece_paths[line_id])
            target_duration = float(line["target_duration"])
            aligned_output = aligned_dir / f"line_{line_id:04d}.wav"
            public_output = public_lines_dir / f"line_{line_id:04d}.wav"
            line_warnings = list(warnings)
            line_flags = list(group_flags)
            repair_info = repair_raw_piece_if_needed(
                raw_output,
                line_id,
                str(line.get("text", "")),
                target_duration,
                config,
                logger=logger,
            )
            line_flags.extend(repair_info.get("flags", []))
            if repair_info.get("status") in {"wrong_pause_auto_cut", "wrong_pause_manual_review"}:
                line_warnings.append(f"pause repair status: {repair_info.get('status')}")

            raw_duration, alignment_metadata = align_raw_line_audio(
                raw_output,
                aligned_output,
                public_output,
                target_duration,
                config,
                logger=logger,
                action_prefix="combined_sentence",
            )
            line_flags.extend(alignment_metadata.get("quality_flags", []))
            line_warnings.extend(alignment_metadata.get("alignment_warnings", []))
            line_status = group_status
            if "wrong_pause_manual_review" in line_flags:
                line_status = "wrong_pause_manual_review"
            elif "wrong_pause_auto_cut" in line_flags:
                line_status = "wrong_pause_auto_cut"
            elif group_status.startswith("flagged_after_"):
                line_status = "wrong_pause_manual_review"
            elif group_status == "passed":
                line_status = "combined_sentence_passed"
            elif group_status == "regenerated_then_passed":
                line_status = "combined_sentence_regenerated_then_passed"

            row = {
                "line_id": line_id,
                "speaker": group.speaker,
                "text": line.get("text"),
                "start": line.get("start"),
                "end": line.get("end"),
                "target_duration": target_duration,
                "method": group.emotion.get("emotion_method", "emo_text"),
                "emo_text": group.emotion.get("emo_text"),
                "emo_vector": group.emotion.get("emo_vector"),
                "emo_alpha": group.emotion.get("emo_alpha"),
                "use_random": group.emotion.get("use_random", False),
                "speaker_reference": speaker_reference,
                "raw_output": str(raw_output),
                "aligned_output": str(aligned_output),
                "public_output": str(public_output),
                "raw_duration": raw_duration,
                "final_duration": alignment_metadata["final_duration"],
                "alignment_action": alignment_metadata["alignment_action"],
                "attempts": retry_result["attempt_count"],
                "accepted_attempt": retry_result["accepted_attempt"],
                "pause_attempts": retry_result["attempts"],
                "abnormal_pause_detected": retry_result.get("abnormal_pause_detected", False),
                "final_abnormal_pause_detected": retry_result.get("final_abnormal_pause_detected", False),
                "final_score": retry_result.get("final_score"),
                "generation_status": line_status,
                "pause_repair_attempted": bool((retry_result.get("pause_repair") or {}).get("attempted", False)) or bool(repair_info.get("attempted", False)),
                "pause_repair_status": repair_info.get("status") or (retry_result.get("pause_repair") or {}).get("status", "skipped"),
                "pause_repair": repair_info if repair_info.get("attempted") else retry_result.get("pause_repair"),
                "pause_detection_before_repair": repair_info.get("pause_detection_before_repair") or retry_result.get("pause_detection_before_repair"),
                "pause_detection_after_repair": repair_info.get("pause_detection_after_repair") or retry_result.get("pause_detection_after_repair"),
                "silence_cleanup": {"skipped": "combined sentence pieces are split before duration alignment"},
                "quality_flags": sorted(set(line_flags)),
                "sentence_group_id": group.group_id,
                "sentence_group_line_ids": group.line_ids,
                "combined_text": group.text,
                "combined_audio_uncut": uncut_audio,
                "combined_audio_selected": str(selected_copy),
                "split_boundary_cuts": [cut.to_dict() for cut in split.cuts],
                "warnings": line_warnings,
                "override": {},
            }
            rows.append(row)
        return rows, group_metadata
    except Exception as exc:
        logger.exception("Failed to synthesize sentence group %s", group.group_id)
        group_metadata.update({"error": str(exc), "status": "failed_generation", "warnings": warnings})
        for line in group.lines:
            line_id = int(line["id"])
            rows.append(
                {
                    "line_id": line_id,
                    "speaker": group.speaker,
                    "text": line.get("text"),
                    "start": line.get("start"),
                    "end": line.get("end"),
                    "target_duration": line.get("target_duration"),
                    "generation_status": "failed_generation",
                    "error": str(exc),
                    "warnings": warnings,
                    "sentence_group_id": group.group_id,
                    "sentence_group_line_ids": group.line_ids,
                }
            )
        return rows, group_metadata


def _duration_scale_is_close(raw_duration: float, target_duration: float, config: DubbingConfig) -> bool:
    if target_duration <= 0:
        return True
    ratio = abs(raw_duration - target_duration) / target_duration
    return ratio <= float(config.model_duration_control.regenerate_if_ratio_outside)


def _clamp_duration_scale(value: float, config: DubbingConfig) -> tuple[float, list[str]]:
    duration_config = config.model_duration_control
    low = float(duration_config.min_duration_scale)
    high = float(duration_config.max_duration_scale)
    if high < low:
        high = low
    clamped = max(low, min(high, float(value)))
    flags: list[str] = []
    if clamped > float(value):
        flags.append("duration_scale_clamped_low")
    if clamped < float(value):
        flags.append("duration_scale_clamped_high")
    return clamped, flags


def _flatten_attempts(pass_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for record in pass_records:
        for attempt in record.get("pause_attempts", []):
            annotated = dict(attempt)
            annotated["duration_pass"] = record.get("pass_index")
            annotated["duration_scale"] = record.get("duration_scale")
            attempts.append(annotated)
    return attempts


def _pause_repair_status(record: dict[str, Any] | None) -> str:
    if not record:
        return "skipped"
    repair = record.get("pause_repair") or {}
    if isinstance(repair, dict):
        return str(repair.get("status") or "skipped")
    return "skipped"


def _unit_status_from_retry(status: str, quality_flags: list[str]) -> str:
    if "wrong_pause_manual_review" in quality_flags:
        return "wrong_pause_manual_review"
    if "wrong_pause_auto_cut" in quality_flags or "pause_repaired" in quality_flags:
        return "wrong_pause_auto_cut"
    if status == "passed":
        return "passed"
    if status == "regenerated_then_passed":
        return "regenerated_then_passed"
    if status.startswith("flagged_after_"):
        return "wrong_pause_manual_review"
    return status


def _generate_unit_audio_pass(
    unit: GenerationUnit,
    speaker_reference: str,
    wrapper: IndexTTS2Wrapper,
    raw_dir: Path,
    attempts_root: Path,
    pass_index: int,
    duration_scale: float,
    config: DubbingConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    pass_label = f"duration_pass_{pass_index}"
    pass_attempt_dir = ensure_dir(attempts_root / unit.unit_id / pass_label)
    pass_raw_output = raw_dir / f"{unit.unit_id}_raw_pass_{pass_index}.wav"
    base_emo_alpha = float(unit.emotion.get("emo_alpha", 0.55))

    def infer_attempt(path: Path, attempt_number: int) -> None:
        retry_alpha = base_emo_alpha
        retry_use_random: bool | None = None
        retry_interval_silence: int | None = None
        if attempt_number > 1:
            retry_alpha = max(0.0, base_emo_alpha * 0.92)
            retry_use_random = False
            retry_interval_silence = 120
        wrapper.infer_line(
            speaker_reference,
            unit.text,
            path,
            unit.emotion,
            emo_alpha_override=retry_alpha,
            use_random_override=retry_use_random,
            interval_silence=retry_interval_silence,
            duration_scale=duration_scale,
        )

    retry_result = generate_with_pause_retries(
        unit.unit_id,
        unit.text,
        unit.span_target_duration,
        pass_attempt_dir,
        config,
        infer_attempt,
        logger=logger,
    )
    selected_audio = retry_result.get("selected_audio")
    if not selected_audio:
        raise RuntimeError(str(retry_result.get("error") or "all generation attempts failed"))

    cleanup = tighten_silences(
        Path(str(selected_audio)),
        pass_raw_output,
        unit.text,
        config.silence_cleanup,
        logger=logger,
    )
    return {
        "pass_index": pass_index,
        "duration_scale": float(duration_scale),
        "raw_path": str(pass_raw_output),
        "raw_duration": cleanup.output_duration,
        "silence_cleanup": cleanup.to_dict(),
        "cleanup_warnings": cleanup.warnings,
        "retry_status": retry_result.get("status"),
        "pause_attempt_count": retry_result.get("attempt_count", 0),
        "accepted_attempt": retry_result.get("accepted_attempt"),
        "pause_attempts": retry_result.get("attempts", []),
        "abnormal_pause_detected": retry_result.get("abnormal_pause_detected", False),
        "final_abnormal_pause_detected": retry_result.get("final_abnormal_pause_detected", False),
        "final_score": retry_result.get("final_score"),
        "pause_repair": retry_result.get("pause_repair"),
        "pause_detection_before_repair": retry_result.get("pause_detection_before_repair"),
        "pause_detection_after_repair": retry_result.get("pause_detection_after_repair"),
        "selected_audio": selected_audio,
    }


def synthesize_unit(
    unit: GenerationUnit,
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
    wrapper: IndexTTS2Wrapper | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger("stable_dubbing.tts")
    output = Path(output_dir)
    raw_dir = ensure_dir(output / "lines" / "raw")
    public_lines_dir = ensure_dir(output / "lines")
    attempts_root = ensure_dir(
        output / "attempts" if config.pause_detection.save_attempts else output / "work" / "attempts"
    )
    raw_output = raw_dir / f"{unit.unit_id}_raw.wav"
    aligned_output = public_lines_dir / f"{unit.unit_id}.wav"
    source_line_indices = [int(line_id) for line_id in unit.source_line_indices]
    speaker_reference = speaker_map.get(unit.speaker)
    warnings: list[str] = list(unit.warnings)
    quality_flags: list[str] = list(unit.warnings)
    generation_status = "failed_generation"
    pass_records: list[dict[str, Any]] = []
    duration_scale_flags: list[str] = []
    selected_record: dict[str, Any] | None = None

    metadata: dict[str, Any] = {
        "unit_id": unit.unit_id,
        "group_id": unit.unit_id if unit.is_group else None,
        "line_id": source_line_indices[0] if source_line_indices else None,
        "line_ids": source_line_indices,
        "source_line_indices": source_line_indices,
        "speaker": unit.speaker,
        "text": unit.text,
        "start": unit.start,
        "end": unit.end,
        "target_duration": unit.span_target_duration,
        "span_target_duration": unit.span_target_duration,
        "summed_line_target_duration": unit.summed_line_target_duration,
        "method": unit.emotion.get("emotion_method", "emo_text"),
        "emo_text": unit.emotion.get("emo_text"),
        "emo_vector": unit.emotion.get("emo_vector"),
        "emo_alpha": unit.emotion.get("emo_alpha"),
        "use_random": unit.emotion.get("use_random", False),
        "emotion": unit.emotion,
        "emotion_blend": unit.emotion.get("emotion_blend"),
        "emotion_source": unit.emotion.get("emotion_source"),
        "speaker_reference": speaker_reference,
        "raw_output": str(raw_output),
        "aligned_output": str(aligned_output),
        "public_output": str(aligned_output),
        "raw_audio_path": str(raw_output),
        "aligned_audio_path": str(aligned_output),
        "raw_duration": None,
        "final_duration": None,
        "alignment_action": None,
        "attempts": 0,
        "accepted_attempt": None,
        "pause_attempts": [],
        "pause_retry_count": 0,
        "abnormal_pause_detected": False,
        "final_abnormal_pause_detected": False,
        "final_score": None,
        "generation_status": generation_status,
        "pause_repair_attempted": False,
        "pause_repair_status": "skipped",
        "pause_repair": None,
        "pause_detection_before_repair": None,
        "pause_detection_after_repair": None,
        "silence_cleanup": None,
        "quality_flags": quality_flags,
        "warnings": warnings,
        "is_group": unit.is_group,
        "source_grouping_method": unit.source_grouping_method,
        "model_duration_control_enabled": bool(config.model_duration_control.enabled),
        "model_duration_control_used": False,
        "duration_scale": float(config.model_duration_control.first_pass_scale),
        "duration_scale_requested": float(config.model_duration_control.first_pass_scale),
        "duration_scale_clamped": False,
        "duration_scale_attempts": [],
        "final_ffmpeg_correction_used": False,
        "final_ffmpeg_speed_factor": None,
        "sentence_group_id": unit.unit_id if unit.is_group else None,
        "sentence_group_line_ids": source_line_indices if unit.is_group else [],
        "combined_text": unit.text if unit.is_group else None,
        "split_boundary_cuts": [],
    }

    try:
        if not speaker_reference:
            raise RuntimeError(f"No speaker reference mapped for {unit.speaker!r}")
        if unit.span_target_duration <= 0:
            raise RuntimeError(f"Unit {unit.unit_id} span duration must be greater than zero")
        if wrapper is None:
            wrapper = IndexTTS2Wrapper(config, logger=logger)
        wrapper.initialize()

        first_scale, first_flags = _clamp_duration_scale(config.model_duration_control.first_pass_scale, config)
        duration_scale_flags.extend(first_flags)
        max_attempts = max(1, int(config.model_duration_control.max_model_duration_attempts))
        selected_record = _generate_unit_audio_pass(
            unit,
            speaker_reference,
            wrapper,
            raw_dir,
            attempts_root,
            1,
            first_scale,
            config,
            logger,
        )
        pass_records.append(selected_record)
        warnings.extend(selected_record.get("cleanup_warnings", []))

        while (
            config.model_duration_control.enabled
            and len(pass_records) < max_attempts
            and not _duration_scale_is_close(
                float(selected_record.get("raw_duration", 0.0)),
                unit.span_target_duration,
                config,
            )
            and float(selected_record.get("raw_duration", 0.0)) > 0
        ):
            requested_scale = (
                float(selected_record.get("duration_scale", 1.0))
                * unit.span_target_duration
                / float(selected_record["raw_duration"])
            )
            next_scale, clamp_flags = _clamp_duration_scale(requested_scale, config)
            duration_scale_flags.extend(clamp_flags)
            pass_index = len(pass_records) + 1
            logger.info(
                "Unit %s raw_duration=%.3f target=%.3f; regenerating with duration_scale=%.3f",
                unit.unit_id,
                float(selected_record["raw_duration"]),
                unit.span_target_duration,
                next_scale,
            )
            try:
                selected_record = _generate_unit_audio_pass(
                    unit,
                    speaker_reference,
                    wrapper,
                    raw_dir,
                    attempts_root,
                    pass_index,
                    next_scale,
                    config,
                    logger,
                )
                pass_records.append(selected_record)
                warnings.extend(selected_record.get("cleanup_warnings", []))
            except Exception as exc:
                warnings.append(f"model duration regeneration failed: {exc}")
                logger.exception("Unit %s duration-scale regeneration failed", unit.unit_id)
                selected_record = pass_records[-1]
                break

        shutil.copy2(str(selected_record["raw_path"]), raw_output)
        raw_duration = audio_duration(raw_output)
        result = render_aligned_audio(
            raw_output,
            aligned_output,
            unit.span_target_duration,
            config.alignment,
            logger=logger,
            force=True,
        )
        warnings.extend(result.warnings)

        final_speed_factor = raw_duration / unit.span_target_duration if unit.span_target_duration > 0 else None
        if final_speed_factor is not None and raw_duration > unit.span_target_duration:
            metadata["final_ffmpeg_speed_factor"] = final_speed_factor
            if final_speed_factor > float(config.model_duration_control.final_ffmpeg_max_speed_factor):
                quality_flags.append("final_speedup_gt_1_15")
        quality_flags.extend(duration_quality_flags(raw_duration, unit.span_target_duration, config))
        quality_flags.extend(duration_scale_flags)
        if any(record.get("abnormal_pause_detected") for record in pass_records):
            quality_flags.append("abnormal_pause_detected")
        if any(record.get("pause_repair") for record in pass_records):
            repair_statuses = [_pause_repair_status(record) for record in pass_records]
            if any(status in {"repaired_passed", "repaired_still_flagged"} for status in repair_statuses):
                quality_flags.append("pause_repaired")

        retry_status = str(selected_record.get("retry_status") or "passed")
        generation_status = _unit_status_from_retry(retry_status, quality_flags)
        flat_attempts = _flatten_attempts(pass_records)
        pause_repair = selected_record.get("pause_repair")
        metadata.update(
            {
                "raw_duration": raw_duration,
                "final_duration": result.final_duration,
                "alignment_action": result.alignment_action,
                "attempts": sum(int(record.get("pause_attempt_count", 0)) for record in pass_records),
                "accepted_attempt": selected_record.get("accepted_attempt"),
                "pause_attempts": flat_attempts,
                "pause_retry_count": len(flat_attempts),
                "abnormal_pause_detected": any(record.get("abnormal_pause_detected") for record in pass_records),
                "final_abnormal_pause_detected": bool(selected_record.get("final_abnormal_pause_detected", False)),
                "final_score": selected_record.get("final_score"),
                "generation_status": generation_status,
                "pause_repair": pause_repair,
                "pause_repair_attempted": bool((pause_repair or {}).get("attempted", False)),
                "pause_repair_status": _pause_repair_status(selected_record),
                "pause_detection_before_repair": selected_record.get("pause_detection_before_repair"),
                "pause_detection_after_repair": selected_record.get("pause_detection_after_repair"),
                "silence_cleanup": selected_record.get("silence_cleanup"),
                "quality_flags": sorted(set(quality_flags)),
                "warnings": warnings,
                "model_duration_control_used": len(pass_records) > 1,
                "duration_scale": float(selected_record.get("duration_scale", first_scale)),
                "duration_scale_requested": (
                    float(pass_records[-2].get("duration_scale", 1.0))
                    * unit.span_target_duration
                    / float(pass_records[-2]["raw_duration"])
                    if len(pass_records) > 1 and float(pass_records[-2].get("raw_duration", 0.0)) > 0
                    else first_scale
                ),
                "duration_scale_clamped": bool(duration_scale_flags),
                "duration_scale_attempts": [
                    {
                        "pass_index": record.get("pass_index"),
                        "duration_scale": record.get("duration_scale"),
                        "raw_duration": record.get("raw_duration"),
                        "retry_status": record.get("retry_status"),
                        "accepted_attempt": record.get("accepted_attempt"),
                    }
                    for record in pass_records
                ],
                "final_ffmpeg_correction_used": result.alignment_action in {"time_stretch", "pad_silence_end"},
            }
        )
    except Exception as exc:
        logger.exception("Failed to synthesize unit %s", unit.unit_id)
        metadata.update(
            {
                "error": str(exc),
                "attempts": sum(int(record.get("pause_attempt_count", 0)) for record in pass_records),
                "pause_attempts": _flatten_attempts(pass_records),
                "warnings": warnings,
                "generation_status": "failed_generation",
                "duration_scale_attempts": [
                    {
                        "pass_index": record.get("pass_index"),
                        "duration_scale": record.get("duration_scale"),
                        "raw_duration": record.get("raw_duration"),
                        "retry_status": record.get("retry_status"),
                    }
                    for record in pass_records
                ],
            }
        )
    return metadata


def synthesize_lines(
    lines: list[dict[str, Any]],
    emotions: list[dict[str, Any]],
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
    logger: logging.Logger | None = None,
    manual_groups_path: str | Path | None = None,
    manual_groups_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    logger = logger or logging.getLogger("stable_dubbing.tts")
    output = Path(output_dir)
    metadata_path = output / "work" / "line_generation_metadata.jsonl"
    group_metadata_path = output / "work" / "group_generation_metadata.jsonl"
    sentence_groups_path = output / "work" / "sentence_groups.json"
    generation_units_path = output / "work" / "generation_units.json"
    wrapper = IndexTTS2Wrapper(config, logger=logger)
    wrapper.initialize()
    metadata_rows: list[dict[str, Any]] = []

    generation_units = build_generation_units(
        lines,
        emotions,
        manual_groups_path=manual_groups_path,
        manual_groups_data=manual_groups_data,
        enable_auto_groups=bool(config.sentence_grouping.enabled),
    )
    write_generation_units(sentence_groups_path, generation_units)
    write_generation_units(generation_units_path, generation_units)

    for unit in generation_units:
        metadata_rows.append(
            synthesize_unit(
                unit,
                speaker_map,
                output,
                config,
                wrapper=wrapper,
                logger=logger,
            )
        )

    write_jsonl(metadata_path, metadata_rows)
    write_jsonl(group_metadata_path, [row for row in metadata_rows if row.get("is_group")])
    write_json(generation_units_path, metadata_rows)
    return metadata_rows
