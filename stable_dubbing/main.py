from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio_assemble import assemble_aligned_track, assemble_generation_units_manifest, concatenate_raw_lines
from .config import DubbingConfig, load_config, str_to_bool
from .emotion_plan import (
    apply_line_override,
    get_line_by_id,
    load_emo_vector_file,
    load_emotion_plan,
    parse_emo_vector,
    write_updated_emotion_plan,
)
from .emotion_prepare import (
    validate_emotion_items,
    wait_for_emotion_edits,
    write_emotion_file,
)
from .evaluation import create_mos_rating_sheet, run_evaluation
from .generation_report import replace_metadata_row, update_line_report, write_generation_report
from .generation_units import build_generation_units, write_generation_units
from .pause_review import build_pause_review_manifest, serve_pause_review
from .report import write_quality_report
from .recombine import recombine_lines
from .speaker_map import SpeakerMapError, build_speaker_map
from .subtitle_parser import parse_subtitle
from .tts_indextts2 import synthesize_line, synthesize_lines
from .utils import ensure_dir, probe_media_duration, read_json, read_jsonl, setup_logging, write_json, write_jsonl
from .video_mux import mux_video


def _find_first(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or "run"


def create_run_output_dir(base_output_dir: Path, run_name: str | None = None) -> Path:
    base = ensure_dir(base_output_dir)
    name = _sanitize_run_name(run_name) if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base / name
    if not candidate.exists():
        return ensure_dir(candidate)
    for suffix in range(2, 1000):
        candidate = base / f"{name}_{suffix:02d}"
        if not candidate.exists():
            return ensure_dir(candidate)
    raise RuntimeError(f"Could not create a unique run output directory under {base}")


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_dir = Path(args.input_dir).resolve() if args.input_dir else None
    video = Path(args.video).resolve() if args.video else None
    script = Path(args.script).resolve() if args.script else None
    refs_dir = Path(args.refs_dir).resolve() if args.refs_dir else None

    if input_dir:
        video = video or _find_first(input_dir, ["*.mp4"])
        script = script or _find_first(input_dir, ["*.ass", "*.txt"])
        refs_dir = refs_dir or input_dir / "refs"
        output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "output"
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd() / "output"

    missing = []
    if not video:
        missing.append("video")
    if not script:
        missing.append("script")
    if not refs_dir:
        missing.append("refs_dir")
    if missing:
        raise ValueError(f"Missing required input path(s): {', '.join(missing)}")

    return {
        "input_dir": input_dir or Path.cwd(),
        "video": video,
        "script": script,
        "refs_dir": refs_dir,
        "output_dir": output_dir,
    }


def apply_cli_overrides(config: DubbingConfig, args: argparse.Namespace) -> DubbingConfig:
    if getattr(args, "language", None):
        config.language = args.language
    if getattr(args, "pause_for_emotion_edit", None) is not None:
        config.pause_for_emotion_edit = bool(str_to_bool(args.pause_for_emotion_edit))
    if getattr(args, "no_pause", False):
        config.pause_for_emotion_edit = False
    if getattr(args, "use_fp16", False):
        config.use_fp16 = True
    if getattr(args, "use_cuda_kernel", False):
        config.use_cuda_kernel = True
    if getattr(args, "use_deepspeed", False):
        config.use_deepspeed = True
    if getattr(args, "mix_original", False):
        config.mix_original = True
    if getattr(args, "indextts_repo_path", None):
        config.indextts_repo_path = args.indextts_repo_path
    if getattr(args, "indextts_cfg_path", None):
        config.indextts_cfg_path = args.indextts_cfg_path
    if getattr(args, "indextts_model_dir", None):
        config.indextts_model_dir = args.indextts_model_dir
    if getattr(args, "non_strict_speaker_refs", False):
        config.strict_speaker_refs = False
    if getattr(args, "max_pause_retries", None) is not None:
        config.pause_detection.max_retries = max(1, int(args.max_pause_retries))
    if getattr(args, "pause_threshold_ms", None) is not None:
        config.pause_detection.min_pause_ms = max(1, int(args.pause_threshold_ms))
    if getattr(args, "silence_db_offset", None) is not None:
        config.pause_detection.silence_db_offset = float(args.silence_db_offset)
    if getattr(args, "disable_pause_check", False):
        config.pause_detection.enabled = False
    if getattr(args, "save_attempts", False):
        config.pause_detection.save_attempts = True
    if getattr(args, "use_asr_alignment", False):
        config.pause_detection.use_asr_alignment = True
    if getattr(args, "enable_sentence_grouping", False):
        config.sentence_grouping.enabled = True
    if getattr(args, "disable_sentence_grouping", False):
        config.sentence_grouping.enabled = False
    if getattr(args, "auto_groups", False):
        config.sentence_grouping.enabled = True
    if getattr(args, "disable_auto_groups", False):
        config.sentence_grouping.enabled = False
    if getattr(args, "enable_pause_repair", False):
        config.pause_repair.enabled = True
    if getattr(args, "disable_pause_repair", False):
        config.pause_repair.enabled = False
    if getattr(args, "pause_repair_keep_ms", None) is not None:
        config.pause_repair.target_keep_ms = max(1, int(args.pause_repair_keep_ms))
    if getattr(args, "pause_repair_min_keep_ms", None) is not None:
        config.pause_repair.min_keep_ms = max(1, int(args.pause_repair_min_keep_ms))
    if getattr(args, "pause_repair_max_keep_ms", None) is not None:
        config.pause_repair.max_keep_ms = max(1, int(args.pause_repair_max_keep_ms))
    if config.pause_repair.max_keep_ms < config.pause_repair.min_keep_ms:
        config.pause_repair.max_keep_ms = config.pause_repair.min_keep_ms
    if getattr(args, "whisperx_model", None):
        config.sentence_grouping.whisperx_model = args.whisperx_model
    if getattr(args, "whisperx_language", None):
        config.sentence_grouping.whisperx_language = args.whisperx_language
    if getattr(args, "whisperx_device", None):
        config.sentence_grouping.whisperx_device = args.whisperx_device
    if getattr(args, "whisperx_compute_type", None):
        config.sentence_grouping.whisperx_compute_type = args.whisperx_compute_type
    if getattr(args, "dry_run", False):
        config.pause_for_emotion_edit = False
    return config


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use_fp16", action="store_true", help="Enable IndexTTS2 FP16 inference")
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Enable IndexTTS2 CUDA kernel")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable IndexTTS2 DeepSpeed")
    parser.add_argument("--indextts_repo_path", help="Path to official index-tts repo")
    parser.add_argument("--indextts_cfg_path", help="Path to checkpoints/config.yaml")
    parser.add_argument("--indextts_model_dir", help="Path to checkpoints directory")


def _add_pause_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-pause-retries", type=int, default=None, help="Max attempts per line when abnormal pauses are detected")
    parser.add_argument("--pause-threshold-ms", type=int, default=None, help="Minimum internal silence length to inspect")
    parser.add_argument("--silence-db-offset", type=float, default=None, help="Silence threshold as average dBFS minus this offset")
    parser.add_argument("--disable-pause-check", action="store_true", help="Disable abnormal pause detection and retries")
    parser.add_argument("--save-attempts", action="store_true", help="Save generated attempts under output/attempts")
    parser.add_argument("--use-asr-alignment", action="store_true", help="Use Whisper word timestamps when available")
    parser.add_argument("--enable-pause-repair", action="store_true", help="Enable targeted abnormal pause repair")
    parser.add_argument("--disable-pause-repair", action="store_true", help="Disable targeted abnormal pause repair")
    parser.add_argument("--pause-repair-keep-ms", type=int, default=None, help="Target silence kept for repaired abnormal pauses")
    parser.add_argument("--pause-repair-min-keep-ms", type=int, default=None, help="Minimum silence kept during pause repair")
    parser.add_argument("--pause-repair-max-keep-ms", type=int, default=None, help="Maximum silence kept during pause repair")
    parser.add_argument("--enable-sentence-grouping", action="store_true", help="Enable combined sentence generation")
    parser.add_argument("--disable-sentence-grouping", action="store_true", help="Disable combined sentence generation")
    parser.add_argument("--manual-groups", default=None, help="JSON file describing manual generation-unit groups")
    parser.add_argument("--auto-groups", action="store_true", help="Enable automatic continuation grouping")
    parser.add_argument("--disable-auto-groups", action="store_true", help="Disable automatic continuation grouping")
    parser.add_argument("--whisperx-model", default=None, help="WhisperX ASR model name for group boundary alignment")
    parser.add_argument("--whisperx-language", default=None, help="WhisperX language code for group boundary alignment")
    parser.add_argument("--whisperx-device", default=None, help="WhisperX device override, e.g. cuda or cpu")
    parser.add_argument("--whisperx-compute-type", default=None, help="WhisperX compute type override")


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_dir", help="Folder containing video.mp4, script.ass/script.txt, and refs/")
    parser.add_argument("--video", help="Explicit MP4 video path")
    parser.add_argument("--script", help="Explicit .ass or .txt subtitle/script path")
    parser.add_argument("--refs_dir", help="Explicit speaker reference audio directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--run_name", help="Optional name for this run's output subfolder")
    parser.add_argument(
        "--flat_output_dir",
        action="store_true",
        help="Write directly to output_dir instead of creating a per-run subfolder",
    )
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--language", default=None, help="Language for evaluation: auto, en, zh, etc.")
    parser.add_argument("--pause_for_emotion_edit", default=None, help="true/false")
    parser.add_argument("--no_pause", action="store_true", help="Do not pause after generating emotions_to_edit.json")
    parser.add_argument("--emotion-json", help="Use this user-provided emotion JSON directly")
    parser.add_argument("--resume_from_emotion_file", help="Use an existing emotions_to_edit.json")
    parser.add_argument("--dry_run", action="store_true", help="Parse, prepare emotions, map speakers, and write report without TTS")
    parser.add_argument("--mix_original", action="store_true", help="Mix original video audio with synthesized audio")
    parser.add_argument("--non_strict_speaker_refs", action="store_true", help="Allow fallback speaker refs")
    _add_model_args(parser)
    _add_pause_detection_args(parser)
    parser.set_defaults(command="generate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {"regenerate-line", "regenerate-lines", "recombine", "review-pauses"}:
        parser = argparse.ArgumentParser(description="IndexTTS2 stable emotion dubbing pipeline.")
        subparsers = parser.add_subparsers(dest="command", required=True)

        regen = subparsers.add_parser("regenerate-line", help="Regenerate one line from an emotion JSON")
        regen.add_argument("--emotion-json", required=True, help="Emotion JSON to load")
        regen.add_argument("--line-id", required=True, type=int, help="Line id to regenerate")
        regen.add_argument("--output-dir", required=True, help="Existing run output directory")
        regen.add_argument("--speaker-map", help="speaker_map.json path; defaults to output-dir/work/speaker_map.json")
        regen.add_argument("--refs-dir", help="Fallback speaker refs directory if speaker_map.json is missing")
        regen.add_argument("--config", help="YAML config file")
        regen.add_argument("--emo-alpha", type=float, help="Override emo_alpha for this regeneration")
        regen.add_argument("--emo-vector", help="Override with an inline JSON emotion vector")
        regen.add_argument("--emo-vector-file", help="Override with an emotion vector JSON file")
        regen.add_argument("--updated-emotion-json", help="Where to write the updated emotion JSON")
        regen.add_argument("--non_strict_speaker_refs", action="store_true", help="Allow fallback speaker refs")
        _add_model_args(regen)
        _add_pause_detection_args(regen)

        regen_lines = subparsers.add_parser("regenerate-lines", help="Regenerate a sentence group or selected lines")
        regen_lines.add_argument("--emotion-json", required=True, help="Emotion JSON to load")
        regen_lines.add_argument("--line-ids", required=True, help="Comma-separated line ids, e.g. 4,5,6")
        regen_lines.add_argument("--output-dir", required=True, help="Existing run output directory")
        regen_lines.add_argument("--speaker-map", help="speaker_map.json path; defaults to output-dir/work/speaker_map.json")
        regen_lines.add_argument("--refs-dir", help="Fallback speaker refs directory if speaker_map.json is missing")
        regen_lines.add_argument("--config", help="YAML config file")
        regen_lines.add_argument("--non_strict_speaker_refs", action="store_true", help="Allow fallback speaker refs")
        _add_model_args(regen_lines)
        _add_pause_detection_args(regen_lines)

        recombine = subparsers.add_parser("recombine", help="Recombine final line wav files")
        recombine.add_argument("--emotion-json", required=True, help="Emotion JSON for line order and timestamps")
        recombine.add_argument("--lines-dir", required=True, help="Directory containing final line wav files")
        recombine.add_argument("--output", required=True, help="Output combined wav path")
        recombine.add_argument("--gap-ms", type=int, default=100, help="Gap between lines when not using timestamps")
        recombine.add_argument("--use-timestamps", type=str_to_bool, default=None, help="true/false; auto when omitted")
        recombine.add_argument("--crossfade-ms", type=int, default=0, help="Crossfade between adjacent lines when gap is 0")

        review = subparsers.add_parser("review-pauses", help="Open a local waveform review tool for flagged pauses")
        review.add_argument("--output-dir", required=True, help="Existing run output directory")
        review.add_argument("--host", default="127.0.0.1", help="Review server host")
        review.add_argument("--port", type=int, default=8765, help="Review server port")
        return parser.parse_args(raw_args)

    parser = argparse.ArgumentParser(description="IndexTTS2 stable emotion dubbing pipeline.")
    _add_generation_args(parser)
    return parser.parse_args(raw_args)


def run_generation(args: argparse.Namespace, argv: list[str] | None = None) -> int:
    paths = resolve_paths(args)
    config = apply_cli_overrides(load_config(args.config), args)
    output_base_dir = ensure_dir(paths["output_dir"])
    output_dir = output_base_dir if args.flat_output_dir else create_run_output_dir(output_base_dir, args.run_name)
    work_dir = ensure_dir(output_dir / "work")
    audio_dir = ensure_dir(output_dir / "audio")
    logger = setup_logging(output_dir)
    random.seed(config.random_seed)

    command = "python -m stable_dubbing.main " + " ".join(sys.argv[1:] if argv is None else argv)
    logger.info("Starting pipeline command=%s", command)
    logger.info("Resolved paths: %s", {key: str(value) for key, value in paths.items()})

    all_warnings: list[str] = []
    try:
        parse_result = parse_subtitle(paths["script"], config)
        lines = [line.to_dict() for line in parse_result.lines]
        write_json(work_dir / "script_structured.json", lines)
        all_warnings.extend(warning["warning"] for warning in parse_result.warnings)
        all_warnings.extend(f"unresolved speaker: {speaker}" for speaker in parse_result.unresolved_speakers)
        logger.info("Parsed %s subtitle lines", len(lines))

        try:
            speaker_map, speaker_warnings = build_speaker_map(
                lines,
                paths["refs_dir"],
                work_dir / "speaker_map.json",
                strict=config.strict_speaker_refs,
                interactive=not args.dry_run,
            )
            all_warnings.extend(speaker_warnings)
        except SpeakerMapError:
            logger.exception("Speaker reference mapping failed")
            raise

        direct_emotion_json = args.emotion_json or args.resume_from_emotion_file
        if direct_emotion_json:
            emotion_path = Path(direct_emotion_json).resolve()
            emotions = load_emotion_plan(emotion_path)
        else:
            emotion_path = write_emotion_file(lines, work_dir / "emotions_to_edit.json", config)
            if config.pause_for_emotion_edit:
                emotions = wait_for_emotion_edits(emotion_path)
            else:
                emotions = read_json(emotion_path)
                errors = validate_emotion_items(emotions)
                if errors:
                    raise ValueError("Generated emotion file failed validation: " + "; ".join(errors))
        write_json(work_dir / "warnings.json", all_warnings)

        metadata_rows: list[dict[str, Any]] = []
        final_outputs = {
            "final_video": "",
            "aligned_wav": str(audio_dir / "voice_cast_aligned.wav"),
            "raw_concatenated_wav": str(audio_dir / "voice_cast_raw_concatenated.wav"),
            "emotion_json": str(emotion_path),
            "metadata_jsonl": str(work_dir / "line_generation_metadata.jsonl"),
            "group_metadata_jsonl": str(work_dir / "group_generation_metadata.jsonl"),
            "sentence_groups_json": str(work_dir / "sentence_groups.json"),
            "generation_units_json": str(work_dir / "generation_units.json"),
            "pause_review_manifest_json": str(work_dir / "pause_review_manifest.json"),
            "generation_report_json": str(output_dir / "generation_report.json"),
            "generation_report_md": str(output_dir / "generation_report.md"),
        }

        if args.dry_run:
            generation_units = build_generation_units(
                lines,
                emotions,
                manual_groups_path=getattr(args, "manual_groups", None),
                enable_auto_groups=bool(config.sentence_grouping.enabled),
            )
            metadata_rows = [unit.to_dict() for unit in generation_units]
            metadata_by_id: dict[int, dict[str, Any]] = {}
            write_jsonl(work_dir / "line_generation_metadata.jsonl", metadata_rows)
            write_generation_units(work_dir / "generation_units.json", generation_units)
            write_generation_report(output_dir, metadata_rows, command=command, emotion_json=str(emotion_path))
            create_mos_rating_sheet(lines, metadata_by_id, output_dir / "evaluation" / "mos_rating_sheet.csv")
            evaluation_summary = {
                "mos_sheet": str(output_dir / "evaluation" / "mos_rating_sheet.csv"),
                "content_consistency": {"skipped": "dry run: ASR was not executed"},
                "sim": {"skipped": "dry run: SIM was not executed"},
            }
        else:
            metadata_rows = synthesize_lines(
                lines,
                emotions,
                speaker_map,
                output_dir,
                config,
                logger=logger,
                manual_groups_path=getattr(args, "manual_groups", None),
            )
            write_generation_report(output_dir, metadata_rows, command=command, emotion_json=str(emotion_path))
            failed_rows = [row for row in metadata_rows if row.get("error")]
            successful_rows = [row for row in metadata_rows if not row.get("error")]
            if not successful_rows:
                write_json(work_dir / "warnings.json", all_warnings)
                all_warnings.append(
                    f"TTS generation failed for all {len(failed_rows)} generation units. "
                    f"See {work_dir / 'line_generation_metadata.jsonl'} and {output_dir / 'logs' / 'pipeline.log'}."
                )
                evaluation_summary = {
                    "content_consistency": {"skipped": "all line generation failed"},
                    "sim": {"skipped": "all line generation failed"},
                    "mos_sheet": "",
                }
            else:
                raw_paths = [row["raw_output"] for row in metadata_rows if row.get("raw_output") and not row.get("error")]
                concat_warnings = concatenate_raw_lines(
                    raw_paths, audio_dir / "voice_cast_raw_concatenated.wav", logger=logger
                )
                all_warnings.extend(concat_warnings)

                video_duration = probe_media_duration(paths["video"])
                if video_duration is None:
                    video_duration = max((float(line["end"]) for line in lines), default=0.0)
                    all_warnings.append("video duration unavailable; using last subtitle end for audio bed")
                if any(row.get("unit_id") for row in metadata_rows):
                    assembly_warnings = assemble_generation_units_manifest(
                        work_dir / "generation_units.json",
                        video_duration,
                        audio_dir / "voice_cast_aligned.wav",
                        sample_rate=config.sample_rate,
                        logger=logger,
                    )
                else:
                    assembly_warnings = assemble_aligned_track(
                        lines,
                        {
                            int(row["line_id"]): row["aligned_output"]
                            for row in metadata_rows
                            if row.get("line_id") is not None
                            and row.get("aligned_output")
                            and not row.get("error")
                        },
                        video_duration,
                        audio_dir / "voice_cast_aligned.wav",
                        sample_rate=config.sample_rate,
                        logger=logger,
                    )
                all_warnings.extend(assembly_warnings)

                final_video = output_dir / "final_dubbed.mp4"
                mux_video(
                    paths["video"],
                    audio_dir / "voice_cast_aligned.wav",
                    final_video,
                    mix_original=config.mix_original,
                    logger=logger,
                )
                final_outputs["final_video"] = str(final_video)
                evaluation_summary = run_evaluation(lines, metadata_rows, speaker_map, output_dir, config)

            write_generation_report(output_dir, metadata_rows, command=command, emotion_json=str(emotion_path))

        write_json(work_dir / "warnings.json", all_warnings)
        write_quality_report(
            output_dir=output_dir,
            config=config,
            command=command,
            input_paths={
                "video": str(paths["video"]),
                "script": str(paths["script"]),
                "refs_dir": str(paths["refs_dir"]),
            },
            lines=lines,
            speaker_map=speaker_map,
            unresolved_speakers=parse_result.unresolved_speakers,
            warnings=all_warnings,
            metadata_rows=metadata_rows,
            evaluation_summary=evaluation_summary,
            final_outputs=final_outputs,
            dry_run=args.dry_run,
        )
        build_pause_review_manifest(output_dir, write=True)
        logger.info("Pipeline finished. Output directory: %s", output_dir)
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        raise


def _load_speaker_map_for_regeneration(
    args: argparse.Namespace,
    output_dir: Path,
    line: dict[str, Any],
    config: DubbingConfig,
) -> dict[str, str]:
    speaker_map_path = Path(args.speaker_map).resolve() if args.speaker_map else output_dir / "work" / "speaker_map.json"
    if speaker_map_path.exists():
        return read_json(speaker_map_path)
    if args.refs_dir:
        speaker_map, _warnings = build_speaker_map(
            [line],
            Path(args.refs_dir).resolve(),
            speaker_map_path,
            strict=config.strict_speaker_refs,
            interactive=False,
        )
        return speaker_map
    raise FileNotFoundError(
        f"Could not find speaker map at {speaker_map_path}. Pass --speaker-map or --refs-dir."
    )


def regenerate_line_command(args: argparse.Namespace) -> int:
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    work_dir = ensure_dir(output_dir / "work")
    logger = setup_logging(output_dir)
    config = apply_cli_overrides(load_config(args.config), args)
    random.seed(config.random_seed)

    emotion_path = Path(args.emotion_json).resolve()
    emotions = load_emotion_plan(emotion_path)
    line = get_line_by_id(emotions, args.line_id)
    override_notes: dict[str, Any] = {}
    updated_emotions = emotions

    emo_vector = None
    if args.emo_vector and args.emo_vector_file:
        raise ValueError("Use either --emo-vector or --emo-vector-file, not both.")
    if args.emo_vector:
        emo_vector = parse_emo_vector(args.emo_vector)
    elif args.emo_vector_file:
        emo_vector = load_emo_vector_file(args.emo_vector_file)

    if args.emo_alpha is not None or emo_vector is not None:
        updated_emotions, line, override_notes = apply_line_override(
            emotions,
            args.line_id,
            emo_alpha=args.emo_alpha,
            emo_vector=emo_vector,
        )
        updated_path = (
            write_json(Path(args.updated_emotion_json).resolve(), updated_emotions)
            if args.updated_emotion_json
            else write_updated_emotion_plan(updated_emotions, output_dir)
        )
        override_notes["updated_emotion_json"] = str(updated_path)
        logger.info("Wrote updated emotion JSON with line override to %s", updated_path)

    speaker_map = _load_speaker_map_for_regeneration(args, output_dir, line, config)
    row = synthesize_line(
        line,
        line,
        speaker_map,
        output_dir,
        config,
        logger=logger,
        override=override_notes,
    )
    metadata_path = work_dir / "line_generation_metadata.jsonl"
    metadata_rows = replace_metadata_row(read_jsonl(metadata_path), row)
    write_jsonl(metadata_path, metadata_rows)
    update_line_report(output_dir, row)
    logger.info("Regenerated line %s status=%s final_audio=%s", args.line_id, row.get("generation_status"), row.get("public_output"))
    return 0


def _parse_line_ids(value: str) -> list[int]:
    line_ids = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not line_ids:
        raise ValueError("--line-ids must include at least one line id")
    return line_ids


def _replace_group_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("group_id")): row for row in existing if row.get("group_id")}
    for row in new_rows:
        if row.get("group_id"):
            by_id[str(row["group_id"])] = row
    return sorted(by_id.values(), key=lambda item: str(item.get("group_id", "")))


def _row_source_ids(row: dict[str, Any]) -> set[int]:
    values = row.get("source_line_indices") or row.get("line_ids") or row.get("sentence_group_line_ids")
    if values:
        return {int(value) for value in values}
    if row.get("line_id") is not None:
        return {int(row["line_id"])}
    return set()


def _merge_rows_replacing_source_ids(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    source_ids: set[int],
) -> list[dict[str, Any]]:
    merged = [row for row in existing if not _row_source_ids(row).intersection(source_ids)]
    merged.extend(new_rows)
    return sorted(
        merged,
        key=lambda item: (
            float(item.get("start", 0.0)),
            min(_row_source_ids(item) or {int(item.get("line_id", 0) or 0)}),
        ),
    )


def _manual_groups_data_for_selected(args: argparse.Namespace, line_ids: list[int]) -> dict[str, Any] | None:
    selected_ids = {int(line_id) for line_id in line_ids}
    manual_groups_path = getattr(args, "manual_groups", None)
    if manual_groups_path:
        data = read_json(Path(manual_groups_path).resolve())
        groups: list[dict[str, Any]] = []
        for group in data.get("groups", []):
            group_line_ids = [int(line_id) for line_id in group.get("lines", [])]
            if group_line_ids and set(group_line_ids).issubset(selected_ids):
                copied = dict(group)
                copied["lines"] = group_line_ids
                groups.append(copied)
        return {"groups": groups}
    if len(line_ids) > 1:
        return {"groups": [{"id": f"u_{line_ids[0]:04d}_{line_ids[-1]:04d}", "lines": line_ids}]}
    return None


def regenerate_lines_command(args: argparse.Namespace) -> int:
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    work_dir = ensure_dir(output_dir / "work")
    logger = setup_logging(output_dir)
    config = apply_cli_overrides(load_config(args.config), args)
    random.seed(config.random_seed)

    line_ids = _parse_line_ids(args.line_ids)
    emotion_path = Path(args.emotion_json).resolve()
    emotions = load_emotion_plan(emotion_path)
    selected_lines = [get_line_by_id(emotions, line_id) for line_id in line_ids]

    speaker_map_path = Path(args.speaker_map).resolve() if args.speaker_map else output_dir / "work" / "speaker_map.json"
    if speaker_map_path.exists():
        speaker_map = read_json(speaker_map_path)
    elif args.refs_dir:
        speaker_map, _warnings = build_speaker_map(
            selected_lines,
            Path(args.refs_dir).resolve(),
            speaker_map_path,
            strict=config.strict_speaker_refs,
            interactive=False,
        )
    else:
        raise FileNotFoundError(f"Could not find speaker map at {speaker_map_path}. Pass --speaker-map or --refs-dir.")

    existing_rows = read_jsonl(work_dir / "line_generation_metadata.jsonl")
    existing_groups = read_jsonl(work_dir / "group_generation_metadata.jsonl")
    sentence_groups_path = work_dir / "sentence_groups.json"
    existing_sentence_groups = read_json(sentence_groups_path) if sentence_groups_path.exists() else []
    manual_groups_data = _manual_groups_data_for_selected(args, line_ids)
    new_rows = synthesize_lines(
        selected_lines,
        selected_lines,
        speaker_map,
        output_dir,
        config,
        logger=logger,
        manual_groups_data=manual_groups_data,
    )
    selected_ids = set(line_ids)
    merged_rows = _merge_rows_replacing_source_ids(existing_rows, new_rows, selected_ids)
    write_jsonl(work_dir / "line_generation_metadata.jsonl", merged_rows)
    write_json(work_dir / "generation_units.json", merged_rows)

    new_groups = read_jsonl(work_dir / "group_generation_metadata.jsonl")
    write_jsonl(work_dir / "group_generation_metadata.jsonl", _replace_group_rows(existing_groups, new_groups))
    new_sentence_groups = read_json(sentence_groups_path) if sentence_groups_path.exists() else []
    sentence_by_id = {
        str(item.get("group_id") or item.get("unit_id")): item
        for item in existing_sentence_groups
        if item.get("group_id") or item.get("unit_id")
    }
    for item in new_sentence_groups:
        key = item.get("group_id") or item.get("unit_id")
        if key:
            sentence_by_id[str(key)] = item
    write_json(
        sentence_groups_path,
        sorted(sentence_by_id.values(), key=lambda item: str(item.get("group_id") or item.get("unit_id") or "")),
    )
    write_generation_report(output_dir, merged_rows, emotion_json=str(emotion_path))
    build_pause_review_manifest(output_dir, write=True)
    logger.info("Regenerated lines %s", line_ids)
    return 0


def review_pauses_command(args: argparse.Namespace) -> int:
    serve_pause_review(args.output_dir, host=args.host, port=args.port)
    return 0


def recombine_command(args: argparse.Namespace) -> int:
    emotions = load_emotion_plan(args.emotion_json)
    report = recombine_lines(
        emotions,
        args.lines_dir,
        args.output,
        gap_ms=args.gap_ms,
        use_timestamps=args.use_timestamps,
        crossfade_ms=args.crossfade_ms,
    )
    item_label = "unit" if "combined_unit_count" in report else "line"
    item_count = report.get("combined_unit_count", report.get("combined_line_count", 0))
    print(f"Recombined {item_count} {item_label}(s) into {report['output']}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = getattr(args, "command", "generate")
    if command == "regenerate-line":
        return regenerate_line_command(args)
    if command == "regenerate-lines":
        return regenerate_lines_command(args)
    if command == "review-pauses":
        return review_pauses_command(args)
    if command == "recombine":
        return recombine_command(args)
    return run_generation(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
