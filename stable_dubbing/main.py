from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio_assemble import assemble_aligned_track, concatenate_raw_lines
from .config import DubbingConfig, load_config, str_to_bool
from .emotion_prepare import (
    load_validated_emotions,
    validate_emotion_items,
    wait_for_emotion_edits,
    write_emotion_file,
)
from .evaluation import create_mos_rating_sheet, run_evaluation
from .report import write_quality_report
from .speaker_map import SpeakerMapError, build_speaker_map
from .subtitle_parser import parse_subtitle
from .tts_indextts2 import synthesize_lines
from .utils import ensure_dir, probe_media_duration, read_json, setup_logging, write_json, write_jsonl
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
    if args.language:
        config.language = args.language
    if args.pause_for_emotion_edit is not None:
        config.pause_for_emotion_edit = bool(str_to_bool(args.pause_for_emotion_edit))
    if args.no_pause:
        config.pause_for_emotion_edit = False
    if args.use_fp16:
        config.use_fp16 = True
    if args.use_cuda_kernel:
        config.use_cuda_kernel = True
    if args.use_deepspeed:
        config.use_deepspeed = True
    if args.mix_original:
        config.mix_original = True
    if args.indextts_repo_path:
        config.indextts_repo_path = args.indextts_repo_path
    if args.indextts_cfg_path:
        config.indextts_cfg_path = args.indextts_cfg_path
    if args.indextts_model_dir:
        config.indextts_model_dir = args.indextts_model_dir
    if args.non_strict_speaker_refs:
        config.strict_speaker_refs = False
    if args.dry_run:
        config.pause_for_emotion_edit = False
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS2 stable emotion dubbing pipeline.")
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
    parser.add_argument("--resume_from_emotion_file", help="Use an existing emotions_to_edit.json")
    parser.add_argument("--dry_run", action="store_true", help="Parse, prepare emotions, map speakers, and write report without TTS")
    parser.add_argument("--mix_original", action="store_true", help="Mix original video audio with synthesized audio")
    parser.add_argument("--use_fp16", action="store_true", help="Enable IndexTTS2 FP16 inference")
    parser.add_argument("--use_cuda_kernel", action="store_true", help="Enable IndexTTS2 CUDA kernel")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable IndexTTS2 DeepSpeed")
    parser.add_argument("--indextts_repo_path", help="Path to official index-tts repo")
    parser.add_argument("--indextts_cfg_path", help="Path to checkpoints/config.yaml")
    parser.add_argument("--indextts_model_dir", help="Path to checkpoints directory")
    parser.add_argument("--non_strict_speaker_refs", action="store_true", help="Allow fallback speaker refs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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

        if args.resume_from_emotion_file:
            emotion_path = Path(args.resume_from_emotion_file).resolve()
            emotions = load_validated_emotions(emotion_path)
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
        }

        if args.dry_run:
            metadata_by_id: dict[int, dict[str, Any]] = {}
            write_jsonl(work_dir / "line_generation_metadata.jsonl", metadata_rows)
            create_mos_rating_sheet(lines, metadata_by_id, output_dir / "evaluation" / "mos_rating_sheet.csv")
            evaluation_summary = {
                "mos_sheet": str(output_dir / "evaluation" / "mos_rating_sheet.csv"),
                "content_consistency": {"skipped": "dry run: ASR was not executed"},
                "sim": {"skipped": "dry run: SIM was not executed"},
            }
        else:
            metadata_rows = synthesize_lines(lines, emotions, speaker_map, output_dir, config, logger=logger)
            failed_rows = [row for row in metadata_rows if row.get("error")]
            successful_rows = [row for row in metadata_rows if not row.get("error")]
            if not successful_rows:
                write_json(work_dir / "warnings.json", all_warnings)
                raise RuntimeError(
                    f"TTS generation failed for all {len(failed_rows)} subtitle lines. "
                    f"See {work_dir / 'line_generation_metadata.jsonl'} and {output_dir / 'logs' / 'pipeline.log'}."
                )
            raw_paths = [row["raw_output"] for row in metadata_rows if row.get("raw_output") and not row.get("error")]
            concat_warnings = concatenate_raw_lines(
                raw_paths, audio_dir / "voice_cast_raw_concatenated.wav", logger=logger
            )
            all_warnings.extend(concat_warnings)

            video_duration = probe_media_duration(paths["video"])
            if video_duration is None:
                video_duration = max((float(line["end"]) for line in lines), default=0.0)
                all_warnings.append("video duration unavailable; using last subtitle end for audio bed")
            aligned_by_id = {
                int(row["line_id"]): row["aligned_output"]
                for row in metadata_rows
                if row.get("aligned_output") and not row.get("error")
            }
            assembly_warnings = assemble_aligned_track(
                lines,
                aligned_by_id,
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
        logger.info("Pipeline finished. Output directory: %s", output_dir)
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
