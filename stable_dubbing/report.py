from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import DubbingConfig, config_to_dict
from .utils import collect_environment, ensure_dir, write_json


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_generation(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in metadata_rows if not row.get("error")]
    failed = [row for row in metadata_rows if row.get("error")]
    diffs = [
        abs(float(row.get("raw_duration", 0.0)) - float(row.get("target_duration", 0.0)))
        for row in successful
        if row.get("raw_duration") is not None
    ]
    return {
        "successful_lines": len(successful),
        "failed_lines": len(failed),
        "successful_units": len(successful),
        "failed_units": len(failed),
        "total_attempts": sum(int(row.get("attempts", 0)) for row in metadata_rows),
        "average_raw_duration_difference": _mean(diffs),
        "alignment_actions": dict(Counter(row.get("alignment_action", "unknown") for row in successful)),
        "duration_scales": [
            row.get("duration_scale")
            for row in successful
            if row.get("duration_scale") is not None
        ],
    }


def write_quality_report(
    output_dir: str | Path,
    config: DubbingConfig,
    command: str,
    input_paths: dict[str, str],
    lines: list[dict[str, Any]],
    speaker_map: dict[str, str],
    unresolved_speakers: list[str],
    warnings: list[str],
    metadata_rows: list[dict[str, Any]],
    evaluation_summary: dict[str, Any],
    final_outputs: dict[str, str],
    dry_run: bool = False,
) -> tuple[Path, Path]:
    output = ensure_dir(output_dir)
    repo_path = config.indextts_repo_path
    model_dir = config.indextts_model_dir or str(Path(repo_path) / "checkpoints")
    environment = collect_environment(repo_path, model_dir)
    generation_summary = summarize_generation(metadata_rows)
    speakers = sorted({line["speaker"] for line in lines})

    report_json = {
        "project": "stable_emotion_dubbing_indextts2",
        "dry_run": dry_run,
        "input_paths": input_paths,
        "environment": environment,
        "subtitle_line_count": len(lines),
        "speakers_found": speakers,
        "speaker_map": speaker_map,
        "unresolved_speakers": unresolved_speakers,
        "generation_summary": generation_summary,
        "evaluation_summary": evaluation_summary,
        "warnings": warnings,
        "reproducibility": {
            "command": command,
            "config": config_to_dict(config),
            "random_seed": config.random_seed,
            "output_dir": str(output),
        },
        "final_outputs": final_outputs,
    }
    json_path = write_json(output / "quality_report.json", report_json)

    cc = evaluation_summary.get("content_consistency", {})
    sim = evaluation_summary.get("sim", {})
    mos = evaluation_summary.get("mos")
    mos_sheet = evaluation_summary.get("mos_sheet")

    lines_md = [
        "# Quality Report",
        "",
        f"Project: stable_emotion_dubbing_indextts2",
        f"Mode: {'dry run' if dry_run else 'full generation'}",
        "",
        "## Inputs",
        f"- Video: {input_paths.get('video', '')}",
        f"- Script: {input_paths.get('script', '')}",
        f"- Speaker refs: {input_paths.get('refs_dir', '')}",
        "",
        "## IndexTTS2 And Environment",
        f"- Repo path: {environment['indextts_repo_path']}",
        f"- Commit: {environment['indextts_commit']}",
        f"- Model checkpoint path: {environment['model_checkpoint_path']}",
        f"- OS: {environment['os']}",
        f"- Python: {environment['python_version']}",
        f"- Torch: {environment.get('torch_version')}",
        f"- CUDA available: {environment.get('cuda_available')}",
        f"- GPU: {environment.get('gpu_name') or 'none detected'}",
        f"- FFmpeg: {environment['ffmpeg_version']}",
        "",
        "## Subtitle And Speakers",
        f"- Subtitle lines: {len(lines)}",
        f"- Speakers found: {', '.join(speakers) if speakers else 'none'}",
        f"- Missing or unresolved speakers: {', '.join(unresolved_speakers) if unresolved_speakers else 'none'}",
        f"- Emotion file: {final_outputs.get('emotion_json', '')}",
        "",
        "## Generation Summary",
        f"- Successful units: {generation_summary['successful_units']}",
        f"- Failed units: {generation_summary['failed_units']}",
        f"- Total attempts: {generation_summary['total_attempts']}",
        f"- Average raw duration difference: {generation_summary['average_raw_duration_difference']}",
        f"- Alignment actions: {generation_summary['alignment_actions']}",
        "",
        "## Content Consistency",
    ]
    if cc.get("skipped"):
        lines_md.append(f"- Skipped: {cc['skipped']}")
    else:
        lines_md.extend(
            [
                f"- ASR engine: {cc.get('engine', '')}",
                f"- Mean WER: {cc.get('mean_wer')}",
                f"- Mean CER: {cc.get('mean_cer')}",
                f"- Results: {cc.get('path', '')}",
            ]
        )

    lines_md.extend(["", "## SIM"])
    if sim.get("skipped"):
        lines_md.append(f"- Skipped: {sim['skipped']}")
    elif sim.get("speaker_summary"):
        for row in sim["speaker_summary"]:
            lines_md.append(f"- {row['speaker']}: mean={row['mean']:.4f}, n={row['number_of_lines']}")
    else:
        lines_md.append("- No SIM results available.")

    lines_md.extend(["", "## MOS"])
    if mos:
        lines_md.append(f"- Human MOS provided. Overall: {mos.get('overall')}")
    else:
        lines_md.append(f"- Human MOS not provided. Rating sheet: {mos_sheet or 'not created'}")

    lines_md.extend(["", "## Warnings"])
    if warnings:
        lines_md.extend(f"- {warning}" for warning in warnings)
    else:
        lines_md.append("- none")

    lines_md.extend(
        [
            "",
            "## Reproducibility",
            f"- Command: `{command}`",
            f"- Random seed: {config.random_seed}",
            f"- Output directory: {output}",
            "",
            "## Final Outputs",
            f"- Final video: {final_outputs.get('final_video', '')}",
            f"- Aligned WAV: {final_outputs.get('aligned_wav', '')}",
            f"- Raw concatenated WAV: {final_outputs.get('raw_concatenated_wav', '')}",
            f"- Emotion JSON: {final_outputs.get('emotion_json', '')}",
            f"- Metadata JSONL: {final_outputs.get('metadata_jsonl', '')}",
        ]
    )
    md_path = output / "quality_report.md"
    md_path.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    return md_path, json_path
