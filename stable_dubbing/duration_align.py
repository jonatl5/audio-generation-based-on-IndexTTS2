from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import AlignmentConfig
from .utils import audio_duration, ensure_dir, run_command


@dataclass
class AlignmentDecision:
    action: str
    should_regenerate: bool
    raw_duration: float
    target_duration: float
    over_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    output_path: str
    raw_duration: float
    final_duration: float
    alignment_action: str
    warnings: list[str] = field(default_factory=list)


def choose_alignment_action(
    raw_duration: float, target_duration: float, config: AlignmentConfig
) -> AlignmentDecision:
    if target_duration <= 0:
        return AlignmentDecision(
            action="invalid_target_duration",
            should_regenerate=False,
            raw_duration=raw_duration,
            target_duration=target_duration,
            warnings=["target duration must be greater than zero"],
        )
    if raw_duration <= target_duration:
        return AlignmentDecision(
            action="pad_silence_end",
            should_regenerate=False,
            raw_duration=raw_duration,
            target_duration=target_duration,
        )
    over_ratio = (raw_duration - target_duration) / target_duration
    if over_ratio >= config.regenerate_if_long_over_ratio:
        return AlignmentDecision(
            action="regenerate",
            should_regenerate=True,
            raw_duration=raw_duration,
            target_duration=target_duration,
            over_ratio=over_ratio,
            warnings=[f"audio is {over_ratio:.1%} longer than target"],
        )
    return AlignmentDecision(
        action="time_stretch",
        should_regenerate=False,
        raw_duration=raw_duration,
        target_duration=target_duration,
        over_ratio=over_ratio,
        warnings=(
            [f"audio is {over_ratio:.1%} longer than preferred stretch ratio"]
            if over_ratio > config.stretch_if_long_under_ratio
            else []
        ),
    )


def build_atempo_filter(speed_factor: float) -> str:
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    factors: list[float] = []
    remaining = speed_factor
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def render_aligned_audio(
    input_path: str | Path,
    output_path: str | Path,
    target_duration: float,
    config: AlignmentConfig,
    logger: logging.Logger | None = None,
    force: bool = False,
    action_label: str | None = None,
) -> AlignmentResult:
    source = Path(input_path)
    target = Path(output_path)
    ensure_dir(target.parent)
    raw_duration = audio_duration(source)
    decision = choose_alignment_action(raw_duration, target_duration, config)
    if decision.should_regenerate and not force:
        return AlignmentResult(
            output_path=str(source),
            raw_duration=raw_duration,
            final_duration=raw_duration,
            alignment_action=decision.action,
            warnings=decision.warnings,
        )

    warnings = list(decision.warnings)
    if raw_duration <= target_duration:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            "apad",
            "-t",
            f"{target_duration:.6f}",
            str(target),
        ]
        action = "pad_silence_end"
    else:
        speed_factor = raw_duration / target_duration
        if speed_factor > config.max_stretch_speed_factor:
            warnings.append(
                f"speed factor {speed_factor:.3f} exceeds preferred cap {config.max_stretch_speed_factor:.3f}"
            )
        atempo = build_atempo_filter(speed_factor)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            f"{atempo},apad",
            "-t",
            f"{target_duration:.6f}",
            str(target),
        ]
        action = "time_stretch"
    run_command(cmd, logger=logger)
    final_duration = audio_duration(target)
    return AlignmentResult(
        output_path=str(target),
        raw_duration=raw_duration,
        final_duration=final_duration,
        alignment_action=action_label or action,
        warnings=warnings,
    )
