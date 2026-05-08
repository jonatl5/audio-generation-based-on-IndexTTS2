from __future__ import annotations

import logging
from pathlib import Path

from .utils import ensure_dir, run_command


def mux_video(
    video_path: str | Path,
    aligned_audio_path: str | Path,
    output_path: str | Path,
    mix_original: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    video = Path(video_path)
    audio = Path(aligned_audio_path)
    target = Path(output_path)
    ensure_dir(target.parent)

    if mix_original:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-filter_complex",
            "[0:a:0][1:a:0]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ]
    run_command(cmd, logger=logger)
    if logger:
        logger.info("Final dubbed video written to %s", target)
    return target

