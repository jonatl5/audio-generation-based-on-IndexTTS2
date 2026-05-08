from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def setup_logging(output_dir: str | Path) -> logging.Logger:
    logs_dir = ensure_dir(Path(output_dir) / "logs")
    log_path = logs_dir / "pipeline.log"
    logger = logging.getLogger("stable_dubbing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Logging initialized at %s", log_path)
    return logger


def run_command(cmd: list[str], logger: logging.Logger | None = None) -> subprocess.CompletedProcess[str]:
    if logger:
        logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        if logger:
            logger.error("Command failed with code %s", completed.returncode)
            logger.error("stdout: %s", completed.stdout.strip())
            logger.error("stderr: %s", completed.stderr.strip())
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n{completed.stderr}"
        )
    if logger and completed.stdout.strip():
        logger.debug("stdout: %s", completed.stdout.strip())
    return completed


def ffmpeg_version() -> str:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"], text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return "ffmpeg not found"
    if completed.returncode != 0:
        return "ffmpeg version unavailable"
    return completed.stdout.splitlines()[0] if completed.stdout else "ffmpeg version unavailable"


def probe_media_duration(path: str | Path) -> float | None:
    media_path = Path(path)
    if not media_path.exists():
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return float(completed.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def audio_duration(path: str | Path) -> float:
    audio_path = Path(path)
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
            return frames / float(rate)
        except (wave.Error, EOFError, FileNotFoundError):
            pass
    probed = probe_media_duration(audio_path)
    if probed is None:
        raise RuntimeError(f"Could not determine audio duration for {audio_path}")
    return probed


def get_git_commit(repo_path: str | Path) -> str:
    path = Path(repo_path)
    if not path.exists():
        return "unavailable"
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip()


def collect_environment(repo_path: str | Path, model_dir: str | Path) -> dict[str, Any]:
    torch_info: dict[str, Any] = {
        "torch_version": "not installed",
        "cuda_available": False,
        "gpu_name": "",
    }
    try:
        import torch

        torch_info["torch_version"] = torch.__version__
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch_info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on local install
        torch_info["torch_error"] = str(exc)

    return {
        "os": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "ffmpeg_version": ffmpeg_version(),
        "indextts_repo_path": str(Path(repo_path)),
        "indextts_commit": get_git_commit(repo_path),
        "model_checkpoint_path": str(Path(model_dir)),
        **torch_info,
    }


def relative_or_absolute(path: str | Path, base: str | Path | None = None) -> str:
    candidate = Path(path)
    if base is not None:
        try:
            return os.path.relpath(candidate, Path(base))
        except ValueError:
            return str(candidate)
    return str(candidate)

