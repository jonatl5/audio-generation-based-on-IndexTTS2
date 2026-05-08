from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import DubbingConfig
from .duration_align import choose_alignment_action, render_aligned_audio
from .silence_cleanup import tighten_silences
from .utils import audio_duration, ensure_dir, write_jsonl


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
    ) -> None:
        self.initialize()
        assert self.tts is not None
        method = emotion.get("emotion_method", "emo_text")
        emo_alpha = float(emo_alpha_override if emo_alpha_override is not None else emotion.get("emo_alpha", 0.55))
        use_random = bool(emotion.get("use_random", False))
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": spk_audio_prompt,
            "text": text,
            "output_path": str(output_path),
            "emo_alpha": emo_alpha,
            "use_random": use_random,
            "verbose": True,
        }
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


def synthesize_lines(
    lines: list[dict[str, Any]],
    emotions: list[dict[str, Any]],
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    logger = logger or logging.getLogger("stable_dubbing.tts")
    output = Path(output_dir)
    raw_dir = ensure_dir(output / "lines" / "raw")
    aligned_dir = ensure_dir(output / "lines" / "aligned")
    public_lines_dir = ensure_dir(output / "lines")
    metadata_path = output / "work" / "line_generation_metadata.jsonl"
    wrapper = IndexTTS2Wrapper(config, logger=logger)
    wrapper.initialize()
    emotion_by_id = {int(item["id"]): item for item in emotions}
    metadata_rows: list[dict[str, Any]] = []

    for line in lines:
        line_id = int(line["id"])
        emotion = emotion_by_id[line_id]
        speaker = line["speaker"]
        target_duration = float(line["target_duration"])
        raw_output = raw_dir / f"line_{line_id:04d}.wav"
        aligned_output = aligned_dir / f"line_{line_id:04d}.wav"
        public_output = public_lines_dir / f"line_{line_id:04d}.wav"
        warnings: list[str] = []
        attempts = 0
        alignment_action = ""
        final_duration: float | None = None
        raw_duration: float | None = None
        speaker_reference = speaker_map.get(speaker)

        metadata = {
            "line_id": line_id,
            "speaker": speaker,
            "text": line["text"],
            "start": line["start"],
            "end": line["end"],
            "target_duration": target_duration,
            "method": emotion.get("emotion_method", "emo_text"),
            "emo_text": emotion.get("emo_text"),
            "emo_vector": emotion.get("emo_vector"),
            "emo_alpha": emotion.get("emo_alpha"),
            "use_random": emotion.get("use_random", False),
            "speaker_reference": speaker_reference,
            "raw_output": str(raw_output),
            "aligned_output": str(aligned_output),
            "raw_duration": None,
            "final_duration": None,
            "alignment_action": None,
            "attempts": 0,
            "silence_cleanup": None,
            "warnings": warnings,
        }

        try:
            if not speaker_reference:
                raise RuntimeError(f"No speaker reference mapped for {speaker!r}")
            attempts = 1
            candidate = raw_dir / f"line_{line_id:04d}_attempt_1.wav"
            wrapper.infer_line(speaker_reference, line["text"], candidate, emotion)
            cleanup = tighten_silences(
                candidate,
                raw_output,
                line["text"],
                config.silence_cleanup,
                logger=logger,
            )
            raw_duration = cleanup.output_duration
            metadata["silence_cleanup"] = cleanup.to_dict()
            warnings.extend(cleanup.warnings)
            decision = choose_alignment_action(raw_duration, target_duration, config.alignment)
            logger.info(
                "Line %s single pass raw_duration=%.3f target=%.3f action=%s",
                line_id,
                raw_duration,
                target_duration,
                decision.action,
            )

            split_done = False
            if _exceeds_stretch_cap(raw_duration, target_duration, config):
                speed_factor = raw_duration / target_duration
                segments = split_text_by_punctuation(line["text"])
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
            final_duration = result.final_duration
            alignment_action = result.alignment_action
            warnings.extend(result.warnings)
            metadata.update(
                {
                    "raw_duration": raw_duration,
                    "final_duration": final_duration,
                    "alignment_action": alignment_action,
                    "attempts": attempts,
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            logger.exception("Failed to synthesize line %s", line_id)
            metadata.update({"error": str(exc), "attempts": attempts, "warnings": warnings})
        metadata_rows.append(metadata)

    write_jsonl(metadata_path, metadata_rows)
    return metadata_rows
