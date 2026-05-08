from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class AlignmentConfig:
    stretch_if_long_under_ratio: float = 1.0
    regenerate_if_long_over_ratio: float = 999.0
    max_regen_attempts: int = 1
    allow_text_shortening: bool = False
    max_stretch_speed_factor: float = 1.5


@dataclass
class EvaluationConfig:
    run_asr: bool = True
    run_sim: bool = True
    create_mos_sheet: bool = True


@dataclass
class SilenceCleanupConfig:
    enabled: bool = True
    min_silence_duration: float = 0.18
    silence_threshold_db: float = -45.0
    keep_internal_silence: float = 0.06
    keep_comma_silence: float = 0.18
    keep_clause_silence: float = 0.22
    keep_sentence_silence: float = 0.30
    keep_edge_silence: float = 0.02
    preserve_punctuation_pauses: bool = True


@dataclass
class DubbingConfig:
    dialogue_styles: list[str] = field(default_factory=lambda: ["Default"])
    skip_styles: list[str] = field(
        default_factory=lambda: ["画面字-[左]", "画面字-[中]", "画面字-[右]", "集数"]
    )
    strict_speaker_refs: bool = True
    default_emo_alpha: float = 0.55
    max_emo_alpha: float = 0.8
    use_random: bool = False
    language: str = "auto"
    sample_rate: int = 24000
    output_audio_format: str = "wav"
    pause_for_emotion_edit: bool = True
    use_fp16: bool = False
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    indextts_repo_path: str = "third_party/index-tts"
    indextts_cfg_path: str = ""
    indextts_model_dir: str = ""
    random_seed: int = 1234
    mix_original: bool = False
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    silence_cleanup: SilenceCleanupConfig = field(default_factory=SilenceCleanupConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _deep_update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    field_map = {f.name: f for f in fields(instance)}
    for key, value in updates.items():
        if key not in field_map:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path | None = None) -> DubbingConfig:
    config = DubbingConfig()
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return _deep_update_dataclass(config, data)


def config_to_dict(config: DubbingConfig) -> dict[str, Any]:
    return asdict(config)
