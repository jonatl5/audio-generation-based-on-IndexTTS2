from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import DubbingConfig
from .utils import ensure_dir, read_json, read_jsonl, write_json


MOS_COLUMNS = [
    "line_id",
    "speaker",
    "text",
    "audio_path",
    "naturalness_1_5",
    "emotional_fidelity_1_5",
    "timbre_stability_1_5",
    "av_sync_1_5",
    "comments",
]


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def normalize_for_cer(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_for_cer(reference))
    hyp = list(normalize_for_cer(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    try:
        import jiwer

        return float(jiwer.wer(reference, hypothesis))
    except Exception:
        ref = re.findall(r"\w+", reference.lower())
        hyp = re.findall(r"\w+", hypothesis.lower())
        if not ref:
            return 0.0 if not hyp else 1.0
        return edit_distance(ref, hyp) / len(ref)


def detect_metric_language(text: str, configured: str) -> str:
    if configured and configured.lower() not in {"auto", ""}:
        return configured.lower()
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return "zh" if cjk_chars >= max(1, len(text) // 5) else "en"


class AsrEngine:
    def __init__(self, language: str = "auto") -> None:
        self.language = None if language == "auto" else language
        self.engine_name = ""
        self.model: Any = None
        self.skip_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            try:
                self.model = WhisperModel("small", device="auto", compute_type="auto")
                self.engine_name = "faster-whisper"
                return
            except Exception as exc:
                auto_error = str(exc)
                self.model = WhisperModel("small", device="cpu", compute_type="int8")
                self.engine_name = "faster-whisper-cpu-int8"
                return
        except Exception as exc:
            first_error = str(exc)
            if "auto_error" in locals():
                first_error = f"auto failed ({auto_error}); cpu failed ({exc})"
        try:
            import whisper

            self.model = whisper.load_model("small")
            self.engine_name = "openai-whisper"
            return
        except Exception as exc:
            self.skip_reason = f"ASR unavailable: faster-whisper failed ({first_error}); openai-whisper failed ({exc})"

    def transcribe(self, path: str | Path) -> str:
        if not self.model:
            return ""
        if self.engine_name == "faster-whisper":
            segments, _info = self.model.transcribe(str(path), language=self.language)
            return " ".join(segment.text.strip() for segment in segments).strip()
        result = self.model.transcribe(str(path), language=self.language)
        return str(result.get("text", "")).strip()


def create_mos_rating_sheet(
    lines: list[dict[str, Any]],
    metadata_by_id: dict[int, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOS_COLUMNS)
        writer.writeheader()
        for line in lines:
            meta = metadata_by_id.get(int(line["id"]), {})
            writer.writerow(
                {
                    "line_id": line["id"],
                    "speaker": line["speaker"],
                    "text": line["text"],
                    "audio_path": meta.get("aligned_output") or meta.get("raw_output") or "",
                    "naturalness_1_5": "",
                    "emotional_fidelity_1_5": "",
                    "timbre_stability_1_5": "",
                    "av_sync_1_5": "",
                    "comments": "",
                }
            )
    return target


def compute_mos_summary(mos_sheet: str | Path) -> dict[str, Any]:
    categories = [
        "naturalness_1_5",
        "emotional_fidelity_1_5",
        "timbre_stability_1_5",
        "av_sync_1_5",
    ]
    values: dict[str, list[float]] = {category: [] for category in categories}
    with Path(mos_sheet).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for category in categories:
                raw = (row.get(category) or "").strip()
                if not raw:
                    continue
                try:
                    score = float(raw)
                except ValueError:
                    continue
                if 1.0 <= score <= 5.0:
                    values[category].append(score)
    summary = {
        category: (sum(scores) / len(scores) if scores else None)
        for category, scores in values.items()
    }
    all_scores = [score for scores in values.values() for score in scores]
    summary["overall"] = sum(all_scores) / len(all_scores) if all_scores else None
    summary["rated_items"] = max((len(scores) for scores in values.values()), default=0)
    return summary


def run_content_consistency(
    lines: list[dict[str, Any]],
    metadata_by_id: dict[int, dict[str, Any]],
    output_dir: str | Path,
    language: str,
) -> dict[str, Any]:
    target = Path(output_dir) / "content_consistency.csv"
    ensure_dir(target.parent)
    engine = AsrEngine(language)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"path": str(target), "engine": engine.engine_name}

    if engine.skip_reason:
        summary["skipped"] = engine.skip_reason
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["line_id", "speaker", "reference_text", "asr_text", "wer", "cer", "language", "notes"],
            )
            writer.writeheader()
            for line in lines:
                writer.writerow(
                    {
                        "line_id": line["id"],
                        "speaker": line["speaker"],
                        "reference_text": line["text"],
                        "asr_text": "",
                        "wer": "",
                        "cer": "",
                        "language": detect_metric_language(line["text"], language),
                        "notes": engine.skip_reason,
                    }
                )
        return summary

    for line in lines:
        meta = metadata_by_id.get(int(line["id"]), {})
        audio_path = meta.get("raw_output") or meta.get("aligned_output")
        notes = ""
        asr_text = ""
        line_wer: float | str = ""
        line_cer: float | str = ""
        metric_language = detect_metric_language(line["text"], language)
        if not audio_path or not Path(audio_path).exists():
            notes = "audio missing"
        else:
            asr_text = engine.transcribe(audio_path)
            if metric_language.startswith("zh"):
                line_cer = cer(line["text"], asr_text)
            else:
                line_wer = wer(line["text"], asr_text)
        rows.append(
            {
                "line_id": line["id"],
                "speaker": line["speaker"],
                "reference_text": line["text"],
                "asr_text": asr_text,
                "wer": line_wer,
                "cer": line_cer,
                "language": metric_language,
                "notes": notes,
            }
        )

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["line_id", "speaker", "reference_text", "asr_text", "wer", "cer", "language", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    numeric_wer = [float(row["wer"]) for row in rows if row["wer"] != ""]
    numeric_cer = [float(row["cer"]) for row in rows if row["cer"] != ""]
    summary["mean_wer"] = sum(numeric_wer) / len(numeric_wer) if numeric_wer else None
    summary["mean_cer"] = sum(numeric_cer) / len(numeric_cer) if numeric_cer else None
    summary["worst_cases"] = sorted(
        rows,
        key=lambda row: max(float(row["wer"] or 0), float(row["cer"] or 0)),
        reverse=True,
    )[:5]
    return summary


def run_speaker_similarity(
    lines: list[dict[str, Any]],
    speaker_map: dict[str, str],
    metadata_by_id: dict[int, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    eval_dir = Path(output_dir)
    by_line_path = eval_dir / "sim_by_line.csv"
    by_speaker_path = eval_dir / "sim_by_speaker.csv"
    ensure_dir(eval_dir)
    try:
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except Exception as exc:
        return {"skipped": f"SIM skipped because SpeechBrain/Torch is unavailable: {exc}"}

    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(eval_dir / "speechbrain_ecapa"),
            local_strategy=LocalStrategy.COPY,
        )
    except Exception as exc:
        return {"skipped": f"SIM skipped because speaker embedding model could not load: {exc}"}

    def load_signal(path: str | Path):
        try:
            signal, sample_rate = torchaudio.load(str(path))
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                signal = torchaudio.functional.resample(signal, sample_rate, 16000)
            return signal.squeeze(0)
        except Exception:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
            samples = torch.tensor(audio.get_array_of_samples(), dtype=torch.float32)
            peak = float(1 << (8 * audio.sample_width - 1))
            return samples / peak

    def embedding(path: str | Path):
        with torch.no_grad():
            signal = load_signal(path)
            return classifier.encode_batch(signal).squeeze()

    ref_embeddings: dict[str, Any] = {}
    for speaker, ref_path in speaker_map.items():
        if Path(ref_path).exists():
            ref_embeddings[speaker] = embedding(ref_path)

    rows: list[dict[str, Any]] = []
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for line in lines:
        speaker = line["speaker"]
        meta = metadata_by_id.get(int(line["id"]), {})
        generated = meta.get("raw_output") or meta.get("aligned_output")
        if speaker not in ref_embeddings or not generated or not Path(generated).exists():
            continue
        gen_emb = embedding(generated)
        similarity = torch.nn.functional.cosine_similarity(
            ref_embeddings[speaker].flatten(), gen_emb.flatten(), dim=0
        ).item()
        rows.append(
            {
                "line_id": line["id"],
                "speaker": speaker,
                "audio_path": generated,
                "sim": similarity,
            }
        )
        by_speaker[speaker].append(similarity)

    with by_line_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["line_id", "speaker", "audio_path", "sim"])
        writer.writeheader()
        writer.writerows(rows)

    speaker_rows = []
    for speaker, scores in by_speaker.items():
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
        speaker_rows.append(
            {
                "speaker": speaker,
                "mean": mean,
                "std": std,
                "min": min(scores),
                "max": max(scores),
                "number_of_lines": len(scores),
            }
        )
    with by_speaker_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["speaker", "mean", "std", "min", "max", "number_of_lines"]
        )
        writer.writeheader()
        writer.writerows(speaker_rows)
    return {
        "by_line_path": str(by_line_path),
        "by_speaker_path": str(by_speaker_path),
        "speaker_summary": speaker_rows,
    }


def run_evaluation(
    lines: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    speaker_map: dict[str, str],
    output_dir: str | Path,
    config: DubbingConfig,
) -> dict[str, Any]:
    eval_dir = ensure_dir(Path(output_dir) / "evaluation")
    metadata_by_id = {int(row["line_id"]): row for row in metadata_rows if "line_id" in row}
    summary: dict[str, Any] = {}
    if config.evaluation.create_mos_sheet:
        mos_path = create_mos_rating_sheet(lines, metadata_by_id, eval_dir / "mos_rating_sheet.csv")
        summary["mos_sheet"] = str(mos_path)
        mos_summary = compute_mos_summary(mos_path)
        if mos_summary.get("overall") is not None:
            summary["mos"] = mos_summary
    if config.evaluation.run_asr:
        summary["content_consistency"] = run_content_consistency(
            lines, metadata_by_id, eval_dir, config.language
        )
    if config.evaluation.run_sim:
        summary["sim"] = run_speaker_similarity(lines, speaker_map, metadata_by_id, eval_dir)
    return summary


def rerun_completed_output_evaluation(
    output_dir: str | Path,
    config: DubbingConfig | None = None,
    update_quality_report: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    config = config or DubbingConfig()
    lines = read_json(output / "work" / "script_structured.json")
    metadata_rows = read_jsonl(output / "work" / "line_generation_metadata.jsonl")
    speaker_map = read_json(output / "work" / "speaker_map.json")
    summary = run_evaluation(lines, metadata_rows, speaker_map, output, config)

    if update_quality_report:
        report_path = output / "quality_report.json"
        if report_path.exists():
            report = read_json(report_path)
            report["evaluation_summary"] = summary
            write_json(report_path, report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed dubbing output.")
    parser.add_argument("--mos_sheet", help="Path to output/evaluation/mos_rating_sheet.csv")
    parser.add_argument("--output_dir", help="Completed output folder containing work/ and lines/")
    parser.add_argument("--language", default=None, help="Override evaluation language: auto, en, zh, etc.")
    parser.add_argument("--skip_asr", action="store_true", help="Skip ASR content consistency")
    parser.add_argument("--skip_sim", action="store_true", help="Skip speaker similarity")
    parser.add_argument("--skip_mos", action="store_true", help="Skip MOS sheet creation/summary")
    parser.add_argument(
        "--no_update_quality_report",
        action="store_true",
        help="Do not update quality_report.json when using --output_dir",
    )
    args = parser.parse_args()

    if args.output_dir:
        config = DubbingConfig()
        if args.language:
            config.language = args.language
        if args.skip_asr:
            config.evaluation.run_asr = False
        if args.skip_sim:
            config.evaluation.run_sim = False
        if args.skip_mos:
            config.evaluation.create_mos_sheet = False
        summary = rerun_completed_output_evaluation(
            args.output_dir,
            config=config,
            update_quality_report=not args.no_update_quality_report,
        )
        output_path = Path(args.output_dir) / "evaluation" / "evaluation_summary.json"
    elif args.mos_sheet:
        summary = compute_mos_summary(args.mos_sheet)
        output_path = Path(args.mos_sheet).with_name("mos_summary.json")
    else:
        parser.error("one of --output_dir or --mos_sheet is required")
        return

    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Evaluation summary written to {output_path}")


if __name__ == "__main__":
    main()
