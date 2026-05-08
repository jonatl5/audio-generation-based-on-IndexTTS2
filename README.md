# stable_emotion_dubbing_indextts2

`stable_emotion_dubbing_indextts2` is a command-line dubbing pipeline for turning a silent or no-original-sound MP4 plus `.ass` or `.txt` subtitles into a voice-casted, timestamp-aligned dubbed MP4 using the official IndexTTS2 implementation.

The workflow is intentionally editable:

1. Parse subtitle lines into structured JSON.
2. Create `output/run_YYYYMMDD_HHMMSS/work/emotions_to_edit.json`.
3. Let you edit `emo_text`, `emo_alpha`, or an 8-value `emo_vector`.
4. Reload and validate that emotion file.
5. Generate per-line IndexTTS2 audio using one speaker reference file per character.
6. Align line duration to subtitle timing.
7. Assemble raw and timestamp-aligned WAV files.
8. Mux the final MP4.
9. Write evaluation files and a quality report without fabricating missing scores.

## Official IndexTTS2 API Confirmed

This project uses the official repository: <https://github.com/index-tts/index-tts>.

The current IndexTTS2 constructor signature in `indextts/infer_v2.py` is:

```python
IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=False,
    device=None,
    use_cuda_kernel=None,
    use_deepspeed=False,
    use_accel=False,
    use_torch_compile=False,
)
```

The current `tts.infer(...)` signature is:

```python
tts.infer(
    spk_audio_prompt,
    text,
    output_path,
    emo_audio_prompt=None,
    emo_alpha=1.0,
    emo_vector=None,
    use_emo_text=False,
    emo_text=None,
    use_random=False,
    interval_silence=200,
    verbose=False,
    max_text_tokens_per_segment=120,
    stream_return=False,
    more_segment_before=0,
    **generation_kwargs,
)
```

IndexTTS2 internally exposes `qwen_emo.inference(emo_text)` when `use_emo_text=True`; this converts a human-readable emotion description into vectors ordered as:

```text
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
```

This project does not assume IndexTTS2 can generate a human-readable emotion file from your script. It creates a first-pass editable emotion-description file with local rules, then passes your edited `emo_text` or `emo_vector` into IndexTTS2.

## Input Folder Format

```text
input_folder/
  video.mp4
  script.ass
  refs/
    Jiang Han.wav
    Lu Ting.wav
    Succubus.wav
```

Reference audio file names should match subtitle speaker names after normalization. These all match `Jiang Han`:

```text
Jiang Han.wav
Jiang_Han.wav
Jiang Han(男).wav
```

Supported reference extensions: `.wav`, `.mp3`, `.flac`, `.m4a`.

## Subtitle Formats

`.ass` is preferred. The parser:

- Reads the `[Events]` section.
- Reads the `Format:` line and uses field names.
- Splits `Dialogue:` rows carefully so commas inside `Text` are preserved.
- Keeps `Style == "Default"` by default.
- Skips screen text styles such as `画面字-[左]`, `画面字-[中]`, `画面字-[右]`, and `集数`.
- Parses names like `Jiang Han(男)` into speaker `Jiang Han`, gender `男`.
- Splits safe multi-speaker dash lines such as `Manager(男)&Beastmaster 1(男)` with `- A - B`.

`.txt` supports either:

```text
[00:00:01.200 --> 00:00:03.800] Speaker: text
00:00:01.200 --> 00:00:03.800 | Speaker | text
```

## Setup

This repository intentionally does not commit model checkpoints, generated audio,
or the official IndexTTS2 source tree. The setup scripts clone IndexTTS2 into
`third_party/index-tts`, install this pipeline into the same uv environment, and
download the IndexTTS2 checkpoints.

Required system tools:

- `git`
- `git-lfs`
- `ffmpeg`
- `uv`
- `python`
- NVIDIA driver/CUDA-capable GPU recommended for practical speed

Python dependencies installed by the setup scripts:

- pipeline runtime: `pydub`, `PyYAML`, `jiwer`
- ASR evaluation: `faster-whisper`, `openai-whisper`
- speaker similarity evaluation: `speechbrain`, `torch`, `torchaudio`
- test runner: `pytest`
- official IndexTTS2 dependencies from `third_party/index-tts`

### Install System Tools With Bash

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg python3 python3-pip curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs install
```

macOS with Homebrew:

```bash
brew install git git-lfs ffmpeg uv python
git lfs install
```

Then clone and set up:

```bash
git clone https://github.com/jonatl5/audio-generation-based-on-IndexTTS2.git
cd audio-generation-based-on-IndexTTS2
bash setup_env.sh
```

### Install System Tools With PowerShell

Windows with winget:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id Gyan.FFmpeg -e
winget install --id Astral.UV -e
winget install --id Python.Python.3.10 -e
git lfs install
```

Then clone and set up:

```powershell
git clone https://github.com/jonatl5/audio-generation-based-on-IndexTTS2.git
cd audio-generation-based-on-IndexTTS2
.\setup_env.ps1
```

If PowerShell blocks local scripts, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

DeepSpeed can be difficult on native Windows. The setup script first tries
`uv sync --all-extras`; if optional DeepSpeed dependencies fail, it falls back
to the IndexTTS2 webui extra. CPU fallback is supported by IndexTTS2 but can be
very slow.

## Run The Pipeline

Input-folder mode:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main --input_dir /path/to/input_folder
```

PowerShell:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main --input_dir "C:\path\to\input_folder"
```

By default each invocation writes to a new subfolder such as
`/path/to/input_folder/output/run_20260507_142500`. Use `--run_name` for a
stable human-readable run folder name, or `--flat_output_dir` to write directly
to `--output_dir` with the previous flat layout.

Explicit paths:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main \
  --video /path/video.mp4 \
  --script /path/script.ass \
  --refs_dir /path/refs \
  --output_dir /path/output \
  --language auto \
  --pause_for_emotion_edit true
```

Skip the edit pause:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main --input_dir /path/to/input_folder --no_pause
```

Resume from an edited emotion file:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main \
  --input_dir /path/to/input_folder \
  --resume_from_emotion_file /path/output/run_20260507_142500/work/emotions_to_edit.json
```

Dry run without IndexTTS2 generation:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main --input_dir examples/input_template --dry_run
```

After `setup_env.ps1` creates the official IndexTTS2 uv environment, use that
environment for full synthesis on Windows:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main --input_dir "C:\path\to\input_folder"
```

This matters because the official IndexTTS2 package requires Python `>=3.10`
and its own matched `torch`/`torchaudio` stack. Running with a separate system
Python can cause import errors such as `ModuleNotFoundError: No module named
'torchaudio'`.

## Rerun Evaluation For A Completed Run

Use this when TTS already finished and you only want to refresh
`content_consistency.csv`, `sim_by_line.csv`, `sim_by_speaker.csv`,
`mos_rating_sheet.csv`, and `evaluation_summary.json`.

Bash:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.evaluation \
  --output_dir /path/to/input_folder/output/source_test2_take1 \
  --language en
```

PowerShell:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.evaluation `
  --output_dir "C:\path\to\input_folder\output\source_test2_take1" `
  --language en
```

## Emotion File

After parsing, the pipeline writes:

```text
output/run_YYYYMMDD_HHMMSS/work/emotions_to_edit.json
```

Example item:

```json
{
  "id": 1,
  "speaker": "Jiang Han",
  "start": 0.06,
  "end": 3.31,
  "target_duration": 3.25,
  "text": "I awakened a supremely rare human-form Succubus",
  "emotion_method": "emo_text",
  "emo_text": "calm narration, slightly mysterious, steady pace",
  "emo_alpha": 0.55,
  "use_random": false,
  "emo_vector": null
}
```

Use `emotion_method: "emo_text"` with `emo_text`, or use `emotion_method: "emo_vector"` with an 8-value `emo_vector`. `emo_alpha` must be between `0.0` and `1.0`; this project defaults to `0.55` and avoids going above `0.8` in its local suggestions because strong emotion can reduce stability and cloning fidelity.

Default duration alignment uses one TTS generation per line, then pads or
time-stretches to the subtitle timing. The built-in alignment defaults are:

```yaml
alignment:
  stretch_if_long_under_ratio: 1.0
  regenerate_if_long_over_ratio: 999.0
  max_regen_attempts: 1
  max_stretch_speed_factor: 1.5
```

If a line would need a speed-up beyond `max_stretch_speed_factor`, the pipeline
tries punctuation-based segmentation before final time-stretching.

The pipeline also tightens excessive generated pauses before duration alignment.
It detects silences like this ffmpeg pattern:

```bash
ffmpeg -i line_0008.wav -af "silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-45dB" line_0008_tight.wav
```

By default it preserves a small number of the longest internal pauses when the
subtitle text has comma, clause, or sentence punctuation, so meaningful pauses
survive while accidental word gaps are shortened:

```yaml
silence_cleanup:
  enabled: true
  min_silence_duration: 0.18
  silence_threshold_db: -45.0
  keep_internal_silence: 0.06
  keep_comma_silence: 0.18
  keep_clause_silence: 0.22
  keep_sentence_silence: 0.30
  keep_edge_silence: 0.02
  preserve_punctuation_pauses: true
```

## Output Layout

```text
output/
  run_20260507_142500/
    final_dubbed.mp4
    audio/
      voice_cast_raw_concatenated.wav
      voice_cast_aligned.wav
    lines/
      line_0001.wav
      raw/
        line_0001.wav
      aligned/
        line_0001.wav
    work/
      script_structured.json
      emotions_to_edit.json
      speaker_map.json
      line_generation_metadata.jsonl
      warnings.json
    evaluation/
      content_consistency.csv
      sim_by_line.csv
      sim_by_speaker.csv
      mos_rating_sheet.csv
    logs/
      pipeline.log
    quality_report.md
    quality_report.json
```

## Evaluation

Content consistency:

- Uses `faster-whisper` if available.
- Falls back to `openai-whisper` if available.
- Computes WER for English with `jiwer`.
- Computes CER for Chinese with a local edit-distance implementation.
- If ASR is unavailable, it writes a clear skipped reason.

Timbre stability SIM:

- Uses SpeechBrain ECAPA-TDNN speaker embeddings when available.
- Computes cosine similarity between reference and generated line audio.
- Writes line-level and speaker-level CSV files.
- If the model cannot install or run, SIM is skipped with a clear reason.

MOS:

- Human MOS is not fabricated.
- The pipeline creates `output/evaluation/mos_rating_sheet.csv`.
- After filling it, run:

```bash
python -m stable_dubbing.evaluation --mos_sheet output/evaluation/mos_rating_sheet.csv
```

## Known Limitations

- IndexTTS2 emotion text control guides emotion but may not perfectly match video acting.
- Strong emotion can reduce voice cloning fidelity.
- Automatic MOS is not true MOS unless human raters score it.
- Subtitle timing may be too short for natural speech, causing segmentation or stretching.
- Multi-speaker subtitles may need manual correction.
- The official IndexTTS2 README notes precise duration-control research, but this public wrapper still performs practical post-generation alignment.
