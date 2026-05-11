# Assessment Worklog: IndexTTS2 Dubbing Pipeline

This is my worklog for the audio generation assessment.

I built a subtitle-based dubbing pipeline around IndexTTS2. The goal was to take a muted video, an `.ass` subtitle file, and a folder of speaker reference audio, then generate a dubbed video with one synthetic voice per character.

The current repo contains the pipeline code, setup scripts, tests, and a finished sample video:

```text
finished video v1.mp4
```

That video is stored with Git LFS because it is too large for normal GitHub file storage.

## What I Built

I created a Python project called `stable_emotion_dubbing_indextts2`.

The pipeline does these steps:

1. Parse `.ass` or `.txt` subtitle files.
2. Split safe multi-speaker subtitle lines into separate speaker lines.
3. Build a speaker map from subtitle speaker names to reference audio files.
4. Generate an editable `emotions_to_edit.json` file.
5. Let the user edit `emo_text`, `emo_alpha`, or paste an 8-value `emo_vector`.
6. Build generation units from subtitle lines.
7. Run IndexTTS2 once for each generation unit.
8. Tighten incorrect silence inside each generated unit.
9. Align each unit to its subtitle timing window.
10. Assemble the final audio track from units.
11. Mux the generated voice track back into the video.
12. Write evaluation files and a quality report.

The main package is:

```text
stable_dubbing/
```

It includes modules for subtitle parsing, emotion preparation, speaker mapping, TTS generation, silence cleanup, duration alignment, audio assembly, video muxing, evaluation, reports, and logging.



## Emotion Control

IndexTTS2 supports emotion in two ways:

```text
emo_text
emo_vector
```

The emotion vector order is:

```text
[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
```

At first, the pipeline wrote simple emotion text like:

```text
calm narration, slightly mysterious, steady pace
```

But I saw that IndexTTS2 often interpreted that as mostly:

```text
calm: 1.0
```

So "mysterious" didn't really show up as a useful emotion. For better control, I prepared manual `emo_vector` values line by line.

I kept the original editable JSON flow. The model still creates `emotions_to_edit.json` first, then the user can paste better vectors into that file before generation continues.

## Manual Sentence Units

The biggest pacing improvement came from changing what the pipeline sends to TTS.

The older version generated every subtitle row by itself. That was simple, but it caused a lot of bad pauses. Many subtitle rows are only wrapped pieces of the same sentence, so the model saw fragments instead of a natural sentence. It would often pause between words that should have stayed connected.

I changed the pipeline to work with generation units.

A generation unit can be:

```text
one subtitle line
several contiguous subtitle lines that belong to the same sentence
```

For grouped lines, the pipeline now generates the combined text once and keeps that audio as one file. It does not split the generated audio back into separate per-line files. The unit is placed on the final timeline from the first line's start time to the last line's end time.

Manual groups live in a JSON file:

```json
{
  "groups": [
    {
      "id": "u_0004_0006",
      "lines": [4, 5, 6]
    },
    {
      "id": "u_0012_0013",
      "lines": [12, 13],
      "text": "Optional manually edited combined text."
    }
  ]
}
```

The command uses it like this:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main `
  --input_dir "C:\path\to\input_folder" `
  --manual-groups "C:\path\to\manual_groups.json"
```

Manual groups override automatic grouping. Lines that are not listed become single-line units.

The grouping file is validated before generation. Groups cannot overlap, must point to real lines, must be contiguous by default, and cannot cross speakers unless `allow_cross_speaker` is set.

When the pipeline joins wrapped subtitle text, it joins with spaces unless punctuation already gives the model a reason to pause. It does not add commas by default, because commas made IndexTTS2 pause in places where I wanted continuous speech.

## Duration And Speed Handling

One big issue was speed.

IndexTTS2 often generated audio that was longer than the subtitle time slot. The first fix was ffmpeg time stretching. It worked, but it was a blunt fix. If the audio needed a large speed-up, the voice could sound compressed, and the bad pauses were still baked into the waveform.

So I looked at how other IndexTTS2 tools controlled speed. The useful clue came from the ComfyUI IndexTTS2 wrapper. It did not just stretch the finished waveform. It changed the target mel length before diffusion and vocoder generation.

The model starts with semantic speech codes from GPT. Then IndexTTS2 estimates the target mel length with a rule close to this:

```python
target_lengths = code_lens * 1.72
```

I patched the local IndexTTS2 inference path to accept `duration_scale` and apply it there:

```python
target_lengths = code_lens * 1.72 * duration_scale
```

Smaller `duration_scale` asks the model to generate shorter, faster speech. Larger `duration_scale` asks it to generate longer, slower speech.

This is better than only using ffmpeg `atempo` because the model speaks closer to the intended pace from the start. ffmpeg is still used, but only for the last small correction.

The pipeline now does this for each unit:

1. Generate once at `duration_scale = 1.0`.
2. Measure the raw duration.
3. Compare it to the unit span duration.
4. If the gap is large, regenerate the whole unit with a clamped `duration_scale`.
5. Use light ffmpeg alignment only at the end.

The related config is:

```yaml
model_duration_control:
  enabled: true
  first_pass_scale: 1.0
  min_duration_scale: 0.70
  max_duration_scale: 1.30
  regenerate_if_ratio_outside: 0.08
  max_model_duration_attempts: 2
  final_ffmpeg_max_speed_factor: 1.15
```

The run metadata records the scale that was used, whether it was clamped, and whether the final correction used padding or ffmpeg.

## Silence Cleanup

Another problem was incorrect pauses between words.

Some generated lines had pauses that were not caused by commas, periods, or natural sentence breaks. Those pauses made the line too long and sounded awkward.

I added a silence cleanup step after each unit is generated and before any final duration correction happens.

The idea is similar to this ffmpeg command:

```bash
ffmpeg -i "line_0008.wav" -af "silenceremove=stop_periods=-1:stop_duration=0.18:stop_threshold=-45dB" "line_0008_tight.wav"
```

But I made it a little more careful.

The pipeline detects silence longer than about `0.18s` below `-45dB`. It shortens accidental word gaps to about `0.06s`. It also tries to preserve pauses that match punctuation:

```text
comma: about 0.18s
clause break: about 0.22s
period/question/exclamation: about 0.30s
```

This now runs at the generation-unit level. If a grouped unit is one sentence, the detector looks at the whole generated sentence, not a chopped-up line.

This is useful, but it is not perfect. Sometimes it can create a jump cut in the voice because we are cutting silence after generation. The model-side duration control is a cleaner fix when the problem is overall speaking pace.

## Evaluation Work

I added evaluation outputs:

```text
content_consistency.csv
sim_by_line.csv
sim_by_speaker.csv
mos_rating_sheet.csv
evaluation_summary.json
```

Content consistency uses Whisper ASR and computes WER or CER.

Speaker similarity uses SpeechBrain speaker embeddings.

MOS is not automatically faked. The pipeline writes a sheet for human rating.

I also added a command to rerun only the evaluation for a completed output folder. This helps when generation already took a long time and I only want to refresh the reports.

## Current Pitfalls

The biggest pitfall is still generation time.

IndexTTS2 can take a long time to finish one full version of the video. On my machine, some short lines still took much longer than expected. The bottleneck looked like the mel generation stage, especially `s2mel_time`.

Another pitfall is that the speed control patch touches the local IndexTTS2 inference code. The official public API still does not expose a clean speed parameter, so the project has to pass `duration_scale` into the model path that computes mel length.

Reference audio also matters a lot. IndexTTS2 copies more than voice identity. It can copy speaking speed, rhythm, and pause style from the reference audio. When I used slower reference audio, the generated lines also became slower and had more pauses.

So better reference audio is important. A clean 10-30 second normal-paced sample is better than a slow, emotional, noisy, or silence-heavy sample.

## Why I Stopped Here

Due to the time limit, I stopped after getting a complete working pipeline, a finished v1 video, manual emotion vector support, one-generation alignment, silence cleanup, and evaluation.

There is still room to improve quality.

The current result is usable as a first version, but it is not the final version I would ship for production dubbing.

## Future Experiments

The next thing I would try is better reference audio.

I would record or collect clean speaker references with:

```text
10-30 seconds per speaker
one speaker only
normal speaking speed
low background noise
minimal long silence
consistent microphone quality
```

I would also experiment with `emo_alpha`.

Lowering `emo_alpha` may reduce some incorrect pauses because the model may act less strongly and speak more steadily. But lowering it too much can make the line sound flat, so it needs testing by speaker and by scene.

A useful future experiment would be:

1. Pick a small set of difficult lines.
2. Generate each line at several `emo_alpha` values.
3. Measure duration, number of silence gaps, WER, and human preference.
4. Pick a speaker-specific or scene-specific `emo_alpha` range.

I would also make the IndexTTS2 duration patch cleaner if the upstream project later exposes an official speed or duration parameter.

## How To Use This Pipeline

Clone the repo:

```bash
git clone https://github.com/jonatl5/audio-generation-based-on-IndexTTS2.git
cd audio-generation-based-on-IndexTTS2
```

If you want the finished sample video too, make sure Git LFS is installed before cloning, or run:

```bash
git lfs install
git lfs pull
```

## Environment Setup

You need these system tools:

```text
git
git-lfs
ffmpeg
uv
python
```

A CUDA GPU is strongly recommended. CPU can work, but it will be slow.

### Bash Setup

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg python3 python3-pip curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs install
bash setup_env.sh
```

macOS with Homebrew:

```bash
brew install git git-lfs ffmpeg uv python
git lfs install
bash setup_env.sh
```

### PowerShell Setup

Windows with winget:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id Gyan.FFmpeg -e
winget install --id Astral.UV -e
winget install --id Python.Python.3.10 -e
git lfs install
.\setup_env.ps1
```

If PowerShell blocks the setup script, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The setup script clones the official IndexTTS2 repo into:

```text
third_party/index-tts
```

It also installs the pipeline into the IndexTTS2 virtual environment and downloads the checkpoints.

## Input File Format

Use this folder layout:

```text
input_folder/
  video.mp4
  script.ass
  refs/
    Speaker 1.wav
    Speaker 2.wav
```

The subtitle speaker names should match the reference filenames.

For example, this subtitle speaker:

```text
Jiang Han
```

should have one of these reference files:

```text
refs/Jiang Han.wav
refs/Jiang Han.mp3
refs/Jiang_Han.wav
```

Supported reference formats:

```text
.wav
.mp3
.flac
.m4a
```

Again, `.wav` is safest.

## Run The Pipeline

Bash:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.main \
  --input_dir /path/to/input_folder
```

PowerShell:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main `
  --input_dir "C:\path\to\input_folder"
```

The pipeline creates a new run folder under:

```text
input_folder/output/
```

Example:

```text
input_folder/output/run_20260507_142500/
```

You can set a run name:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main `
  --input_dir "C:\path\to\input_folder" `
  --run_name source_test_take1
```

## Emotion Editing Flow

By default, the pipeline writes:

```text
output/<run_name>/work/emotions_to_edit.json
```

Then it pauses.

You can edit:

```text
emo_text
emo_alpha
emo_vector
```

Then save the file and press Enter in the terminal.

To resume from an existing edited emotion file:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.main `
  --input_dir "C:\path\to\input_folder" `
  --run_name source_test_take2 `
  --resume_from_emotion_file "C:\path\to\emotions_to_edit.json"
```

## Rerun Evaluation Only

Bash:

```bash
third_party/index-tts/.venv/bin/python -m stable_dubbing.evaluation \
  --output_dir /path/to/input_folder/output/source_test_take1 \
  --language en
```

PowerShell:

```powershell
.\third_party\index-tts\.venv\Scripts\python.exe -m stable_dubbing.evaluation `
  --output_dir "C:\path\to\input_folder\output\source_test_take1" `
  --language en
```

## Final Notes

I got the system working end to end, but the quality is still tied heavily to IndexTTS2 behavior and the reference audio.

The current silence cleanup helps, but it is still a post-processing patch.

The better long-term fix is either stronger model-side pacing controls or better reference audio plus a tuned emotion strategy.
