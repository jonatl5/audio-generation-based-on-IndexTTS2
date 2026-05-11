from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.recombine import recombine_lines
from stable_dubbing.utils import audio_duration, write_json


def _write_silence(path: Path, duration_ms: int) -> None:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise unittest.SkipTest("pydub is not installed")
    exported = AudioSegment.silent(duration=duration_ms).export(path, format="wav")
    exported.close()


class TestRecombine(unittest.TestCase):
    def test_recombine_uses_updated_line_audio(self) -> None:
        plan = [
            {
                "id": 1,
                "speaker": "A",
                "start": 0.0,
                "end": 0.1,
                "target_duration": 0.1,
                "text": "One",
                "emotion_method": "emo_text",
                "emo_text": "calm",
                "emo_alpha": 0.55,
                "use_random": False,
                "emo_vector": None,
            },
            {
                "id": 2,
                "speaker": "B",
                "start": 0.1,
                "end": 0.2,
                "target_duration": 0.1,
                "text": "Two",
                "emotion_method": "emo_text",
                "emo_text": "calm",
                "emo_alpha": 0.55,
                "use_random": False,
                "emo_vector": None,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            lines_dir = temp / "lines"
            lines_dir.mkdir()
            _write_silence(lines_dir / "line_0001.wav", 100)
            _write_silence(lines_dir / "line_0002.wav", 100)
            first = temp / "first.wav"
            recombine_lines(plan, lines_dir, first, gap_ms=50, use_timestamps=False)

            _write_silence(lines_dir / "line_0002.wav", 300)
            second = temp / "second.wav"
            recombine_lines(plan, lines_dir, second, gap_ms=50, use_timestamps=False)

            self.assertGreater(audio_duration(second), audio_duration(first))

    def test_recombine_prefers_generation_units_manifest(self) -> None:
        plan = [
            {
                "id": 1,
                "speaker": "A",
                "start": 0.0,
                "end": 0.1,
                "target_duration": 0.1,
                "text": "One",
                "emotion_method": "emo_text",
                "emo_text": "calm",
                "emo_alpha": 0.55,
                "use_random": False,
                "emo_vector": None,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            lines_dir = temp / "lines"
            work_dir = temp / "work"
            lines_dir.mkdir()
            work_dir.mkdir()
            unit_audio = lines_dir / "u_0001_0002.wav"
            _write_silence(unit_audio, 250)
            write_json(
                work_dir / "generation_units.json",
                [
                    {
                        "unit_id": "u_0001_0002",
                        "source_line_indices": [1, 2],
                        "start": 0.0,
                        "end": 0.25,
                        "span_target_duration": 0.25,
                        "aligned_audio_path": str(unit_audio),
                    }
                ],
            )

            output = temp / "combined.wav"
            report = recombine_lines(plan, lines_dir, output, use_timestamps=True)

            self.assertEqual(report["combined_unit_count"], 1)
            self.assertEqual(report["units"][0]["unit_id"], "u_0001_0002")
            self.assertAlmostEqual(audio_duration(output), 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
