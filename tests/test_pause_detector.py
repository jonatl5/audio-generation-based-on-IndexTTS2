from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.pause_detector import analyze_abnormal_pauses, classify_pause_allowed


def _build_audio(path: Path, pause_ms: int = 500) -> None:
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        raise unittest.SkipTest("pydub is not installed")

    tone = Sine(440).to_audio_segment(duration=300).apply_gain(-12)
    audio = tone + AudioSegment.silent(duration=pause_ms) + tone
    exported = audio.export(path, format="wav")
    exported.close()


class TestPauseDetector(unittest.TestCase):
    def test_detects_internal_silence_without_punctuation_as_abnormal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pause.wav"
            _build_audio(path)

            result = analyze_abnormal_pauses(
                str(path),
                "Succubus made a mistake",
                min_pause_ms=350,
                silence_db_offset=25,
                use_asr_alignment=False,
            )

        self.assertTrue(result["has_abnormal_pause"])
        self.assertEqual(result["abnormal_pause_count"], 1)
        self.assertEqual(result["pauses"][0]["after_word"], "Succubus")
        self.assertEqual(result["pauses"][0]["before_word"], "made")

    def test_punctuation_boundary_is_allowed(self) -> None:
        result = classify_pause_allowed(
            "Succubus, made a mistake",
            pause_start_sec=0.3,
            pause_end_sec=0.8,
            duration_sec=1.1,
        )

        self.assertTrue(result["allowed"])

    def test_no_punctuation_boundary_is_abnormal(self) -> None:
        result = classify_pause_allowed(
            "Succubus made a mistake",
            pause_start_sec=0.3,
            pause_end_sec=0.8,
            duration_sec=1.1,
        )

        self.assertFalse(result["allowed"])


if __name__ == "__main__":
    unittest.main()
