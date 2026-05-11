from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.config import DubbingConfig
from stable_dubbing.tts_indextts2 import generate_with_pause_retries


def _write_tiny_wav(path: Path) -> None:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise unittest.SkipTest("pydub is not installed")
    exported = AudioSegment.silent(duration=300).export(path, format="wav")
    exported.close()


class TestGenerationRetry(unittest.TestCase):
    def test_retry_loop_stops_after_passing_attempt(self) -> None:
        config = DubbingConfig()
        calls: list[int] = []
        analyses = [
            {"duration_sec": 1.0, "has_abnormal_pause": True, "abnormal_pause_count": 1, "total_abnormal_pause_sec": 0.4, "pauses": []},
            {"duration_sec": 1.0, "has_abnormal_pause": False, "abnormal_pause_count": 0, "total_abnormal_pause_sec": 0.0, "pauses": []},
        ]

        def infer(path: Path, attempt: int) -> None:
            calls.append(attempt)
            _write_tiny_wav(path)

        def analyze(*_args, **_kwargs):
            return analyses[len(calls) - 1]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_with_pause_retries(1, "hello world", 1.0, temp_dir, config, infer, analyze_func=analyze)

        self.assertEqual(calls, [1, 2])
        self.assertEqual(result["status"], "regenerated_then_passed")
        self.assertEqual(result["accepted_attempt"], 2)

    def test_retry_loop_flags_after_four_failed_attempts(self) -> None:
        config = DubbingConfig()
        config.pause_detection.max_retries = 4
        config.pause_repair.enabled = False
        calls: list[int] = []

        def infer(path: Path, attempt: int) -> None:
            calls.append(attempt)
            _write_tiny_wav(path)

        def analyze(*_args, **_kwargs):
            return {
                "duration_sec": 1.0,
                "has_abnormal_pause": True,
                "abnormal_pause_count": 1,
                "total_abnormal_pause_sec": 0.4,
                "pauses": [{"allowed": False}],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_with_pause_retries(1, "hello world", 1.0, temp_dir, config, infer, analyze_func=analyze)

        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(result["status"], "flagged_after_4_attempts")
        self.assertEqual(result["attempt_count"], 4)

    def test_failed_retries_can_auto_cut_wrong_pause(self) -> None:
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
        except ImportError:
            self.skipTest("pydub is not installed")

        config = DubbingConfig()
        config.pause_detection.max_retries = 2
        config.pause_repair.target_keep_ms = 140
        config.pause_repair.fade_ms = 0
        config.pause_repair.crossfade_ms = 0
        calls: list[int] = []

        def infer(path: Path, attempt: int) -> None:
            calls.append(attempt)
            tone = Sine(440).to_audio_segment(duration=300).apply_gain(-12)
            exported = (tone + AudioSegment.silent(duration=500) + tone).export(path, format="wav")
            exported.close()

        def analyze(audio_path: str, *_args, **_kwargs):
            if str(audio_path).endswith("_repaired.wav"):
                return {
                    "duration_sec": 0.74,
                    "min_pause_ms": 350,
                    "has_abnormal_pause": False,
                    "abnormal_pause_count": 0,
                    "total_abnormal_pause_sec": 0.0,
                    "pauses": [],
                }
            return {
                "duration_sec": 1.1,
                "min_pause_ms": 350,
                "silence_thresh": -35.0,
                "has_abnormal_pause": True,
                "abnormal_pause_count": 1,
                "total_abnormal_pause_sec": 0.5,
                "pauses": [
                    {
                        "start_sec": 0.3,
                        "end_sec": 0.8,
                        "duration_ms": 500,
                        "allowed": False,
                        "reason": "long internal pause without nearby punctuation",
                        "word_before_pause": "hello",
                        "word_after_pause": "world",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_with_pause_retries(1, "hello world", 1.0, temp_dir, config, infer, analyze_func=analyze)

        self.assertEqual(calls, [1, 2])
        self.assertEqual(result["status"], "wrong_pause_auto_cut")
        self.assertEqual(result["pause_repair"]["status"], "repaired_passed")
        self.assertTrue(str(result["selected_audio"]).endswith("_repaired.wav"))


if __name__ == "__main__":
    unittest.main()
