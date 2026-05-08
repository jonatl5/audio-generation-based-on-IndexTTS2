from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.config import SilenceCleanupConfig
from stable_dubbing.silence_cleanup import _internal_pause_keeps_ms, tighten_silences
from stable_dubbing.utils import audio_duration


class TestSilenceCleanup(unittest.TestCase):
    def test_internal_pause_keeps_follow_punctuation(self) -> None:
        config = SilenceCleanupConfig()

        keeps = _internal_pause_keeps_ms("Hah. Interesting, kid", config)

        self.assertEqual(keeps, [300, 180])

    def test_tighten_silences_preserves_one_punctuation_pause(self) -> None:
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
        except ImportError:
            self.skipTest("pydub is not installed")

        config = SilenceCleanupConfig()
        tone = Sine(440).to_audio_segment(duration=120).apply_gain(-12)
        audio = (
            tone
            + AudioSegment.silent(duration=260)
            + tone
            + AudioSegment.silent(duration=260)
            + tone
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            target = Path(temp_dir) / "target.wav"
            audio.export(source, format="wav")

            result = tighten_silences(source, target, "First, second third", config)

            self.assertTrue(result.applied)
            self.assertLess(audio_duration(target), audio_duration(source))
            self.assertEqual(result.detected_silences, 2)
            self.assertEqual(result.protected_pauses, 1)


if __name__ == "__main__":
    unittest.main()
