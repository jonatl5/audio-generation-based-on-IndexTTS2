from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stable_dubbing.config import DubbingConfig
from stable_dubbing.tts_indextts2 import synthesize_lines
from stable_dubbing.utils import audio_duration, read_json


def _line(line_id: int, text: str, start: float, end: float) -> dict:
    return {
        "id": line_id,
        "speaker": "A",
        "start": start,
        "end": end,
        "target_duration": end - start,
        "text": text,
        "emotion_method": "emo_text",
        "emo_text": "calm",
        "emo_alpha": 0.55,
        "use_random": False,
        "emo_vector": None,
    }


class FakeWrapper:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def initialize(self) -> None:
        pass

    def infer_line(self, _speaker_ref, _text, output_path, _emotion, **kwargs) -> None:
        try:
            from pydub.generators import Sine
        except ImportError:
            raise unittest.SkipTest("pydub is not installed")
        duration_scale = float(kwargs.get("duration_scale") or 1.0)
        exported = Sine(440).to_audio_segment(duration=int(round(300 * duration_scale))).export(output_path, format="wav")
        exported.close()


class TestUnitSynthesis(unittest.TestCase):
    def test_manual_group_writes_one_unit_file_and_manifest(self) -> None:
        lines = [
            _line(4, "With her own consciousness", 0.0, 0.2),
            _line(5, "that a poor student like me", 0.2, 0.35),
            _line(6, "could never afford it", 0.35, 0.5),
        ]
        config = DubbingConfig()
        config.pause_detection.enabled = False
        config.silence_cleanup.enabled = False
        config.evaluation.run_asr = False
        config.evaluation.run_sim = False

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            with patch("stable_dubbing.tts_indextts2.IndexTTS2Wrapper", FakeWrapper):
                rows = synthesize_lines(
                    lines,
                    lines,
                    {"A": "speaker.wav"},
                    output,
                    config,
                    manual_groups_data={"groups": [{"id": "u_0004_0006", "lines": [4, 5, 6]}]},
                )

            self.assertEqual(len(rows), 1)
            self.assertTrue((output / "lines" / "u_0004_0006.wav").exists())
            self.assertTrue((output / "lines" / "raw" / "u_0004_0006_raw.wav").exists())
            self.assertFalse((output / "lines" / "line_0004.wav").exists())
            self.assertFalse((output / "lines" / "line_0005.wav").exists())
            self.assertFalse((output / "lines" / "line_0006.wav").exists())
            manifest = read_json(output / "work" / "generation_units.json")
            self.assertEqual(manifest[0]["source_line_indices"], [4, 5, 6])
            self.assertAlmostEqual(manifest[0]["start"], 0.0)
            self.assertAlmostEqual(manifest[0]["end"], 0.5)
            self.assertAlmostEqual(audio_duration(output / "lines" / "u_0004_0006.wav"), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
