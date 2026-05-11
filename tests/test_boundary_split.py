from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.boundary_split import build_boundary_cuts, split_combined_audio_to_lines
from stable_dubbing.utils import audio_duration


def _line(line_id: int, text: str, duration: float = 1.0) -> dict:
    return {
        "id": line_id,
        "speaker": "A",
        "start": float(line_id - 1),
        "end": float(line_id - 1) + duration,
        "target_duration": duration,
        "text": text,
    }


def _export_audio(path: Path, with_silence: bool = True) -> None:
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        raise unittest.SkipTest("pydub is not installed")
    tone = Sine(440).to_audio_segment(duration=300).apply_gain(-12)
    if with_silence:
        audio = tone + AudioSegment.silent(duration=500) + tone + AudioSegment.silent(duration=500) + tone
    else:
        audio = tone + tone + tone
    exported = audio.export(path, format="wav")
    exported.close()


class TestBoundarySplit(unittest.TestCase):
    def test_selects_silence_midpoint_between_boundary_words(self) -> None:
        lines = [_line(1, "Hello world"), _line(2, "this continues"), _line(3, "again")]
        words = [
            {"word": "Hello", "start": 0.02, "end": 0.10},
            {"word": "world", "start": 0.20, "end": 0.30},
            {"word": "this", "start": 0.82, "end": 0.92},
            {"word": "continues", "start": 1.00, "end": 1.10},
            {"word": "again", "start": 1.62, "end": 1.72},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "combined.wav"
            _export_audio(path)
            cuts, _silence, fallback, flags = build_boundary_cuts(
                path,
                lines,
                word_timestamps=words,
                min_pause_ms=200,
                silence_db_offset=25,
            )

        self.assertFalse(fallback)
        self.assertEqual(flags, [])
        self.assertEqual([cut.method for cut in cuts], ["silence_midpoint", "silence_midpoint"])
        self.assertAlmostEqual(cuts[0].cut_sec, 0.55, delta=0.08)
        self.assertAlmostEqual(cuts[1].cut_sec, 1.35, delta=0.08)

    def test_uses_word_gap_midpoint_when_no_silence_matches(self) -> None:
        lines = [_line(1, "Hello world"), _line(2, "this continues")]
        words = [
            {"word": "Hello", "start": 0.02, "end": 0.10},
            {"word": "world", "start": 0.20, "end": 0.30},
            {"word": "this", "start": 0.50, "end": 0.60},
            {"word": "continues", "start": 0.65, "end": 0.75},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "combined.wav"
            _export_audio(path, with_silence=False)
            cuts, _silence, fallback, flags = build_boundary_cuts(
                path,
                lines,
                word_timestamps=words,
                min_pause_ms=200,
                silence_db_offset=25,
            )

        self.assertTrue(fallback)
        self.assertIn("word_gap_midpoint_fallback", flags)
        self.assertEqual(cuts[0].method, "word_gap_midpoint")
        self.assertAlmostEqual(cuts[0].cut_sec, 0.4, delta=0.01)

    def test_uses_proportional_fallback_without_words(self) -> None:
        lines = [_line(1, "Hello world", duration=1.0), _line(2, "this continues", duration=3.0)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "combined.wav"
            _export_audio(path, with_silence=False)
            cuts, _silence, fallback, flags = build_boundary_cuts(
                path,
                lines,
                word_timestamps=[],
                min_pause_ms=200,
                silence_db_offset=25,
            )

        self.assertTrue(fallback)
        self.assertIn("proportional_duration_fallback", flags)
        self.assertEqual(cuts[0].method, "proportional_duration")

    def test_exports_line_pieces(self) -> None:
        lines = [_line(1, "Hello world"), _line(2, "this continues")]
        words = [
            {"word": "Hello", "start": 0.02, "end": 0.10},
            {"word": "world", "start": 0.20, "end": 0.30},
            {"word": "this", "start": 0.82, "end": 0.92},
            {"word": "continues", "start": 1.00, "end": 1.10},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            path = temp / "combined.wav"
            _export_audio(path)
            result = split_combined_audio_to_lines(
                path,
                lines,
                temp / "raw",
                word_timestamps=words,
                min_pause_ms=200,
                silence_db_offset=25,
            )

            self.assertTrue(Path(result.piece_paths[1]).exists())
            self.assertTrue(Path(result.piece_paths[2]).exists())
            self.assertGreater(audio_duration(result.piece_paths[1]), 0)


if __name__ == "__main__":
    unittest.main()
