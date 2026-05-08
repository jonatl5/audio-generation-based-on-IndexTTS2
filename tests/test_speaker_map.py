from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.speaker_map import build_speaker_map, normalize_speaker_name


class TestSpeakerMap(unittest.TestCase):
    def test_normalize_speaker_name(self) -> None:
        self.assertEqual(normalize_speaker_name("Jiang_Han(男).wav"), "jiang han")

    def test_speaker_map_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "refs"
            refs.mkdir()
            (refs / "Jiang_Han(男).wav").write_bytes(b"placeholder")
            (refs / "Succubus.wav").write_bytes(b"placeholder")
            lines = [
                {"speaker": "Jiang Han"},
                {"speaker": "Succubus"},
            ]
            mapping, warnings = build_speaker_map(
                lines, refs, root / "speaker_map.json", strict=True, interactive=False
            )
            self.assertEqual(warnings, [])
            self.assertTrue(mapping["Jiang Han"].endswith("Jiang_Han(男).wav"))
            self.assertTrue(mapping["Succubus"].endswith("Succubus.wav"))


if __name__ == "__main__":
    unittest.main()

