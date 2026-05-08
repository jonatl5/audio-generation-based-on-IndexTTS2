from __future__ import annotations

import unittest
from pathlib import Path

from stable_dubbing.config import DubbingConfig
from stable_dubbing.subtitle_parser import (
    ass_timestamp_to_seconds,
    clean_ass_text,
    parse_speaker_name,
    parse_subtitle,
)


def sample_ass_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    attached = project_root.parent / "data_source" / "source_test1.ass"
    if attached.exists():
        return attached
    return project_root / "examples" / "input_template" / "script.ass"


class TestAssParser(unittest.TestCase):
    def test_ass_timestamp_conversion(self) -> None:
        self.assertAlmostEqual(ass_timestamp_to_seconds("0:01:36.58"), 96.58)

    def test_ass_tag_cleaning(self) -> None:
        self.assertEqual(
            clean_ass_text(r"{\pos(170,488)}Hello\Nthere\\nagain"),
            "Hello there again",
        )

    def test_speaker_gender_parsing(self) -> None:
        self.assertEqual(parse_speaker_name("Jiang Han(男)"), ("Jiang Han", "男"))
        self.assertEqual(parse_speaker_name("Succubus(女)"), ("Succubus", "女"))
        self.assertEqual(parse_speaker_name(""), ("Unknown", None))

    def test_style_filtering_and_multi_speaker_split(self) -> None:
        result = parse_subtitle(sample_ass_path(), DubbingConfig())
        lines = [line.to_dict() for line in result.lines]
        self.assertGreater(len(lines), 0)
        self.assertTrue(all(line["style"] == "Default" for line in lines))
        self.assertNotIn("Episode 1", {line["text"] for line in lines})
        self.assertNotIn("Jiang Han", [line["text"] for line in lines if line["style"] != "Default"])

        multi_lines = [
            line
            for line in lines
            if line["source"]["line_index"] and "split from multi-speaker subtitle" in line["warnings"]
        ]
        self.assertGreaterEqual(len(multi_lines), 2)
        self.assertEqual(multi_lines[0]["speaker"], "Manager")
        self.assertEqual(multi_lines[1]["speaker"], "Beastmaster 1")
        self.assertLess(multi_lines[0]["end"], multi_lines[1]["end"])


if __name__ == "__main__":
    unittest.main()

