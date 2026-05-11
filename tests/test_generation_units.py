from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.generation_units import build_generation_units, join_unit_text
from stable_dubbing.utils import write_json


def _line(line_id: int, speaker: str, text: str, start: float, end: float) -> dict:
    return {
        "id": line_id,
        "speaker": speaker,
        "start": start,
        "end": end,
        "target_duration": end - start,
        "text": text,
        "emotion_method": "emo_vector",
        "emo_text": "calm",
        "emo_alpha": 0.5,
        "use_random": False,
        "emo_vector": [0, 0, 0, 0, 0, 0, 0, 1],
    }


class TestGenerationUnits(unittest.TestCase):
    def test_manual_group_overrides_auto_and_keeps_unlisted_lines_single(self) -> None:
        lines = [
            _line(4, "A", "With her own consciousness, she knew all too well", 12.34, 14.0),
            _line(5, "A", "that a poor student like me", 14.0, 15.0),
            _line(6, "A", "could never afford it", 15.0, 17.88),
            _line(7, "A", "another line", 18.1, 20.0),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual_groups.json"
            write_json(path, {"groups": [{"id": "u_0004_0006", "lines": [6, 4, 5]}]})
            units = build_generation_units(lines, lines, manual_groups_path=path, enable_auto_groups=True)

        self.assertEqual([unit.unit_id for unit in units], ["u_0004_0006", "u_0007"])
        self.assertEqual(units[0].source_line_indices, [4, 5, 6])
        self.assertEqual(units[0].start, 12.34)
        self.assertEqual(units[0].end, 17.88)
        self.assertAlmostEqual(units[0].span_target_duration, 5.54)
        self.assertAlmostEqual(units[0].summed_line_target_duration, 5.54)
        self.assertTrue(units[0].is_group)
        self.assertEqual(units[0].source_grouping_method, "manual")
        self.assertFalse(units[1].is_group)

    def test_manual_group_text_override(self) -> None:
        lines = [_line(12, "A", "first", 0.0, 1.0), _line(13, "A", "second", 1.0, 2.0)]
        units = build_generation_units(
            lines,
            lines,
            manual_groups_data={"groups": [{"lines": [12, 13], "text": "Edited combined text."}]},
        )

        self.assertEqual(units[0].text, "Edited combined text.")

    def test_manual_group_rejects_overlap_and_cross_speaker(self) -> None:
        lines = [_line(1, "A", "hello", 0.0, 1.0), _line(2, "B", "world", 1.0, 2.0)]

        with self.assertRaisesRegex(ValueError, "crosses speakers"):
            build_generation_units(lines, lines, manual_groups_data={"groups": [{"lines": [1, 2]}]})
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build_generation_units(
                lines,
                lines,
                manual_groups_data={
                    "groups": [
                        {"lines": [1]},
                        {"lines": [1, 2], "allow_cross_speaker": True},
                    ]
                },
            )

    def test_emotion_vector_and_alpha_are_duration_weighted(self) -> None:
        first = _line(1, "A", "First", 0.0, 1.0)
        second = _line(2, "A", "second", 1.0, 3.0)
        first["emo_alpha"] = 0.2
        second["emo_alpha"] = 0.8
        first["emo_vector"] = [1, 0, 0, 0, 0, 0, 0, 0]
        second["emo_vector"] = [0, 0, 0, 0, 0, 0, 0, 1]

        unit = build_generation_units(
            [first, second],
            [first, second],
            manual_groups_data={"groups": [{"lines": [1, 2]}]},
        )[0]

        self.assertEqual(unit.emotion["emotion_source"], "blended")
        self.assertAlmostEqual(unit.emotion["emo_alpha"], 0.6)
        self.assertAlmostEqual(unit.emotion["emo_vector"][0], 1 / 3, places=5)
        self.assertAlmostEqual(unit.emotion["emo_vector"][7], 2 / 3, places=5)

    def test_join_unit_text_does_not_insert_commas(self) -> None:
        self.assertEqual(
            join_unit_text([{"text": "But her first words"}, {"text": "were to refuse forming a pact"}]),
            "But her first words were to refuse forming a pact",
        )


if __name__ == "__main__":
    unittest.main()
