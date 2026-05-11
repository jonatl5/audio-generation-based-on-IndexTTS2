from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.emotion_plan import (
    apply_line_override,
    get_line_by_id,
    load_emotion_plan,
)
from stable_dubbing.generation_report import replace_metadata_row
from stable_dubbing.utils import write_json


class TestEmotionPlan(unittest.TestCase):
    def test_load_emotion_plan_normalizes_target_duration(self) -> None:
        items = [
            {
                "id": 12,
                "speaker": "Jiang Han",
                "start": 1.0,
                "end": 4.25,
                "text": "But this Succubus made the wrong choice.",
                "emotion_method": "emo_text",
                "emo_text": "cold, controlled",
                "emo_alpha": 0.55,
                "use_random": False,
                "emo_vector": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_json(Path(temp_dir) / "emotion.json", items)
            loaded = load_emotion_plan(path)

        self.assertEqual(loaded[0]["id"], 12)
        self.assertAlmostEqual(loaded[0]["target_duration"], 3.25)

    def test_apply_line_override_only_changes_requested_line(self) -> None:
        items = [
            {
                "id": 1,
                "speaker": "A",
                "start": 0.0,
                "end": 1.0,
                "target_duration": 1.0,
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
                "start": 1.0,
                "end": 2.0,
                "target_duration": 1.0,
                "text": "Two",
                "emotion_method": "emo_text",
                "emo_text": "calm",
                "emo_alpha": 0.55,
                "use_random": False,
                "emo_vector": None,
            },
        ]

        updated, line, notes = apply_line_override(items, 2, emo_alpha=0.35)

        self.assertEqual(get_line_by_id(updated, 1)["emo_alpha"], 0.55)
        self.assertEqual(line["emo_alpha"], 0.35)
        self.assertEqual(notes["emo_alpha"], 0.35)
        self.assertEqual(items[1]["emo_alpha"], 0.55)

    def test_regenerate_metadata_replacement_only_changes_one_line(self) -> None:
        rows = [
            {"line_id": 1, "generation_status": "passed", "final_score": 0.0},
            {"line_id": 2, "generation_status": "passed", "final_score": 0.1},
        ]
        updated = replace_metadata_row(rows, {"line_id": 2, "generation_status": "regenerated_then_passed"})

        self.assertEqual(updated[0], rows[0])
        self.assertEqual(updated[1]["generation_status"], "regenerated_then_passed")


if __name__ == "__main__":
    unittest.main()
