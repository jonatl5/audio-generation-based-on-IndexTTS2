from __future__ import annotations

import unittest

from stable_dubbing.sentence_groups import build_sentence_groups, join_group_text


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


class TestSentenceGroups(unittest.TestCase):
    def test_lowercase_continuation_groups_same_speaker_lines(self) -> None:
        lines = [
            _line(4, "Jiang Han", "With her own consciousness, she knew all too well", 0.0, 2.0),
            _line(5, "Jiang Han", "that a poor student like me", 2.0, 3.0),
            _line(6, "Jiang Han", "could never afford to upkeep a high-tier Beast like her", 3.0, 5.0),
        ]

        groups = build_sentence_groups(lines, lines)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].line_ids, [4, 5, 6])
        self.assertEqual(
            groups[0].text,
            "With her own consciousness, she knew all too well that a poor student like me could never afford to upkeep a high-tier Beast like her",
        )

    def test_does_not_group_across_speakers(self) -> None:
        lines = [
            _line(1, "A", "Hello", 0.0, 1.0),
            _line(2, "B", "there", 1.0, 2.0),
        ]

        groups = build_sentence_groups(lines, lines)

        self.assertEqual([group.line_ids for group in groups], [[1], [2]])

    def test_join_normalizes_trailing_commas(self) -> None:
        lines = [
            {"text": "Jiang Han,"},
            {"text": "don't worry"},
        ]

        self.assertEqual(join_group_text(lines), "Jiang Han, don't worry")

    def test_different_vectors_are_duration_weighted(self) -> None:
        first = _line(1, "A", "First", 0.0, 1.0)
        second = _line(2, "A", "second", 1.0, 3.0)
        first["emo_vector"] = [1, 0, 0, 0, 0, 0, 0, 0]
        second["emo_vector"] = [0, 0, 0, 0, 0, 0, 0, 1]

        group = build_sentence_groups([first, second], [first, second])[0]

        self.assertEqual(group.emotion["emotion_blend"], "duration_weighted_by_target_duration")
        self.assertAlmostEqual(group.emotion["emo_vector"][0], 1 / 3, places=5)
        self.assertAlmostEqual(group.emotion["emo_vector"][7], 2 / 3, places=5)


if __name__ == "__main__":
    unittest.main()
