from __future__ import annotations

import unittest

from stable_dubbing.config import AlignmentConfig
from stable_dubbing.duration_align import choose_alignment_action


class TestDurationAlign(unittest.TestCase):
    def test_shorter_pads(self) -> None:
        decision = choose_alignment_action(2.0, 3.0, AlignmentConfig())
        self.assertEqual(decision.action, "pad_silence_end")
        self.assertFalse(decision.should_regenerate)

    def test_default_single_pass_prefers_stretching(self) -> None:
        decision = choose_alignment_action(6.0, 3.0, AlignmentConfig())
        self.assertEqual(decision.action, "time_stretch")
        self.assertFalse(decision.should_regenerate)

    def test_between_stretch_and_regen_thresholds_stretches_with_warning(self) -> None:
        config = AlignmentConfig(stretch_if_long_under_ratio=0.10, regenerate_if_long_over_ratio=0.20)
        decision = choose_alignment_action(3.45, 3.0, config)
        self.assertEqual(decision.action, "time_stretch")
        self.assertFalse(decision.should_regenerate)
        self.assertTrue(decision.warnings)

    def test_over_regen_threshold_regenerates(self) -> None:
        config = AlignmentConfig(stretch_if_long_under_ratio=0.10, regenerate_if_long_over_ratio=0.20)
        decision = choose_alignment_action(3.7, 3.0, config)
        self.assertEqual(decision.action, "regenerate")
        self.assertTrue(decision.should_regenerate)


if __name__ == "__main__":
    unittest.main()
