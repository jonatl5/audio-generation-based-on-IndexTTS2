from __future__ import annotations

import unittest

from stable_dubbing.main import parse_args


class TestCliGroupCommands(unittest.TestCase):
    def test_parse_review_pauses(self) -> None:
        args = parse_args(["review-pauses", "--output-dir", "out", "--port", "9000"])

        self.assertEqual(args.command, "review-pauses")
        self.assertEqual(args.output_dir, "out")
        self.assertEqual(args.port, 9000)

    def test_parse_regenerate_lines(self) -> None:
        args = parse_args(
            [
                "regenerate-lines",
                "--line-ids",
                "4,5,6",
                "--output-dir",
                "out",
                "--emotion-json",
                "emotions.json",
            ]
        )

        self.assertEqual(args.command, "regenerate-lines")
        self.assertEqual(args.line_ids, "4,5,6")
        self.assertEqual(args.emotion_json, "emotions.json")

    def test_parse_manual_groups_for_generation(self) -> None:
        args = parse_args(["--manual-groups", "manual_groups.json", "--disable-auto-groups", "--dry_run"])

        self.assertEqual(args.command, "generate")
        self.assertEqual(args.manual_groups, "manual_groups.json")
        self.assertTrue(args.disable_auto_groups)


if __name__ == "__main__":
    unittest.main()
