from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_dubbing.main import create_run_output_dir


class TestMainPaths(unittest.TestCase):
    def test_create_run_output_dir_uses_unique_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first = create_run_output_dir(base, "chapter 1")
            second = create_run_output_dir(base, "chapter 1")

        self.assertEqual(first.name, "chapter_1")
        self.assertEqual(second.name, "chapter_1_02")


if __name__ == "__main__":
    unittest.main()
