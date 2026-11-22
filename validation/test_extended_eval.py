"""Unit tests for the manual extended evaluation and HPatches harness."""
from pathlib import Path
import tempfile
import unittest

import extended_eval


class ExtendedEvalTests(unittest.TestCase):
    def test_parse_suites_canonical_order_and_validation(self):
        self.assertEqual(extended_eval.parse_suites("all"), ["configs", "stress", "scaling", "hpatches"])
        self.assertEqual(extended_eval.parse_suites("hpatches,configs,configs"), ["configs", "hpatches"])
        with self.assertRaises(ValueError):
            extended_eval.parse_suites("unknown_suite")
        with self.assertRaises(ValueError):
            extended_eval.parse_suites("  ")

    def test_parse_threads(self):
        self.assertEqual(extended_eval.parse_threads("1, 4, 8"), [1, 4, 8])
        with self.assertRaises(ValueError):
            extended_eval.parse_threads("0,4")

    def test_discover_hpatches_scenes_and_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "hpatches-sequences-release"
            for name in ("i_ajuntament", "v_adam", "v_boat", "misc_dir"):
                (root / name).mkdir(parents=True)
            self.assertEqual(extended_eval.find_hpatches_root(directory), root)
            scenes = extended_eval.discover_hpatches_scenes(root)
            self.assertEqual([s.name for s in scenes], ["i_ajuntament", "v_adam", "v_boat"])
            limited = extended_eval.discover_hpatches_scenes(root, limit=2)
            self.assertEqual([s.name for s in limited], ["i_ajuntament", "v_adam"])


if __name__ == "__main__":
    unittest.main()
