"""Harness integrity regressions; run with normal Python and with -O."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from common import Gates, execute
from real_world import matching_stats, project, resize_image
from validate import angles, frame_pairs


class ReportingTests(unittest.TestCase):
    def test_failed_check_survives_optimized_python(self):
        code = "from common import Gates; r={}; g=Gates(r); g.require(False,'corrupted descriptors'); raise SystemExit(0 if len(r['failures']) == 1 else 9)"
        for flags in ([], ["-O"]):
            result = subprocess.run([sys.executable, *flags, "-c", code],
                                    cwd=Path(__file__).parent, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_failed_rerun_replaces_stale_success_and_records_all_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"passed":true,"run_id":"old"}')
            def runner(args, report, gates):
                running = json.loads(output.read_text())
                self.assertFalse(running["passed"])
                self.assertEqual(running["state"], "running")
                self.assertNotEqual(running["run_id"], "old")
                gates.require(False, "descriptor cosine", 0.)
                gates.require(False, "cross-library retrieval", 0)
                gates.require(True, "unrelated good check")
            with contextlib.redirect_stdout(io.StringIO()):
                status = execute(SimpleNamespace(output=output), runner, "negative control")
            result = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["state"], "complete")
            self.assertEqual(len(result["failures"]), 2)
            self.assertEqual(len(result["checks"]), 3)
            self.assertIn("src/SIFTFeatures.jl", result["source_sha256"])

    def test_exception_leaves_an_error_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            def runner(*args):
                raise RuntimeError("image decode failed")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = execute(SimpleNamespace(output=output), runner, "exception control")
            result = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["state"], "error")
            self.assertEqual(result["error"], "RuntimeError: image decode failed")

    def test_success_requires_no_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            def runner(args, report, gates):
                gates.require(True, "valid descriptors")
            with contextlib.redirect_stdout(io.StringIO()):
                status = execute(SimpleNamespace(output=output), runner, "positive control")
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(output.read_text())["passed"])

    def test_nonfinite_results_are_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            def runner(args, report, gates):
                report["metrics"] = {"cosine": float("nan"), "error": float("inf")}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = execute(SimpleNamespace(output=output), runner, "nonfinite control")
            result = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["state"], "error")
            self.assertEqual(result["metrics"], {"cosine": None, "error": None})
            self.assertFalse(list(Path(directory).glob(".result.json*")))


class GeometryTests(unittest.TestCase):
    def test_grid_pairing_equals_independent_dense_pairing(self):
        rng = np.random.default_rng(11)
        ref = np.column_stack((rng.uniform(0, 40, (160, 2)), rng.uniform(1, 5, 160),
                               rng.uniform(0, 2*np.pi, 160)))
        j = ref.copy()
        j[:, :2] += rng.uniform(-1.1, 1.1, (len(j), 2))
        j[:, 2] *= rng.uniform(.93, 1.07, len(j))
        j[:, 3] += rng.uniform(-.12, .12, len(j))
        j = np.vstack((j, j[:8]))  # ties and duplicate candidates
        pos = np.linalg.norm(j[:, None, :2] - ref[None, :, :2], axis=2)
        scale = np.abs(np.log(j[:, None, 2] / ref[None, :, 2]))
        angle = angles(j[:, None, 3], ref[None, :, 3])
        candidates = np.argwhere((pos < 1) & (scale < .05) & (angle < np.deg2rad(5)))
        ordered = sorted(candidates.tolist(), key=lambda p: pos[tuple(p)] + 10*scale[tuple(p)] + angle[tuple(p)])
        used_i, used_r, expected = set(), set(), []
        for i, r in ordered:
            if i not in used_i and r not in used_r:
                expected.append((i, r)); used_i.add(i); used_r.add(r)
        self.assertEqual(frame_pairs(j, ref).tolist(), [list(p) for p in expected])
        self.assertEqual(frame_pairs(np.empty((0, 4)), ref).shape, (0, 2))

    def test_resize_coordinates_match_pixel_centroids(self):
        # A 2×2 block maps to one pixel under exact factor-two area resampling.
        image = np.zeros((40, 60), np.uint8)
        image[12:14, 20:22] = 255
        small, C = resize_image(image, 30)
        self.assertEqual(int(small[6, 10]), 255)
        np.testing.assert_allclose(project(np.array([[20.5, 12.5]]), C), [[10., 6.]])
        unchanged, identity = resize_image(image, 0)
        self.assertIs(unchanged, image)
        np.testing.assert_equal(identity, np.eye(3))

    def test_known_geometry_detects_deliberate_bad_matches(self):
        a = np.array([[10, 10], [40, 10], [10, 40], [40, 40], [25, 25]], float)
        H = np.array([[1, 0, 7], [0, 1, -4], [0, 0, 1]], float)
        b = project(a, H)
        correct = matching_stats([(i, i) for i in range(5)], a, b, H, (60, 60))
        bad = matching_stats([(i, (i+1) % 5) for i in range(5)], a, b, H, (60, 60))
        self.assertEqual(correct["correct_within_3px"], 5)
        self.assertLess(correct["homography_mean_probe_error_px"], 1e-5)
        self.assertEqual(bad["correct_within_3px"], 0)


if __name__ == "__main__":
    unittest.main()
