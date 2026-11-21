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

from common import Gates, execute, load_csv, load_julia_env
from real_world import matching_stats, project, resize_image
import update_reports
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


class ReportGeneratorTests(unittest.TestCase):
    def test_load_csv_handles_header_only_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_csv = Path(directory) / "empty.matches.csv"
            empty_csv.write_text("query,train,distance\n", encoding="utf-8")
            arr = load_csv(empty_csv, 3)
            self.assertEqual(arr.shape, (0, 3))

            nonempty_csv = Path(directory) / "one.matches.csv"
            nonempty_csv.write_text("query,train,distance\n0,2,0.125\n", encoding="utf-8")
            arr2 = load_csv(nonempty_csv, 3)
            self.assertEqual(arr2.shape, (1, 3))
            self.assertAlmostEqual(arr2[0, 2], 0.125)

    def test_load_julia_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / "julia-environment.tsv"
            env_path.write_text("ImageCore\t0.10.5\nColorTypes\t0.12.5\nStaticArrays\t1.9.25\n", encoding="utf-8")
            env = load_julia_env(env_path)
            self.assertEqual(env["ColorTypes"], "0.12.5")
            self.assertEqual(env["StaticArrays"], "1.9.25")
            self.assertEqual(load_julia_env(Path(directory) / "missing.tsv"), {})

    def test_update_reports_dynamic_versions_and_none_homography(self):
        self.assertEqual(update_reports.fmt_err(None), "N/A")
        self.assertEqual(update_reports.fmt_err(3.067), "3.07")

        jenv = {"ImageCore": "0.10.5", "ColorTypes": "0.12.5", "StaticArrays": "1.9.25"}
        rw = {
            "started_utc": "2026-11-21T00:00:00+00:00",
            "julia": "julia version 1.13.4",
            "opencv": "5.0.0",
            "numpy": "2.5.5",
            "python": "3.13.18",
            "python_optimization": 1,
            "run_id": "testrun",
            "passed": True,
            "checks": [{"name": "ok", "passed": True, "detail": ""}],
            "failures": [],
            "julia_environment": jenv,
            "sequences": [{"name": "graf", "condition": "viewpoint"}],
            "images": {
                "graf_1": {
                    "native_count": 100, "opencv_count": 100, "paired_count": 100,
                    "native_coverage": 1.0, "opencv_coverage": 1.0,
                    "position_px": {"p95": 0.0001},
                    "relative_scale_error": {"p95": 1e-5},
                    "angle_error_deg": {"p95": 0.001},
                    "detected_descriptor_cosine": {"median": 0.99999},
                    "native_sift": {"seconds": 0.1, "bytes": 1024 * 1024},
                    "opencv_seconds": 0.1,
                }
            },
            "pairs": {
                "graf_2": {
                    "source": "graf_1", "target": "graf_2",
                    "native_match": {"seconds": 0.01},
                    "native": {
                        "matches": 2, "correct_within_3px": 0,
                        "homography_accurate_3px": False,
                        "homography_mean_probe_error_px": None,
                    },
                    "opencv": {
                        "matches": 2, "correct_within_3px": 0,
                        "homography_accurate_3px": False,
                        "homography_mean_probe_error_px": None,
                    },
                }
            },
            "summary": {
                "images": 1, "pairs": 1,
                "native_features": 100, "opencv_features": 100, "paired_frames": 100,
                "native_matches": 2, "opencv_matches": 2,
                "native_correct_matches_3px": 0, "opencv_correct_matches_3px": 0,
                "native_accurate_homographies_3px": 0, "opencv_accurate_homographies_3px": 0,
            },
        }
        comp = {
            "started_utc": "2026-11-21T00:00:00+00:00",
            "julia": "julia version 1.13.4",
            "opencv": "5.0.0",
            "numpy": "2.5.5",
            "python": "3.13.18",
            "cpu": "TestCPU",
            "passed": True,
            "checks": [],
            "julia_environment": jenv,
            "images": {
                "graf_1": {
                    "shape": [640, 800], "julia_features": 100, "opencv_features": 100,
                    "julia_seconds": {"median": 0.1},
                    "opencv_seconds": {"median": 0.1},
                    "opencv_unit_descriptor_seconds": {"median": 0.1},
                    "julia_over_opencv_unit_time": 1.0,
                    "julia_allocated_bytes": {"minimum": 1024 * 1024},
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            val_md = d / "validation.md"
            base_md = d / "baseline.md"
            readme_md = d / "README.md"
            readme_md.write_text("# Top\n\n## Comparative benchmarks\nold\n\n## Installation\nend\n", encoding="utf-8")

            update_reports.update_real_world_docs(rw, val_md, base_md, julia_test_count=555)
            update_reports.update_readme(comp, rw, readme_md, julia_test_count=555)

            val_text = val_md.read_text(encoding="utf-8")
            base_text = base_md.read_text(encoding="utf-8")
            readme_text = readme_md.read_text(encoding="utf-8")

            self.assertIn("N/A / N/A", val_text)
            self.assertIn("555 Julia checks", val_text)
            self.assertNotIn("170 Julia", val_text)
            self.assertIn("ColorTypes 0.12.5, StaticArrays 1.9.25", base_text)
            self.assertIn("| Metric | SIFTFeatures.jl | OpenCV 5.0.0 |", readme_text)
            self.assertNotIn("OpenCV 4.10", readme_text)
            self.assertIn("555 Julia checks", readme_text)


if __name__ == "__main__":
    unittest.main()
