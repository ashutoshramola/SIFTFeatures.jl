#!/usr/bin/env python3
"""Repeated Julia/OpenCV SIFT timings on img1 from each Oxford sequence."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from common import execute, sha256


def summarize(values):
    return {"median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "samples": values}


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unavailable"


def run(args, report, gates):
    if args.samples < 3 or args.warmups < 1 or args.threads < 1:
        raise ValueError("Require at least 3 samples, 1 warmup, and 1 thread")
    cv2.setNumThreads(args.threads)
    manifest_path = ROOT / "validation/oxford-sources.json"
    manifest = json.loads(manifest_path.read_text())
    report.update({
        "cpu": cpu_name(), "opencv": cv2.__version__, "numpy": np.__version__,
        "julia": subprocess.check_output([args.julia, "--startup-file=no", "--version"], text=True).strip(),
        "dataset_manifest_sha256": sha256(manifest_path),
        "configuration": {"samples": args.samples, "warmups_per_image": args.warmups,
                          "image_selection": "img1 from each of eight sequences, original resolution",
                          "julia_threads": args.threads, "opencv_threads": cv2.getNumThreads(),
                          "octave_layers": 3, "contrast_threshold": .04,
                          "edge_threshold": 10, "sigma": 1.6, "nfeatures": 0,
                          "first_octave": -1, "enable_precise_upscale": True},
        "timing_scope": {
            "julia": "sift(UInt8 image): detection and unquantized unit 128xN Float32 descriptors",
            "opencv": "detectAndCompute: detection and quantized Float32 descriptors",
            "opencv_unit_descriptors": "detectAndCompute plus transpose-copy to 128xN and unit-L2 normalization",
            "excluded": "image decoding, file interchange, warmups, forced GC, output checks, and report writing",
            "allocation": "Julia total allocated bytes per call; not peak memory; OpenCV memory unmeasured",
            "order": "Julia image blocks first, OpenCV image blocks second; timing is machine-specific",
            "gc": "forced GC precedes each sample; GC during the timed call is included"},
        "images": {}})
    for path in (Path(__file__), ROOT / "benchmark/compare.jl"):
        report["source_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    images = {}
    for sequence in manifest["sequences"]:
        scene = sequence["name"]
        filename = next(f for f in sequence["files_sha256"]
                        if f.startswith("img1.") and f.endswith((".ppm", ".pgm")))
        path = args.data_dir / scene / filename
        if sha256(path) != sequence["files_sha256"][filename]:
            raise ValueError(f"{scene}/{filename}: source checksum mismatch")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not decode " + str(path))
        name = scene + "_1"
        images[name] = image
        image.tofile(work / (name + ".u8"))
        (work / (name + ".shape")).write_text(f"{image.shape[0]} {image.shape[1]}\n")
        report["images"][name] = {"shape": list(image.shape),
                                 "source_sha256": sha256(path),
                                 "input_u8_sha256": sha256(work / (name + ".u8"))}
    (work / "images.txt").write_text("\n".join(images) + "\n")
    env = dict(os.environ, JULIA_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([args.julia, "--startup-file=no", f"--threads={args.threads}", f"--project={ROOT}",
                    str(ROOT / "benchmark/compare.jl"), str(work), str(args.samples),
                    str(args.warmups)], check=True, env=env)
    report["julia_environment"] = dict(line.split("\t", 1) for line in
                                       (work / "julia-environment.tsv").read_text().splitlines())
    with (work / "julia-samples.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reference = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=.04,
                                edgeThreshold=10, sigma=1.6, enable_precise_upscale=True)
    for name, image in images.items():
        native = [row for row in rows if row["image"] == name]
        gates.require(len(native) == args.samples, name + ": Julia sample count")
        seconds, normalized_seconds, counts = [], [], []
        for sample in range(-args.warmups, args.samples):
            gc.collect()
            start = time.perf_counter()
            keypoints, descriptors = reference.detectAndCompute(image, None)
            detect_seconds = time.perf_counter() - start
            if descriptors is None:
                raise ValueError(name + ": no OpenCV descriptors")
            unit = descriptors.T.copy()
            unit /= np.maximum(np.linalg.norm(unit, axis=0, keepdims=True), np.finfo(np.float32).tiny)
            full_seconds = time.perf_counter() - start
            if sample >= 0:
                seconds.append(detect_seconds)
                normalized_seconds.append(full_seconds)
                counts.append(len(keypoints))
            if unit.shape != (128, len(keypoints)) or not np.isfinite(unit).all():
                raise ValueError(name + ": invalid OpenCV descriptors")
        gates.require(len(set(counts)) == 1, name + ": deterministic OpenCV counts")
        native_counts = {int(row["features"]) for row in native}
        gates.require(len(native_counts) == 1, name + ": deterministic Julia counts")
        julia_count = next(iter(native_counts))
        gates.require(abs(julia_count-counts[0]) <= max(julia_count, counts[0])*.03,
                      name + ": comparable feature counts")
        result = report["images"][name]
        result.update({"julia_features": julia_count, "opencv_features": counts[0],
                       "julia_seconds": summarize([float(r["seconds"]) for r in native]),
                       "julia_allocated_bytes": summarize([int(r["allocated_bytes"]) for r in native]),
                       "julia_gc_seconds": summarize([float(r["gc_seconds"]) for r in native]),
                       "opencv_seconds": summarize(seconds),
                       "opencv_unit_descriptor_seconds": summarize(normalized_seconds)})
        result["julia_over_opencv_unit_time"] = (result["julia_seconds"]["median"] /
                                                 result["opencv_unit_descriptor_seconds"]["median"])
        print(f"{name}: Julia {1000*result['julia_seconds']['median']:.1f} ms; "
              f"OpenCV + unit conversion {1000*result['opencv_unit_descriptor_seconds']['median']:.1f} ms",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "benchmark-work")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/comparison-results.json")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    sys.exit(execute(args, run, "Repeated Julia/OpenCV timing comparison"))
