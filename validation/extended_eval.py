#!/usr/bin/env python3
"""Manual extended evaluation: non-default SIFTConfig parity, stress tests,
multi-threaded scaling, and optional HPatches (116 scenes, 696 photos, 580 pairs).

This script is intentionally opt-in (manual CI workflow_dispatch or local runs)
and never commits datasets to the repository.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile

import cv2
import numpy as np

from common import ROOT, execute, load_csv, load_julia_env, write_report
from real_world import load_frames, matching_stats, reference_features, resize_image
from validate import angles, frame_pairs, normalized, ratio_matches, summarize

HPATCHES_URL = (
    "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/"
    "hpatches-sequences-release.zip"
)
VALID_SUITES = ("configs", "stress", "scaling", "hpatches")

# Non-default SIFTConfig parity matrix matched to cv2.SIFT_create parameters.
# Note: first_octave=0 has no direct OpenCV SIFT_create flag, so it is checked
# for structural invariants (valid coordinates, octave >= 0, unit descriptors).
CONFIG_SPECS = [
    {"tag": "nfeatures_500", "nfeatures": 500, "octave_layers": 3,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "nfeatures_2000", "nfeatures": 2000, "octave_layers": 3,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "contrast_0.02", "nfeatures": 0, "octave_layers": 3,
     "contrast_threshold": 0.02, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "edge_5.0", "nfeatures": 0, "octave_layers": 3,
     "contrast_threshold": 0.04, "edge_threshold": 5.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "layers_4", "nfeatures": 0, "octave_layers": 4,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "layers_5", "nfeatures": 0, "octave_layers": 5,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "sigma_2.0", "nfeatures": 0, "octave_layers": 3,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 2.0, "first_octave": -1,
     "compare_opencv": True},
    {"tag": "first_octave_0", "nfeatures": 0, "octave_layers": 3,
     "contrast_threshold": 0.04, "edge_threshold": 10.0, "sigma": 1.6, "first_octave": 0,
     "compare_opencv": False},
]


def parse_suites(spec):
    if spec.strip().lower() == "all":
        return list(VALID_SUITES)
    chosen = [s.strip().lower() for s in spec.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in VALID_SUITES]
    if unknown:
        raise ValueError(f"Unknown suite(s): {unknown}. Expected subset of {VALID_SUITES} or 'all'.")
    if not chosen:
        raise ValueError("At least one suite must be selected.")
    # Preserve canonical order without duplicates.
    return [s for s in VALID_SUITES if s in chosen]


def parse_threads(spec):
    threads = [int(t.strip()) for t in spec.split(",") if t.strip()]
    if not threads or any(t < 1 for t in threads):
        raise ValueError(f"Invalid --threads list: {spec!r}")
    return threads


def synthetic_fallback_image(shape=(700, 1000)):
    """Deterministic multi-scale textured image used when Oxford data is absent."""
    h, w = shape
    rng = np.random.default_rng(2026)
    base = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base += 0.35 + 0.15 * np.sin(xx / 19.0) * np.cos(yy / 23.0)
    for _ in range(180):
        cx = float(rng.uniform(20, w - 20))
        cy = float(rng.uniform(20, h - 20))
        rad = float(rng.uniform(4.0, 28.0))
        amp = float(rng.uniform(-0.35, 0.35))
        base += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * rad * rad))
    base += rng.normal(0.0, 0.04, (h, w)).astype(np.float32)
    return np.clip(np.round(base * 255.0), 0, 255).astype(np.uint8)


def load_base_photo(data_dir, scene, number=1):
    directory = Path(data_dir) / scene
    if directory.is_dir():
        matches = [p for p in directory.glob(f"img{number}.*") if p.suffix in (".ppm", ".pgm", ".png")]
        if matches:
            img = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img, str(matches[0])
    return synthetic_fallback_image(), "synthetic-fallback"


def write_u8(work_dir, name, image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image.tofile(work_dir / f"{name}.u8")
    (work_dir / f"{name}.shape").write_text(f"{image.shape[0]} {image.shape[1]}\n", encoding="utf-8")


def run_julia_mode(julia_bin, mode, work_dir, extra_args=(), threads=1):
    cmd = [
        julia_bin, "--startup-file=no", f"--project={ROOT}",
        f"-t", str(threads),
        str(ROOT / "validation/extended_eval.jl"), mode, str(work_dir), *map(str, extra_args),
    ]
    subprocess.run(cmd, check=True)


def compare_to_opencv(j, jd, ref, rd):
    pairs = frame_pairs(j, ref)
    cov_j = len(pairs) / max(1, len(j))
    cov_r = len(pairs) / max(1, len(ref))
    stats = {
        "native_count": int(len(j)),
        "opencv_count": int(len(ref)),
        "paired_count": int(len(pairs)),
        "native_coverage": float(cov_j),
        "opencv_coverage": float(cov_r),
        "descriptor_max_norm_error": float(np.max(np.abs(np.linalg.norm(jd, axis=1) - 1.0))) if len(jd) else 0.0,
    }
    if len(pairs) > 0:
        i, r = pairs.T
        stats["position_px"] = summarize(np.linalg.norm(j[i, :2] - ref[r, :2], axis=1))
        stats["relative_scale_error"] = summarize(np.abs(j[i, 2] / ref[r, 2] - 1.0))
        stats["angle_error_deg"] = summarize(np.rad2deg(angles(j[i, 3], ref[r, 3])))
        stats["detected_descriptor_cosine"] = summarize(np.sum(jd[i] * rd[r], axis=1))
    return stats


def run_configs_suite(args, report, gates):
    work = args.work_dir / "configs"
    work.mkdir(parents=True, exist_ok=True)
    img, source = load_base_photo(args.data_dir, "bikes", 1)
    write_u8(work, "config_input", img)
    lines = []
    for cfg in CONFIG_SPECS:
        lines.append("\t".join(str(cfg[k]) for k in (
            "tag", "nfeatures", "octave_layers", "contrast_threshold",
            "edge_threshold", "sigma", "first_octave"
        )))
    (work / "configs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_julia_mode(args.julia, "configs", work, threads=1)

    cv2.setNumThreads(1)
    results = {"source_image": source, "shape": list(img.shape), "cases": {}}
    for cfg in CONFIG_SPECS:
        tag = cfg["tag"]
        j = load_frames(work / f"{tag}.kp.csv")
        jd = np.fromfile(work / f"{tag}.desc.f32", dtype=np.float32).reshape(-1, 128)
        gates.require(len(j) == len(jd) and len(j) > 50, f"config:{tag} feature count", len(j))
        gates.require(np.all(np.isfinite(jd)) and np.all(jd >= 0), f"config:{tag} finite nonnegative descriptors")
        norm_err = float(np.max(np.abs(np.linalg.norm(jd, axis=1) - 1.0)))
        gates.require(norm_err < 2e-6, f"config:{tag} unit descriptors", norm_err)
        if cfg["nfeatures"] > 0:
            gates.require(len(j) <= cfg["nfeatures"], f"config:{tag} nfeatures cap", len(j))

        if cfg["compare_opencv"]:
            sift = cv2.SIFT_create(
                nfeatures=cfg["nfeatures"],
                nOctaveLayers=cfg["octave_layers"],
                contrastThreshold=cfg["contrast_threshold"],
                edgeThreshold=cfg["edge_threshold"],
                sigma=cfg["sigma"],
                enable_precise_upscale=True,
            )
            ref, rd = reference_features(sift, img)
            stats = compare_to_opencv(j, jd, ref, rd)
            min_cov = min(stats["native_coverage"], stats["opencv_coverage"])
            # nfeatures truncation can differ on borderline response ties at the exact cutoff.
            cov_floor = 0.96 if cfg["nfeatures"] > 0 else 0.985
            gates.require(min_cov >= cov_floor, f"config:{tag} frame coverage", f"{min_cov:.4%}")
            gates.require(stats["detected_descriptor_cosine"]["median"] > 0.999,
                          f"config:{tag} descriptor cosine", stats["detected_descriptor_cosine"]["median"])
            results["cases"][tag] = stats
            print(f"[config] {tag:15s}: {len(j):5d}/{len(ref):5d} kp, cov={min_cov:.4%}, "
                  f"cos={stats['detected_descriptor_cosine']['median']:.6f}", flush=True)
        else:
            min_oct = int(np.min(j[:, 5])) if len(j) else 0
            gates.require(min_oct >= 0, f"config:{tag} min octave >= 0", min_oct)
            results["cases"][tag] = {
                "native_count": int(len(j)),
                "min_octave": min_oct,
                "descriptor_max_norm_error": norm_err,
            }
            print(f"[config] {tag:15s}: {len(j):5d} kp, min_octave={min_oct}", flush=True)
    report["configs"] = results


def run_stress_suite(args, report, gates):
    work = args.work_dir / "stress"
    work.mkdir(parents=True, exist_ok=True)
    base, source = load_base_photo(args.data_dir, "bikes", 1)
    rng = np.random.default_rng(42)
    cases = {
        "clean_u8_u16": base,
        "noise_sigma15": np.clip(base.astype(np.float32) + rng.normal(0.0, 15.0, base.shape), 0, 255).astype(np.uint8),
        "noise_sigma30": np.clip(base.astype(np.float32) + rng.normal(0.0, 30.0, base.shape), 0, 255).astype(np.uint8),
        "low_light_0.15x": np.clip(np.round(base.astype(np.float32) * 0.15), 0, 255).astype(np.uint8),
        "pano_wide_127x1023": cv2.resize(base, (1023, 127), interpolation=cv2.INTER_AREA),
        "pano_tall_1023x127": cv2.resize(base, (127, 1023), interpolation=cv2.INTER_AREA),
    }
    for name, image in cases.items():
        write_u8(work, name, image)
    (work / "stress_images.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
    run_julia_mode(args.julia, "stress", work, threads=1)

    u16_parity = {}
    with (work / "u16_parity.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            u16_parity[row["name"]] = {
                "identical_kp": row["identical_kp"] == "true",
                "identical_desc": row["identical_desc"] == "true",
                "max_desc_diff": float(row["max_desc_diff"]),
            }

    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10,
                           sigma=1.6, enable_precise_upscale=True)
    results = {"source_image": source, "cases": {}}
    for name, image in cases.items():
        j = load_frames(work / f"{name}.kp.csv")
        jd = np.fromfile(work / f"{name}.desc.f32", dtype=np.float32).reshape(-1, 128)
        ref, rd = reference_features(sift, image)
        stats = compare_to_opencv(j, jd, ref, rd)
        stats["shape"] = list(image.shape)
        stats["u16_parity"] = u16_parity[name]
        results["cases"][name] = stats

        gates.require(u16_parity[name]["identical_kp"] and u16_parity[name]["identical_desc"],
                      f"stress:{name} UInt16 bitwise parity", u16_parity[name])
        gates.require(stats["descriptor_max_norm_error"] < 2e-6,
                      f"stress:{name} unit descriptors", stats["descriptor_max_norm_error"])
        if len(ref) >= 10:
            min_cov = min(stats["native_coverage"], stats["opencv_coverage"])
            gates.require(min_cov >= 0.97, f"stress:{name} frame coverage", f"{min_cov:.4%}")
            gates.require(stats["detected_descriptor_cosine"]["median"] > 0.999,
                          f"stress:{name} descriptor cosine", stats["detected_descriptor_cosine"]["median"])
            print(f"[stress] {name:19s}: {len(j):5d}/{len(ref):5d} kp, cov={min_cov:.4%}, "
                  f"u16_exact={u16_parity[name]['identical_desc']}", flush=True)
        else:
            gates.require(abs(len(j) - len(ref)) <= 2, f"stress:{name} low-count parity", f"{len(j)}/{len(ref)}")
            print(f"[stress] {name:19s}: {len(j):5d}/{len(ref):5d} kp (low-contrast), "
                  f"u16_exact={u16_parity[name]['identical_desc']}", flush=True)
    report["stress"] = results


def run_scaling_suite(args, report, gates):
    work = args.work_dir / "scaling"
    work.mkdir(parents=True, exist_ok=True)
    trees, source = load_base_photo(args.data_dir, "trees", 1)
    images = {
        "trees_1": trees,
        "1080p": cv2.resize(trees, (1920, 1080), interpolation=cv2.INTER_CUBIC),
        "4K": cv2.resize(trees, (3840, 2160), interpolation=cv2.INTER_CUBIC),
    }
    for name, image in images.items():
        write_u8(work, name, image)
    (work / "scaling_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")

    threads_list = parse_threads(args.threads)
    results = {"source_image": source, "threads": threads_list,
               "warmups": args.warmups, "samples": args.samples, "runs": {}}
    baseline_hashes = {}

    for t in threads_list:
        run_julia_mode(args.julia, "scaling", work, extra_args=(args.warmups, args.samples), threads=t)
        cv2.setNumThreads(t)
        sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10,
                               sigma=1.6, enable_precise_upscale=True)
        row_map = {}
        with (work / f"scaling_t{t}.tsv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                row_map[row["name"]] = row

        t_key = str(t)
        results["runs"][t_key] = {}
        for name, image in images.items():
            for _ in range(args.warmups):
                sift.detectAndCompute(image, None)
            cv_times = []
            for _ in range(args.samples):
                t0 = time.perf_counter()
                _, desc = sift.detectAndCompute(image, None)
                if desc is not None and len(desc):
                    _ = normalized(desc).T.copy()
                cv_times.append((time.perf_counter() - t0) * 1000.0)
            cv_med = statistics.median(cv_times)
            j_row = row_map[name]
            j_med = float(j_row["median_ms"])
            hashes = (j_row["kp_hash"], j_row["desc_hash"])
            if name not in baseline_hashes:
                baseline_hashes[name] = hashes
            else:
                gates.require(hashes == baseline_hashes[name],
                              f"scaling:{name} bitwise determinism t={t}",
                              f"{hashes} vs {baseline_hashes[name]}")
            ratio = j_med / cv_med if cv_med > 0 else float("inf")
            results["runs"][t_key][name] = {
                "shape": list(image.shape),
                "features": int(j_row["features"]),
                "julia_median_ms": j_med,
                "opencv_median_ms": cv_med,
                "julia_over_opencv": ratio,
                "kp_hash": j_row["kp_hash"],
                "desc_hash": j_row["desc_hash"],
            }
            print(f"[scaling t={t}] {name:8s}: Julia={j_med:7.1f} ms  OpenCV={cv_med:7.1f} ms  "
                  f"ratio={ratio:.2f}x", flush=True)
    report["scaling"] = results


def find_hpatches_root(hpatches_dir):
    root = Path(hpatches_dir)
    if (root / "i_ajuntament").is_dir():
        return root
    nested = root / "hpatches-sequences-release"
    if (nested / "i_ajuntament").is_dir():
        return nested
    return None


def ensure_hpatches(hpatches_dir):
    found = find_hpatches_root(hpatches_dir)
    if found is not None:
        return found
    hpatches_dir = Path(hpatches_dir)
    hpatches_dir.mkdir(parents=True, exist_ok=True)
    archive = hpatches_dir / "hpatches-sequences-release.zip"
    if not archive.is_file():
        print(f"Downloading HPatches archive to {archive} ...", flush=True)
        tmp_archive = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(HPATCHES_URL, tmp_archive)
        tmp_archive.replace(archive)
    print(f"Extracting {archive} ...", flush=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(hpatches_dir)
    found = find_hpatches_root(hpatches_dir)
    if found is None:
        raise RuntimeError(f"HPatches sequences not found under {hpatches_dir} after extraction")
    return found


def discover_hpatches_scenes(hp_root, limit=0):
    scenes = sorted(
        p for p in Path(hp_root).iterdir()
        if p.is_dir() and (p.name.startswith("i_") or p.name.startswith("v_"))
    )
    if limit > 0:
        scenes = scenes[:limit]
    return scenes


def run_hpatches_suite(args, report, gates):
    hp_root = ensure_hpatches(args.hpatches_dir)
    scenes = discover_hpatches_scenes(hp_root, limit=args.hpatches_limit)
    gates.require(len(scenes) > 0, "hpatches: discovered scenes", len(scenes))

    work = args.work_dir / "hpatches"
    work.mkdir(parents=True, exist_ok=True)
    images, coordinates, homographies = {}, {}, {}
    image_names, pair_lines = [], []
    for scene_dir in scenes:
        scene = scene_dir.name
        for num in range(1, 7):
            img_path = scene_dir / f"{num}.ppm"
            raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise ValueError(f"Could not decode {img_path}")
            name = f"{scene}_{num}"
            resized, C = resize_image(raw, args.hpatches_max_side)
            images[name] = resized
            coordinates[name] = C
            write_u8(work, name, resized)
            image_names.append(name)
            if num > 1:
                H = np.loadtxt(scene_dir / f"H_1_{num}")
                homographies[name] = C @ H @ np.linalg.inv(coordinates[f"{scene}_1"])
                pair_lines.append(f"{scene}_1 {name}")
    (work / "images.txt").write_text("\n".join(image_names) + "\n", encoding="utf-8")
    (work / "pairs.txt").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    run_julia_mode(args.julia, "hpatches", work, threads=args.hpatches_threads)
    julia_wall_s = time.perf_counter() - t0

    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10,
                           sigma=1.6, enable_precise_upscale=True)

    totals = {
        "scenes": len(scenes), "images": len(images), "pairs": len(homographies),
        "max_side": args.hpatches_max_side, "julia_wall_seconds": julia_wall_s,
        "native_features": 0, "opencv_features": 0, "paired_frames": 0,
        "native_matches": 0, "opencv_matches": 0,
        "native_correct_3px": 0, "opencv_correct_3px": 0,
        "native_homographies_3px": 0, "opencv_homographies_3px": 0,
        "matcher_exact_pairs": 0,
    }
    by_split = {
        "illumination": {"scenes": 0, "pairs": 0, "native_matches": 0, "opencv_matches": 0,
                         "native_correct_3px": 0, "opencv_correct_3px": 0,
                         "native_homographies_3px": 0, "opencv_homographies_3px": 0},
        "viewpoint": {"scenes": 0, "pairs": 0, "native_matches": 0, "opencv_matches": 0,
                      "native_correct_3px": 0, "opencv_correct_3px": 0,
                      "native_homographies_3px": 0, "opencv_homographies_3px": 0},
    }
    worst_cos_median = 1.0
    max_norm_err = 0.0

    for scene_dir in scenes:
        scene = scene_dir.name
        split = "illumination" if scene.startswith("i_") else "viewpoint"
        by_split[split]["scenes"] += 1
        native_scene, ref_scene = {}, {}
        for num in range(1, 7):
            name = f"{scene}_{num}"
            img = images[name]
            j = load_frames(work / f"{name}.kp.csv")
            jd = np.fromfile(work / f"{name}.desc.f32", dtype=np.float32).reshape(-1, 128)
            ref, rd = reference_features(sift, img)
            native_scene[name] = (j, jd)
            ref_scene[name] = (ref, rd)
            pairs = frame_pairs(j, ref)
            totals["native_features"] += len(j)
            totals["opencv_features"] += len(ref)
            totals["paired_frames"] += len(pairs)
            if len(jd):
                nerr = float(np.max(np.abs(np.linalg.norm(jd, axis=1) - 1.0)))
                if nerr > max_norm_err:
                    max_norm_err = nerr
            if len(pairs):
                i, r = pairs.T
                cos_med = float(np.median(np.sum(jd[i] * rd[r], axis=1)))
                if cos_med < worst_cos_median:
                    worst_cos_median = cos_med
        src = f"{scene}_1"
        a, ad = native_scene[src]
        ra, rad = ref_scene[src]
        for num in range(2, 7):
            name = f"{scene}_{num}"
            b, bd = native_scene[name]
            rb, rbd = ref_scene[name]
            H = homographies[name]
            matches = load_csv(work / f"{name}.matches.csv", 3)
            jpairs = [tuple(map(int, p[:2])) for p in matches]
            check = ratio_matches(ad, bd)
            rpairs = ratio_matches(rad, rbd)
            if set(jpairs) == set(check):
                totals["matcher_exact_pairs"] += 1
            nj = matching_stats(jpairs, a, b, H, images[src].shape)
            nr = matching_stats(rpairs, ra, rb, H, images[src].shape)
            totals["native_matches"] += nj["matches"]
            totals["opencv_matches"] += nr["matches"]
            totals["native_correct_3px"] += nj["correct_within_3px"]
            totals["opencv_correct_3px"] += nr["correct_within_3px"]
            totals["native_homographies_3px"] += int(nj["homography_accurate_3px"])
            totals["opencv_homographies_3px"] += int(nr["homography_accurate_3px"])
            sp = by_split[split]
            sp["pairs"] += 1
            sp["native_matches"] += nj["matches"]
            sp["opencv_matches"] += nr["matches"]
            sp["native_correct_3px"] += nj["correct_within_3px"]
            sp["opencv_correct_3px"] += nr["correct_within_3px"]
            sp["native_homographies_3px"] += int(nj["homography_accurate_3px"])
            sp["opencv_homographies_3px"] += int(nr["homography_accurate_3px"])

    coverage = totals["paired_frames"] / max(1, min(totals["native_features"], totals["opencv_features"]))
    j_prec = totals["native_correct_3px"] / max(1, totals["native_matches"])
    o_prec = totals["opencv_correct_3px"] / max(1, totals["opencv_matches"])
    totals.update({
        "frame_coverage": coverage,
        "worst_image_descriptor_cosine_median": worst_cos_median,
        "max_descriptor_norm_error": max_norm_err,
        "native_precision_3px": j_prec,
        "opencv_precision_3px": o_prec,
        "splits": by_split,
    })
    gates.require(max_norm_err < 2e-6, "hpatches: unit descriptors", max_norm_err)
    gates.require(coverage >= 0.99, "hpatches: pooled frame coverage", f"{coverage:.4%}")
    gates.require(worst_cos_median > 0.999, "hpatches: worst descriptor cosine median", worst_cos_median)
    gates.require(totals["matcher_exact_pairs"] == totals["pairs"],
                  "hpatches: same-descriptor matcher agreement",
                  f"{totals['matcher_exact_pairs']}/{totals['pairs']}")
    gates.require(abs(j_prec - o_prec) <= 0.005, "hpatches: pooled precision parity",
                  f"{j_prec:.4%} vs {o_prec:.4%}")
    gates.require(totals["native_homographies_3px"] >= totals["opencv_homographies_3px"] - 5,
                  "hpatches: accurate homographies parity",
                  f"{totals['native_homographies_3px']}/{totals['opencv_homographies_3px']}")
    print(f"[hpatches] {totals['scenes']} scenes, {totals['pairs']} pairs: "
          f"cov={coverage:.4%}, matcher={totals['matcher_exact_pairs']}/{totals['pairs']}, "
          f"prec={j_prec:.3%} vs {o_prec:.3%}, homographies={totals['native_homographies_3px']}/{totals['opencv_homographies_3px']}",
          flush=True)
    report["hpatches"] = totals


def run(args, report, gates):
    suites = parse_suites(args.suites)
    report.update({
        "suites": suites,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "julia": subprocess.check_output([args.julia, "--startup-file=no", "--version"], text=True).strip(),
    })
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if "configs" in suites:
        run_configs_suite(args, report, gates)
    if "stress" in suites:
        run_stress_suite(args, report, gates)
    if "scaling" in suites:
        run_scaling_suite(args, report, gates)
    if "hpatches" in suites:
        run_hpatches_suite(args, report, gates)
    for sub in ("configs", "stress", "scaling", "hpatches"):
        env_path = args.work_dir / sub / "julia-environment.tsv"
        if env_path.is_file():
            report["julia_environment"] = load_julia_env(env_path)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default="configs,stress,scaling",
                        help="Comma-separated subset of configs,stress,scaling,hpatches or 'all'")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "validation-data",
                        help="Oxford dataset directory (falls back to synthetic texture if absent)")
    parser.add_argument("--hpatches-dir", type=Path, default=ROOT / "hpatches-data",
                        help="Directory to cache/read extracted HPatches sequences (gitignored)")
    parser.add_argument("--hpatches-max-side", type=int, default=800,
                        help="Max image side for HPatches (0 = original resolution; default 800)")
    parser.add_argument("--hpatches-limit", type=int, default=0,
                        help="Optional max number of HPatches scenes to run (0 = all 116)")
    parser.add_argument("--hpatches-threads", type=int, default=4,
                        help="Julia thread count for HPatches extraction (default 4)")
    parser.add_argument("--threads", default="1,4",
                        help="Comma-separated thread counts for scaling suite (default 1,4)")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "extended-eval-work")
    parser.add_argument("--output", type=Path, default=ROOT / "extended-eval-work/extended-results.json")
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    raise SystemExit(execute(args, run, "extended evaluation"))


if __name__ == "__main__":
    main()
