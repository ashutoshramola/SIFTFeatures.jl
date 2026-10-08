#!/usr/bin/env python3
"""Real-photo SIFT parity and known-homography matching on Oxford sequences."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from common import ROOT, execute, sha256, write_report
from validate import angles, frame_pairs, geometric_errors, normalized, ratio_matches, summarize


def resize_image(image, maximum):
    h, w = image.shape
    if maximum == 0 or max(h, w) <= maximum:
        return image, np.eye(3)
    scale = maximum / max(h, w)
    hh, ww = max(2, round(h*scale)), max(2, round(w*scale))
    sx, sy = ww / w, hh / h
    # OpenCV resize maps pixel centers with a half-pixel offset.
    C = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
    return cv2.resize(image, (ww, hh), interpolation=cv2.INTER_AREA), C


def project(points, H):
    p = np.c_[points, np.ones(len(points))] @ H.T
    return p[:, :2] / p[:, 2:]


def load_frames(path):
    if len(path.read_text().splitlines()) == 1:
        return np.empty((0, 7), np.float64)
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def reference_features(sift, image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    frames = np.asarray([[*k.pt, k.size/2, np.deg2rad(k.angle), k.response,
                          k.octave & 255, (k.octave >> 8) & 255] for k in keypoints],
                        dtype=np.float64).reshape(-1, 7)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)
    return frames, normalized(descriptors)


def matching_stats(pairs, a, b, H, shape):
    errors = geometric_errors(pairs, a, b, H)
    stats = {"matches": len(pairs), "correct_within_3px": int(np.sum(errors < 3)),
             "precision_3px": float(np.mean(errors < 3)) if len(errors) else 0.,
             "error_px": summarize(errors), "homography_found": False,
             "homography_accurate_3px": False,
             "homography_mean_probe_error_px": None,
             "homography_max_probe_error_px": None, "ransac_inliers": 0}
    if len(pairs) < 4:
        return stats
    pa = np.float32([a[i, :2] for i, _ in pairs])
    pb = np.float32([b[j, :2] for _, j in pairs])
    cv2.setRNGSeed(7)
    fit, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3., maxIters=10000, confidence=.999)
    if fit is None:
        return stats
    h, w = shape
    probes = np.array([[.1*(w-1), .1*(h-1)], [.9*(w-1), .1*(h-1)],
                       [.1*(w-1), .9*(h-1)], [.9*(w-1), .9*(h-1)],
                       [.5*(w-1), .5*(h-1)]])
    delta = np.linalg.norm(project(probes, fit) - project(probes, H), axis=1)
    stats.update({"homography_found": True, "ransac_inliers": int(mask.sum()),
                  "homography_accurate_3px": bool(np.mean(delta) < 3),
                  "homography_mean_probe_error_px": float(np.mean(delta)),
                  "homography_max_probe_error_px": float(np.max(delta)),
                  "homography": fit.tolist()})
    return stats


def save_matches(output, name, images, a, b, pairs, H):
    # An optional diagnostic plot; matplotlib is not needed for validation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = name.split("_")[0] + "_1"
    left, right = images[source], images[name]
    h = max(left.shape[0], right.shape[0])
    joined = np.full((h, left.shape[1]+right.shape[1]), 245, np.uint8)
    joined[:left.shape[0], :left.shape[1]] = left
    joined[:right.shape[0], left.shape[1]:] = right
    errors = geometric_errors(pairs, a, b, H)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(joined, cmap="gray", vmin=0, vmax=255)
    # Show a deterministic, spaced sample, including incorrect matches.
    indices = np.linspace(0, len(pairs)-1, min(60, len(pairs)), dtype=int)
    for ix in indices:
        i, j = pairs[ix]
        p, q = a[i, :2], b[j, :2] + [left.shape[1], 0]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#18c977" if errors[ix] < 3 else "#fb5b58",
                lw=.65, alpha=.8)
    ax.set_title(f"{source} → {name}: {len(pairs)} Julia matches; "
                 f"{np.sum(errors<3)} within 3 px of supplied homography\n"
                 "Green: correct by reference geometry · Red: incorrect · up to 60 shown")
    ax.axis("off")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (name + "-matches.png"), dpi=150)
    plt.close(fig)


def run(args, report, gates):
    if args.max_side != 0 and args.max_side < 32:
        raise ValueError("--max-side must be 0 (original size) or at least 32")
    cv2.setNumThreads(1)
    manifest_path = Path(__file__).with_name("oxford-sources.json")
    manifest = json.loads(manifest_path.read_text())
    sequences = manifest["sequences"]
    if args.sequences:
        selected = set(args.sequences.split(","))
        available = {s["name"] for s in sequences}
        if not selected <= available:
            raise ValueError("Unknown sequence: " + str(selected - available))
        sequences = [s for s in sequences if s["name"] in selected]
    report.update({"dataset": manifest["dataset"], "dataset_homepage": manifest["homepage"],
                   "dataset_manifest_sha256": sha256(manifest_path),
                   "opencv": cv2.__version__, "numpy": np.__version__,
                   "julia": subprocess.check_output([args.julia, "--startup-file=no", "--version"], text=True).strip(),
                   "configuration": {"max_side": args.max_side, "octave_layers": 3,
                                     "contrast_threshold": .04, "edge_threshold": 10,
                                     "sigma": 1.6, "first_octave": -1, "nfeatures": 0,
                                     "precise_upscale": True, "ratio": .8, "mutual": True,
                                     "geometric_threshold_px": 3.},
                   "sequences": sequences, "images": {}, "pairs": {}})
    images, original_shapes, coordinates, homographies = {}, {}, {}, {}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    work = args.work_dir
    # Real-photo runs never use the synthetic harness's fixed descriptor probes.
    (work / "probes.csv").unlink(missing_ok=True)
    for sequence in sequences:
        scene = sequence["name"]
        directory = args.data_dir / scene
        for filename, digest in sequence["files_sha256"].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f"{scene}/{filename}: dataset checksum mismatch")
        for number in range(1, 7):
            path = next(p for p in directory.glob(f"img{number}.*") if p.suffix in (".ppm", ".pgm"))
            original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise ValueError("Could not decode " + str(path))
            name = f"{scene}_{number}"
            images[name], coordinates[name] = resize_image(original, args.max_side)
            original_shapes[name] = list(original.shape)
            images[name].tofile(work / (name + ".u8"))
            (work / (name + ".shape")).write_text(f"{images[name].shape[0]} {images[name].shape[1]}")
            if number > 1:
                H = np.loadtxt(directory / f"H1to{number}p")
                homographies[name] = coordinates[name] @ H @ np.linalg.inv(coordinates[scene + "_1"])
    (work / "images.txt").write_text("\n".join(images) + "\n")
    (work / "pairs.txt").write_text("\n".join(f"{name.split('_')[0]}_1 {name}" for name in homographies) + "\n")
    subprocess.run([args.julia, "--startup-file=no", f"--project={ROOT}",
                    str(ROOT / "validation/export.jl"), str(work)], check=True)
    performance = {}
    with (work / "performance.csv").open() as f:
        for row in csv.DictReader(f):
            performance[(row["kind"], row["name"])] = {"seconds": float(row["seconds"]), "bytes": int(row["bytes"])}
    sift = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=.04, edgeThreshold=10,
                          sigma=1.6, enable_precise_upscale=True)
    sift.detectAndCompute(next(iter(images.values())), None)  # warm OpenCV
    native, reference = {}, {}
    for name, image in images.items():
        j = load_frames(work / (name + ".kp.csv"))
        jd = np.fromfile(work / (name + ".desc.f32"), dtype=np.float32).reshape(-1, 128)
        start = time.perf_counter()
        ref, rd = reference_features(sift, image)
        cv_time = time.perf_counter() - start
        native[name], reference[name] = (j, jd), (ref, rd)
        stats = {"original_shape": original_shapes[name], "evaluated_shape": list(image.shape),
                 "input_u8_sha256": sha256(work / (name + ".u8")),
                 "native_count": len(j), "opencv_count": len(ref),
                 "native_sift": performance[("sift", name)], "opencv_seconds": cv_time}
        report["images"][name] = stats
        gates.require(len(j) == len(jd), name + ": descriptor count")
        gates.require(np.all(np.isfinite(jd)) and np.all(jd >= 0), name + ": finite nonnegative descriptors")
        norm_error = float(np.max(np.abs(np.linalg.norm(jd, axis=1)-1))) if len(jd) else None
        stats["descriptor_max_norm_error"] = norm_error
        gates.require(norm_error is not None and norm_error < 2e-6, name + ": unit descriptors", norm_error)
        pairs = frame_pairs(j, ref)
        stats["paired_count"] = len(pairs)
        stats["native_coverage"] = len(pairs) / max(1, len(j))
        stats["opencv_coverage"] = len(pairs) / max(1, len(ref))
        if not gates.require(len(pairs) > 30, name + ": paired frames", len(pairs)):
            continue
        i, r = pairs.T
        stats.update({"position_px": summarize(np.linalg.norm(j[i,:2]-ref[r,:2], axis=1)),
                      "relative_scale_error": summarize(np.abs(j[i,2]/ref[r,2]-1)),
                      "angle_error_deg": summarize(np.rad2deg(angles(j[i,3], ref[r,3]))),
                      "detected_descriptor_cosine": summarize(np.sum(jd[i]*rd[r], axis=1))})
        keys = [cv2.KeyPoint(float(p[0]), float(p[1]), float(2*p[2]), float(np.rad2deg(p[3]) % 360),
                            float(p[4]), (int(p[5]) & 255) | (int(p[6]) << 8)) for p in j]
        _, fixed = sift.compute(image, keys)
        stats["fixed_frame_descriptor_cosine"] = summarize(np.sum(jd*normalized(fixed), axis=1))
        cross = ratio_matches(jd, rd)
        good = sum(np.linalg.norm(j[q,:2]-ref[t,:2]) < 1 and
                   abs(np.log(j[q,2]/ref[t,2])) < .05 and
                   angles(j[q,3],ref[t,3]) < np.deg2rad(5) for q,t in cross)
        stats["cross_matches"] = len(cross)
        stats["cross_match_precision"] = good / max(1, len(cross))
        gates.require(min(stats["native_coverage"], stats["opencv_coverage"]) >= .97, name+": frame coverage", min(stats["native_coverage"],stats["opencv_coverage"]))
        for metric, limit in (("position_px", .01), ("relative_scale_error", .005), ("angle_error_deg", .1)):
            gates.require(stats[metric]["p95"] < limit, name+": "+metric+" p95", stats[metric]["p95"])
        gates.require(stats["detected_descriptor_cosine"]["median"] > .999, name+": detected descriptor cosine", stats["detected_descriptor_cosine"]["median"])
        gates.require(stats["fixed_frame_descriptor_cosine"]["p05"] > .995, name+": fixed-frame descriptor cosine", stats["fixed_frame_descriptor_cosine"]["p05"])
        gates.require(len(cross) >= .8*min(len(j),len(ref)), name+": cross-library retrieval count", len(cross))
        gates.require(stats["cross_match_precision"] >= .95, name+": cross-library precision", stats["cross_match_precision"])
        print(f"{name}: {len(j)}/{len(ref)} features, {len(pairs)} paired, "
              f"coverage {min(stats['native_coverage'],stats['opencv_coverage']):.4%}, "
              f"descriptor cosine {stats['detected_descriptor_cosine']['median']:.8f}", flush=True)
        write_report(args.output, report)
    for name, H in homographies.items():
        source = name.split("_")[0] + "_1"
        a, ad = native[source]; b, bd = native[name]
        ra, rad = reference[source]; rb, rbd = reference[name]
        matches = load_frames(work / (name + ".matches.csv"))
        pairs = [tuple(map(int, p[:2])) for p in matches]
        check = ratio_matches(ad, bd)
        rpairs = ratio_matches(rad, rbd)
        stats = {"source": source, "target": name, "known_homography": H.tolist(),
                 "native_match": performance[("match", name)],
                 "same_descriptor_matcher_pairs_equal": set(pairs) == set(check),
                 "native": matching_stats(pairs, a, b, H, images[source].shape),
                 "opencv": matching_stats(rpairs, ra, rb, H, images[source].shape)}
        report["pairs"][name] = stats
        gates.require(stats["same_descriptor_matcher_pairs_equal"], name+": same-descriptor matcher equality")
        # Challenging viewpoint pairs can defeat standard SIFT in both libraries.
        # Score accuracy against supplied geometry and gate *relative* behavior.
        nj, nr = stats["native"], stats["opencv"]
        gates.require(nj["correct_within_3px"] >= .9*nr["correct_within_3px"]-2,
                      name+": correct matches comparable", f"{nj['correct_within_3px']}/{nr['correct_within_3px']}")
        gates.require(nj["precision_3px"] >= nr["precision_3px"]-.05,
                      name+": precision comparable", f"{nj['precision_3px']}/{nr['precision_3px']}")
        gates.require(nj["matches"] >= .9*nr["matches"]-2 and nj["matches"] <= 1.1*nr["matches"]+2,
                      name+": match count comparable", f"{nj['matches']}/{nr['matches']}")
        if nr["homography_accurate_3px"]:
            gates.require(nj["homography_accurate_3px"], name+": accurate homography when OpenCV succeeds",
                          nj["homography_mean_probe_error_px"])
        if args.figures and name in ("graf_4", "leuven_6", "bikes_6", "bark_6"):
            save_matches(args.figures, name, images, a, b, pairs, H)
        print(f"{source} → {name}: {nj['matches']}/{nr['matches']} matches, "
              f"{nj['correct_within_3px']}/{nr['correct_within_3px']} correct within 3 px", flush=True)
        write_report(args.output, report)
    report["summary"] = {
        "images": len(report["images"]), "pairs": len(report["pairs"]),
        "native_features": sum(s["native_count"] for s in report["images"].values()),
        "opencv_features": sum(s["opencv_count"] for s in report["images"].values()),
        "paired_frames": sum(s["paired_count"] for s in report["images"].values()),
        "native_matches": sum(s["native"]["matches"] for s in report["pairs"].values()),
        "opencv_matches": sum(s["opencv"]["matches"] for s in report["pairs"].values()),
        "native_correct_matches_3px": sum(s["native"]["correct_within_3px"] for s in report["pairs"].values()),
        "opencv_correct_matches_3px": sum(s["opencv"]["correct_within_3px"] for s in report["pairs"].values()),
        "native_accurate_homographies_3px": sum(s["native"]["homography_accurate_3px"] for s in report["pairs"].values()),
        "opencv_accurate_homographies_3px": sum(s["opencv"]["homography_accurate_3px"] for s in report["pairs"].values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="Interchange files retained for inspection")
    parser.add_argument("--max-side", type=int, default=640, help="0 keeps original resolution")
    parser.add_argument("--sequences", help="Optional comma-separated subset")
    parser.add_argument("--figures", type=Path, help="Optional diagnostic plots; requires matplotlib")
    parser.add_argument("--output", default=str(ROOT / "validation/real-world-results.json"))
    sys.exit(execute(parser.parse_args(), run, "Oxford real-photo reference"))
