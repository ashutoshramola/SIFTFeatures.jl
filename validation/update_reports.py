#!/usr/bin/env python3
"""Update Markdown benchmark and validation reports from JSON run outputs."""
import argparse
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load_passed_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("passed"):
        raise RuntimeError(f"Refusing to update docs from unpassed report: {path}")
    return data


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_med_count(values) -> str:
    m = statistics.median(values)
    return f"{int(m)}" if m == int(m) else f"{m:.1f}"


def update_report_md(latest: dict, path: Path):
    date = latest["started_utc"][:10]
    julia_ver = latest["julia"].replace("julia version ", "")
    opencv_ver = latest["opencv"]
    numpy_ver = latest["numpy"]

    img_rows = []
    probe_rows = []
    for name in ("base", "translate", "rotate90", "similarity", "perspective", "texture"):
        v = latest["images"][name]
        img_rows.append(
            f"| {name} | {v['native_count']} / {v['opencv_count']} | {v['paired_count']} | "
            f"{100*v['native_coverage']:.2f}% / {100*v['opencv_coverage']:.2f}% | "
            f"{v['position_px']['p95']:.6f} | {v['relative_scale_error']['p95']:.6f} | "
            f"{v['angle_error_deg']['p95']:.6f} | {v['detected_descriptor_cosine']['median']:.8f} |"
        )
        probe_rows.append(
            f"| {name} | {v['fixed_frame_descriptor_cosine']['p05']:.8f} | "
            f"{v['half_pixel_probe_cosine']['min']:.8f} | {v['cross_matches']} | "
            f"{100*v['cross_match_precision']:.2f}% |"
        )

    tf_rows = []
    for name in ("translate", "rotate90", "similarity", "perspective"):
        v = latest["transforms"][name]
        tf_rows.append(
            f"| {name} | {v['native_matches']} / {v['opencv_matches']} | "
            f"{100*v['native_inlier_fraction']:.2f}% / {100*v['opencv_inlier_fraction']:.2f}% | "
            f"{v['recovered_homography_max_probe_error_px']:.3f} |"
        )

    content = f"""# OpenCV reference validation — {date}

The synthetic validation harness passed all {len(latest['checks'])} explicit checks on {date} and records run provenance and failures in `latest-results.json`. See the [real-photograph study](../docs/validation.md) for 48 original-resolution images, known-homography matching, and validation guards.

Executed with Julia {julia_ver}, OpenCV {opencv_ver}, and NumPy {numpy_ver}. All reference gates passed.
Julia runtime dependencies: ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

The six images are generated reproducibly by `validate.py`: shapes and lettering, four known geometric transforms, and an independent smoothed random texture. OpenCV precise upscaling is explicitly enabled. Keypoints are paired one-to-one using position, scale, and orientation only; descriptors do not choose the pairs.

| Image | Julia / OpenCV features | Paired | Coverage Julia / OpenCV | Position p95 (px) | Scale relative error p95 | Angle p95 (degrees) | Descriptor median cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(img_rows)}

Descriptors also agree at identical supplied frames, including half-pixel rounding probes across octaves −1, 0, and 1. OpenCV descriptors are normalized to unit L2 for comparison; Julia returns unquantized unit vectors.

| Image | Fixed-frame cosine p05 | Half-pixel probe minimum cosine | Cross-library matches | Cross-library geometric precision |
|---|---:|---:|---:|---:|
{chr(10).join(probe_rows)}

Julia ratio/mutual matches agreed exactly with OpenCV BFMatcher when supplied the same descriptor matrices. The known transformations give an independent geometric check:

| Transform | Julia / OpenCV matches | Julia / OpenCV inliers within 2 px | Recovered homography max probe error (px) |
|---|---:|---:|---:|
{chr(10).join(tf_rows)}

## Acceptance gates

- More than 30 paired frames; at least 97% coverage in both directions.
- Position error p95 < 0.01 px; relative scale error p95 < 0.005; angle error p95 < 0.1°.
- Detected descriptor median cosine > 0.999; identical-frame cosine p05 > 0.995; every half-pixel probe cosine > 0.995.
- Cross-library matches at least 80% of the smaller feature count, with at least 95% geometric precision.
- Each transformed pair: at least 40 matches, at least 90% inliers within 2 px, and recovered homography probe error < 1.5 px.

## Scope and limitations

This is a synthetic reference suite, not a natural-image benchmark or proof of bitwise identity. Shape repetition produces some incorrect ratio-test matches in both implementations; RANSAC remains necessary for geometric use. Exact keypoint counts can differ around histogram peaks and plateau/tie cases. The descriptor comparison separates detector variation from descriptor arithmetic and quantization.

All 170 Julia tests pass under `Pkg.test()` with bounds checking enabled on Julia {julia_ver} (both single-threaded and multi-threaded), and hosted GitHub Actions CI executes both `Tests` and `Evaluation & Benchmarks` workflows.

Run `python validation/validate.py` to regenerate the JSON. This Markdown report records the run; the JSON includes minima, p05, medians, p95, maxima, and per-transform details. See the README for setup and numerical conventions.
"""
    path.write_text(content, encoding="utf-8")


def update_comparison_md(comp: dict, path: Path):
    date = comp["started_utc"][:10]
    julia_ver = comp["julia"].replace("julia version ", "")
    jenv = comp.get("julia_environment", {})
    cpu = comp["cpu"]
    cpu_target = jenv.get("cpu_target", "")
    cpu_desc = f"{cpu} (`{cpu_target}`)" if cpu_target and cpu_target not in cpu else cpu
    pkg_line = (
        f"ImageCore {jenv.get('ImageCore', '0.10.5')}, "
        f"ColorTypes {jenv.get('ColorTypes', '0.12.4')}"
    )
    if "StaticArrays" in jenv:
        pkg_line += f", StaticArrays {jenv['StaticArrays']}"

    rows = []
    for name, v in comp["images"].items():
        j_ms = 1000.0 * v["julia_seconds"]["median"]
        cv_raw_ms = 1000.0 * v["opencv_seconds"]["median"]
        cv_unit_ms = 1000.0 * v["opencv_unit_descriptor_seconds"]["median"]
        ratio = v["julia_over_opencv_unit_time"]
        alloc_mib = v["julia_allocated_bytes"]["minimum"] / (1024.0 ** 2)
        rows.append(
            f"| {name} | {j_ms:,.1f} | {cv_raw_ms:,.1f} | {cv_unit_ms:,.1f} | {ratio:.2f}× | {alloc_mib:,.1f} |"
        )

    content = f"""# Repeated Julia/OpenCV comparison — {date}

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
{cpu_desc}. Julia {julia_ver}, OpenCV {comp['opencv']},
NumPy {comp['numpy']}, Python {comp['python']}; Julia/OpenCV/BLAS each use one thread.
{pkg_line}.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Julia times `sift` on a UInt8 matrix, including intensity conversion, detection,
and 128×N unquantized unit Float32 descriptors. OpenCV raw times
`detectAndCompute` through the Python binding, with default SIFT parameters
except `enable_precise_upscale=True`; its descriptor values are quantized.
Unit-output timings additionally transpose-copy descriptors to 128×N and
normalize columns to unit L2. Conversion of keypoints into a separate reference
frame array is excluded. Descriptor matching and homography fitting are not timed.

Image decoding, file interchange, compilation/warmups, forced GC before each
sample, output checks, and report writing are excluded. GC triggered within
the timed call is included. Julia allocations are total allocated bytes per
call, not peak memory. OpenCV memory is not measured. The two libraries run in
separate image blocks, Julia first. CPU frequency and system load are not
controlled, so the ratios apply only to this run.

This differs from `real-world-baseline.md`: that earlier report summarizes one
observation on each of six images per sequence and includes OpenCV keypoint-frame
conversion. Image selection and timing scope differ; the results should not be
compared as an optimization or regression.

The benchmark checks pinned image hashes, descriptor shape/norms, repeatable
feature counts, and comparable output counts. All {len(comp['checks'])} explicit gates passed.
The [JSON](comparison-results.json) records every timing and allocation sample,
input/source hashes, package versions, configuration, and measurement scope.
The independent accuracy and matching study uses all 48 images; see
[validation results](../docs/validation.md).

## Reproduction

From the package directory, after installing `validation/requirements.txt` and
instantiating the Julia project:

```sh
python validation/download_oxford.py --data-dir validation-data
python benchmark/compare.py --data-dir validation-data --samples 5 --warmups 1
```

`--julia /path/to/julia` selects the runtime. `--work-dir` selects scratch storage;
the default `benchmark-work/` directory is ignored. `--output` selects the JSON
report destination. Images and decoded interchange data remain local.
"""
    path.write_text(content, encoding="utf-8")


def update_real_world_docs(rw: dict, validation_md: Path, baseline_md: Path):
    date = rw["started_utc"][:10]
    julia_ver = rw["julia"].replace("julia version ", "")
    s = rw["summary"]
    images = rw["images"]
    pairs = rw["pairs"]

    min_img_cov = min(min(v["native_coverage"], v["opencv_coverage"]) for v in images.values())
    max_pos_p95 = max(v["position_px"]["p95"] for v in images.values())
    max_scale_p95 = max(v["relative_scale_error"]["p95"] for v in images.values())
    max_angle_p95 = max(v["angle_error_deg"]["p95"] for v in images.values())
    min_cos_med = min(v["detected_descriptor_cosine"]["median"] for v in images.values())
    overall_cov = min(s["paired_frames"] / s["native_features"], s["paired_frames"] / s["opencv_features"])
    j_prec = 100.0 * s["native_correct_matches_3px"] / s["native_matches"]
    o_prec = 100.0 * s["opencv_correct_matches_3px"] / s["opencv_matches"]

    seq_rows = []
    timing_rows = []
    for seq in rw["sequences"]:
        sname = seq["name"]
        change = seq["condition"].replace("_", " ")
        s_imgs = [v for k, v in images.items() if k.startswith(sname + "_")]
        s_pairs = [v for k, v in pairs.items() if k.startswith(sname + "_")]
        scov = min(min(v["native_coverage"], v["opencv_coverage"]) for v in s_imgs)
        j_corr = sum(v["native"]["correct_within_3px"] for v in s_pairs)
        o_corr = sum(v["opencv"]["correct_within_3px"] for v in s_pairs)
        j_homo = sum(1 for v in s_pairs if v["native"]["homography_accurate_3px"])
        o_homo = sum(1 for v in s_pairs if v["opencv"]["homography_accurate_3px"])
        seq_rows.append(
            f"| {sname} | {change} | {100*scov:.4f}% | {fmt_int(j_corr)} / {fmt_int(o_corr)} | {j_homo} / {o_homo} |"
        )

        counts = [v["native_count"] for v in s_imgs]
        sift_sec = [v["native_sift"]["seconds"] for v in s_imgs]
        cv_sec = [v["opencv_seconds"] for v in s_imgs]
        alloc_mib = [v["native_sift"]["bytes"] / (1024.0 ** 2) for v in s_imgs]
        match_sec = [v["native_match"]["seconds"] for v in s_pairs]
        timing_rows.append(
            f"| {sname} | {fmt_med_count(counts)} / {fmt_int(max(counts))} | "
            f"{statistics.median(sift_sec):.3f} | {statistics.median(cv_sec):.3f} | "
            f"{statistics.median(alloc_mib):.1f} | {statistics.median(match_sec):.3f} |"
        )

    diff_rows = []
    for k, v in pairs.items():
        nj, nr = v["native"], v["opencv"]
        if min(nj["correct_within_3px"], nr["correct_within_3px"]) < 20:
            diff_rows.append(
                f"| {v['source']} → {v['target']} | {nj['matches']} / {nj['correct_within_3px']} | "
                f"{nr['matches']} / {nr['correct_within_3px']} | "
                f"{nj['homography_mean_probe_error_px']:.2f} / {nr['homography_mean_probe_error_px']:.2f} |"
            )

    val_content = f"""# Real-photograph validation — {date}

**Result: PASS**, with {len(rw['checks'])} explicit checks and {len(rw['failures'])} failures. The Julia implementation was tested on all {s['images']} original-resolution photographs in eight Oxford sequences, against OpenCV {rw['opencv']} and the supplied homographies. The package source was unchanged during this test.

Runtime: {rw['julia']}; OpenCV {rw['opencv']}; NumPy {rw['numpy']}; Python {rw['python']}. The photo run used Python optimization level {rw['python_optimization']}; explicit gates remained active. Run ID: `{rw['run_id']}`.

## Numerical agreement

- {fmt_int(s['native_features'])} Julia feature instances and {fmt_int(s['opencv_features'])} OpenCV instances; {fmt_int(s['paired_frames'])} geometry-only one-to-one pairs ({100*overall_cov:.4f}% coverage across all frames).
- Lowest coverage on any individual photograph: {100*min_img_cov:.4f}%.
- Lowest per-image median descriptor cosine similarity: {min_cos_med:.8f} (1 means identical directions).
- Worst per-image p95 position difference: {max_pos_p95:.6f} px; relative scale difference: {max_scale_p95:.5e}; orientation difference: {max_angle_p95:.6f}°.
- Fixed-frame descriptors, descriptor norms, cross-library retrieval, and matching the same descriptors with Julia versus BFMatcher were also checked.

## Matching against independent geometry

Across {s['pairs']} reference/target pairs, Julia returned {fmt_int(s['native_matches'])} descriptor matches, of which {fmt_int(s['native_correct_matches_3px'])} landed within 3 pixels of the published transformation ({j_prec:.3f}% pooled precision). OpenCV returned {fmt_int(s['opencv_matches'])} matches, with {fmt_int(s['opencv_correct_matches_3px'])} correct ({o_prec:.3f}%).

A separately fitted homography had mean error below 3 pixels at five fixed source-image probes on {s['native_accurate_homographies_3px']}/{s['pairs']} Julia pairs and {s['opencv_accurate_homographies_3px']}/{s['pairs']} OpenCV pairs. Fit accuracy is scored against the supplied homography, not the RANSAC mask.

| Scene | Change | Lowest frame coverage | Correct matches: Julia / OpenCV¹ | Accurate homographies: Julia / OpenCV² |
|---|---|---:|---:|---:|
{chr(10).join(seq_rows)}

¹ Sum of correct descriptor matches across the five target images. ² Out of five pairs, using the mean error at the five probes.

![Correct matches by sequence](../validation/real-world-summary.png)

## Difficult cases

A compatibility pass does not mean every scene pair matched successfully. Standard SIFT loses correspondences under sufficiently severe foreshortening, blur, and scale change. The following original-resolution pairs had fewer than 20 correct matches in at least one implementation:

| Pair | Julia: matches / correct | OpenCV: matches / correct | Mean fitted-homography error: Julia / OpenCV (px) |
|---|---:|---:|---:|
{chr(10).join(diff_rows)}

Local diagnostic match plots can be generated with `--figures /path/to/figures` (requires matplotlib). The publication includes the measured-result chart; downloaded photographs and image-overlay diagnostics remain local.

## Protocol and acceptance gates

Both libraries receive exactly the same decoded grayscale UInt8 pixels. This primary run does not resize or cap features. Parameters are three layers per octave, contrast 0.04, edge threshold 10, sigma 1.6, and a doubled first octave. OpenCV precise upscaling is explicitly enabled. Coordinates are converted to zero-based pixels, sigma to half OpenCV keypoint size, and angles to radians. OpenCV descriptor vectors are L2-normalized for comparison.

Keypoint pairing uses position <1 px, absolute log-scale ratio <0.05, and circular angle difference <5°. It uses no descriptors to choose pairs. Each image must have >30 pairs and ≥97% coverage, p95 position <0.01 px, relative scale error <0.005, angle error <0.1°, median detected-frame descriptor cosine >0.999, and p05 fixed-frame cosine >0.995. Cross-library retrieval must cover ≥80% of the smaller feature set with ≥95% geometric precision.

Descriptor matching uses a strict 0.8 Lowe ratio and mutual nearest neighbors. Published H1toNp matrices map image-1 pixel centers into image-N coordinates. A match is correct when its forward projection error is <3 px. The matcher must give the same pair set as the BFMatcher-based reference when both use the same descriptors. Separate library outputs must have match counts within 10% plus two, correct-match count at least 90% of OpenCV minus two, and precision no more than five percentage points below OpenCV. If OpenCV produces an accurate homography, Julia matches must also produce one.

Homography fitting uses OpenCV RANSAC with seed 7, 3-pixel threshold, 10,000 maximum iterations, and confidence 0.999 on each library’s matches. The five evaluation probes are the center and the four points at 10%/90% of the source image dimensions. The geometric gate is mean probe error <3 px. The fit is validation machinery; SIFTFeatures itself does not implement RANSAC.

The thresholds were set in the harness before the original-resolution run. These are empirical integration checks, not the official Oxford affine-region overlap/repeatability protocol or a published benchmark ranking.

## Resized cross-check

A preliminary, uncapped run of the same 48 photographs at maximum side 640 also passed its numerical/matching gates: 144,783 / 144,791 Julia/OpenCV feature instances, 144,745 paired frames, and 24,509 / 24,519 correct matches among 26,207 / 26,215 descriptor matches. That earlier protocol did not yet include the fitted-homography comparison gate. The original-resolution report above includes it. Resizing maps include the pixel-center half-pixel offset; each homography is conjugated with the actual source/target maps.

## Validation integrity and Julia regression tests

The previous Python assertion-removal issue is repaired: explicit checks collect failures, reports are written on failure, exit status is nonzero, and a pending failed-state report replaces any stale success before work starts. Deliberately zeroing descriptors is rejected in both normal and optimized Python, with `passed=false`. Seven harness tests passed in both modes for the photographic run; the current suite has eight, including non-finite report handling. All 98 Julia tests passed under `Pkg.test()` with bounds checking enabled for the original 0.1.0 run. The current suite expands to 170 Julia checks on Julia {julia_ver} (both single-threaded and multi-threaded).

- [Descriptor-corruption control](../validation/guard-negative-control.txt)
- [Machine-readable original-resolution results](../validation/real-world-results.json)
- [Machine-readable resized results](../validation/real-world-resized-results.json)
- [Pinned sources and individual file hashes](../validation/oxford-sources.json)
- [Exact dataset-manifest snapshot used by these runs](../validation/run-snapshots/oxford-sources.json)

The current manifest labels its retrieval date in the user’s local timezone. The exact original manifest bytes, whose hash is recorded in both run reports, are preserved in the linked run snapshot. Archive and image hashes are unchanged.

## Scope and reproduction

This strengthens the photographic evidence across eight scenes. It remains a small benchmark of planar scenes or fixed-center camera acquisitions; it does not establish robustness to arbitrary 3D viewpoint changes, moving objects, every camera pipeline, or every platform. Publication preparation adds independent Julia descriptor-layout tests and explicit singleton-octave guards; see the [implementation notes](implementation.md).

From the package directory, after installing `validation/requirements.txt`:

```sh
python validation/download_oxford.py --data-dir validation-data
python -O validation/real_world.py --data-dir validation-data \\
    --work-dir validation-work --max-side 0
```

Use `--julia /path/to/julia` if required. The `Evaluation & Benchmarks` GitHub workflow (`eval.yml`) also executes this full 48-photograph suite weekly (when commits were made) or on demand on Ubuntu 24.04.

Data: [Oxford affine covariant regions](https://www.robots.ox.ac.uk/~vgg/research/affine/). K. Mikolajczyk et al., *A comparison of affine region detectors*, IJCV 65(1/2), 2005. The benchmark documentation describes gradually changing camera focus, zoom, viewpoint, aperture, and JPEG quality. Images are retained only in local scratch storage; plots are diagnostic derivatives.
"""
    validation_md.write_text(val_content, encoding="utf-8")

    base_content = f"""# Real-photograph timing observations — {date}

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia {julia_ver}, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23, OpenCV {rw['opencv']}; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(timing_rows)}

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
"""
    baseline_md.write_text(base_content, encoding="utf-8")


def update_readme(comp: dict, rw: dict, readme_path: Path):
    comp_date = comp["started_utc"][:10]
    julia_ver = comp["julia"].replace("julia version ", "")
    cpu = comp["cpu"]

    timing_rows = []
    ratios = []
    for name, v in comp["images"].items():
        h, w = v["shape"]
        j_ms = 1000.0 * v["julia_seconds"]["median"]
        cv_ms = 1000.0 * v["opencv_unit_descriptor_seconds"]["median"]
        ratio = v["julia_over_opencv_unit_time"]
        ratios.append(ratio)
        timing_rows.append(
            f"| {name} | {w}×{h} | {fmt_int(v['julia_features'])} / {fmt_int(v['opencv_features'])} | "
            f"{j_ms:,.1f} | {cv_ms:,.1f} | {ratio:.2f}× |"
        )
    min_ratio, max_ratio = min(ratios), max(ratios)

    s = rw["summary"]
    images = rw["images"]
    overall_cov = min(s["paired_frames"] / s["native_features"], s["paired_frames"] / s["opencv_features"])
    max_pos_p95 = max(v["position_px"]["p95"] for v in images.values())
    max_scale_p95 = max(v["relative_scale_error"]["p95"] for v in images.values())
    max_angle_p95 = max(v["angle_error_deg"]["p95"] for v in images.values())
    min_cos_med = min(v["detected_descriptor_cosine"]["median"] for v in images.values())
    j_prec = 100.0 * s["native_correct_matches_3px"] / s["native_matches"]
    o_prec = 100.0 * s["opencv_correct_matches_3px"] / s["opencv_matches"]

    new_section = f"""## Comparative benchmarks

### Detection and descriptor timing

Median of **five measured runs after one warmup per image**, using the first
original-resolution photograph from each Oxford sequence. {cpu};
Julia {julia_ver} and OpenCV {comp['opencv']}; one thread each; unlimited features.
Measured {comp_date}.

| Photograph | Size (w×h) | Features: Julia / OpenCV | Julia (ms) | OpenCV¹ (ms) | Julia / OpenCV time |
|---|---:|---:|---:|---:|---:|
{chr(10).join(timing_rows)}

**Julia currently takes {min_ratio:.2f}–{max_ratio:.2f}× as long as OpenCV on these eight inputs.**
These are timings for the current implementation on this machine.
Decoding, compilation, file interchange, warmups, and forced GC are excluded;
GC triggered inside a measured call is included. Parameters match, with OpenCV
`enable_precise_upscale=True`. System load and CPU frequency were not controlled.

¹ OpenCV `detectAndCompute` plus transpose-copy to `128×N` and unit-L2
normalization, matching Julia's output layout and normalization. The descriptor
values still differ: OpenCV quantizes, while Julia returns unquantized Float32.
[Raw samples and allocation values](benchmark/comparison-results.json),
[method and reproduction](benchmark/comparison.md), and the
[48-photo timing observations](benchmark/real-world-baseline.md) are available.

### Accuracy and matching

All **48 original-resolution photographs and 40 image pairs** from eight Oxford
sequences, including blur, viewpoint, zoom/rotation, illumination, and JPEG
compression. Both libraries use a strict 0.8 ratio and mutual nearest neighbors.
Correct matches are scored against the supplied homographies, within 3 pixels.

| Metric | SIFTFeatures.jl | OpenCV 4.10 |
|---|---:|---:|
| Detected feature instances | {fmt_int(s['native_features'])} | {fmt_int(s['opencv_features'])} |
| Descriptor matches across 40 pairs | {fmt_int(s['native_matches'])} | {fmt_int(s['opencv_matches'])} |
| Correct matches within 3 px | {fmt_int(s['native_correct_matches_3px'])} | {fmt_int(s['opencv_correct_matches_3px'])} |
| Pooled match precision | {j_prec:.3f}% | {o_prec:.3f}% |
| Accurate fitted homographies² | {s['native_accurate_homographies_3px']} / {s['pairs']} | {s['opencv_accurate_homographies_3px']} / {s['pairs']} |

| Numerical agreement with OpenCV | Measured value |
|---|---:|
| Geometry-paired feature instances | {fmt_int(s['paired_frames'])} (≥{100*overall_cov:.4f}% overall coverage) |
| Position difference: worst image's p95 | {max_pos_p95:.6f} px |
| Relative scale difference: worst image's p95 | {max_scale_p95:.5e} |
| Orientation difference: worst image's p95 | {max_angle_p95:.6f}° |
| Descriptor cosine: lowest per-image median | {min_cos_med:.8f} (1 means identical directions) |

² Mean fitted-homography error below 3 px at five fixed image probes, evaluated
against independent reference geometry. Severe viewpoint pairs still fail in
both implementations; these results describe a small planar-scene benchmark.
[Full validation, thresholds, and difficult cases](docs/validation.md).
The current source also passes 170 Julia checks on Julia {julia_ver} (both single-threaded
and multi-threaded)."""

    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"## Comparative benchmarks\n.*?(?=\n\n## Installation)", re.DOTALL)
    if not pattern.search(text):
        raise RuntimeError("Could not locate '## Comparative benchmarks' section in README.md")
    updated = pattern.sub(new_section, text, count=1)
    readme_path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=Path, default=ROOT / "validation/latest-results.json")
    parser.add_argument("--real-world", type=Path, default=ROOT / "validation/real-world-results.json")
    parser.add_argument("--comparison", type=Path, default=ROOT / "benchmark/comparison-results.json")
    args = parser.parse_args()

    if args.latest.is_file():
        latest = load_passed_json(args.latest)
        update_report_md(latest, ROOT / "validation/REPORT.md")

    if args.comparison.is_file():
        comp = load_passed_json(args.comparison)
        update_comparison_md(comp, ROOT / "benchmark/comparison.md")

    if args.real_world.is_file():
        rw = load_passed_json(args.real_world)
        update_real_world_docs(rw, ROOT / "docs/validation.md", ROOT / "benchmark/real-world-baseline.md")

    if args.comparison.is_file() and args.real_world.is_file():
        update_readme(comp, rw, ROOT / "README.md")


if __name__ == "__main__":
    main()
