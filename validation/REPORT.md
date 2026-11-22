# OpenCV reference validation — 2026-11-22

The synthetic validation harness passed all 84 explicit checks on 2026-11-22 and records run provenance and failures in `latest-results.json`. See the [real-photograph study](../docs/validation.md) for 48 original-resolution images, known-homography matching, and validation guards.

Executed with Julia 1.13.4, OpenCV 5.0.0, and NumPy 2.5.5. All reference gates passed.
Julia runtime dependencies: ColorTypes 0.12.5, Colors 0.13.4, FixedPointNumbers 0.8.6, StaticArrays 1.9.25.

The six images are generated reproducibly by `validate.py`: shapes and lettering, four known geometric transforms, and an independent smoothed random texture. OpenCV precise upscaling is explicitly enabled. Keypoints are paired one-to-one using position, scale, and orientation only; descriptors do not choose the pairs.

| Image | Julia / OpenCV features | Paired | Coverage Julia / OpenCV | Position p95 (px) | Scale relative error p95 | Angle p95 (degrees) | Descriptor median cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 729 / 733 | 729 | 100.00% / 99.45% | 0.000084 | 0.000018 | 0.002757 | 0.99998558 |
| translate | 694 / 698 | 694 | 100.00% / 99.43% | 0.000098 | 0.000018 | 0.000768 | 0.99998593 |
| rotate90 | 712 / 718 | 712 | 100.00% / 99.16% | 0.000088 | 0.000018 | 0.002845 | 0.99998569 |
| similarity | 527 / 527 | 527 | 100.00% / 100.00% | 0.000108 | 0.000016 | 0.001564 | 0.99998498 |
| perspective | 607 / 606 | 605 | 99.67% / 99.83% | 0.000091 | 0.000017 | 0.003112 | 0.99998504 |
| texture | 3353 / 3351 | 3351 | 99.94% / 100.00% | 0.000032 | 0.000006 | 0.000072 | 0.99998021 |

Descriptors also agree at identical supplied frames, including half-pixel rounding probes across octaves −1, 0, and 1. OpenCV descriptors are normalized to unit L2 for comparison; Julia returns unquantized unit vectors.

| Image | Fixed-frame cosine p05 | Half-pixel probe minimum cosine | Cross-library matches | Cross-library geometric precision |
|---|---:|---:|---:|---:|
| base | 0.99997973 | 0.99961668 | 728 | 100.00% |
| translate | 0.99997932 | 0.99997997 | 693 | 100.00% |
| rotate90 | 0.99997962 | 0.99998140 | 712 | 100.00% |
| similarity | 0.99997962 | 0.99998194 | 527 | 100.00% |
| perspective | 0.99997973 | 0.99995726 | 606 | 99.83% |
| texture | 0.99997729 | 0.99997723 | 3351 | 100.00% |

Julia ratio/mutual matches agreed exactly with OpenCV BFMatcher when supplied the same descriptor matrices. The known transformations give an independent geometric check:

| Transform | Julia / OpenCV matches | Julia / OpenCV inliers within 2 px | Recovered homography max probe error (px) |
|---|---:|---:|---:|
| translate | 620 / 622 | 99.68% / 99.68% | 0.017 |
| rotate90 | 671 / 672 | 99.85% / 99.85% | 0.013 |
| similarity | 316 / 314 | 93.04% / 93.63% | 0.039 |
| perspective | 384 / 383 | 96.61% / 96.61% | 0.104 |

## Acceptance gates

- More than 30 paired frames; at least 97% coverage in both directions.
- Position error p95 < 0.01 px; relative scale error p95 < 0.005; angle error p95 < 0.1°.
- Detected descriptor median cosine > 0.999; identical-frame cosine p05 > 0.995; every half-pixel probe cosine > 0.995.
- Cross-library matches at least 80% of the smaller feature count, with at least 95% geometric precision.
- Each transformed pair: at least 40 matches, at least 90% inliers within 2 px, and recovered homography probe error < 1.5 px.

## Scope and limitations

This is a synthetic reference suite, not a natural-image benchmark or proof of bitwise identity. Shape repetition produces some incorrect ratio-test matches in both implementations; RANSAC remains necessary for geometric use. Exact keypoint counts can differ around histogram peaks and plateau/tie cases. The descriptor comparison separates detector variation from descriptor arithmetic and quantization.

Under `Pkg.test()` with bounds checking enabled, all 602 Julia checks pass on Julia 1.13.4 (both single-threaded and multi-threaded), and hosted GitHub Actions CI executes both `Tests` and `Evaluation & Benchmarks` workflows.

Run `python validation/validate.py` to regenerate the JSON. This Markdown report records the run; the JSON includes minima, p05, medians, p95, maxima, and per-transform details. See the README for setup and numerical conventions.
