# OpenCV reference validation — 2026-10-15

The synthetic validation harness passed all 84 explicit checks on 2026-10-15 and records run provenance and failures in `latest-results.json`. See the [real-photograph study](../docs/validation.md) for 48 original-resolution images, known-homography matching, and validation guards.

Executed locally and in GitHub Actions CI with Julia 1.13.1, OpenCV 4.10.0, and NumPy 2.x. All reference gates passed.
Julia runtime dependencies: ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

The six images are generated reproducibly by `validate.py`: shapes and lettering, four known geometric transforms, and an independent smoothed random texture. OpenCV precise upscaling is explicitly enabled. Keypoints are paired one-to-one using position, scale, and orientation only; descriptors do not choose the pairs.

| Image | Julia / OpenCV features | Paired | Coverage Julia / OpenCV | Position p95 (px) | Scale relative error p95 | Angle p95 (degrees) | Descriptor median cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 730 / 731 | 728 | 99.73% / 99.59% | 0.000066 | 0.000014 | 0.003375 | 0.99998546 |
| translate | 698 / 701 | 698 | 100.00% / 99.57% | 0.000066 | 0.000015 | 0.001050 | 0.99998569 |
| rotate90 | 722 / 722 | 720 | 99.72% / 99.72% | 0.000068 | 0.000015 | 0.002053 | 0.99998558 |
| similarity | 532 / 532 | 532 | 100.00% / 100.00% | 0.000079 | 0.000011 | 0.000159 | 0.99998510 |
| perspective | 624 / 623 | 623 | 99.84% / 100.00% | 0.000071 | 0.000011 | 0.000198 | 0.99998504 |
| texture | 3353 / 3351 | 3351 | 99.94% / 100.00% | 0.000026 | 0.000005 | 0.000065 | 0.99998021 |

Descriptors also agree at identical supplied frames, including half-pixel rounding probes across octaves −1, 0, and 1. OpenCV descriptors are normalized to unit L2 for comparison; Julia returns unquantized unit vectors.

| Image | Fixed-frame cosine p05 | Half-pixel probe minimum cosine | Cross-library matches | Cross-library geometric precision |
|---|---:|---:|---:|---:|
| base | 0.99997962 | 0.99961686 | 728 | 100.00% |
| translate | 0.99997962 | 0.99998045 | 698 | 100.00% |
| rotate90 | 0.99997985 | 0.99998057 | 720 | 100.00% |
| similarity | 0.99997950 | 0.99998105 | 532 | 100.00% |
| perspective | 0.99997979 | 0.99995267 | 623 | 100.00% |
| texture | 0.99997729 | 0.99997723 | 3351 | 100.00% |

Julia ratio/mutual matches agreed exactly with OpenCV BFMatcher when supplied the same descriptor matrices. The known transformations give an independent geometric check:

| Transform | Julia / OpenCV matches | Julia / OpenCV inliers within 2 px | Recovered homography max probe error (px) |
|---|---:|---:|---:|
| translate | 625 / 627 | 100.00% / 100.00% | 0.018 |
| rotate90 | 676 / 675 | 99.85% / 99.85% | 0.012 |
| similarity | 305 / 305 | 94.43% / 94.43% | 0.171 |
| perspective | 387 / 386 | 96.90% / 96.89% | 0.075 |

## Acceptance gates

- More than 30 paired frames; at least 97% coverage in both directions.
- Position error p95 < 0.01 px; relative scale error p95 < 0.005; angle error p95 < 0.1°.
- Detected descriptor median cosine > 0.999; identical-frame cosine p05 > 0.995; every half-pixel probe cosine > 0.995.
- Cross-library matches at least 80% of the smaller feature count, with at least 95% geometric precision.
- Each transformed pair: at least 40 matches, at least 90% inliers within 2 px, and recovered homography probe error < 1.5 px.

## Scope and limitations

This is a synthetic reference suite, not a natural-image benchmark or proof of bitwise identity. Shape repetition produces some incorrect ratio-test matches in both implementations; RANSAC remains necessary for geometric use. Exact keypoint counts can differ around histogram peaks and plateau/tie cases. The descriptor comparison separates detector variation from descriptor arithmetic and quantization.

All 170 Julia tests pass under `Pkg.test()` with bounds checking enabled on Julia 1.13.1 (both single-threaded and multi-threaded), and hosted GitHub Actions CI executes both `Tests` and `Real photographs` workflows.

Run `python validation/validate.py` to regenerate the JSON. This Markdown report records the local run; the JSON includes minima, p05, medians, p95, maxima, and per-transform details. See the README for setup and numerical conventions.
