# Repeated Julia/OpenCV comparison — 2026-11-22

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
AMD EPYC 7763 64-Core Processor (`znver3`). Julia 1.13.4, OpenCV 5.0.0,
NumPy 2.5.5, Python 3.13.18; Julia/OpenCV/BLAS each use one thread.
ColorTypes 0.12.5, Colors 0.13.4, FixedPointNumbers 0.8.6, StaticArrays 1.9.25.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 129.2 | 187.0 | 188.4 | 0.69× | 116.8 |
| trees_1 | 314.1 | 425.9 | 430.5 | 0.73× | 124.3 |
| graf_1 | 100.6 | 136.6 | 137.5 | 0.73× | 85.7 |
| wall_1 | 245.5 | 334.5 | 337.5 | 0.73× | 122.4 |
| bark_1 | 115.6 | 151.4 | 152.7 | 0.76× | 67.5 |
| boat_1 | 196.0 | 268.0 | 270.8 | 0.72× | 102.1 |
| leuven_1 | 93.8 | 122.4 | 123.2 | 0.76× | 90.0 |
| ubc_1 | 133.8 | 182.6 | 184.3 | 0.73× | 87.8 |

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
feature counts, and comparable output counts. All 32 explicit gates passed.
The independent accuracy and matching study uses all 48 images; see
[historical v1.0.0 validation details](../docs/validation.md).

> **Historical note (`v1.0.0`)**: Starting in `v1.1.0`, the repository removed the
> Python/OpenCV comparison harness to stay 100% pure Julia. To run the ongoing
> pure-Julia Oxford accuracy and timing suite:
>
> ```sh
> julia --project=. -t 1 validation/eval_pure.jl --oxford validation-data --download --compare-rootsift --benchmark
> ```
