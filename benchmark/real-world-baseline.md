# Real-photograph timing observations — 2026-10-15

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.1, ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23, OpenCV 4.10.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 988.5 / 3,226 | 0.540 | 0.107 | 340.6 | 0.032 |
| trees | 10497 / 15,234 | 1.220 | 0.320 | 349.0 | 1.789 |
| graf | 3562.5 / 4,568 | 0.585 | 0.153 | 252.6 | 0.134 |
| wall | 10175 / 10,403 | 1.006 | 0.286 | 300.0 | 1.366 |
| bark | 4180.5 / 4,589 | 0.520 | 0.140 | 195.2 | 0.226 |
| boat | 5530 / 8,357 | 0.679 | 0.179 | 285.7 | 0.563 |
| leuven | 1577.5 / 2,313 | 0.438 | 0.101 | 264.2 | 0.042 |
| ubc | 5554 / 7,221 | 0.628 | 0.179 | 253.9 | 0.426 |

The implementation uses in-place `imfilter!` into Float32 buffers, `StaticArrays` 3D localization, `Memory` work buffers, and `@simd` Float64 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
