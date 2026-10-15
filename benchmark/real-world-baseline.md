# Real-photograph timing observations — 2026-10-15

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.2, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23, OpenCV 4.10.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.219 | 0.158 | 117.5 | 0.020 |
| trees | 10498.5 / 15,232 | 0.482 | 0.446 | 125.6 | 1.086 |
| graf | 3562.5 / 4,567 | 0.238 | 0.189 | 88.5 | 0.085 |
| wall | 10175.5 / 10,405 | 0.424 | 0.369 | 108.6 | 0.795 |
| bark | 4179.5 / 4,589 | 0.205 | 0.184 | 69.0 | 0.134 |
| boat | 5530 / 8,356 | 0.293 | 0.235 | 100.9 | 0.348 |
| leuven | 1577.5 / 2,311 | 0.172 | 0.130 | 91.4 | 0.029 |
| ubc | 5553.5 / 7,221 | 0.268 | 0.228 | 89.9 | 0.260 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
