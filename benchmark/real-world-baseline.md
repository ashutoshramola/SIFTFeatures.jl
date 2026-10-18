# Real-photograph timing observations — 2026-10-18

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.2, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23, OpenCV 4.10.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.197 | 0.133 | 117.5 | 0.016 |
| trees | 10498.5 / 15,232 | 0.428 | 0.383 | 125.6 | 0.847 |
| graf | 3562.5 / 4,567 | 0.209 | 0.167 | 88.5 | 0.066 |
| wall | 10175.5 / 10,405 | 0.386 | 0.325 | 108.6 | 0.623 |
| bark | 4179.5 / 4,589 | 0.182 | 0.163 | 69.0 | 0.105 |
| boat | 5530 / 8,356 | 0.253 | 0.208 | 100.9 | 0.272 |
| leuven | 1577.5 / 2,311 | 0.155 | 0.117 | 91.4 | 0.022 |
| ubc | 5553.5 / 7,221 | 0.248 | 0.202 | 89.9 | 0.204 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
