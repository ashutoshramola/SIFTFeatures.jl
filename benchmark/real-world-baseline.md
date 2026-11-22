# Real-photograph timing observations — 2026-11-22

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.4, ColorTypes 0.12.5, Colors 0.13.4, FixedPointNumbers 0.8.6, StaticArrays 1.9.25, OpenCV 5.0.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.141 | 0.122 | 114.9 | 0.020 |
| trees | 10498.5 / 15,232 | 0.279 | 0.420 | 123.2 | 1.101 |
| graf | 3562.5 / 4,567 | 0.133 | 0.183 | 86.7 | 0.085 |
| wall | 10175.5 / 10,405 | 0.268 | 0.375 | 106.5 | 0.804 |
| bark | 4179.5 / 4,589 | 0.131 | 0.180 | 67.7 | 0.138 |
| boat | 5530 / 8,356 | 0.177 | 0.237 | 98.8 | 0.354 |
| leuven | 1577.5 / 2,311 | 0.090 | 0.131 | 89.5 | 0.029 |
| ubc | 5553.5 / 7,221 | 0.159 | 0.228 | 88.1 | 0.264 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
