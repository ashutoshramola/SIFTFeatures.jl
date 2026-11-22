# Real-photograph timing observations — 2026-11-22

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.4, ColorTypes 0.12.5, Colors 0.13.4, FixedPointNumbers 0.8.6, StaticArrays 1.9.25, OpenCV 5.0.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.132 | 0.125 | 114.9 | 0.021 |
| trees | 10498.5 / 15,232 | 0.296 | 0.427 | 123.2 | 1.089 |
| graf | 3562.5 / 4,567 | 0.127 | 0.188 | 86.7 | 0.085 |
| wall | 10175.5 / 10,405 | 0.273 | 0.382 | 106.5 | 0.807 |
| bark | 4179.5 / 4,589 | 0.131 | 0.173 | 67.7 | 0.137 |
| boat | 5530 / 8,356 | 0.175 | 0.239 | 98.8 | 0.349 |
| leuven | 1577.5 / 2,311 | 0.098 | 0.132 | 89.5 | 0.029 |
| ubc | 5553.5 / 7,221 | 0.160 | 0.229 | 88.1 | 0.264 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
