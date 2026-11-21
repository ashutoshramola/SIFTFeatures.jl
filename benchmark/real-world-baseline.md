# Real-photograph timing observations — 2026-11-21

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.4, ColorTypes 0.12.5, Colors 0.13.4, FixedPointNumbers 0.8.6, StaticArrays 1.9.25, OpenCV 5.0.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.134 | 0.084 | 114.8 | 0.013 |
| trees | 10498.5 / 15,232 | 0.375 | 0.315 | 123.0 | 0.827 |
| graf | 3562.5 / 4,567 | 0.156 | 0.139 | 86.6 | 0.057 |
| wall | 10175.5 / 10,405 | 0.327 | 0.281 | 106.4 | 0.606 |
| bark | 4179.5 / 4,589 | 0.145 | 0.138 | 67.6 | 0.100 |
| boat | 5530 / 8,356 | 0.199 | 0.174 | 98.7 | 0.272 |
| leuven | 1577.5 / 2,311 | 0.104 | 0.096 | 89.3 | 0.019 |
| ubc | 5553.5 / 7,221 | 0.184 | 0.168 | 88.0 | 0.202 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
