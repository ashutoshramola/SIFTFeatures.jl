# Real-photograph timing observations — 2026-11-14

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.4, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23, OpenCV 5.0.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 987.5 / 3,224 | 0.153 | 0.136 | 114.8 | 0.016 |
| trees | 10498.5 / 15,232 | 0.371 | 0.385 | 123.0 | 0.871 |
| graf | 3562.5 / 4,567 | 0.165 | 0.163 | 86.6 | 0.066 |
| wall | 10175.5 / 10,405 | 0.340 | 0.321 | 106.4 | 0.628 |
| bark | 4179.5 / 4,589 | 0.159 | 0.151 | 67.6 | 0.105 |
| boat | 5530 / 8,356 | 0.206 | 0.203 | 98.7 | 0.272 |
| leuven | 1577.5 / 2,311 | 0.107 | 0.112 | 89.3 | 0.022 |
| ubc | 5553.5 / 7,221 | 0.211 | 0.198 | 88.0 | 0.203 |

The implementation uses a separable symmetric Float32 FIR Gaussian filter with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD orientation/descriptor kernels with padded trilinear splatting, and `4×1` register-blocked Float32 L2 distance accumulation.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
