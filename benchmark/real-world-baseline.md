# Real-photograph timing observations — 2026-10-09

This extends the synthetic baseline with the eight Oxford sequences used for real-photo validation. Each row summarizes six original-resolution photographs with unlimited features. Timings are one observational sample per photograph after a separate Julia/OpenCV warmup, not a controlled multi-repeat speed benchmark.

Julia 1.13.1, ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.3, OpenCV 4.10.0; Julia and OpenCV each used one thread. Julia timings exclude file interchange. OpenCV timing includes detection, descriptor computation, conversion to the reference frame array, and unit normalization. Hardware and system load were not controlled; avoid treating these numbers as a universal speed ratio.

| Sequence | Features: median / maximum | Julia SIFT median (s) | OpenCV observation median (s) | Julia allocations median (MiB) | Julia matcher median (s) |
|---|---:|---:|---:|---:|---:|
| bikes | 989 / 3,226 | 0.803 | 0.219 | 511.1 | 0.274 |
| trees | 10498 / 15,233 | 1.680 | 0.561 | 551.2 | 15.628 |
| graf | 3562 / 4,571 | 0.842 | 0.223 | 393.6 | 1.667 |
| wall | 10176 / 10,404 | 1.628 | 0.413 | 478.2 | 16.437 |
| bark | 4180 / 4,590 | 0.836 | 0.209 | 305.8 | 2.780 |
| boat | 5530 / 8,357 | 1.010 | 0.275 | 444.0 | 7.079 |
| leuven | 1578 / 2,313 | 0.650 | 0.152 | 401.2 | 0.622 |
| ubc | 5554 / 7,221 | 0.976 | 0.264 | 395.8 | 5.621 |

The implementation retains the audited Float64 filtering intermediates and brute-force Float64 L2 distance accumulation. This test adds correctness evidence and records realistic feature densities; it does not optimize the implementation. SIMD/GPU work remains deferred.

[Full raw observations](../validation/real-world-results.json) and [geometric validation report](../docs/validation.md).
