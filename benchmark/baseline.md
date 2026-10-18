# Local correctness-first benchmark

Julia 1.13.2; znver4; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 2.64 | 2.62 | 2.4 |
| 128×128 | 77 | Full SIFT | 4.89 | 4.87 | 2.81 |
| 128×128 | 77 | Self matching + mutual check | 0.06 | 0.05 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 11.58 | 11.45 | 9.59 |
| 256×256 | 98 | Full SIFT | 15.94 | 15.83 | 11.01 |
| 256×256 | 98 | Self matching + mutual check | 0.08 | 0.08 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 46.09 | 46.04 | 38.34 |
| 512×512 | 156 | Full SIFT | 56.24 | 56.14 | 43.82 |
| 512×512 | 156 | Self matching + mutual check | 0.2 | 0.19 | 0.02 |
