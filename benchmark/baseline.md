# Local correctness-first benchmark

Julia 1.13.4; znver3; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 1.65 | 1.64 | 2.4 |
| 128×128 | 77 | Full SIFT | 3.28 | 3.27 | 2.89 |
| 128×128 | 77 | Self matching + mutual check | 0.07 | 0.07 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 7.72 | 7.42 | 9.59 |
| 256×256 | 98 | Full SIFT | 11.27 | 11.23 | 10.91 |
| 256×256 | 98 | Self matching + mutual check | 0.11 | 0.1 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 37.36 | 37.19 | 38.34 |
| 512×512 | 156 | Full SIFT | 47.98 | 47.86 | 42.96 |
| 512×512 | 156 | Self matching + mutual check | 0.24 | 0.24 | 0.02 |
