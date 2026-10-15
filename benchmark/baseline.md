# Local correctness-first benchmark

Julia 1.13.2; znver3; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 3.07 | 3.04 | 2.4 |
| 128×128 | 77 | Full SIFT | 5.88 | 5.86 | 2.81 |
| 128×128 | 77 | Self matching + mutual check | 0.07 | 0.07 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 13.31 | 13.3 | 9.59 |
| 256×256 | 98 | Full SIFT | 18.7 | 18.64 | 11.01 |
| 256×256 | 98 | Self matching + mutual check | 0.1 | 0.1 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 47.31 | 47.05 | 38.34 |
| 512×512 | 156 | Full SIFT | 59.31 | 59.13 | 43.82 |
| 512×512 | 156 | Self matching + mutual check | 0.24 | 0.24 | 0.02 |
