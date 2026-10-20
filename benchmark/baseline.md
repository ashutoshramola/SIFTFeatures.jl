# Local correctness-first benchmark

Julia 1.13.2; znver4; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 3.36 | 3.33 | 2.4 |
| 128×128 | 77 | Full SIFT | 6.38 | 6.35 | 2.81 |
| 128×128 | 77 | Self matching + mutual check | 0.07 | 0.07 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 14.59 | 14.43 | 9.59 |
| 256×256 | 98 | Full SIFT | 20.55 | 20.41 | 11.01 |
| 256×256 | 98 | Self matching + mutual check | 0.1 | 0.1 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 56.29 | 56.14 | 38.34 |
| 512×512 | 156 | Full SIFT | 69.01 | 68.81 | 43.83 |
| 512×512 | 156 | Self matching + mutual check | 0.24 | 0.23 | 0.02 |
