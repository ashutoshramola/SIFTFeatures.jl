# Local correctness-first benchmark

Julia 1.13.4; graniterapids; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 1.26 | 1.24 | 2.4 |
| 128×128 | 77 | Full SIFT | 2.47 | 2.46 | 2.89 |
| 128×128 | 77 | Self matching + mutual check | 0.05 | 0.05 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 5.52 | 5.43 | 9.59 |
| 256×256 | 98 | Full SIFT | 8.92 | 8.85 | 10.91 |
| 256×256 | 98 | Self matching + mutual check | 0.09 | 0.08 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 24.34 | 24.26 | 38.34 |
| 512×512 | 156 | Full SIFT | 37.34 | 36.81 | 42.96 |
| 512×512 | 156 | Self matching + mutual check | 0.17 | 0.17 | 0.02 |
