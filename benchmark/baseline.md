# Local correctness-first benchmark

Julia 1.13.4; znver3; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 2.09 | 2.01 | 2.4 |
| 128×128 | 77 | Full SIFT | 3.59 | 3.45 | 2.89 |
| 128×128 | 77 | Self matching + mutual check | 0.08 | 0.07 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 7.87 | 7.86 | 9.59 |
| 256×256 | 98 | Full SIFT | 11.5 | 11.37 | 10.91 |
| 256×256 | 98 | Self matching + mutual check | 0.12 | 0.1 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 38.07 | 37.55 | 38.34 |
| 512×512 | 156 | Full SIFT | 48.89 | 48.72 | 42.96 |
| 512×512 | 156 | Self matching + mutual check | 0.28 | 0.26 | 0.02 |
