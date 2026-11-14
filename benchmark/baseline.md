# Local correctness-first benchmark

Julia 1.13.4; znver4; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 1.34 | 1.34 | 2.4 |
| 128×128 | 77 | Full SIFT | 3.58 | 3.52 | 2.75 |
| 128×128 | 77 | Self matching + mutual check | 0.06 | 0.05 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 6.75 | 6.6 | 9.59 |
| 256×256 | 98 | Full SIFT | 10.99 | 10.86 | 10.76 |
| 256×256 | 98 | Self matching + mutual check | 0.08 | 0.08 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 32.29 | 31.84 | 38.34 |
| 512×512 | 156 | Full SIFT | 44.55 | 43.37 | 42.81 |
| 512×512 | 156 | Self matching + mutual check | 0.19 | 0.19 | 0.02 |
