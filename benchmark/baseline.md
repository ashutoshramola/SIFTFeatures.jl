# Local correctness-first benchmark

Julia 1.13.2; znver3; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 11.22 | 10.89 | 7.14 |
| 128×128 | 77 | Full SIFT | 19.65 | 19.54 | 8.88 |
| 128×128 | 77 | Self matching + mutual check | 0.19 | 0.18 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 45.36 | 45.2 | 26.46 |
| 256×256 | 98 | Full SIFT | 60.04 | 59.95 | 33.22 |
| 256×256 | 98 | Self matching + mutual check | 0.29 | 0.28 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 159.16 | 157.44 | 101.97 |
| 512×512 | 156 | Full SIFT | 188.51 | 182.06 | 128.78 |
| 512×512 | 156 | Self matching + mutual check | 0.68 | 0.51 | 0.02 |
