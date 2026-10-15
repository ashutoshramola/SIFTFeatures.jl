# Local correctness-first benchmark

Julia 1.13.2; znver3; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 10.86 | 10.69 | 7.14 |
| 128×128 | 77 | Full SIFT | 19.53 | 19.4 | 8.88 |
| 128×128 | 77 | Self matching + mutual check | 0.18 | 0.16 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 44.74 | 44.69 | 26.46 |
| 256×256 | 98 | Full SIFT | 59.49 | 59.38 | 33.22 |
| 256×256 | 98 | Self matching + mutual check | 0.28 | 0.24 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 159.21 | 158.4 | 101.97 |
| 512×512 | 156 | Full SIFT | 182.2 | 181.67 | 128.78 |
| 512×512 | 156 | Self matching + mutual check | 0.69 | 0.52 | 0.02 |
