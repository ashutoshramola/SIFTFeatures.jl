# Local correctness-first benchmark

Julia 1.13.4; graniterapids; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 1.25 | 1.2 | 2.4 |
| 128×128 | 77 | Full SIFT | 3.49 | 3.44 | 2.75 |
| 128×128 | 77 | Self matching + mutual check | 0.05 | 0.05 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 5.77 | 5.57 | 9.59 |
| 256×256 | 98 | Full SIFT | 9.69 | 9.65 | 10.76 |
| 256×256 | 98 | Self matching + mutual check | 0.08 | 0.07 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 24.51 | 23.1 | 38.34 |
| 512×512 | 156 | Full SIFT | 37.05 | 36.34 | 42.83 |
| 512×512 | 156 | Self matching + mutual check | 0.19 | 0.17 | 0.02 |
