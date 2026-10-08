# Local correctness-first benchmark

Julia 1.13.1; tigerlake; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 14.36 | 13.19 | 11.43 |
| 128×128 | 77 | Full SIFT | 28.31 | 27.41 | 13.4 |
| 128×128 | 77 | Self matching + mutual check | 0.91 | 0.83 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 63.41 | 61.13 | 42.35 |
| 256×256 | 98 | Full SIFT | 84.85 | 81.23 | 49.44 |
| 256×256 | 98 | Self matching + mutual check | 1.55 | 1.42 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 299.26 | 275.66 | 163.97 |
| 512×512 | 156 | Full SIFT | 344.7 | 329.23 | 191.37 |
| 512×512 | 156 | Self matching + mutual check | 3.78 | 3.4 | 0.02 |
