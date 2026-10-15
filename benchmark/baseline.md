# Local correctness-first benchmark

Julia 1.13.1; raptorlake; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 8.45 | 7.68 | 7.14 |
| 128×128 | 77 | Full SIFT | 15.3 | 14.58 | 8.88 |
| 128×128 | 77 | Self matching + mutual check | 0.1 | 0.09 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 34.91 | 34.4 | 26.47 |
| 256×256 | 98 | Full SIFT | 47.6 | 47.2 | 33.22 |
| 256×256 | 98 | Self matching + mutual check | 0.16 | 0.15 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 148.81 | 138.19 | 101.98 |
| 512×512 | 156 | Full SIFT | 173.46 | 160.84 | 128.79 |
| 512×512 | 156 | Self matching + mutual check | 0.39 | 0.34 | 0.02 |
