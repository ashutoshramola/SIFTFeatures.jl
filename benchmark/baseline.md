# Local correctness-first benchmark

Julia 1.13.2; raptorlake; 1 Julia thread(s).
Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.

| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |
|---|---:|---|---:|---:|---:|
| 128×128 | 77 | Gaussian pyramid | 2.25 | 1.9 | 2.4 |
| 128×128 | 77 | Full SIFT | 4.3 | 3.65 | 2.81 |
| 128×128 | 77 | Self matching + mutual check | 0.04 | 0.04 | 0.01 |
| 256×256 | 98 | Gaussian pyramid | 8.21 | 6.8 | 9.59 |
| 256×256 | 98 | Full SIFT | 11.94 | 10.96 | 11.01 |
| 256×256 | 98 | Self matching + mutual check | 0.07 | 0.05 | 0.01 |
| 512×512 | 156 | Gaussian pyramid | 34.67 | 28.02 | 38.34 |
| 512×512 | 156 | Full SIFT | 38.6 | 37.69 | 43.83 |
| 512×512 | 156 | Self matching + mutual check | 0.15 | 0.15 | 0.02 |
