# Pure-Julia Validation & Benchmark Report — 2026-11-22

Runtime: Julia `1.13.4` (`graniterapids`, `1` thread(s)). Zero Python or OpenCV dependencies.

## Oxford 48-Photograph Ground-Truth Accuracy (40 pairs)

| Mode | Features | Matches | Correct (≤3 px) | Pooled Precision | Accurate Homographies |
|---|---:|---:|---:|---:|---:|
| SIFT (`ratio=0.80`) | 249542 | 38552 | 34989 | 90.758% | 33 / 40 |
| RootSIFT (`ratio=0.80`) | 249542 | 39588 | 36706 | 92.720% | 34 / 40 |

### Per-Sequence Breakdown (SIFT / RootSIFT)

| Scene | Correct Matches (SIFT) | Homographies (SIFT) | Correct Matches (RootSIFT) | Homographies (RootSIFT) |
|---|---:|---:|---:|---:|
| bikes | 1845 / 2109 | 5 / 5 | 1908 / 2129 | 5 / 5 |
| trees | 3722 / 4585 | 3 / 5 | 4061 / 4862 | 4 / 5 |
| graf | 1434 / 2013 | 3 / 5 | 1582 / 2046 | 3 / 5 |
| wall | 10473 / 11006 | 3 / 5 | 10876 / 11360 | 3 / 5 |
| bark | 2440 / 2615 | 5 / 5 | 2475 / 2586 | 5 / 5 |
| boat | 4914 / 5508 | 4 / 5 | 5211 / 5584 | 4 / 5 |
| leuven | 3084 / 3326 | 5 / 5 | 3239 / 3420 | 5 / 5 |
| ubc | 7077 / 7390 | 5 / 5 | 7354 / 7601 | 5 / 5 |

## Oxford Timing Benchmark (`1` thread(s))

| Photograph | Size (w×h) | Features | Median (ms) | Min (ms) | Allocated (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 1000×700 | 3224 | 108.1 | 105.8 | 116.8 |
| trees_1 | 1000×700 | 12477 | 222.6 | 219.0 | 124.3 |
| graf_1 | 800×640 | 2729 | 82.0 | 80.1 | 85.7 |
| wall_1 | 1000×700 | 9175 | 183.2 | 177.7 | 122.4 |
| bark_1 | 765×512 | 3757 | 78.3 | 77.0 | 67.5 |
| boat_1 | 850×680 | 8356 | 150.6 | 146.3 | 102.1 |
| leuven_1 | 900×600 | 2310 | 78.4 | 76.9 | 90.0 |
| ubc_1 | 800×640 | 5037 | 106.5 | 105.9 | 87.8 |

