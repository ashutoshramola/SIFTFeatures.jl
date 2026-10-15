# Repeated Julia/OpenCV comparison — 2026-10-15

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (`raptorlake`). Julia 1.13.2, OpenCV 4.10.0,
NumPy 2.5.4, Python 3.15.0; Julia/OpenCV/BLAS each use one thread.
ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 201.1 | 164.4 | 165.7 | 1.21× | 119.3 |
| trees_1 | 420.6 | 343.0 | 350.2 | 1.20× | 126.7 |
| graf_1 | 166.1 | 121.0 | 123.7 | 1.34× | 87.5 |
| wall_1 | 352.5 | 284.3 | 289.3 | 1.22× | 124.8 |
| bark_1 | 162.3 | 140.1 | 141.5 | 1.15× | 68.8 |
| boat_1 | 274.2 | 242.4 | 247.7 | 1.11× | 104.0 |
| leuven_1 | 144.3 | 115.9 | 116.9 | 1.23× | 91.9 |
| ubc_1 | 193.2 | 169.5 | 171.4 | 1.13× | 89.6 |

Julia times `sift` on a UInt8 matrix, including intensity conversion, detection,
and 128×N unquantized unit Float32 descriptors. OpenCV raw times
`detectAndCompute` through the Python binding, with default SIFT parameters
except `enable_precise_upscale=True`; its descriptor values are quantized.
Unit-output timings additionally transpose-copy descriptors to 128×N and
normalize columns to unit L2. Conversion of keypoints into a separate reference
frame array is excluded. Descriptor matching and homography fitting are not timed.

Image decoding, file interchange, compilation/warmups, forced GC before each
sample, output checks, and report writing are excluded. GC triggered within
the timed call is included. Julia allocations are total allocated bytes per
call, not peak memory. OpenCV memory is not measured. The two libraries run in
separate image blocks, Julia first. CPU frequency and system load are not
controlled, so the ratios apply only to this run.

This differs from `real-world-baseline.md`: that earlier report summarizes one
observation on each of six images per sequence and includes OpenCV keypoint-frame
conversion. Image selection and timing scope differ; the results should not be
compared as an optimization or regression.

The benchmark checks pinned image hashes, descriptor shape/norms, repeatable
feature counts, and comparable output counts. All 32 explicit gates passed.
The [JSON](comparison-results.json) records every timing and allocation sample,
input/source hashes, package versions, configuration, and measurement scope.
The independent accuracy and matching study uses all 48 images; see
[validation results](../docs/validation.md).

## Reproduction

From the package directory, after installing `validation/requirements.txt` and
instantiating the Julia project:

```sh
python validation/download_oxford.py --data-dir validation-data
python benchmark/compare.py --data-dir validation-data --samples 5 --warmups 1
```

`--julia /path/to/julia` selects the runtime. `--work-dir` selects scratch storage;
the default `benchmark-work/` directory is ignored. `--output` selects the JSON
report destination. Images and decoded interchange data remain local.
