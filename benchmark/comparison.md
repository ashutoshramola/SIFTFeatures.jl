# Repeated Julia/OpenCV comparison — 2026-10-20

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
AMD EPYC 9V74 80-Core Processor (`znver4`). Julia 1.13.2, OpenCV 5.0.0,
NumPy 2.5.4, Python 3.13.16; Julia/OpenCV/BLAS each use one thread.
ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 273.5 | 204.6 | 205.6 | 1.33× | 119.4 |
| trees_1 | 595.7 | 451.7 | 455.3 | 1.31× | 126.8 |
| graf_1 | 221.0 | 143.2 | 144.0 | 1.53× | 87.5 |
| wall_1 | 477.8 | 358.5 | 361.6 | 1.32× | 124.9 |
| bark_1 | 212.8 | 161.2 | 162.3 | 1.31× | 68.8 |
| boat_1 | 382.8 | 281.9 | 284.3 | 1.35× | 104.1 |
| leuven_1 | 208.4 | 126.8 | 127.4 | 1.64× | 91.9 |
| ubc_1 | 283.7 | 193.8 | 195.2 | 1.45× | 89.6 |

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
