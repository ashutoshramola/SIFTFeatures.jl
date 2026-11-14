# Repeated Julia/OpenCV comparison — 2026-11-14

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
AMD EPYC 9V74 80-Core Processor (`znver4`). Julia 1.13.4, OpenCV 5.0.0,
NumPy 2.5.4, Python 3.13.18; Julia/OpenCV/BLAS each use one thread.
ImageCore 0.10.5, ColorTypes 0.12.5, StaticArrays 1.9.25.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 155.0 | 164.1 | 164.9 | 0.94× | 116.7 |
| trees_1 | 384.3 | 369.1 | 372.3 | 1.03× | 124.2 |
| graf_1 | 135.6 | 116.7 | 117.3 | 1.16× | 85.6 |
| wall_1 | 317.5 | 292.1 | 294.4 | 1.08× | 122.2 |
| bark_1 | 146.2 | 130.1 | 131.0 | 1.12× | 67.3 |
| boat_1 | 254.2 | 227.7 | 229.8 | 1.11× | 101.9 |
| leuven_1 | 115.5 | 104.2 | 104.7 | 1.10× | 89.9 |
| ubc_1 | 173.4 | 157.5 | 158.6 | 1.09× | 87.7 |

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
