# Repeated Julia/OpenCV comparison — 2026-10-15

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
AMD EPYC 7763 64-Core Processor (`znver3`). Julia 1.13.2, OpenCV 4.10.0,
NumPy 2.4.6, Python 3.11.17; Julia/OpenCV/BLAS each use one thread.
ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 242.2 | 214.6 | 216.0 | 1.12× | 119.3 |
| trees_1 | 520.5 | 444.1 | 449.0 | 1.16× | 126.7 |
| graf_1 | 193.2 | 152.1 | 153.2 | 1.26× | 87.5 |
| wall_1 | 417.4 | 359.3 | 362.9 | 1.15× | 124.8 |
| bark_1 | 186.2 | 163.6 | 165.1 | 1.13× | 68.8 |
| boat_1 | 334.6 | 284.1 | 287.3 | 1.16× | 104.0 |
| leuven_1 | 186.3 | 138.5 | 139.4 | 1.34× | 91.9 |
| ubc_1 | 244.9 | 196.8 | 198.8 | 1.23× | 89.6 |

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
