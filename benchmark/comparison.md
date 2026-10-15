# Repeated Julia/OpenCV comparison — 2026-10-15

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
AMD EPYC 7763 64-Core Processor (`znver3`). Julia 1.13.2, OpenCV 4.10.0,
NumPy 2.4.6, Python 3.11.17; Julia/OpenCV/BLAS each use one thread.
ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 796.2 | 213.5 | 214.7 | 3.71× | 342.5 |
| trees_1 | 1,577.0 | 447.5 | 452.3 | 3.49× | 350.2 |
| graf_1 | 614.5 | 152.6 | 153.9 | 3.99× | 251.5 |
| wall_1 | 1,285.6 | 362.8 | 366.4 | 3.51× | 348.2 |
| bark_1 | 593.5 | 167.2 | 168.8 | 3.52× | 194.9 |
| boat_1 | 1,058.2 | 284.5 | 287.8 | 3.68× | 289.0 |
| leuven_1 | 573.2 | 141.1 | 142.2 | 4.03× | 264.8 |
| ubc_1 | 774.8 | 201.3 | 203.3 | 3.81× | 253.6 |

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
