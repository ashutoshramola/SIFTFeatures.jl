# Repeated Julia/OpenCV comparison — 2026-10-09

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz. Julia 1.13.1, OpenCV 4.10.0,
NumPy 2.3.5, Python 3.14.4; Julia/OpenCV/BLAS each use one thread.
ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.3.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 1,247.5 | 222.2 | 224.7 | 5.55× | 525.8 |
| trees_1 | 2,417.9 | 474.1 | 486.4 | 4.97× | 557.9 |
| graf_1 | 941.1 | 166.2 | 167.6 | 5.62× | 388.3 |
| wall_1 | 1,917.8 | 353.0 | 361.6 | 5.30× | 546.3 |
| bark_1 | 866.7 | 167.9 | 170.6 | 5.08× | 304.8 |
| boat_1 | 1,578.9 | 297.8 | 305.3 | 5.17× | 457.0 |
| leuven_1 | 976.3 | 152.1 | 153.1 | 6.38× | 405.4 |
| ubc_1 | 1,239.8 | 226.9 | 230.4 | 5.38× | 393.7 |

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
