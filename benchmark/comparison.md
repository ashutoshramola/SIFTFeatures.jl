# Repeated Julia/OpenCV comparison — 2026-10-15

Five measured samples after one warmup per image, using img1 from each of eight
Oxford sequences at original resolution. The samples run sequentially on
Intel(R) Core(TM) i9-14900HX (`raptorlake`). Julia 1.13.1, OpenCV 4.10.0,
NumPy 2.5.4, Python 3.15.0; Julia/OpenCV/BLAS each use one thread.
ImageFiltering 0.7.12, ImageCore 0.10.5, ColorTypes 0.12.4, StaticArrays 1.9.23.

| Photograph | Julia median (ms) | OpenCV raw median (ms) | OpenCV unit-output median (ms) | Julia / unit-output time | Julia minimum allocation (MiB) |
|---|---:|---:|---:|---:|---:|
| bikes_1 | 594.0 | 147.5 | 148.6 | 4.00× | 342.5 |
| trees_1 | 1,156.2 | 316.7 | 324.7 | 3.56× | 350.2 |
| graf_1 | 481.2 | 118.5 | 119.5 | 4.03× | 251.5 |
| wall_1 | 938.2 | 255.3 | 261.3 | 3.59× | 348.3 |
| bark_1 | 478.9 | 123.9 | 125.4 | 3.82× | 195.0 |
| boat_1 | 807.2 | 213.1 | 218.2 | 3.70× | 289.0 |
| leuven_1 | 465.4 | 105.1 | 105.8 | 4.40× | 264.8 |
| ubc_1 | 583.5 | 151.3 | 154.3 | 3.78× | 253.6 |

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
