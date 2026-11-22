# Changelog

## 1.1.1 — 2026-11-23

- Fixed overflow (`Float16` / large floats) and underflow (tiny `Float32` / subnormals) in `rootsift!` and `rootsift` by scaling through `sqrt(peak)` in widened precision while preserving zero allocations.
- Avoided quadratic `2n × 2n` left-singular allocation in `fit_homography_dlt` (`examples/homography_ransac.jl`) by using thin SVD (`full=(n == 4)`) for `n >= 5` pairs.
- Included HPatches accuracy tables in Markdown reports written by `validation/eval_pure.jl --write-report` and added regression tests for all three fixes.

## 1.1.0 — 2026-11-22

- **100% pure-Julia self-sufficient testing & evaluation** (removed Python/OpenCV harness dependencies):
  - Added `Aqua.jl` package quality assurance suite (`test`-only dependency).
  - Added self-contained pure-Julia synthetic geometric warp tests (translation, rotation, similarity, perspective for both SIFT and RootSIFT), cross-platform `v1.0.0` golden regression anchors across 8 `SIFTConfig` presets, and stress invariants (`UInt16`, noise, low-light, panoramic).
  - Added pure-Julia dataset downloader/SHA-256 verifier (`validation/download_datasets.jl` + `validation/oxford-sources.toml`) and pure-Julia real-world ground-truth evaluator (`validation/eval_pure.jl`) with native `.ppm`/`.pgm` decoding, area resizing, and Hartley-normalized DLT + 2-pass RANSAC homography evaluation for Oxford (48 photos) and HPatches (116 scenes / 696 photos).
  - Simplified `ci.yml`, `eval.yml`, and `extended-eval.yml` to run 100% in pure Julia.

## 1.0.0 — 2026-11-22

- **SIMD performance pipeline (`0.67–0.77×` OpenCV 5.0.0 single-threaded)**:
  - 8-wide SIMD 6-neighbor axial extremum prefilter in `_scan_octave_columns!`.
  - Flat-patch SIMD orientation assignment in `OrientationWorkspace` / `_orientations!`.
  - L1-compact branchless flat-patch `_descriptor!` with 2-wide `SVector{2,Float32}` scatter into a `6×6×10` padded histogram.
- **Optional RootSIFT (Hellinger) descriptors**: added `SIFTConfig(rootsift=true)`, `descriptor(...; rootsift=true)`, and exported `rootsift` / `rootsift!` helpers for higher matching precision and homography accuracy while preserving OpenCV-identical defaults.
- **Public reusable workspaces & hardened validation**: exported/public `OrientationWorkspace` and `DescriptorWorkspace` with dynamic growth and zero-allocation reuse; tightened input validation across `gaussian_blur!`, `orientations!`, `descriptor!`, `localize`, and `dog_pyramid`.
- **Correctness & memory-safety fixes**:
  - Fixed out-of-bounds descriptor workspace sizing for large-scale keypoints after early octave-level release.
  - Replaced `_exp_neg_f32` with an exact Remez degree-6 minimax fit (`< 4e-7` abs error on `[0, 1.5625]`).
  - Wrapped all raw-pointer SIMD loops in `GC.@preserve`.
  - Reduced out-of-range descriptor angles in `Float64` (`rem2pi`) before `Float32` conversion.
- **Packaging, CI & evaluation**:
  - Replaced `ImageCore` with direct `Colors` + `FixedPointNumbers` dependencies; added `.gitattributes`; expanded CI across Ubuntu (`-t 1` & `-t auto`), macOS, Windows, and `examples/run_all.jl`.
  - Added manual-only `Extended Evaluation & HPatches` workflow (`extended-eval.yml`) and `validation/extended_eval.py` covering non-default `SIFTConfig`s, stress tests, multi-thread scaling, and the 116-scene HPatches benchmark.

## 0.2.0 — 2026-10-16

- Julia 1.13+ native optimizations: separable symmetric Float32 FIR Gaussian
  blur with DoG buffer reuse, `StaticArrays` 3D localization, branchless SIMD
  orientation and descriptor kernels with padded trilinear splatting, `4×1`
  register-blocked Float32 L2 matching, and bitwise-deterministic multi-threading.
- Removed `ImageFiltering` dependency; added `public` API declarations and
  in-place `gaussian_blur!`, `orientations!`, and `descriptor!`.

## 0.1.0

- Native Julia Gaussian/DoG scale space, subpixel localization, edge rejection,
  multiple orientation assignment, and 128-dimensional SIFT descriptors.
- L2 matching with strict Lowe ratio and optional mutual nearest neighbors.
- Focused JuliaImages dependencies and one-based AbstractMatrix input APIs.
- Analytical Julia tests plus OpenCV comparison on synthetic images and eight
  photographic benchmark sequences with supplied homographies.
- Explicit octave-size guards, unique-exact-match handling for tiny ratios,
  and independent descriptor-layout/interpolation anchors.
- Validation reports with failure checks active in optimized Python, negative
  controls, input/source hashes, and stale-result protection.
- Julia/OpenCV CI, an optional photographic workflow, and timing observations.
