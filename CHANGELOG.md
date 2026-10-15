# Changelog

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
