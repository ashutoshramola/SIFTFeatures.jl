# Changelog

## 0.1.0 — unreleased

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
