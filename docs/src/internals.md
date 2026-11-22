```@meta
CurrentModule = SIFTFeatures
```

# Implementation & Architecture

`SIFTFeatures.jl` follows Lowe (2004) with OpenCV-compatible conventions (`enable_precise_upscale=True`).

## Scale-Space Pipeline

1. **Input normalization & doubling**: [`grayscale_float`](@ref) validates finite `[0, 1]` intensities into `Matrix{Float32}`. When `first_octave = -1`, [`double_image`](@ref) upsamples `2×` on a pixel-aligned grid (`x -> 2x`).
2. **Gaussian & DoG pyramids**: [`gaussian_blur!`](@ref) applies a separable symmetric 4-sigma `Float32` FIR filter with reflect-101 border handling using 32-row SIMD strips (`4 × SVector{8,Float32}`) and preallocated buffers. Each octave builds `S + 3` Gaussian levels and `S + 2` Difference-of-Gaussians levels, reusing pure-DoG levels in-place and releasing them immediately after scanning.
3. **SIMD extremum detection & 3D localization**: `_scan_octave_columns!` runs an 8-wide SIMD 6-neighbor axial prefilter (an exact necessary condition for a 26-neighbor extremum that rejects >96% of 8-pixel blocks) before verifying all 26 neighbors and solving the 3D quadratic Taylor expansion with stack-allocated `StaticArrays` (`SVector{3,Float64}`, `SMatrix{3,3,Float64,9}`). Candidates converging to the same discrete extremum are deduplicated.
4. **Flat-patch SIMD orientation assignment**: [`orientations!`](@ref) packs the 2D Gaussian-weighted gradient window into contiguous L1-resident buffers in [`OrientationWorkspace`](@ref), computes `atan2` and magnitude in one flat SIMD sweep, smooths the 36-bin histogram, and extracts all interpolated peaks ≥ 80% of the maximum.
5. **L1-compact branchless descriptor extraction**: [`descriptor!`](@ref) packs valid rotated-patch columns flat into [`DescriptorWorkspace`](@ref), transforms gradients and trilinear weights in-place across 6 arrays using a Remez degree-6 minimax `exp(-x)` fit (`< 4e-7` error on `[0, 1.5625]`) and branchless `atan2`, scatters via 2-wide `SVector{2,Float32}` updates into a `6×6×10` padded histogram, clips at `0.2`, and normalizes (L2 for SIFT, or L1 + `sqrt` for RootSIFT).
6. **Register-blocked L2 matching**: [`match_descriptors`](@ref) uses `4×1` register-blocked `Float32` squared-distance accumulation with deterministic chunked parallelization.

## OpenCV Compatibility & Differences

- **Precision**: `SIFTFeatures.jl` returns unquantized `Float32` unit vectors, whereas OpenCV quantizes descriptors near `512×` a unit vector.
- **Exactness**: Bitwise equality with OpenCV is not promised due to `atan2` approximation, floating-point accumulation order, strict plateau handling, and initial contrast prefilter rounding (`0.5 * c / S` vs OpenCV's floored 8-bit prefilter).
- **Validation**: Continuous validation covers synthetic geometric transforms, the 48-photograph Oxford affine benchmark (`eval.yml`), and an opt-in manual extended suite covering non-default `SIFTConfig`s, stress cases (`UInt16`, noise, low light, panoramic aspect ratios), multi-thread scaling, and the 116-scene HPatches benchmark (`extended-eval.yml`).

## References

- D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, International Journal of Computer Vision 60(2):91–110, 2004.
- R. Arandjelović and A. Zisserman, *Three things everyone should know to improve object retrieval*, CVPR 2012 (RootSIFT).
- K. Mikolajczyk et al., *A comparison of affine region detectors*, IJCV 65(1/2):43–72, 2005 (Oxford benchmark).
- V. Balntas et al., *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*, CVPR 2017.
