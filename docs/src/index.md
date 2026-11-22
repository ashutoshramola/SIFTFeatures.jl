```@meta
CurrentModule = SIFTFeatures
```

# SIFTFeatures.jl

SIFT and RootSIFT feature detection, description, and matching in pure Julia.

`SIFTFeatures.jl` is a native Julia 1.13+ implementation of David Lowe's Scale-Invariant Feature Transform (SIFT) with optional RootSIFT (Hellinger) descriptors, bitwise-deterministic multi-threading, and OpenCV-validated accuracy. It depends only on `ColorTypes`, `Colors`, `FixedPointNumbers`, `LinearAlgebra`, and `StaticArrays`—no Python or OpenCV runtime is required.

## Highlights

- **Fast pure-Julia SIMD kernels**: 0.67–0.77× OpenCV 5.0.0 runtime single-threaded on the Oxford photographic benchmark, scaling deterministically across threads.
- **OpenCV-validated accuracy**: ≥99.96% feature parity and 0.99998+ median descriptor cosine against OpenCV 5.0.0 across Oxford (48 photos) and HPatches (116 scenes / 696 photos).
- **Optional RootSIFT**: Set `SIFTConfig(rootsift=true)` or call [`rootsift`](@ref) / [`rootsift!`](@ref) for higher matching precision and homography accuracy at negligible cost.
- **Zero-allocation low-level kernels**: Public [`OrientationWorkspace`](@ref), [`DescriptorWorkspace`](@ref), [`gaussian_blur!`](@ref), [`orientations!`](@ref), [`descriptor!`](@ref), and [`rootsift!`](@ref) for custom pipelines.

## Installation

Requires Julia 1.13 or newer.

```julia
using Pkg
Pkg.add(url="https://github.com/ashutoshramola/SIFTFeatures.jl")
```

## Quickstart

```julia
using SIFTFeatures

# Non-periodic multi-scale textured image (also accepts UInt8/UInt16 or JuliaImages Gray/RGB)
img = Float32[
    0.5 + 0.25 * sin(0.0012 * x^2 + 0.09y) * cos(0.0015 * y^2 - 0.08x) +
          0.20 * exp(-((x - 70)^2 + (y - 55)^2) / 90) -
          0.20 * exp(-((x - 130)^2 + (y - 110)^2) / 140)
    for y in 1:161, x in 1:193
]

# Practical starting point for registration: capped features + RootSIFT descriptors
cfg = SIFTConfig(nfeatures=1000, rootsift=true)
kp1, desc1 = sift(img; config=cfg)
kp2, desc2 = sift(rotl90(img); config=cfg)
matches = match_descriptors(desc1, desc2; ratio=0.8, crosscheck=true)

# Verify recovered 90° rotation alignment: (x, y) -> (y, width + 1 - x)
w = size(img, 2)
errors = [hypot(kp2[m.train].x - kp1[m.query].y,
                kp2[m.train].y - (w + 1 - kp1[m.query].x)) for m in matches]
println("Matches: ", length(matches), ", within 1 px: ", count(<(1.0), errors))
```
