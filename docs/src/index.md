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

# Accepts Float32/Float64 in [0, 1], UInt8/UInt16, or JuliaImages Gray/RGB/RGBA matrices.
img = Float32[exp(-((x - 50.3)^2 + (y - 49.7)^2) / 32) for y in 1:101, x in 1:101]

# Detect keypoints and 128×N unit-L2 Float32 descriptors
keypoints, descriptors = sift(img)

# Match descriptors across views with Lowe's strict ratio test and mutual check
kp2, desc2 = sift(rotl90(img))
matches = match_descriptors(descriptors, desc2; ratio=0.8, crosscheck=true)

# Optional RootSIFT (Hellinger) descriptors for higher matching accuracy
kp_r, desc_r = sift(img; config=SIFTConfig(rootsift=true))
```
