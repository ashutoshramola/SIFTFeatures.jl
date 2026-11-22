```@meta
CurrentModule = SIFTFeatures
```

# User Guide

## Input Types and Conventions

[`sift`](@ref) and [`gaussian_pyramid`](@ref) accept any one-based `AbstractMatrix` (including views and transposes) containing:

- **Floating-point or real intensities** in `[0, 1]` (not rescaled; out-of-range or non-finite values throw `ArgumentError`).
- **Unsigned integers** (`UInt8`, `UInt16`, `UInt32`, `UInt64`), automatically normalized by `typemax`.
- **JuliaImages color types** (`Gray`, `RGB`, `RGBA`, `Normed`), converted to luminance via `ColorTypes.Gray(color(v))` with alpha ignored.

Offset-indexed arrays are rejected with `ArgumentError`. Empty or constant images return an empty keypoint vector and a `128×0` `Matrix{Float32}`.

### Coordinate, Scale, and Angle Conventions

Each [`SIFTKeypoint`](@ref) stores `(y, x, sigma, angle, response, octave, layer)`:

| Field | Convention | OpenCV Equivalent |
|---|---|---|
| `y`, `x` | One-based `(row, col)` pixel centers in the original image; `(1, 1)` is the top-left pixel | `pt = (x - 1, y - 1)` |
| `sigma` | Gaussian standard deviation in original-image pixels | `size = 2 * sigma` |
| `angle` | Clockwise radians from `+x` in `[0, 2π)` | `angle = rad2deg(angle)` |
| `octave` | Signed octave index (`-1` when `first_octave = -1`) | `octave & 255` (signed int8) |
| `layer` | Zero-based scale layer within the octave | `(octave >> 8) & 255` |

Descriptors are returned as a `128×N Matrix{Float32}` with one unit-L2 column per keypoint. Within each column, orientation bin (`0..7`) varies fastest, followed by spatial column (`0..3`) and spatial row (`0..3`).

## Configuring `SIFTConfig`

[`SIFTConfig`](@ref) uses OpenCV-compatible defaults:

```julia
config = SIFTConfig(
    nfeatures = 0,             # 0 = unlimited; positive caps strongest orientation instances
    octave_layers = 3,         # DoG intervals per octave
    contrast_threshold = 0.04, # Divided internally by octave_layers
    edge_threshold = 10.0,     # Lower values reject more edge-like responses
    sigma = 1.6,               # Base Gaussian sigma at the first octave
    assumed_input_sigma = 0.5, # Assumed blur of the input image
    first_octave = -1,         # -1 doubles input (2×); 0 starts at original resolution
    rootsift = false,          # true enables RootSIFT (Hellinger) descriptors
)
kp, desc = sift(img; config)
```

### Recommended Presets

| Workload | Suggested `SIFTConfig` |
|---|---|
| **Maximum recall / photogrammetry** | `SIFTConfig()` or `SIFTConfig(rootsift=true)` |
| **Image retrieval / database indexing** | `SIFTConfig(nfeatures=500, rootsift=true)` |
| **Low-latency video / robotics** | `SIFTConfig(first_octave=0, nfeatures=500)` |
| **Dark / low-contrast imagery** | `SIFTConfig(contrast_threshold=0.02, rootsift=true)` |

## RootSIFT (Hellinger) Descriptors

RootSIFT (Arandjelović & Zisserman, 2012) L1-normalizes each nonnegative SIFT descriptor and takes the elementwise square root. Because $\sum_k (\sqrt{d_k / \|d\|_1})^2 = 1$, RootSIFT descriptors remain unit-L2 normalized, so standard Euclidean [`match_descriptors`](@ref) computes the Hellinger kernel directly.

You can enable RootSIFT either during extraction or as a post-processing transform:

```julia
# Computed directly inside sift() / descriptor!():
kp, desc = sift(img; config=SIFTConfig(rootsift=true))

# Or transform existing SIFT descriptors (out-of-place or in-place):
desc_root = rootsift(desc)
rootsift!(desc)
```

On Oxford (40 pairs) and HPatches (580 pairs) at `ratio=0.8`, RootSIFT improves pooled matching precision by ~2 percentage points (from ~90.8% to ~92.8%) and increases correct matches by ~5%.

## Descriptor Matching

[`match_descriptors`](@ref) performs exact brute-force L2 matching with Lowe's **strict** ratio test (`best < ratio * second`):

```julia
matches = match_descriptors(desc_query, desc_train; ratio=0.8, crosscheck=true)
for m in matches
    kp_q = kp_query[m.query]
    kp_t = kp_train[m.train]
    # m.distance is the Euclidean (non-squared) L2 distance
end
```

- `crosscheck=true` retains only mutual nearest neighbors (without applying a reverse ratio test).
- Fewer than two training descriptors or tied nearest neighbors yield no match.

## Multi-Threading and Determinism

Launch Julia with `julia -t auto` (or `-t 4`, `-t 8`, etc.) to parallelize grayscale conversion, `2×` upsampling, Gaussian blurring, extremum scanning, orientation assignment, descriptor extraction, and descriptor matching automatically.

All multi-threaded stages use deterministic chunking and reduction order so outputs are **bit-for-bit identical** across thread counts.

## Zero-Allocation Workspaces

For tight loops calling low-level public kernels directly, reuse [`OrientationWorkspace`](@ref) and [`DescriptorWorkspace`](@ref):

```julia
gp = gaussian_pyramid(img)
ori_ws = SIFTFeatures.OrientationWorkspace()
desc_ws = SIFTFeatures.DescriptorWorkspace()
angles = Float64[]
buf = Vector{Float32}(undef, 128)

SIFTFeatures.orientations!(angles, ori_ws, gp[1][2], 32, 32, 1.6)
SIFTFeatures.descriptor!(buf, desc_ws, gp[1][2], 32.0, 32.0, 1.6, angles[1]; rootsift=true)
```

Both workspaces grow dynamically if an unusually large `sigma` exceeds their initial capacity and allocate zero bytes on subsequent calls.
