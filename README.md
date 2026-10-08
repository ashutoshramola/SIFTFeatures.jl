# SIFTFeatures.jl

Native Julia SIFT detection, floating-point subpixel keypoints, 128-dimensional
descriptors, and L2 matching. Uses `ImageCore`, `ImageFiltering`, and `ColorTypes`
with ordinary Julia arrays. The package runs without Python or OpenCV.

Version 0.1.0 is a reference implementation focused on correctness. OpenCV
comparison tests cover synthetic images and 48 photographs from eight Oxford
benchmark sequences. See [validation results](docs/validation.md) and
[implementation notes](docs/implementation.md).

## Installation

Requires Julia 1.10 or newer. From a local checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/SIFTFeatures.jl")
```

Once the repository is published, install the unregistered package by URL:

```julia
using Pkg
Pkg.add(url="https://github.com/ashutoshramola/SIFTFeatures.jl")
```

## Usage

```julia
using SIFTFeatures

# Loaded JuliaImages Gray/RGB matrices also work.
img = Float32[exp(-((x-50.3)^2+(y-49.7)^2)/32) for y in 1:101, x in 1:101]
keypoints, descriptors = sift(img)

kp2, desc2 = sift(rotl90(img))
matches = match_descriptors(descriptors, desc2; ratio=0.8, crosscheck=true)

config = SIFTConfig(nfeatures=1000, octave_layers=3,
                    contrast_threshold=0.04, edge_threshold=10, sigma=1.6)
kp, desc = sift(img; config)
```

## Data conventions

- Inputs are one-based `AbstractMatrix` values, including views and transposes,
  of real intensities, unsigned integers, or JuliaImages colors. Unsigned pixels
  are divided by their type's maximum. Other real values are not rescaled.
  RGB is converted to luminance and alpha is ignored. Converted Float32
  intensities must be finite and in `[0,1]`. Offset axes are unsupported.
- `SIFTKeypoint{Float32}` stores `y`, `x`, `sigma`, `angle`, `response`, `octave`,
  and `layer`. Coordinates refer to the original image; `(x=1,y=1)` is its first
  pixel center. `sigma` is Gaussian standard deviation in original-image pixels.
- Angles are clockwise radians from +x in `[0,2π)`. Octaves start at −1 by
  default, and layers are zero-based. OpenCV conversion is
  `pt=(x-1,y-1)`, `size=2sigma`, and `angle=rad2deg(angle)`.
- Descriptors are `128×N Matrix{Float32}`, one unit-L2 column per keypoint.
  Orientation varies fastest within each descriptor, followed by spatial
  column and row. Clipping at 0.2 precedes the final renormalization, which can
  make returned components exceed 0.2.
- Features are sorted by decreasing DoG response, with deterministic geometric
  tie-breaking. `nfeatures=0` means unlimited; a cap counts orientation instances.
- `DescriptorMatch` contains one-based `query`/`train` column indices and
  Euclidean distance. The strict Lowe ratio needs at least two train columns;
  tied nearest neighbors fail. `crosscheck=true` adds a reverse nearest-neighbor
  check, without a reverse ratio test. Reverse ties choose the first query.
- Empty and constant images return empty keypoints and a `128×0` descriptor matrix.

## Configuration

`first_octave=-1` uses aligned bilinear doubling, comparable to OpenCV with
`enable_precise_upscale=True`. Set it to 0 to disable doubling.
`assumed_input_sigma=0.5` specifies input blur; `sigma` must be at least the
effective input blur at the first octave. Gaussian filtering uses finite
four-sigma kernels and reflected borders.

The final contrast cutoff is divided by `octave_layers`, following OpenCV's
parameter convention. Lower `edge_threshold` values reject more edge responses.
Use `sift(img; octaves=n)` or `gaussian_pyramid(img, config; octaves=n)` to override
the automatic octave count. Every requested octave must retain at least two
pixels along each axis. `dog_pyramid(gaussian)` computes adjacent differences.

## Tests and reference validation

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. benchmark/run.jl
```

The Julia tests include analytical localization and descriptor-layout anchors,
orientation/interpolation checks, geometric consistency, and matcher edge cases.
CI runs Julia 1.10 and the current release on Linux.

Python is used only in the optional OpenCV comparison environment:

```sh
python -m pip install -r validation/requirements.txt
python -m unittest discover -s validation -p 'test_*.py'
python -O -m unittest discover -s validation -p 'test_*.py'
python validation/validate.py
python validation/check_guards.py
```

To reproduce the photographic tests with supplied homographies:

```sh
python validation/download_oxford.py --data-dir validation-data
python -O validation/real_world.py --data-dir validation-data \
    --work-dir validation-work --max-side 0
```

`--max-side 0` keeps original resolution; the default maximum side is 640.
`--sequences graf,leuven` selects a subset. `--julia /path/to/julia` selects the
runtime. Optional `--figures /path/to/figures` needs matplotlib and generates
local diagnostic match plots. Dataset archives and extracted images stay local.
The `Real photographs` GitHub workflow can run this benchmark manually.

Validation uses explicit checks that remain active under Python `-O`, writes
failures with a nonzero exit status, and replaces stale success reports before
starting. [Synthetic timing samples](benchmark/baseline.md) and
[photographic timing observations](benchmark/real-world-baseline.md) are recorded;
no SIMD/GPU optimization has been added.

## Development and references

The implementation follows [Lowe's SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf),
with selected [OpenCV 4.10 conventions](https://docs.opencv.org/4.10.0/d7/d60/classcv_1_1SIFT.html).
Photographic data comes from the
[Oxford affine covariant regions benchmark](https://www.robots.ox.ac.uk/~vgg/research/affine/).

Development used OpenAI Codex for implementation and execution, and Claude for
an independent code audit. Maintainer review remains part of preparation for
General registration. See [contribution guidance](CONTRIBUTING.md).

MIT licensed; the package is not yet registered in General.
