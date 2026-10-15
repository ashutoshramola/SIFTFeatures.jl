# Implementation notes

The native Julia pipeline builds `S+3` Gaussian levels and `S+2` DoG levels per
octave, tests 26 neighbors, and iteratively solves a 3D quadratic for location
and scale. It rejects low contrast and spatial Hessian edge responses, and
deduplicates candidates converging to the same lattice extremum.

Orientation assignment uses a smoothed 36-bin histogram, interpolated peaks,
and an 80% peak cutoff. Descriptors use 4×4×8 histograms, Gaussian spatial
weighting, trilinear interpolation, clipping, and renormalization. Returned
keypoints retain subpixel locations; orientation and descriptor windows sample
around the nearest octave pixel.

## OpenCV compatibility

Comparison uses OpenCV 4.10 with precise upscaling explicitly enabled. OpenCV
disables that option by default. Its descriptor vectors are transposed and
unit-normalized for comparison; its floating output is quantized near 512 times
a unit vector, whereas Julia returns unquantized unit vectors.

Bitwise equality is not promised. Differences include exact versus approximate
`atan2`, accumulation order, strict plateau handling, feature-cap tie policy,
and a slightly different initial contrast prefilter. Julia uses `0.5*c/S`;
OpenCV floors the corresponding threshold after scaling by 255. The final
contrast rejection has the same parameter convention. No lost localized
extrema were observed in the tested contrast-prefilter probes.

Even-sized rotations can shift coarse sampling grids and change detected
features. Geometry tests therefore use tolerances rather than requiring equal
feature counts after every transformation.

## Audit follow-up

The initial independent Claude audit found no major default-pipeline algorithm
defect. Reproducers identified and motivated these repairs:

- Validation checks now survive optimized Python and preserve failed reports.
- Independent Julia tests anchor descriptor orientation, spatial bin order,
  interpolation, and clipping with analytical gradients and an impulse image.
- Explicit octave overrides that would produce singleton axes raise `ArgumentError`.
- Unique exact descriptor matches pass every positive ratio, including values
  whose square underflows; duplicate exact nearest neighbors still fail.
- Input-range documentation specifies validation after Float32 conversion.

Gaussian filtering uses a separable symmetric Float32 FIR filter with reflect-101
border handling and preallocated workspace buffers, 3D quadratic localization
solves use stack-allocated `StaticArrays` (`SVector`/`SMatrix`), and orientation,
descriptor, and matcher routines use branchless SIMD passes, padded trilinear
histogram splatting, and `4×1` register-blocked L2 distance accumulation with
deterministic multi-threaded chunking on Julia 1.13+.

## References

- D. Lowe, [Distinctive Image Features from Scale-Invariant Keypoints](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), 2004.
- [OpenCV pyramid implementation](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/sift.dispatch.cpp).
- [OpenCV orientation and descriptor implementation](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/sift.simd.hpp).
