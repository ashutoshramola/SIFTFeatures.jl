# Contributing

Thank you for your interest in contributing to `SIFTFeatures.jl`.

## Before Submitting a Pull Request

Run the pure-Julia test suite (including `Aqua.jl`, synthetic geometric warps,
`v1.0.0` golden anchors, and stress invariants) with bounds checking enabled on
both single-threaded and multi-threaded Julia:

```sh
julia --startup-file=no --project=. --check-bounds=yes -t 1 -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. --check-bounds=yes -t auto -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. -t auto examples/run_all.jl
```

For changes affecting detection, orientation, descriptors, RootSIFT, or matching,
also run the pure-Julia Oxford 48-photograph ground-truth evaluation (and
optionally the 116-scene HPatches benchmark):

```sh
julia --startup-file=no --project=. -t auto validation/eval_pure.jl \
    --oxford validation-data --download --compare-rootsift --benchmark
```

If you modify docstrings or files under `docs/src/`, verify that the
`Documenter.jl` site builds cleanly with zero warnings:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Design and Quality Guidelines

- **APIs and types**: Keep input APIs based on one-based `AbstractMatrix`,
  floating-point subpixel `SIFTKeypoint{Float32}`, and `128×N` unit-L2 `Float32`
  descriptors. Keep default `SIFTConfig()` behavior consistent with the
  OpenCV-verified `v1.0.0` baseline; opt-in extensions (like `rootsift=true`)
  must remain backwards-compatible.
- **Minimal dependencies**: Keep runtime dependencies focused on `ColorTypes`,
  `Colors`, `FixedPointNumbers`, `LinearAlgebra`, and `StaticArrays`—do not add
  heavy image frameworks, Python, or external binary dependencies.
- **Determinism and memory safety**: Multi-threaded paths must remain
  bit-for-bit identical between `-t 1` and `-t auto`. Wrap any raw-pointer SIMD
  operations in `GC.@preserve`, and keep preallocated workspace kernels
  zero-allocation on reuse.
- **Independent tests**: Anchor new behavior with analytical images, known
  geometric transforms, and regression checks—self-matching and unit norms alone
  cannot verify descriptor-bin ordering or scale-space accuracy.
- **Performance changes**: Profile before optimizing, avoid regressions across
  single- and multi-threaded runs, and verify accuracy on the Oxford benchmark
  after any numerical change.
