# Contributing

Run `Pkg.test()` in the package environment before submitting a change. Detector
and descriptor changes should also run `validation/validate.py`; use the Oxford
sequence runner when changes could affect photographic matching.

Keep input APIs based on `AbstractMatrix`, floating subpixel keypoints, and
128×N Float32 descriptors. Runtime dependencies should remain focused on the
JuliaImages components already used by the package.

Tests should anchor behavior independently: analytical images, known geometric
transforms, and external reference comparisons. Self-matching and unit norms
alone cannot establish descriptor-bin order. Reference validation must continue
to fail with deliberately corrupted descriptors under normal and optimized Python.

Explain algorithmic changes using the cited SIFT paper or reference source.
Record relevant runtime versions and numerical tolerances. Profile before
optimizing, and repeat reference checks after any numerical change.
