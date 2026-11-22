```@meta
CurrentModule = SIFTFeatures
```

# API Reference

## Exported Types and Functions

```@docs
SIFTConfig
SIFTKeypoint
DescriptorMatch
sift
match_descriptors
rootsift
rootsift!
gaussian_pyramid
dog_pyramid
```

## Public Low-Level Kernels and Workspaces

These symbols are declared `public` (Julia 1.11+) and accessed as `SIFTFeatures.<name>`.

```@docs
OrientationWorkspace
DescriptorWorkspace
validate
grayscale_float
double_image
gaussian_blur
gaussian_blur!
default_octaves
is_extremum
derivatives
localize
orientations
orientations!
descriptor
descriptor!
angle32
```
