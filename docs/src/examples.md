```@meta
CurrentModule = SIFTFeatures
```

# Examples

The repository includes four self-contained workflow scripts under `examples/` that require no extra packages beyond `SIFTFeatures.jl`'s environment.

Run all four examples at once from a repository checkout:

```sh
julia --threads=auto --project=. examples/run_all.jl
```

## 1. Image Registration & Alignment (`examples/01_image_registration.jl`)

Detects features across two transformed views, finds mutual L2 matches with [`match_descriptors`](@ref), estimates a 3×3 homography using RANSAC (`examples/homography_ransac.jl`), and warps the target image onto the reference grid.

```julia
kp_ref, desc_ref = sift(img_ref)
kp_tgt, desc_tgt = sift(img_target)
matches = match_descriptors(desc_ref, desc_tgt; ratio=0.8, crosscheck=true)
```

## 2. Object / Template Localization (`examples/02_object_detection.jl`)

Locates a planar template inside a larger cluttered scene under 90° rotation and partial occlusion, projecting the four template corners via the fitted homography.

## 3. Visual Place Recognition (`examples/03_place_recognition.jl`)

Indexes reference scenes with a bounded feature budget (`SIFTConfig(nfeatures=400)`) and ranks database candidates by mutual match count against a query image.

```julia
cfg = SIFTConfig(nfeatures=400, rootsift=true)
kp, desc = sift(scene; config=cfg)
```

## 4. Multi-Frame Visual Odometry (`examples/04_visual_odometry.jl`)

Tracks sequential frame-to-frame homographies across a multi-frame trajectory and accumulates pose transforms over multiple hops. For high-throughput video tracking, disabling `2×` upsampling (`SIFTConfig(first_octave=0, nfeatures=800)`) reduces first-octave pixel count by 4×.
