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

Detects features across two views using `SIFTConfig(nfeatures=1000, rootsift=true)`, finds mutual L2 matches with [`match_descriptors`](@ref), fits a 3×3 homography with RANSAC (`examples/homography_ransac.jl`), and warps the target onto the reference grid—returning `found=false` and `H=nothing` when an unrelated distractor lacks sufficient distinct spatial inliers.

```julia
res = register_images(img_ref, img_target; min_inliers=8)
if res.found
    # res.H and res.aligned are verified by >= 8 distinct spatial inlier locations
end
```

## 2. Object / Template Localization (`examples/02_object_detection.jl`)

Locates a planar template inside a cluttered scene under 90° rotation and 45% occlusion, and explicitly demonstrates rejection (`found=false`, `corners=nothing`) on both a blank scene and an unrelated textured distractor.

## 3. Visual Place Recognition (`examples/03_place_recognition.jl`)

Indexes reference scenes with a bounded feature budget (`SIFTConfig(nfeatures=400, rootsift=true)`) and ranks database candidates by mutual match count, returning `best_match=nothing` whenever the top score falls below `min_matches` (rejecting both blank queries and unrelated textured distractors that produce a few accidental matches).

## 4. Multi-Frame Visual Odometry (`examples/04_visual_odometry.jl`)

Tracks sequential frame-to-frame homographies across a multi-frame trajectory and accumulates pose transforms over multiple hops. For high-throughput video tracking, disabling `2×` upsampling (`SIFTConfig(first_octave=0, nfeatures=800)`) reduces first-octave pixel count by 4×.

## When to Trust vs. Reject a Match

Two practical rules prevent false positives when connecting SIFT matching to downstream geometry:

1. **Count distinct spatial locations, not raw SIFT instances**: a single circular blob can emit up to 8 orientation keypoints at the same `(x, y)` coordinate. `distinct_inlier_locations` deduplicates integer `(x, y)` coordinates so geometric support reflects genuinely distinct points across the image.
2. **Test against an unrelated textured distractor, not just a blank image**: blank images yield zero matches, but an unrelated textured photo can easily produce 1–5 accidental mutual matches. Requiring `min_matches >= 10` for retrieval or `min_inliers >= 8` distinct RANSAC inliers for homographies rejects both cleanly.
