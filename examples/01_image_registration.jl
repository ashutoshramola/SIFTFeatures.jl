# Example 1: Image Registration & Alignment (SIFT + Mutual Matching + RANSAC Homography)
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/01_image_registration.jl

include("homography_ransac.jl")

"""
    register_images(img_ref, img_target; config=SIFTConfig(), ratio=0.8, thresh_px=3.0)

Detect SIFT features in `img_ref` and `img_target`, match descriptors with a strict
Lowe ratio and mutual nearest-neighbor check, estimate the 3×3 homography `H_ref_to_target`
via RANSAC, and warp `img_target` back into `img_ref`'s coordinate frame.
"""
function register_images(img_ref::AbstractMatrix, img_target::AbstractMatrix;
                         config::SIFTConfig=SIFTConfig(),
                         ratio::Real=0.8,
                         thresh_px::Real=3.0)
    kp_ref, desc_ref = sift(img_ref; config)
    kp_target, desc_target = sift(img_target; config)
    matches = match_descriptors(desc_ref, desc_target; ratio, crosscheck=true)
    H_ref_to_target, inliers = ransac_homography(kp_ref, kp_target, matches; thresh_px)

    # To warp img_target back onto img_ref's grid, for each pixel (y, x) in aligned,
    # sample img_target at H_ref_to_target * [x, y, 1].
    aligned = Matrix{Float32}(undef, size(img_ref))
    warp_perspective!(aligned, SIFTFeatures.grayscale_float(img_target), H_ref_to_target)

    return (
        H=H_ref_to_target,
        matches=matches,
        inliers=inliers,
        kp_ref=kp_ref,
        kp_target=kp_target,
        aligned=aligned,
    )
end

function run_registration_demo()
    println("=== Example 1: Image Registration & Alignment ===")
    h, w = 320, 400
    img_ref = synthetic_scene(h, w)

    # Construct a known perspective + rotation + scale homography (1-based coordinates)
    theta = deg2rad(14.0)
    scale = 0.88
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0
    c, s = scale * cos(theta), scale * sin(theta)
    H_true = SMatrix{3,3,Float64,9}(
         c,      s,     0.00018,
        -s,      c,    -0.00012,
        cx - c * cx + s * cy + 12.0, cy - s * cx - c * cy + 8.0, 1.0
    )
    H_true /= H_true[3, 3]

    # Warp img_ref by H_true (inverse mapping uses inv(H_true))
    img_target = Matrix{Float32}(undef, h, w)
    warp_perspective!(img_target, img_ref, inv(H_true))

    # Warmup and run registration
    register_images(img_ref, img_target)
    t_sec = @elapsed res = register_images(img_ref, img_target)

    err_px = homography_probe_error(res.H, H_true, h, w)
    n_inliers = count(res.inliers)
    println("  Reference features : ", length(res.kp_ref))
    println("  Target features    : ", length(res.kp_target))
    println("  Mutual L2 matches  : ", length(res.matches))
    println("  RANSAC inliers     : ", n_inliers, " / ", length(res.matches),
            " (", round(100 * n_inliers / max(1, length(res.matches)); digits=1), "%)")
    println("  Mean probe error   : ", round(err_px; digits=4), " px")
    println("  End-to-end time    : ", round(t_sec * 1000; digits=1), " ms")
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_registration_demo()
end
