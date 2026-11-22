# Example 2: Object / Template Localization Under Rotation, Scale & Partial Occlusion
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/02_object_detection.jl

@isdefined(ransac_homography) || include("homography_ransac.jl")

"""
    detect_object(template, scene; config=SIFTConfig(nfeatures=1000, rootsift=true),
                  ratio=0.8, thresh_px=2.5, min_inliers=8)

Locate `template` inside `scene` even under rotation, scale change, or partial occlusion.
Requires at least `min_inliers` (>= 4) distinct spatial inlier locations supporting the
fitted homography; otherwise returns `found=false`, `H=nothing`, and `corners=nothing`.
"""
function detect_object(template::AbstractMatrix, scene::AbstractMatrix;
                       config::SIFTConfig=SIFTConfig(nfeatures=1000, rootsift=true),
                       ratio::Real=0.8,
                       thresh_px::Real=2.5,
                       min_inliers::Int=8)
    min_inliers >= 4 || throw(ArgumentError("min_inliers must be at least 4"))
    kp_t, desc_t = sift(template; config)
    kp_s, desc_s = sift(scene; config)
    matches = match_descriptors(desc_t, desc_s; ratio, crosscheck=true)
    H_est, inliers = ransac_homography(kp_t, kp_s, matches; thresh_px)
    n_distinct = distinct_inlier_locations(kp_t, kp_s, matches, inliers)
    found = n_distinct >= min_inliers

    ht, wt = size(template)
    scene_corners = if found
        template_corners = (
            (1.0, 1.0),
            (Float64(wt), 1.0),
            (Float64(wt), Float64(ht)),
            (1.0, Float64(ht)),
        )
        map(c -> project_point(H_est, c[1], c[2]), template_corners)
    else
        nothing
    end

    return (
        found=found,
        H=found ? H_est : nothing,
        corners=scene_corners,
        matches=matches,
        inliers=inliers,
        distinct_inliers=n_distinct,
        kp_template=kp_t,
        kp_scene=kp_s,
    )
end

function run_object_detection_demo()
    println("=== Example 2: Object Localization Under 90° Rotation & 45% Occlusion ===")
    scene = synthetic_scene(360, 440; seed=1)
    # Extract a 180×220 object template from the scene (rows 91:270, cols 111:330)
    y0, x0 = 91, 111
    ht, wt = 180, 220
    template_upright = scene[y0:(y0 + ht - 1), x0:(x0 + wt - 1)]

    # Rotate the template 90° counter-clockwise (rotl90)
    template_rot = rotl90(template_upright)

    # Occlude the right 45% of the object inside the scene image
    occluded_scene = copy(scene)
    occluded_scene[:, (x0 + 120):end] .= 0.1f0

    H_gt = SMatrix{3,3,Float64,9}(
         0.0,  1.0, 0.0,
        -1.0,  0.0, 0.0,
        Float64(x0 + wt), Float64(y0 - 1), 1.0
    )

    detect_object(template_rot, occluded_scene)
    t_sec = @elapsed res = detect_object(template_rot, occluded_scene)

    err_px = homography_probe_error(res.H, H_gt, size(template_rot, 1), size(template_rot, 2))
    n_inliers = count(res.inliers)
    println("  Template keypoints : ", length(res.kp_template))
    println("  Scene keypoints    : ", length(res.kp_scene))
    println("  Matches / Inliers  : ", length(res.matches), " / ", n_inliers,
            " (", res.distinct_inliers, " distinct locations, ",
            round(100 * n_inliers / max(1, length(res.matches)); digits=1), "% purity)")
    println("  Corner error       : ", round(err_px; digits=4), " px")
    println("  Detection time     : ", round(t_sec * 1000; digits=1), " ms")

    # Rejection checks: blank scene and unrelated textured distractor
    res_blank = detect_object(template_rot, zeros(Float32, 360, 440))
    res_dist = detect_object(template_rot, synthetic_scene(360, 440; seed=99))
    println("  Blank scene        : found=", res_blank.found, " (corners=", res_blank.corners, ")")
    println("  Textured distractor: found=", res_dist.found,
            " (matches=", length(res_dist.matches), ", distinct_inliers=", res_dist.distinct_inliers, ")")
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_object_detection_demo()
end
