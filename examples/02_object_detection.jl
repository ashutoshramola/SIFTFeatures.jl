# Example 2: Object / Template Localization Under Rotation, Scale & Partial Occlusion
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/02_object_detection.jl

include("homography_ransac.jl")

"""
    detect_object(template, scene; config=SIFTConfig(), ratio=0.8, thresh_px=2.5)

Locate `template` inside `scene` even under rotation, scale change, or partial occlusion.
Returns the 3×3 homography `H_template_to_scene`, the 4 projected corner coordinates
in `scene` (`corners_xy`), and match/inlier statistics.
"""
function detect_object(template::AbstractMatrix, scene::AbstractMatrix;
                       config::SIFTConfig=SIFTConfig(),
                       ratio::Real=0.8,
                       thresh_px::Real=2.5)
    kp_t, desc_t = sift(template; config)
    kp_s, desc_s = sift(scene; config)
    matches = match_descriptors(desc_t, desc_s; ratio, crosscheck=true)
    H, inliers = ransac_homography(kp_t, kp_s, matches; thresh_px)

    ht, wt = size(template)
    template_corners = (
        (1.0, 1.0),
        (Float64(wt), 1.0),
        (Float64(wt), Float64(ht)),
        (1.0, Float64(ht)),
    )
    scene_corners = map(c -> project_point(H, c[1], c[2]), template_corners)
    return (
        H=H,
        corners=scene_corners,
        matches=matches,
        inliers=inliers,
        kp_template=kp_t,
        kp_scene=kp_s,
    )
end

function run_object_detection_demo()
    println("=== Example 2: Object Localization Under 90° Rotation & 45% Occlusion ===")
    scene = synthetic_scene(360, 440)
    # Extract a 180×220 object template from the scene (rows 91:270, cols 111:330)
    y0, x0 = 91, 111
    ht, wt = 180, 220
    template_upright = scene[y0:(y0 + ht - 1), x0:(x0 + wt - 1)]

    # Rotate the template 90° counter-clockwise (rotl90)
    template_rot = rotl90(template_upright)

    # Occlude the right 45% of the object inside the scene image
    occluded_scene = copy(scene)
    occluded_scene[:, (x0 + 120):end] .= 0.1f0

    # Ground-truth mapping from 1-based template_rot (size wt × ht = 220 × 180) to scene:
    # In rotl90(A), pixel (r_u, c_u) of upright (ht × wt) moves to:
    #   r_rot = wt - c_u + 1,  c_rot = r_u
    # So x_rot = c_rot = r_u,  y_rot = r_rot = wt - c_u + 1 => c_u = wt + 1 - y_rot, r_u = x_rot
    # And in scene: x_scene = x0 - 1 + c_u = (x0 + wt) - y_rot,  y_scene = y0 - 1 + r_u = (y0 - 1) + x_rot
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
            " (", round(100 * n_inliers / max(1, length(res.matches)); digits=1), "% purity)")
    println("  Corner error       : ", round(err_px; digits=4), " px")
    println("  Detection time     : ", round(t_sec * 1000; digits=1), " ms")
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_object_detection_demo()
end
