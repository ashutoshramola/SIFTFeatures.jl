# Example 4: Multi-Frame Sequential Visual Odometry (Frame-to-Frame Motion Chaining)
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/04_visual_odometry.jl

include("homography_ransac.jl")

"""
    track_sequence(frames; config=SIFTConfig(nfeatures=800), ratio=0.8, thresh_px=2.5)

Track camera motion across a sequence of images `frames[1..N]` by matching each frame
`t-1` to `t`, estimating the step homography `H_{t-1 -> t}` via RANSAC, and chaining
`H_{1 -> t} = H_{t-1 -> t} * H_{1 -> t-1}`.
"""
function track_sequence(frames::AbstractVector{<:AbstractMatrix};
                        config::SIFTConfig=SIFTConfig(nfeatures=800),
                        ratio::Real=0.8,
                        thresh_px::Real=2.5)
    isempty(frames) && throw(ArgumentError("frames must not be empty"))
    n = length(frames)
    cumulative_H = Vector{SMatrix{3,3,Float64,9}}(undef, n)
    step_inliers = Vector{Int}(undef, n - 1)
    cumulative_H[1] = SMatrix{3,3,Float64,9}(I)

    prev_kp, prev_desc = sift(frames[1]; config)
    for t in 2:n
        cur_kp, cur_desc = sift(frames[t]; config)
        matches = match_descriptors(prev_desc, cur_desc; ratio, crosscheck=true)
        H_step, inliers = ransac_homography(prev_kp, cur_kp, matches; thresh_px)
        step_inliers[t - 1] = count(inliers)
        cumulative_H[t] = H_step * cumulative_H[t - 1]
        cumulative_H[t] /= cumulative_H[t][3, 3]
        prev_kp, prev_desc = cur_kp, cur_desc
    end
    return (cumulative_H=cumulative_H, step_inliers=step_inliers)
end

function run_visual_odometry_demo()
    println("=== Example 4: Multi-Frame Sequential Visual Odometry (5 Hops) ===")
    h, w = 280, 360
    base = synthetic_scene(h, w)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0

    # Generate a 6-frame camera trajectory with progressive rotation + translation + scale
    nframes = 6
    frames = Vector{Matrix{Float32}}(undef, nframes)
    gt_H = Vector{SMatrix{3,3,Float64,9}}(undef, nframes)
    frames[1] = base
    gt_H[1] = SMatrix{3,3,Float64,9}(I)

    for t in 2:nframes
        ang = deg2rad(2.5 * (t - 1))
        sc = 1.0 - 0.02 * (t - 1)
        c, s = sc * cos(ang), sc * sin(ang)
        tx = 4.0 * (t - 1)
        ty = 2.5 * (t - 1)
        H = SMatrix{3,3,Float64,9}(
             c,  s, 0.0,
            -s,  c, 0.0,
            cx - c * cx + s * cy + tx, cy - s * cx - c * cy + ty, 1.0
        )
        gt_H[t] = H
        frame = Matrix{Float32}(undef, h, w)
        warp_perspective!(frame, base, inv(H))
        frames[t] = frame
    end

    track_sequence(frames)
    t_sec = @elapsed res = track_sequence(frames)

    drift_px = homography_probe_error(res.cumulative_H[end], gt_H[end], h, w)
    println("  Frames tracked     : ", nframes, " (", nframes - 1, " sequential frame-to-frame hops)")
    println("  Step inliers       : ", join(res.step_inliers, " -> "))
    println("  Final 5-hop drift  : ", round(drift_px; digits=4), " px")
    println("  Avg time per hop   : ", round(t_sec * 1000 / (nframes - 1); digits=1), " ms")
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_visual_odometry_demo()
end
