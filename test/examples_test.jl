module ExampleWorkflowTests

using Test
using SIFTFeatures

include(joinpath(@__DIR__, "..", "examples", "01_image_registration.jl"))
include(joinpath(@__DIR__, "..", "examples", "02_object_detection.jl"))
include(joinpath(@__DIR__, "..", "examples", "03_place_recognition.jl"))

@testset "Example workflows: success, blank, and textured distractor rejection" begin
    # 1. Image registration: known warp succeeds, blank and unrelated distractor reject
    h, w = 240, 300
    ref = synthetic_scene(h, w; seed=1)
    tgt = rotl90(ref)
    res_ok = register_images(ref, tgt)
    @test res_ok.found
    @test res_ok.H !== nothing
    @test res_ok.aligned !== nothing
    @test res_ok.distinct_inliers >= 40

    res_blank = register_images(ref, zeros(Float32, h, w))
    @test !res_blank.found
    @test res_blank.H === nothing
    @test res_blank.aligned === nothing
    @test res_blank.distinct_inliers == 0

    res_dist = register_images(ref, synthetic_scene(h, w; seed=99))
    @test !res_dist.found
    @test res_dist.H === nothing
    @test res_dist.aligned === nothing
    @test_throws ArgumentError register_images(ref, tgt; min_inliers=3)

    # 2. Object detection: occluded rotated template found; blank & distractor rejected
    scene = synthetic_scene(300, 360; seed=1)
    tpl = rotl90(scene[71:220, 91:260])
    occ = copy(scene)
    occ[:, 190:end] .= 0.1f0
    det_ok = detect_object(tpl, occ)
    @test det_ok.found
    @test det_ok.H !== nothing
    @test det_ok.corners !== nothing && length(det_ok.corners) == 4
    @test det_ok.distinct_inliers >= 20

    det_blank = detect_object(tpl, zeros(Float32, 300, 360))
    @test !det_blank.found
    @test det_blank.H === nothing
    @test det_blank.corners === nothing

    det_dist = detect_object(tpl, synthetic_scene(300, 360; seed=99))
    @test !det_dist.found
    @test det_dist.H === nothing
    @test det_dist.corners === nothing
    @test_throws ArgumentError detect_object(tpl, occ; min_inliers=2)

    # Single-location multi-orientation blob must NOT satisfy min_inliers=8 distinct locations
    blob = Float32[exp(-((x - 50.3)^2 + (y - 49.7)^2) / 32) for y in 1:101, x in 1:101]
    det_blob = detect_object(blob, blob; min_inliers=8)
    @test length(det_blob.matches) >= 4
    @test det_blob.distinct_inliers == 1
    @test !det_blob.found && det_blob.corners === nothing

    # 3. Place recognition: distorted match retrieved; blank & textured distractor return nothing
    cfg = SIFTConfig(nfeatures=400, rootsift=true)
    scenes = [
        "scene_alpha" => synthetic_scene(256, 320; seed=1),
        "scene_beta"  => synthetic_scene(256, 320; seed=2),
        "scene_gamma" => synthetic_scene(256, 320; seed=3),
        "scene_delta" => synthetic_scene(256, 320; seed=4),
    ]
    db = build_image_database(scenes; config=cfg)

    q_good = clamp.(SIFTFeatures.gaussian_blur(rotl90(scenes[3].second), 1.4) .* 0.75f0 .+ 0.05f0, 0f0, 1f0)
    r_good = query_image_database(db, q_good; config=cfg, min_matches=10)
    @test r_good.best_match == "scene_gamma"
    @test r_good.best_count >= 100

    r_blank = query_image_database(db, zeros(Float32, 256, 320); config=cfg)
    @test r_blank.best_match === nothing
    @test r_blank.best_count == 0

    r_dist = query_image_database(db, synthetic_scene(256, 320; seed=99); config=cfg, min_matches=10)
    @test r_dist.best_match === nothing
    @test r_dist.best_count < 10
    @test_throws ArgumentError query_image_database(db, q_good; config=cfg, min_matches=0)
end

end
