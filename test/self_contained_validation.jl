using Random: MersenneTwister
using StaticArrays: SMatrix, SVector
using Statistics: median

include(joinpath(@__DIR__, "..", "examples", "homography_ransac.jl"))

@noinline function test_synthetic_geometric_warps()
    src = synthetic_scene(256, 320; seed=7)
    h, w = size(src)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0

    # Ground-truth homographies mapping 1-based src (x, y) -> dst (u, v)
    T_center = SMatrix{3,3,Float64,9}(1, 0, 0,  0, 1, 0,  -cx, -cy, 1)
    T_uncenter = SMatrix{3,3,Float64,9}(1, 0, 0,  0, 1, 0,  cx, cy, 1)

    H_translate = SMatrix{3,3,Float64,9}(1, 0, 0,  0, 1, 0,  11.4, -7.6, 1)

    theta = deg2rad(27.0)
    c_th, s_th = cos(theta), sin(theta)
    H_rotate = T_uncenter * SMatrix{3,3,Float64,9}(c_th, s_th, 0,  -s_th, c_th, 0,  0, 0, 1) * T_center

    sc = 1.18
    theta2 = deg2rad(-19.0)
    c2, s2 = sc * cos(theta2), sc * sin(theta2)
    H_similarity = T_uncenter * SMatrix{3,3,Float64,9}(c2, s2, 0,  -s2, c2, 0,  6.2, -4.8, 1) * T_center

    H_perspective = T_uncenter * SMatrix{3,3,Float64,9}(
        1.04, 0.06, 0.00035,
        -0.05, 0.96, -0.00025,
        5.0, -4.0, 1.0
    ) * T_center

    cases = (
        ("translate",   H_translate,   150, 0.95, 0.4, 0.5),
        ("rotate27",    H_rotate,      120, 0.92, 0.5, 0.8),
        ("similarity",  H_similarity,  100, 0.90, 0.6, 1.0),
        ("perspective", H_perspective, 120, 0.92, 0.5, 0.8),
    )

    for use_root in (false, true)
        cfg = SIFTConfig(rootsift=use_root)
        kp1, d1 = sift(src; config=cfg)
        @test length(kp1) >= 200
        for (name, H_gt, min_matches, min_inlier_frac, max_med_err, max_probe_err) in cases
            dst = zeros(Float32, h, w)
            warp_perspective!(dst, src, inv(H_gt))
            kp2, d2 = sift(dst; config=cfg)
            m = match_descriptors(d1, d2; ratio=0.8, crosscheck=true)
            @test length(m) >= min_matches
            errs = Float64[]
            sizehint!(errs, length(m))
            for t in m
                p = project_point(H_gt, kp1[t.query].x, kp1[t.query].y)
                push!(errs, hypot(p[1] - kp2[t.train].x, p[2] - kp2[t.train].y))
            end
            inlier_frac = count(<(2.0), errs) / length(errs)
            @test inlier_frac >= min_inlier_frac
            @test median(errs) < max_med_err
            H_fit, inliers = ransac_homography(kp1, kp2, m; thresh_px=2.5, iters=1000, rng=MersenneTwister(11))
            @test count(inliers) >= round(Int, 0.8 * length(m))
            @test homography_probe_error(H_fit, H_gt, h, w) < max_probe_err
        end
    end
end

@noinline function test_v1_golden_regression_and_stress()
    # Cross-platform v1.0.0 golden anchors (OpenCV-validated baseline).
    # Tolerances absorb x86 AVX2 vs ARM64 NEON 1-ULP FMA rounding while catching real regressions.
    a = fixture(200, 240)
    golden = (
        (SIFTConfig(),                        206, 38.98513f0,  62.996128f0, 1.5126945f0, 1.5962884f0, 0.0786017f0,  -1, 3),
        (SIFTConfig(rootsift=true),           206, 38.98513f0,  62.996128f0, 1.5126945f0, 1.5962884f0, 0.0786017f0,  -1, 3),
        (SIFTConfig(nfeatures=50),             50, 38.98513f0,  62.996128f0, 1.5126945f0, 1.5962884f0, 0.0786017f0,  -1, 3),
        (SIFTConfig(first_octave=0),          189, 185.07753f0, 32.96733f0,  4.4959936f0, 0.9436812f0, 0.07697025f0,  1, 1),
        (SIFTConfig(contrast_threshold=0.02), 222, 38.98513f0,  62.996128f0, 1.5126945f0, 1.5962884f0, 0.0786017f0,  -1, 3),
        (SIFTConfig(edge_threshold=5.0),      193, 38.98513f0,  62.996128f0, 1.5126945f0, 1.5962884f0, 0.0786017f0,  -1, 3),
        (SIFTConfig(octave_layers=4),         205, 38.98567f0,  62.99629f0,  1.5583949f0, 1.5981562f0, 0.059062455f0,-1, 4),
        (SIFTConfig(sigma=2.0),               203, 38.98528f0,  62.996174f0, 1.5122576f0, 1.6204464f0, 0.07860332f0, -1, 2),
    )
    for (cfg, n_exp, y1, x1, s1, a1, r1, o1, l1) in golden
        kp, d = sift(a; config=cfg)
        @test abs(length(kp) - n_exp) <= 1
        @test size(d) == (128, length(kp))
        @test all(isfinite, d) && all(d .>= 0f0)
        @test all(abs.(vec(sum(abs2, d; dims=1)) .- 1f0) .< 2f-6)
        k1 = kp[1]
        @test k1.octave == o1 && k1.layer == l1
        @test abs(k1.y - y1) < 1f-3
        @test abs(k1.x - x1) < 1f-3
        @test abs(k1.sigma - s1) < 1f-3
        @test abs(k1.angle - a1) < 1f-3
        @test abs(k1.response - r1) < 1f-4
    end

    # Stress invariants: UInt16 bitwise parity, heavy noise, low light, odd panoramic shapes
    rng = MersenneTwister(1234)
    u8 = UInt8.(round.(a .* 255))
    u16 = UInt16.(u8) .* UInt16(257)
    @test sift(u8) == sift(u16)

    noisy = clamp.(a .+ Float32.(0.12 .* randn(rng, size(a))), 0f0, 1f0)
    dark = a .* 0.15f0
    wide = fixture(127, 511)
    tall = fixture(511, 127)
    for img in (noisy, dark, wide, tall)
        kp, d = sift(img)
        @test size(d) == (128, length(kp))
        @test all(isfinite, d) && all(d .>= 0f0)
        if !isempty(kp)
            @test all(abs.(vec(sum(abs2, d; dims=1)) .- 1f0) .< 2f-6)
        end
    end
end

@testset "Self-contained geometric warps, v1.0 golden anchors, and stress" begin
    test_synthetic_geometric_warps()
    test_v1_golden_regression_and_stress()
end
