#!/usr/bin/env julia
"""
Pure-Julia real-world ground-truth SIFT & RootSIFT evaluator and benchmark suite
(zero Python, zero OpenCV, zero external packages beyond SIFTFeatures + stdlibs).

Evaluates Oxford (48 photos / 40 pairs) and/or HPatches (116 scenes / 696 photos / 580 pairs)
directly against each dataset's published ground-truth homographies (`H1toNp` / `H_1_N`),
and optionally runs repeated single/multi-threaded timing benchmarks and writes a Markdown report.

Usage:
    julia --project=. -t auto validation/eval_pure.jl --oxford validation-data --compare-rootsift
    julia --project=. -t auto validation/eval_pure.jl --oxford validation-data --download --benchmark --write-report validation/REPORT.md
    julia --project=. -t auto validation/eval_pure.jl --hpatches hpatches-data --download --compare-rootsift
"""

using SIFTFeatures
using Dates: Dates, today
using LinearAlgebra: inv, norm
using Printf: @printf, @sprintf
using Random: MersenneTwister
using StaticArrays: SMatrix, SVector
using Statistics: median

include(joinpath(@__DIR__, "..", "examples", "homography_ransac.jl"))
include(joinpath(@__DIR__, "download_datasets.jl"))

# --- Pure-Julia Netpbm (.ppm / .pgm, P5/P6) binary image decoder ---
function _read_pnm_token(io::IO)
    buf = UInt8[]
    while !eof(io)
        b = read(io, UInt8)
        if b == UInt8('#')
            readline(io)
            continue
        elseif b in (UInt8(' '), UInt8('\t'), UInt8('\n'), UInt8('\r'))
            !isempty(buf) && break
        else
            push!(buf, b)
        end
    end
    return String(buf)
end

function read_pnm_gray(path::AbstractString)
    open(path, "r") do io
        magic = _read_pnm_token(io)
        (magic == "P5" || magic == "P6") || error("Unsupported Netpbm format $magic in $path")
        w = parse(Int, _read_pnm_token(io))
        h = parse(Int, _read_pnm_token(io))
        maxval = parse(Int, _read_pnm_token(io))
        maxval == 255 || error("Only 8-bit Netpbm supported (got maxval=$maxval in $path)")
        if magic == "P5"
            raw = read(io, w * h)
            return permutedims(reshape(raw, w, h))
        else
            raw = read(io, 3 * w * h)
            img = Matrix{UInt8}(undef, h, w)
            # ITU-R BT.601 luminance
            idx = 1
            @inbounds for y in 1:h, x in 1:w
                r = raw[idx]; g = raw[idx + 1]; b = raw[idx + 2]
                img[y, x] = round(UInt8, 0.299 * r + 0.587 * g + 0.114 * b)
                idx += 3
            end
            return img
        end
    end
end

# --- Exact pixel-center box/area downsampler with homography transform matrix C ---
function resize_gray(img::Matrix{UInt8}, max_side::Int)
    h, w = size(img)
    if max_side <= 0 || max(h, w) <= max_side
        return img, SMatrix{3,3,Float64,9}(1, 0, 0,  0, 1, 0,  0, 0, 1)
    end
    scale = max_side / max(h, w)
    hh = max(2, round(Int, h * scale))
    ww = max(2, round(Int, w * scale))
    sx = ww / w
    sy = hh / h
    C = SMatrix{3,3,Float64,9}(sx, 0.0, 0.0,  0.0, sy, 0.0,  (sx - 1.0) / 2.0, (sy - 1.0) / 2.0, 1.0)
    out = Matrix{UInt8}(undef, hh, ww)
    inv_sx = w / ww
    inv_sy = h / hh
    @inbounds for ox in 1:ww
        x0 = (ox - 1) * inv_sx
        x1 = ox * inv_sx
        ix0 = max(1, floor(Int, x0) + 1)
        ix1 = min(w, ceil(Int, x1))
        for oy in 1:hh
            y0 = (oy - 1) * inv_sy
            y1 = oy * inv_sy
            iy0 = max(1, floor(Int, y0) + 1)
            iy1 = min(h, ceil(Int, y1))
            acc = 0.0
            wsum = 0.0
            for ix in ix0:ix1
                wx = min(Float64(ix), x1) - max(Float64(ix - 1), x0)
                wx <= 0.0 && continue
                for iy in iy0:iy1
                    wy = min(Float64(iy), y1) - max(Float64(iy - 1), y0)
                    wy <= 0.0 && continue
                    wt = wx * wy
                    acc += wt * img[iy, ix]
                    wsum += wt
                end
            end
            out[oy, ox] = clamp(round(Int, acc / wsum), 0, 255)
        end
    end
    return out, C
end

function read_homography(path::AbstractString)
    vals = Float64[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        for tok in split(s)
            push!(vals, parse(Float64, tok))
        end
    end
    length(vals) == 9 || error("Expected 9 floats in $path, got $(length(vals))")
    return SMatrix{3,3,Float64,9}(
        vals[1], vals[4], vals[7],
        vals[2], vals[5], vals[8],
        vals[3], vals[6], vals[9]
    )
end

function score_pair(kp1, d1, kp2, d2, H_gt::SMatrix{3,3,Float64,9}, h1::Int, w1::Int;
                    ratio::Float64=0.8, ransac_iters::Int=3000)
    m = match_descriptors(d1, d2; ratio=ratio, crosscheck=true)
    correct = 0
    @inbounds for t in m
        p = project_point(H_gt, kp1[t.query].x - 1.0, kp1[t.query].y - 1.0)
        dx = p[1] - (kp2[t.train].x - 1.0)
        dy = p[2] - (kp2[t.train].y - 1.0)
        if dx * dx + dy * dy < 9.0
            correct += 1
        end
    end
    accurate_h = false
    probe_err = Inf
    if length(m) >= 4
        H_fit, inliers = ransac_homography(kp1, kp2, m; thresh_px=3.0, iters=ransac_iters,
                                           zero_based=true, rng=MersenneTwister(7))
        if count(inliers) >= 4
            t2 = 9.0
            in2 = BitVector(undef, length(m))
            @inbounds for (idx, t) in enumerate(m)
                p = project_point(H_fit, kp1[t.query].x - 1.0, kp1[t.query].y - 1.0)
                dx = p[1] - (kp2[t.train].x - 1.0)
                dy = p[2] - (kp2[t.train].y - 1.0)
                in2[idx] = (dx * dx + dy * dy < t2)
            end
            if count(in2) >= 4
                pa = Matrix{Float64}(undef, count(in2), 2)
                pb = Matrix{Float64}(undef, count(in2), 2)
                row = 1
                @inbounds for (idx, t) in enumerate(m)
                    if in2[idx]
                        pa[row, 1] = kp1[t.query].x - 1.0; pa[row, 2] = kp1[t.query].y - 1.0
                        pb[row, 1] = kp2[t.train].x - 1.0; pb[row, 2] = kp2[t.train].y - 1.0
                        row += 1
                    end
                end
                H_fit = fit_homography_dlt(pa, pb)
            end
            probe_err = homography_probe_error(H_fit, H_gt, h1, w1; zero_based=true)
            accurate_h = isfinite(probe_err) && probe_err < 3.0
        end
    end
    return (matches=length(m), correct=correct, accurate_h=accurate_h, probe_err=probe_err)
end

function evaluate_dataset(name::String, scene_dirs::Vector{String}, kind::Symbol;
                          max_side::Int=0, rootsift::Bool=false, ratio::Float64=0.8)
    cfg = SIFTConfig(rootsift=rootsift)
    sift(UInt8[mod(17x + 31y + 7x * y, 256) for y in 1:64, x in 1:64]; config=cfg)
    match_descriptors(ones(Float32, 128, 2), zeros(Float32, 128, 3); crosscheck=true)

    tot_features = 0
    tot_matches = 0
    tot_correct = 0
    tot_homog = 0
    tot_pairs = 0
    max_norm_err = 0.0
    sift_time = 0.0
    match_time = 0.0
    scene_rows = NamedTuple[]

    for sdir in scene_dirs
        sname = basename(sdir)
        img_paths = String[]
        for n in 1:6
            if kind == :oxford
                cands = filter(f -> startswith(f, "img$n.") && (endswith(f, ".ppm") || endswith(f, ".pgm")), readdir(sdir))
                isempty(cands) && error("Missing img$n in $sdir")
                push!(img_paths, joinpath(sdir, first(cands)))
            else
                push!(img_paths, joinpath(sdir, "$n.ppm"))
            end
        end

        raw1 = read_pnm_gray(img_paths[1])
        img1, C1 = resize_gray(raw1, max_side)
        invC1 = inv(C1)
        t0 = time_ns()
        kp1, d1 = sift(img1; config=cfg)
        sift_time += (time_ns() - t0) / 1e9
        tot_features += length(kp1)
        if !isempty(kp1)
            nerr = maximum(abs.(vec(sum(abs2, d1; dims=1)) .- 1f0))
            nerr > max_norm_err && (max_norm_err = nerr)
        end

        sc_matches = 0
        sc_correct = 0
        sc_homog = 0
        for n in 2:6
            rawn = read_pnm_gray(img_paths[n])
            imgn, Cn = resize_gray(rawn, max_side)
            h_file = kind == :oxford ? joinpath(sdir, "H1to$(n)p") : joinpath(sdir, "H_1_$(n)")
            H_gt = Cn * read_homography(h_file) * invC1

            t0 = time_ns()
            kpn, dn = sift(imgn; config=cfg)
            sift_time += (time_ns() - t0) / 1e9
            tot_features += length(kpn)
            if !isempty(kpn)
                nerr = maximum(abs.(vec(sum(abs2, dn; dims=1)) .- 1f0))
                nerr > max_norm_err && (max_norm_err = nerr)
            end

            tm = time_ns()
            st = score_pair(kp1, d1, kpn, dn, H_gt, size(img1, 1), size(img1, 2); ratio=ratio)
            match_time += (time_ns() - tm) / 1e9

            tot_pairs += 1
            tot_matches += st.matches
            tot_correct += st.correct
            tot_homog += st.accurate_h ? 1 : 0
            sc_matches += st.matches
            sc_correct += st.correct
            sc_homog += st.accurate_h ? 1 : 0
        end
        push!(scene_rows, (scene=sname, matches=sc_matches, correct=sc_correct, homographies=sc_homog))
    end

    prec = tot_matches > 0 ? 100.0 * tot_correct / tot_matches : 0.0
    label = rootsift ? "RootSIFT" : "SIFT"
    @printf("%-10s | %-8s (ratio=%.2f) | %3d scenes, %3d pairs | %9d kp | %7d/%7d correct (%.3f%%) | %3d/%3d homog | sift=%.2fs match+ransac=%.2fs | max_norm_err=%.2e\n",
            name, label, ratio, length(scene_dirs), tot_pairs, tot_features,
            tot_correct, tot_matches, prec, tot_homog, tot_pairs, sift_time, match_time, max_norm_err)
    return (features=tot_features, matches=tot_matches, correct=tot_correct,
            precision=prec, homographies=tot_homog, pairs=tot_pairs,
            max_norm_err=max_norm_err, sift_time=sift_time, scenes=scene_rows)
end

function benchmark_oxford(oxford_dir::AbstractString; samples::Int=5, warmups::Int=1)
    scenes = ["bikes", "trees", "graf", "wall", "bark", "boat", "leuven", "ubc"]
    rows = NamedTuple[]
    println("\n--- Oxford timing benchmark (Julia $(VERSION), $(Sys.CPU_NAME), threads=$(Threads.nthreads()), median of $samples runs) ---")
    for s in scenes
        sdir = joinpath(oxford_dir, s)
        cand = first(filter(f -> startswith(f, "img1.") && (endswith(f, ".ppm") || endswith(f, ".pgm")), readdir(sdir)))
        img = read_pnm_gray(joinpath(sdir, cand))
        kp = SIFTKeypoint{Float32}[]
        for _ in 1:warmups
            kp, _ = sift(img)
        end
        times = Float64[]
        allocs = Int[]
        for _ in 1:samples
            GC.gc()
            res = @timed sift(img)
            kp = res.value[1]
            push!(times, res.time * 1000.0)
            push!(allocs, res.bytes)
        end
        med_ms = median(times)
        min_ms = minimum(times)
        mib = minimum(allocs) / 2.0^20
        @printf("  %-9s (%4d×%-4d) | %5d kp | median=%6.1f ms | min=%6.1f ms | alloc=%5.1f MiB\n",
                "$(s)_1", size(img, 2), size(img, 1), length(kp), med_ms, min_ms, mib)
        push!(rows, (name="$(s)_1", w=size(img, 2), h=size(img, 1), features=length(kp),
                     median_ms=med_ms, min_ms=min_ms, alloc_mib=mib))
    end
    return rows
end

function write_markdown_report(path::AbstractString, datasets, bench_rows)
    lines = String[
        "# Pure-Julia Validation & Benchmark Report — $(today())",
        "",
        "Runtime: Julia `$(VERSION)` (`$(Sys.CPU_NAME)`, `$(Threads.nthreads())` thread(s)). Zero Python or OpenCV dependencies.",
        "",
    ]
    for (name, r_sift, r_root) in datasets
        photos = r_sift.pairs + length(r_sift.scenes)
        push!(lines, "## $name $photos-Photograph Ground-Truth Accuracy ($(r_sift.pairs) pairs)", "")
        push!(lines, "| Mode | Features | Matches | Correct (≤3 px) | Pooled Precision | Accurate Homographies |")
        push!(lines, "|---|---:|---:|---:|---:|---:|")
        push!(lines, @sprintf("| SIFT (`ratio=0.80`) | %d | %d | %d | %.3f%% | %d / %d |",
                              r_sift.features, r_sift.matches, r_sift.correct, r_sift.precision, r_sift.homographies, r_sift.pairs))
        if !isnothing(r_root)
            push!(lines, @sprintf("| RootSIFT (`ratio=0.80`) | %d | %d | %d | %.3f%% | %d / %d |",
                                  r_root.features, r_root.matches, r_root.correct, r_root.precision, r_root.homographies, r_root.pairs))
        end
        modes = isnothing(r_root) ? "SIFT" : "SIFT / RootSIFT"
        push!(lines, "", "### Per-Sequence Breakdown ($modes)", "")
        push!(lines, "| Scene | Correct Matches (SIFT) | Homographies (SIFT) |" * (isnothing(r_root) ? "" : " Correct Matches (RootSIFT) | Homographies (RootSIFT) |"))
        push!(lines, "|---|---:|---:|" * (isnothing(r_root) ? "" : "---:|---:|"))
        for (idx, row) in enumerate(r_sift.scenes)
            extra = isnothing(r_root) ? "" : @sprintf(" %d / %d | %d / 5 |", r_root.scenes[idx].correct, r_root.scenes[idx].matches, r_root.scenes[idx].homographies)
            push!(lines, @sprintf("| %s | %d / %d | %d / 5 |%s", row.scene, row.correct, row.matches, row.homographies, extra))
        end
        push!(lines, "")
    end
    if !isempty(bench_rows)
        push!(lines, "## Oxford Timing Benchmark (`$(Threads.nthreads())` thread(s))", "")
        push!(lines, "| Photograph | Size (w×h) | Features | Median (ms) | Min (ms) | Allocated (MiB) |")
        push!(lines, "|---|---:|---:|---:|---:|---:|")
        for r in bench_rows
            push!(lines, @sprintf("| %s | %d×%d | %d | %.1f | %.1f | %.1f |",
                                  r.name, r.w, r.h, r.features, r.median_ms, r.min_ms, r.alloc_mib))
        end
        push!(lines, "")
    end
    mkpath(dirname(abspath(path)))
    write(path, join(lines, "\n") * "\n")
    println("Wrote report: $path")
end

function main(args=ARGS)
    oxford_dir = nothing
    hpatches_dir = nothing
    max_side = 0
    hpatches_max_side = 800
    compare_rootsift = false
    do_download = false
    do_benchmark = false
    report_path = nothing
    limit = 0
    samples = 5
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--oxford"
            oxford_dir = args[i + 1]; i += 2
        elseif arg == "--hpatches"
            hpatches_dir = args[i + 1]; i += 2
        elseif arg == "--max-side"
            max_side = parse(Int, args[i + 1]); i += 2
        elseif arg == "--hpatches-max-side"
            hpatches_max_side = parse(Int, args[i + 1]); i += 2
        elseif arg == "--limit"
            limit = parse(Int, args[i + 1]); i += 2
        elseif arg == "--samples"
            samples = parse(Int, args[i + 1]); i += 2
        elseif arg == "--write-report"
            report_path = args[i + 1]; i += 2
        elseif arg == "--compare-rootsift"
            compare_rootsift = true; i += 1
        elseif arg == "--download"
            do_download = true; i += 1
        elseif arg == "--benchmark"
            do_benchmark = true; i += 1
        else
            error("Unknown argument: $arg")
        end
    end
    (isnothing(oxford_dir) && isnothing(hpatches_dir)) &&
        error("Pass --oxford <dir> and/or --hpatches <dir>")

    println("Pure-Julia SIFTFeatures.jl Evaluator (Julia $(VERSION), threads=$(Threads.nthreads()))")
    failed = false
    datasets = NamedTuple[]
    bench_rows = NamedTuple[]

    if !isnothing(oxford_dir)
        do_download && ensure_oxford(oxford_dir)
        scenes = ["bikes", "trees", "graf", "wall", "bark", "boat", "leuven", "ubc"]
        dirs = [joinpath(oxford_dir, s) for s in scenes]
        ox_sift = evaluate_dataset("Oxford", dirs, :oxford; max_side=max_side, rootsift=false)
        (ox_sift.max_norm_err < 2e-6 && ox_sift.precision >= 90.0 && ox_sift.homographies >= 32) || (failed = true)
        ox_root = nothing
        if compare_rootsift
            ox_root = evaluate_dataset("Oxford", dirs, :oxford; max_side=max_side, rootsift=true)
            (ox_root.max_norm_err < 2e-6 && ox_root.precision >= 92.0 && ox_root.homographies >= 33) || (failed = true)
        end
        push!(datasets, (name="Oxford", sift=ox_sift, root=ox_root))
        if do_benchmark
            bench_rows = benchmark_oxford(oxford_dir; samples=samples)
        end
    end

    if !isnothing(hpatches_dir)
        root = do_download ? ensure_hpatches(hpatches_dir) :
               (isdir(joinpath(hpatches_dir, "i_ajuntament")) ? hpatches_dir :
                joinpath(hpatches_dir, "hpatches-sequences-release"))
        dirs = sort([joinpath(root, d) for d in readdir(root)
                     if isdir(joinpath(root, d)) && (startswith(d, "i_") || startswith(d, "v_"))])
        limit > 0 && (dirs = dirs[1:min(limit, length(dirs))])
        r_sift = evaluate_dataset("HPatches", dirs, :hpatches; max_side=hpatches_max_side, rootsift=false)
        (r_sift.max_norm_err < 2e-6 && r_sift.precision >= 89.0) || (failed = true)
        r_root = nothing
        if compare_rootsift
            r_root = evaluate_dataset("HPatches", dirs, :hpatches; max_side=hpatches_max_side, rootsift=true)
            (r_root.max_norm_err < 2e-6 && r_root.precision >= 91.0) || (failed = true)
        end
        push!(datasets, (name="HPatches", sift=r_sift, root=r_root))
    end

    if !isnothing(report_path)
        write_markdown_report(report_path, datasets, bench_rows)
    end

    println(failed ? "PURE-JULIA EVAL: FAIL" : "PURE-JULIA EVAL: PASS")
    return failed ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
