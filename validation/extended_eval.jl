using SIFTFeatures, Statistics

function load_u8(root::AbstractString, name::AbstractString)
    h, w = parse.(Int, split(read(joinpath(root, name * ".shape"), String)))
    return permutedims(reshape(read(joinpath(root, name * ".u8")), w, h))
end

function write_features(root::AbstractString, tag::AbstractString, kp, d)
    open(joinpath(root, tag * ".kp.csv"), "w") do io
        println(io, "x,y,sigma,angle,response,octave,layer")
        for p in kp
            println(io, join((p.x - 1, p.y - 1, p.sigma, p.angle, p.response, p.octave, p.layer), ','))
        end
    end
    write(joinpath(root, tag * ".desc.f32"), reinterpret(UInt8, vec(d)))
end

function write_env(root::AbstractString)
    open(joinpath(root, "julia-environment.tsv"), "w") do io
        for (key, value) in (("julia", VERSION), ("cpu_target", Sys.CPU_NAME),
                             ("julia_threads", Threads.nthreads()))
            println(io, key, '\t', value)
        end
    end
end

function run_configs(root::AbstractString)
    write_env(root)
    a = load_u8(root, "config_input")
    sift(UInt8[mod(17x + 31y + 7x * y, 256) for y in 1:64, x in 1:64])
    for line in readlines(joinpath(root, "configs.tsv"))
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        tag = parts[1]
        cfg = SIFTConfig(
            nfeatures = parse(Int, parts[2]),
            octave_layers = parse(Int, parts[3]),
            contrast_threshold = parse(Float64, parts[4]),
            edge_threshold = parse(Float64, parts[5]),
            sigma = parse(Float64, parts[6]),
            first_octave = parse(Int, parts[7]),
        )
        kp, d = sift(a; config=cfg)
        write_features(root, tag, kp, d)
        println("config ", tag, ": ", length(kp), " features")
    end
end

function run_stress(root::AbstractString)
    write_env(root)
    sift(UInt8[mod(17x + 31y + 7x * y, 256) for y in 1:64, x in 1:64])
    names = readlines(joinpath(root, "stress_images.txt"))
    u16_status = open(joinpath(root, "u16_parity.tsv"), "w")
    println(u16_status, "name\tidentical_kp\tidentical_desc\tmax_desc_diff")
    for name in names
        isempty(strip(name)) && continue
        a = load_u8(root, name)
        kp, d = sift(a)
        write_features(root, name, kp, d)
        # Verify exact UInt16 scaling parity: v * 0x0101 maps 0..255 identically onto 0..65535.
        a16 = UInt16.(a) .* UInt16(257)
        kp16, d16 = sift(a16)
        same_kp = (kp == kp16)
        same_d = (d == d16)
        max_diff = isempty(d) && isempty(d16) ? 0.0f0 :
            (size(d) == size(d16) ? maximum(abs.(d .- d16)) : Inf32)
        println(u16_status, name, '\t', same_kp, '\t', same_d, '\t', max_diff)
        println("stress ", name, ": ", length(kp), " features (u16_exact=", same_kp && same_d, ")")
    end
    close(u16_status)
end

function run_scaling(root::AbstractString, warmups::Int, samples::Int)
    write_env(root)
    names = readlines(joinpath(root, "scaling_images.txt"))
    sift(UInt8[mod(17x + 31y + 7x * y, 256) for y in 1:64, x in 1:64])
    open(joinpath(root, "scaling_t$(Threads.nthreads()).tsv"), "w") do io
        println(io, "name\tthreads\tfeatures\tmedian_ms\tsamples_ms\tkp_hash\tdesc_hash")
        for name in names
            isempty(strip(name)) && continue
            a = load_u8(root, name)
            kp = SIFTKeypoint{Float32}[]
            d = Matrix{Float32}(undef, 128, 0)
            for _ in 1:warmups
                kp, d = sift(a)
            end
            times = Float64[]
            for _ in 1:samples
                GC.gc()
                t0 = time_ns()
                kp, d = sift(a)
                push!(times, (time_ns() - t0) / 1e6)
            end
            med = median(times)
            println(io, join((
                name, Threads.nthreads(), length(kp), med,
                join(round.(times; digits=3), ","),
                string(hash(kp)), string(hash(d))
            ), '\t'))
            println("scaling ", name, " (t=", Threads.nthreads(), "): ",
                    round(med; digits=2), " ms (", length(kp), " features)")
        end
    end
end

function run_hpatches(root::AbstractString)
    write_env(root)
    names = readlines(joinpath(root, "images.txt"))
    pairs = [Tuple(split(line)) for line in readlines(joinpath(root, "pairs.txt")) if !isempty(strip(line))]
    sift(UInt8[mod(17x + 31y + 7x * y, 256) for y in 1:64, x in 1:64])
    match_descriptors(ones(Float32, 128, 2), zeros(Float32, 128, 3); crosscheck=true)
    # Process scene-by-scene to keep peak memory small across all 116 HPatches scenes.
    by_scene = Dict{String, Vector{String}}()
    for name in names
        isempty(strip(name)) && continue
        scene = name[1:findlast('_', name) - 1]
        push!(get!(by_scene, scene, String[]), name)
    end
    pairs_by_scene = Dict{String, Vector{Tuple{String, String}}}()
    for (src, tgt) in pairs
        scene = src[1:findlast('_', src) - 1]
        push!(get!(pairs_by_scene, scene, Tuple{String, String}[]), (src, tgt))
    end
    perf = open(joinpath(root, "performance.csv"), "w")
    println(perf, "kind,name,seconds,bytes")
    for scene in sort!(collect(keys(by_scene)))
        feats = Dict{String, Tuple{Vector{SIFTKeypoint{Float32}}, Matrix{Float32}}}()
        for name in by_scene[scene]
            a = load_u8(root, name)
            measured = @timed sift(a)
            kp, d = measured.value
            println(perf, join(("sift", name, measured.time, measured.bytes), ','))
            feats[name] = (kp, d)
            write_features(root, name, kp, d)
        end
        for (src, tgt) in get(pairs_by_scene, scene, Tuple{String, String}[])
            measured = @timed match_descriptors(feats[src][2], feats[tgt][2]; crosscheck=true)
            m = measured.value
            println(perf, join(("match", tgt, measured.time, measured.bytes), ','))
            open(joinpath(root, tgt * ".matches.csv"), "w") do io
                println(io, "query,train,distance")
                for t in m
                    println(io, "$(t.query - 1),$(t.train - 1),$(t.distance)")
                end
            end
        end
    end
    close(perf)
    println("hpatches: exported ", length(names), " images and ", length(pairs), " pairs")
end

mode = ARGS[1]
root = ARGS[2]
if mode == "configs"
    run_configs(root)
elseif mode == "stress"
    run_stress(root)
elseif mode == "scaling"
    warmups = parse(Int, ARGS[3])
    samples = parse(Int, ARGS[4])
    run_scaling(root, warmups, samples)
elseif mode == "hpatches"
    run_hpatches(root)
else
    error("Unknown mode: $mode")
end
