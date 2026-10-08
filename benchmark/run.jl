using SIFTFeatures
include(joinpath(@__DIR__,"..","test","fixtures.jl"))

# Warm every workload before measuring; these are descriptive baselines, not
# performance gates. @timed records Julia allocations, not process peak RSS.
function measure(f; samples=5)
    f()
    times,bytes = Float64[],Int[]
    for _ in 1:samples
        GC.gc()
        result = @timed f()
        push!(times,result.time)
        push!(bytes,result.bytes)
    end
    sort!(times)
    (median_ms=1000times[cld(samples,2)], minimum_ms=1000minimum(times),
     allocated_mib=minimum(bytes)/2.0^20)
end

function main()
    path = isempty(ARGS) ? joinpath(@__DIR__,"baseline.md") : ARGS[1]
    lines = ["# Local correctness-first benchmark", "",
             "Julia $(VERSION); $(Sys.CPU_NAME); $(Threads.nthreads()) Julia thread(s).",
             "Five warmed samples per workload; image generation excluded. Allocation is the minimum recorded Julia allocation, not peak memory.", "",
             "| Image | Features | Workload | Median ms | Minimum ms | Allocated MiB |",
             "|---|---:|---|---:|---:|---:|"]
    for n in (128,256,512)
        a = fixture(n,n)
        k,d = sift(a)
        for (name,f) in (("Gaussian pyramid",()->gaussian_pyramid(a)),
                         ("Full SIFT",()->sift(a)),
                         ("Self matching + mutual check",()->match_descriptors(d,d;crosscheck=true)))
            r = measure(f)
            push!(lines,"| $(n)×$(n) | $(length(k)) | $name | $(round(r.median_ms;digits=2)) | $(round(r.minimum_ms;digits=2)) | $(round(r.allocated_mib;digits=2)) |")
        end
    end
    mkpath(dirname(abspath(path)))
    write(path,join(lines,"\n")*"\n")
    println(join(lines,"\n"))
end
main()
