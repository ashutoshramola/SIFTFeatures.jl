using SIFTFeatures, LinearAlgebra, ImageFiltering, ImageCore, ColorTypes, StaticArrays

# File loading, compilation/warmup, forced GC, and result checks are outside
# the timed interval. GC triggered by sift itself remains part of that interval.
function main()
    root, samples_text, warmups_text = ARGS
    samples, warmups = parse(Int, samples_text), parse(Int, warmups_text)
    samples >= 3 && warmups >= 1 || error("Require at least 3 samples and 1 warmup")
    BLAS.set_num_threads(1)
    open(joinpath(root, "julia-environment.tsv"), "w") do io
        for (key, value) in (("julia", VERSION), ("cpu_target", Sys.CPU_NAME),
                             ("julia_threads", Threads.nthreads()),
                             ("blas_threads", BLAS.get_num_threads()),
                             ("ImageFiltering", Base.pkgversion(ImageFiltering)),
                             ("ImageCore", Base.pkgversion(ImageCore)),
                             ("ColorTypes", Base.pkgversion(ColorTypes)),
                             ("StaticArrays", Base.pkgversion(StaticArrays)))
            println(io, key, '\t', value)
        end
    end
    open(joinpath(root, "julia-samples.tsv"), "w") do io
        println(io, "image\tsample\tfeatures\tseconds\tallocated_bytes\tgc_seconds")
        for name in readlines(joinpath(root, "images.txt"))
            h, w = parse.(Int, split(read(joinpath(root, name * ".shape"), String)))
            raw = read(joinpath(root, name * ".u8"))
            length(raw) == h*w || error("Invalid image byte count")
            image = permutedims(reshape(raw, w, h))
            for _ in 1:warmups
                sift(image)
            end
            counts = Int[]
            for sample in 1:samples
                GC.gc()
                measured = @timed sift(image)
                keypoints, descriptors = measured.value
                size(descriptors) == (128, length(keypoints)) || error("Descriptor shape")
                eltype(descriptors) == Float32 || error("Descriptor element type")
                all(isfinite, descriptors) || error("Non-finite descriptor")
                all(abs(norm(d)-1) < 2e-5 for d in eachcol(descriptors)) || error("Descriptor norm")
                push!(counts, length(keypoints))
                println(io, join((name, sample, length(keypoints), measured.time,
                                  measured.bytes, measured.gctime), '\t'))
                flush(io)
            end
            all(==(first(counts)), counts) || error("Non-deterministic feature count")
            println(name, ": ", first(counts), " features; ", samples, " timed samples")
        end
    end
end
main()
