using SIFTFeatures
# Small dependency-free interchange: image bytes are row-major, descriptors are
# Julia columns (equivalently row-major N×128), and keypoints are readable CSV.
root = ARGS[1]
names = readlines(joinpath(root,"images.txt"))
features = Dict()
# Warm compilation separately from the observational timings below.
sift(UInt8[mod(17x+31y+7x*y,256) for y in 1:64,x in 1:64])
match_descriptors(ones(Float32,128,2),zeros(Float32,128,3);crosscheck=true)
performance = open(joinpath(root,"performance.csv"),"w")
println(performance,"kind,name,seconds,bytes")
for name in names
    h,w = parse.(Int,split(read(joinpath(root,name*".shape"),String)))
    a = permutedims(reshape(read(joinpath(root,name*".u8")),w,h))
    measured = @timed sift(a)
    kp,d = measured.value
    println(performance,join(("sift",name,measured.time,measured.bytes),','))
    features[name] = (kp,d)
    if isfile(joinpath(root,"probes.csv"))
        probes = [parse.(Float64,split(line,',')) for line in readlines(joinpath(root,"probes.csv"))]
        gp = gaussian_pyramid(a)
        pd = Matrix{Float32}(undef,128,length(probes))
        for (i,p) in enumerate(probes)
            x,y,sigma,angle,octave,layer = p
            scale = 2.0^octave
            pd[:,i] = SIFTFeatures.descriptor(gp[Int(octave)+2][Int(layer)+1],
                1+y/scale,1+x/scale,sigma/scale,angle)
        end
        write(joinpath(root,name*".probes.f32"),reinterpret(UInt8,vec(pd)))
    end
    open(joinpath(root,name*".kp.csv"),"w") do io
        println(io,"x,y,sigma,angle,response,octave,layer")
        for p in kp
            println(io,join((p.x-1,p.y-1,p.sigma,p.angle,p.response,p.octave,p.layer),','))
        end
    end
    write(joinpath(root,name*".desc.f32"),reinterpret(UInt8,vec(d)))
    println(name,": ",length(kp)," features")
end
pairs = isfile(joinpath(root,"pairs.txt")) ?
    [Tuple(split(line)) for line in readlines(joinpath(root,"pairs.txt"))] :
    [(names[1],name) for name in names[2:end]]
for (source,target) in pairs
    measured = @timed match_descriptors(features[source][2],features[target][2];crosscheck=true)
    m = measured.value
    println(performance,join(("match",target,measured.time,measured.bytes),','))
    open(joinpath(root,target*".matches.csv"),"w") do io
        println(io,"query,train,distance")
        for t in m
            println(io,"$(t.query-1),$(t.train-1),$(t.distance)")
        end
    end
    println(source," -> ",target,": ",length(m)," matches")
end
close(performance)
