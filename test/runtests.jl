using Test, SIFTFeatures, LinearAlgebra
using ColorTypes: Gray, RGB
using ImageCore: N0f8
const SF = SIFTFeatures
include("fixtures.jl")

@testset "Inputs and parameters" begin
    for a in (zeros(Float32,0,0),zeros(Float32,0,12),zeros(Float32,1,1),
              fill(0.5f0,32,40),zeros(UInt8,32,40),trues(32,40))
        k,d = sift(a)
        @test isempty(k)
        @test size(d) == (128,0)
        @test eltype(d) == Float32
    end
    a = fixture(80,96)
    @test SF.grayscale_float(Gray.(a)) == a
    @test SF.grayscale_float(RGB.(a,a,a)) ≈ a
    u = UInt8.(round.(a.*255))
    @test SF.grayscale_float(u) ≈ Float32.(u)./255
    @test SF.grayscale_float(Gray{N0f8}.(a)) ≈ Float32.(Gray{N0f8}.(a))
    @test sift(@view a[:,:]) == sift(a)
    @test sift(transpose(a)) == sift(permutedims(a))
    for v in (NaN,Inf,-0.1,1.1)
        @test_throws ArgumentError sift(fill(v,20,20))
    end
    for c in (SIFTConfig(sigma=0),SIFTConfig(sigma=0.8),SIFTConfig(first_octave=1),
              SIFTConfig(nfeatures=-1),SIFTConfig(octave_layers=0),
              SIFTConfig(contrast_threshold=-1),SIFTConfig(edge_threshold=0),
              SIFTConfig(assumed_input_sigma=NaN),SIFTConfig(sigma=Inf))
        @test_throws ArgumentError sift(a;config=c)
    end
    @test_throws ArgumentError sift(a;octaves=-1)
    @test_throws ArgumentError sift(a;octaves=100)
    @test size(last(sift(a;octaves=0))) == (128,0)
    # The final octave may be two pixels wide, but cannot have a singleton axis.
    for first_octave in (-1,0), dims in ((2,2),(2,9),(3,3),(9,2))
        tiny = ones(Float32,dims)
        c = SIFTConfig(;first_octave)
        allowed = first_octave == -1 ? 2 : 1
        @test length(gaussian_pyramid(tiny,c;octaves=allowed)) == allowed
        @test_throws ArgumentError gaussian_pyramid(tiny,c;octaves=allowed+1)
        @test_throws ArgumentError sift(tiny;config=c,octaves=allowed+1)
    end
    @test_throws ArgumentError gaussian_pyramid(ones(Float32,1,20),SIFTConfig(first_octave=0);octaves=1)
end

@testset "Independent descriptor layout and interpolation" begin
    center = 17
    horizontal = Float32[x/64 for y in 1:33,x in 1:33]
    vertical = permutedims(horizontal)
    dh = SF.descriptor(horizontal,center,center,1/3,0)
    dv = SF.descriptor(vertical,center,center,1/3,0)
    # Positive x is bin 0; positive image y is upward-gradient bin 6.
    @test findall(>(0),dh) == collect(1:8:128)
    # Float32 atan can leak roundoff-sized mass into the neighboring axis bin.
    @test findall(>(2e-6),dv) == collect(7:8:128)
    # Rotating the frame 90 degrees puts the vertical gradient in bin 0.
    @test SF.descriptor(vertical,center,center,1/3,pi/2) ≈ dh atol=2e-6
    # A 22.5-degree upward gradient lies halfway between bins 0 and 1.
    diagonal = Float32[0.5+(x-tan(pi/8)*y)/128 for y in 1:33,x in 1:33]
    dd = SF.descriptor(diagonal,center,center,1/3,0)
    @test findall(>(0),dd) == sort(vcat(collect(1:8:128),collect(2:8:128)))
    @test all(isapprox.(dd[1:8:128],dd[2:8:128];atol=2e-6))
    # One impulse at (row=-2,col=+1) has three in-window gradients:
    # (+x at -2,0), (-x at -2,2), and (+upward-y at -1,1).
    # At width=1, half-cell interpolation produces these seven independent
    # row/column/orientation addresses. All seven exceed the 0.2 clip before
    # the final normalization, so their expected values are 1/sqrt(7).
    impulse = zeros(Float32,33,33)
    impulse[center-2,center+1] = 1
    expected = zeros(Float32,128)
    expected[[9,17,19,27,29,51,59]] .= Float32(1/sqrt(7))
    actual = SF.descriptor(impulse,center,center,1/3,0)
    @test actual ≈ expected atol=2e-6
    @test !isapprox(reverse(actual),expected;atol=2e-6)
end

@testset "Scale space and interpolation" begin
    a = reshape(Float32.(1:12)./12,3,4)
    b = SF.double_image(a)
    @test b[1:2:end,1:2:end] == a
    @test b[2,2] ≈ sum(a[1:2,1:2])/4
    @test b[end,end] == a[end,end]
    p = gaussian_pyramid(fixture(81,95);octaves=3)
    @test length(p) == 3
    @test all(length(o)==6 for o in p)
    @test size(p[1][1]) == (162,190)
    @test size(p[3][1]) == (40,47)
    @test p[2][1] == p[1][4][1:2:162,1:2:190]
    d = dog_pyramid(p)
    @test length(d[1]) == 5
    @test d[2][3] == p[2][4]-p[2][3]
    @test maximum(abs,SF.gaussian_blur(ones(Float32,20,20),1.5).-1) < 2e-6
    # Known quadratic extremum with a non-integer x/y/scale maximum.
    dogs = [Float32[1-0.05*((x-10.2)^2+(y-9.7)^2+(l-3.15)^2)
                     for y in 1:21,x in 1:21] for l in 1:5]
    loc = SF.localize(dogs,10,10,3,SIFTConfig())
    @test loc !== nothing
    @test loc.x ≈ 10.2 atol=1e-5
    @test loc.y ≈ 9.7 atol=1e-5
    @test loc.sigma ≈ 1.6*2^((2+0.15)/3) atol=1e-5
    @test SF.is_extremum(dogs,10,10,3)
    negative = [-level for level in dogs]
    @test SF.is_extremum(negative,10,10,3)
    @test SF.localize(negative,10,10,3,SIFTConfig()).x ≈ loc.x
    @test SF.localize([level.*0.001 for level in dogs],10,10,3,SIFTConfig()) === nothing
    plateau = deepcopy(dogs)
    plateau[3][10,11] = plateau[3][10,10]
    @test !SF.is_extremum(plateau,10,10,3)
    @test SF.localize([zeros(Float32,21,21) for _ in 1:5],10,10,3,SIFTConfig()) === nothing
    edge = [Float32[1-0.001*(x-10)^2-0.5*(y-10)^2-0.1*(l-3)^2
                     for y in 1:21,x in 1:21] for l in 1:5]
    @test SF.localize(edge,10,10,3,SIFTConfig()) === nothing
    # Analytical Gaussian blob: the DoG extremum's lower scale is s/sqrt(k).
    blob = Float32[exp(-((x-50.3)^2+(y-49.7)^2)/32) for y in 1:101,x in 1:101]
    bk,_ = sift(blob;config=SIFTConfig(assumed_input_sigma=0))
    @test !isempty(bk)
    @test all(p->hypot(p.x-50.3,p.y-49.7)<0.05,bk)
    @test all(p->abs(p.sigma/(4/2^(1/6))-1)<0.01,bk)
    @test SF.angle32(2pi-1e-9) == 0f0
    @test 0 <= SF.angle32(-1e-5) < 2pi
end

@testset "Detection, descriptors, and geometric consistency" begin
    # Dimensions minus one are divisible by four: a 90-degree rotation preserves
    # the sampling phase through octave 2, isolating rotation from resampling.
    a = fixture(161,193)
    k,d = sift(a)
    @test length(k) > 30
    @test size(d) == (128,length(k))
    @test all(isfinite,d) && all(d .>= 0)
    @test all(abs.(vec(sum(abs2,d;dims=1)).-1) .< 2e-6)
    @test all(p->p.x isa Float32 && p.sigma>0 && 0<=p.angle<2pi,k)
    @test any(p->abs(p.x-round(p.x))>0.05,k)
    @test length(unique((p.x,p.y,p.sigma,p.angle) for p in k)) == length(k)
    @test issorted([p.response for p in k];rev=true)
    limited,ld = sift(a;config=SIFTConfig(nfeatures=12))
    @test limited == k[1:12]
    @test ld == d[:,1:12]
    @test length(first(sift(a;config=SIFTConfig(contrast_threshold=0.1)))) <= length(k)
    k0,d0 = sift(a;config=SIFTConfig(first_octave=0))
    @test !isempty(k0) && minimum(p.octave for p in k0)>=0
    @test size(d0,2)==length(k0)
    kr,dr = sift(rotl90(a))
    m = match_descriptors(d,dr;crosscheck=true)
    @test length(m) >= 0.65length(k)
    errors = [hypot(kr[t.train].x-k[t.query].y,
                    kr[t.train].y-(size(a,2)+1-k[t.query].x)) for t in m]
    @test count(<(1.0),errors) >= 0.95length(m)
    good = [t for t in m if hypot(kr[t.train].x-k[t.query].y,kr[t.train].y-(size(a,2)+1-k[t.query].x))<1]
    @test all(abs(mod(kr[t.train].angle-k[t.query].angle+pi/2+pi,2pi)-pi)<0.08 for t in good)
    ks,ds = sift(SF.double_image(a))
    ms = match_descriptors(d,ds;crosscheck=true)
    @test length(ms) > 20
    @test count(t->hypot(ks[t.train].x-(2k[t.query].x-1),ks[t.train].y-(2k[t.query].y-1))<1.5,ms) > 0.85length(ms)
    # Ramps pin the orientation sign and flat windows must give finite zeros.
    ramp = Float32[x/65 for y in 1:65,x in 1:65]
    @test only(SF.orientations(ramp,33,33,2.0)) ≈ 0 atol=1e-7
    @test only(SF.orientations(permutedims(ramp),33,33,2.0)) ≈ pi/2 atol=1e-7
    @test SF.descriptor(zeros(Float32,30,30),15.,15.,2.,0.) == zeros(Float32,128)
end

@testset "L2 ratio and mutual matching" begin
    a = zeros(Float32,128,2); a[1,2] = 1
    b = zeros(Float32,128,3); b[1,:] = [0.1,0.3,1.1]
    m = match_descriptors(a,b)
    @test [(t.query,t.train) for t in m] == [(1,1),(2,3)]
    @test [t.distance for t in m] ≈ [0.1,0.1]
    # Ratio applies to distances, not squared distances (0.7/1 fails at 0.6).
    @test isempty(match_descriptors(zeros(128,1),hcat(fill(0.7,128),ones(128));ratio=0.6))
    @test isempty(match_descriptors(a,zeros(128,2)))
    @test isempty(match_descriptors(a,b[:,1:1]))
    @test isempty(match_descriptors(zeros(128,0),b))
    many = hcat(a[:,1],a[:,1].+0.001,a[:,2])
    @test [(t.query,t.train) for t in match_descriptors(many,b;crosscheck=true)] == [(2,1),(3,3)]
    exact = zeros(Float32,128,1)
    distinct = hcat(exact,ones(Float32,128,1))
    @test [(t.query,t.train) for t in match_descriptors(exact,distinct;ratio=1e-200)] == [(1,1)]
    @test isempty(match_descriptors(exact,hcat(exact,exact);ratio=1e-200))
    @test_throws DimensionMismatch match_descriptors(zeros(127,2),b)
    @test_throws ArgumentError match_descriptors(a,b;ratio=1)
    @test_throws ArgumentError match_descriptors(a,fill(NaN,128,2))
end
