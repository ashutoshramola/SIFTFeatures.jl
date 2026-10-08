module SIFTFeatures

using ColorTypes: Colorant, Gray, color
using ImageCore: ImageCore
using ImageFiltering: imfilter, KernelFactors
using LinearAlgebra: norm, dot, SingularException

export SIFTConfig, SIFTKeypoint, DescriptorMatch, sift, match_descriptors,
       gaussian_pyramid, dog_pyramid

"""SIFT parameters. `first_octave=-1` doubles the input on a pixel-aligned grid.
Contrast uses OpenCV semantics (the threshold is divided by `octave_layers`).
Descriptors always have 4×4 spatial cells and 8 orientation bins.
"""
Base.@kwdef struct SIFTConfig
    nfeatures::Int = 0
    octave_layers::Int = 3
    contrast_threshold::Float64 = 0.04
    edge_threshold::Float64 = 10.0
    sigma::Float64 = 1.6
    assumed_input_sigma::Float64 = 0.5
    first_octave::Int = -1
end

"""A subpixel SIFT feature. `x` and `y` are one-based column/row coordinates,
`sigma` is the Gaussian standard deviation in original-image pixels, and `angle` is clockwise from +x
in radians, in [0, 2π). `octave` is signed; `layer` is zero-based.
"""
struct SIFTKeypoint{T<:AbstractFloat}
    y::T
    x::T
    sigma::T
    angle::T
    response::T
    octave::Int
    layer::Int
end

"""One-based descriptor column indices and their Euclidean (not squared) distance."""
struct DescriptorMatch
    query::Int
    train::Int
    distance::Float32
end

function validate(c::SIFTConfig)
    c.nfeatures >= 0 || throw(ArgumentError("nfeatures must be nonnegative"))
    c.octave_layers >= 1 || throw(ArgumentError("octave_layers must be positive"))
    c.first_octave in (-1, 0) || throw(ArgumentError("first_octave must be -1 or 0"))
    isfinite(c.contrast_threshold) && c.contrast_threshold >= 0 ||
        throw(ArgumentError("contrast_threshold must be finite and nonnegative"))
    isfinite(c.edge_threshold) && c.edge_threshold > 0 ||
        throw(ArgumentError("edge_threshold must be finite and positive"))
    isfinite(c.assumed_input_sigma) && c.assumed_input_sigma >= 0 ||
        throw(ArgumentError("assumed_input_sigma must be finite and nonnegative"))
    isfinite(c.sigma) && c.sigma > 0 && c.sigma >= c.assumed_input_sigma * 2.0^(-c.first_octave) ||
        throw(ArgumentError("sigma must be positive and at least the input blur at the first octave"))
    return c
end

intensity(v::Real) = Float32(v)
intensity(v::Unsigned) = Float32(v / typemax(typeof(v)))
intensity(v::Colorant) = Float32(Gray(color(v)).val)

function grayscale_float(img::AbstractMatrix{<:Union{Real,Colorant}})
    Base.require_one_based_indexing(img)
    a = Matrix{Float32}(undef, size(img))
    for i in eachindex(a, img)
        v = intensity(img[i])
        isfinite(v) && 0 <= v <= 1 ||
            throw(ArgumentError("image intensities must be finite and in [0,1]; unsigned integers are normalized by typemax"))
        a[i] = v
    end
    return a
end

# Doubling is aligned to the original zero-based pixel grid: x -> 2x.
# At the far edge a half-pixel beyond the final sample uses edge replication.
function double_image(a::Matrix{Float32})
    h, w = size(a)
    b = Matrix{Float32}(undef, 2h, 2w)
    for x in 1:2w, y in 1:2h
        x0, y0 = (x + 1) ÷ 2, (y + 1) ÷ 2
        fx, fy = iseven(x) ? 0.5f0 : 0f0, iseven(y) ? 0.5f0 : 0f0
        x1, y1 = min(x0 + 1, w), min(y0 + 1, h)
        b[y,x] = (1-fy)*((1-fx)*a[y0,x0] + fx*a[y0,x1]) +
                 fy*((1-fx)*a[y1,x0] + fx*a[y1,x1])
    end
    b
end

function gaussian_blur(a::Matrix{Float32}, sigma::Real)
    sigma <= eps(Float32) && return copy(a)
    # A four-sigma finite FIR kernel, with reflect-101 (no repeated endpoint).
    len = (round(Int, 8sigma + 1) | 1)
    kernels = KernelFactors.gaussian((sigma, sigma), (len, len))
    return imfilter(a, kernels, "reflect")
end

function default_octaves(img, c)
    m = minimum(size(img))
    m < 2 && return 0
    max(1, round(Int, log2(m)) - 2 - c.first_octave)
end

"""Build Gaussian octaves, each containing `octave_layers + 3` Float32 matrices.
Input is a one-based AbstractMatrix of normalized real values, unsigned integer
pixels, or JuliaImages colors. Empty input gives an empty pyramid.
"""
function gaussian_pyramid(img::AbstractMatrix, config::SIFTConfig=SIFTConfig(); octaves=nothing)
    c = validate(config)
    a = grayscale_float(img)
    n = isnothing(octaves) ? default_octaves(a, c) : octaves
    n isa Integer && n >= 0 || throw(ArgumentError("octaves must be a nonnegative integer"))
    pyramid = Vector{Matrix{Float32}}[]
    (isempty(a) || n == 0) && return pyramid
    base = c.first_octave == -1 ? double_image(a) : a
    # Reflected filtering requires at least two pixels along each octave axis.
    n <= floor(Int, log2(minimum(size(base)))) ||
        throw(ArgumentError("too many octaves: every octave must have at least two pixels along each axis"))
    blur = sqrt(max(0, c.sigma^2 - (c.assumed_input_sigma * 2.0^(-c.first_octave))^2))
    base = gaussian_blur(base, blur)
    s = c.octave_layers
    for o in 1:n
        levels = Matrix{Float32}[base]
        for l in 1:s+2
            sigma_prev = c.sigma * 2.0^((l-1)/s)
            sigma_next = c.sigma * 2.0^(l/s)
            push!(levels, gaussian_blur(levels[end], sqrt(sigma_next^2-sigma_prev^2)))
        end
        push!(pyramid, levels)
        if o < n
            source = levels[s+1]
            h, w = size(source) .÷ 2
            base = source[1:2:2h, 1:2:2w]
        end
    end
    pyramid
end

"""Adjacent Gaussian differences, with `octave_layers + 2` levels per octave."""
dog_pyramid(g::AbstractVector) = [[o[l+1] .- o[l] for l in 1:length(o)-1] for o in g]

function is_extremum(d, y, x, l)
    v = d[l][y,x]
    iszero(v) && return false
    for k in l-1:l+1, j in x-1:x+1, i in y-1:y+1
        (k == l && j == x && i == y) && continue
        u = d[k][i,j]
        (v > 0 ? v <= u : v >= u) && return false
    end
    true
end

# Derivative order is x, y, scale; solve the 3-D quadratic in Float64.
function derivatives(d, y, x, l)
    p, a, n = d[l-1], d[l], d[l+1]
    v = Float64(a[y,x])
    dx = (Float64(a[y,x+1])-a[y,x-1])/2
    dy = (Float64(a[y+1,x])-a[y-1,x])/2
    ds = (Float64(n[y,x])-p[y,x])/2
    dxx = Float64(a[y,x+1])+a[y,x-1]-2v
    dyy = Float64(a[y+1,x])+a[y-1,x]-2v
    dss = Float64(n[y,x])+p[y,x]-2v
    dxy = (Float64(a[y+1,x+1])-a[y+1,x-1]-a[y-1,x+1]+a[y-1,x-1])/4
    dxs = (Float64(n[y,x+1])-n[y,x-1]-p[y,x+1]+p[y,x-1])/4
    dys = (Float64(n[y+1,x])-n[y-1,x]-p[y+1,x]+p[y-1,x])/4
    [dx,dy,ds], [dxx dxy dxs; dxy dyy dys; dxs dys dss]
end

function localize(d, y, x, l, c)
    h, w = size(d[1])
    for _ in 1:5
        (6 <= y <= h-5 && 6 <= x <= w-5 && 2 <= l <= c.octave_layers+1) || return nothing
        g, H = derivatives(d,y,x,l)
        offset = try
            -(H \ g)
        catch e
            e isa SingularException || rethrow()
            return nothing
        end
        all(isfinite, offset) && maximum(abs, offset) < 1e6 || return nothing
        if maximum(abs, offset) < 0.5
            contrast = d[l][y,x] + dot(g,offset)/2
            abs(contrast)*c.octave_layers >= c.contrast_threshold || return nothing
            detxy = H[1,1]*H[2,2] - H[1,2]^2
            trace = H[1,1]+H[2,2]
            detxy > 0 && trace^2*c.edge_threshold < detxy*(c.edge_threshold+1)^2 || return nothing
            sigma = c.sigma * 2.0^((l-1+offset[3])/c.octave_layers)
            return (y=y+offset[2], x=x+offset[1], sigma=sigma,
                    response=abs(contrast), layer=l, iy=y, ix=x)
        end
        x += round(Int, offset[1])
        y += round(Int, offset[2])
        l += round(Int, offset[3])
    end
    nothing
end

function orientations(a, y, x, sigma)
    hist = zeros(Float64,36)
    radius = round(Int,4.5sigma)
    h,w = size(a)
    for j in -radius:radius, i in -radius:radius
        yy,xx = y+i,x+j
        (2 <= yy < h && 2 <= xx < w) || continue
        gx = a[yy,xx+1]-a[yy,xx-1]
        gy = a[yy-1,xx]-a[yy+1,xx]
        bin = mod(round(Int,atan(gy,gx)*36/(2pi)),36)+1
        hist[bin] += hypot(gx,gy)*exp(-(i*i+j*j)/(2*(1.5sigma)^2))
    end
    sm = [(hist[mod1(b-2,36)] + 4hist[mod1(b-1,36)] + 6hist[b] +
           4hist[mod1(b+1,36)] + hist[mod1(b+2,36)])/16 for b in 1:36]
    peak = maximum(sm)
    angles = Float64[]
    peak > 0 || return angles
    for b in 1:36
        left,mid,right = sm[mod1(b-1,36)],sm[b],sm[mod1(b+1,36)]
        if mid > left && mid > right && mid >= 0.8peak
            offset = (left-right)/(2*(left-2mid+right))
            # Convert the upward-y histogram angle to clockwise image coordinates.
            push!(angles,mod(-(b-1+offset)*2pi/36,2pi))
        end
    end
    angles
end

function descriptor(a, y, x, sigma, angle)
    hist = zeros(Float64,128)
    # Round in zero-based coordinates, then convert back. Ties-to-even would
    # choose the opposite pixel if applied directly to a one-based coordinate.
    cy,cx = round(Int,y-1)+1,round(Int,x-1)+1
    width = 3sigma
    # Internal gradient angles use upward y, as in the standard SIFT convention.
    phi = -angle
    cs,sn = cos(phi)/width,sin(phi)/width
    h,w = size(a)
    radius = min(round(Int,width*sqrt(2)*2.5),ceil(Int,hypot(h,w)))
    for j in -radius:radius, i in -radius:radius
        yy,xx = cy+i,cx+j
        (2 <= yy < h && 2 <= xx < w) || continue
        u,v = j*cs-i*sn,j*sn+i*cs
        rb,cb = v+1.5,u+1.5
        (-1 < rb < 4 && -1 < cb < 4) || continue
        gx = a[yy,xx+1]-a[yy,xx-1]
        gy = a[yy-1,xx]-a[yy+1,xx]
        ob = mod((atan(gy,gx)-phi)*8/(2pi),8)
        mag = hypot(gx,gy)*exp(-(u*u+v*v)/8)
        r0,c0,o0 = floor(Int,rb),floor(Int,cb),floor(Int,ob)
        fr,fc,fo = rb-r0,cb-c0,ob-o0
        for dr in 0:1, dc in 0:1, db in 0:1
            r,cl = r0+dr,c0+dc
            (0 <= r < 4 && 0 <= cl < 4) || continue
            b = mod(o0+db,8)
            weight = (dr==0 ? 1-fr : fr)*(dc==0 ? 1-fc : fc)*(db==0 ? 1-fo : fo)
            hist[(r*4+cl)*8+b+1] += mag*weight
        end
    end
    n = norm(hist)
    n > 0 || return zeros(Float32,128)
    hist .= min.(hist ./ n,0.2)
    Float32.(hist ./ norm(hist))
end

# Rounding a value just below 2π to Float32 can otherwise leave the interval.
function angle32(angle)
    a = Float32(mod(angle,2pi))
    a < 2pi ? a : 0f0
end

"""
    keypoints, descriptors = sift(img; config=SIFTConfig(), octaves=nothing)

Native SIFT detection and description. Returns floating-point keypoints and a
128×N Float32 matrix with unit-L2 columns. Multiple strong orientations create
separate features. `nfeatures=0` keeps all features; otherwise keep the strongest.
"""
function sift(img::AbstractMatrix; config::SIFTConfig=SIFTConfig(), octaves=nothing)
    c = validate(config)
    gp = gaussian_pyramid(img,c;octaves)
    dp = dog_pyramid(gp)
    kp = SIFTKeypoint{Float32}[]
    # Converged candidates at the same lattice extremum are the same feature.
    seen = Set{NTuple{4,Int}}()
    for o in eachindex(dp)
        d = dp[o]
        h,w = size(d[1])
        octave = o-1+c.first_octave
        factor = 2.0^octave
        for l in 2:c.octave_layers+1, x in 6:w-5, y in 6:h-5
            abs(d[l][y,x]) >= 0.5c.contrast_threshold/c.octave_layers || continue
            is_extremum(d,y,x,l) || continue
            p = localize(d,y,x,l,c)
            isnothing(p) && continue
            id = (o,p.layer,p.iy,p.ix)
            id in seen && continue
            push!(seen,id)
            for angle in orientations(gp[o][p.layer],p.iy,p.ix,p.sigma)
                push!(kp,SIFTKeypoint(Float32(1+(p.y-1)*factor),Float32(1+(p.x-1)*factor),
                    Float32(p.sigma*factor),angle32(angle),Float32(p.response),octave,p.layer-1))
            end
        end
    end
    sort!(kp;by=k->(-k.response,k.octave,k.layer,k.y,k.x,k.angle))
    c.nfeatures > 0 && resize!(kp,min(c.nfeatures,length(kp)))
    desc = Matrix{Float32}(undef,128,length(kp))
    for (i,k) in enumerate(kp)
        factor = 2.0^k.octave
        desc[:,i] = descriptor(gp[k.octave-c.first_octave+1][k.layer+1],
            1+(k.y-1)/factor,1+(k.x-1)/factor,k.sigma/factor,k.angle)
    end
    kp,desc
end

"""
    match_descriptors(a, b; ratio=0.8, crosscheck=false)

Brute-force L2 matching for 128×N descriptors. Accept only a *strict* Lowe ratio
between the nearest and second-nearest neighbors. Fewer than two train columns
or tied nearest neighbors yield no match. `crosscheck=true` also requires a
mutual nearest neighbor (it does not apply a reverse ratio test).
"""
function match_descriptors(a::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real}; ratio::Real=0.8, crosscheck::Bool=false)
    Base.require_one_based_indexing(a,b)
    size(a,1) == size(b,1) == 128 || throw(DimensionMismatch("descriptors must have 128 rows"))
    isfinite(ratio) && 0 < ratio < 1 || throw(ArgumentError("ratio must be in (0,1)"))
    all(isfinite,a) && all(isfinite,b) || throw(ArgumentError("descriptors must be finite"))
    matches = DescriptorMatch[]
    size(b,2) >= 2 || return matches
    reverse_idx = zeros(Int,size(b,2))
    reverse_dist = fill(Inf,size(b,2))
    for i in axes(a,2)
        best,second,index = Inf,Inf,0
        for j in axes(b,2)
            d = sum((Float64(a[k,i])-Float64(b[k,j]))^2 for k in 1:128)
            if d < best
                second,best,index = best,d,j
            elseif d < second
                second = d
            end
            if crosscheck && d < reverse_dist[j]
                reverse_dist[j],reverse_idx[j] = d,i
            end
        end
        # A unique exact match passes every positive ratio, including values
        # whose square underflows. Duplicate exact neighbors still fail.
        ((best == 0 && second > 0) || best < ratio^2*second) &&
            push!(matches,DescriptorMatch(i,index,Float32(sqrt(best))))
    end
    crosscheck && filter!(m->reverse_idx[m.train]==m.query,matches)
    matches
end

end
