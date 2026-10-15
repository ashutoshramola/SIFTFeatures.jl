module SIFTFeatures

using ColorTypes: Colorant, Gray, AbstractGray, color
using ImageCore: ImageCore, Normed, gray
using LinearAlgebra: norm, dot, det
using StaticArrays: SVector, SMatrix

export SIFTConfig, SIFTKeypoint, DescriptorMatch, sift, match_descriptors,
       gaussian_pyramid, dog_pyramid

public validate, grayscale_float, double_image, gaussian_blur, gaussian_blur!,
       default_octaves, is_extremum, derivatives, localize, orientations, orientations!,
       descriptor, descriptor!, angle32

"""SIFT parameters. `first_octave=-1` doubles the input on a pixel-aligned grid.
Contrast uses OpenCV semantics (the threshold is divided by `octave_layers`).
Descriptors always have 4×4 spatial cells and 8 orientation bins.
"""
@kwdef struct SIFTConfig
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

Base.CartesianIndex(k::SIFTKeypoint) = CartesianIndex(round(Int, k.y), round(Int, k.x))

"""One-based descriptor column indices and their Euclidean (not squared) distance."""
struct DescriptorMatch
    query::Int
    train::Int
    distance::Float32
end

struct LocalizedCandidate
    o::Int
    octave::Int
    layer::Int
    iy::Int
    ix::Int
    y::Float64
    x::Float64
    sigma::Float64
    response::Float64
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

@inline intensity(v::Real) = Float32(v)
@inline intensity(v::Normed) = Float32(v)
@inline intensity(v::Unsigned) = Float32(reinterpret(Normed{typeof(v), sizeof(v) * 8}, v))
@inline intensity(v::AbstractGray) = Float32(gray(v))
@inline intensity(v::Colorant) = Float32(gray(Gray(color(v))))

@inline _chunk_range(n::Int, chunk::Int, nchunks::Int) =
    (((chunk - 1) * n) ÷ nchunks + 1):((chunk * n) ÷ nchunks)

function _grayscale_float_chunk!(a::Matrix{Float32}, invalid::Memory{Bool},
                                 img::AbstractMatrix, chunk::Int, r::UnitRange{Int})
    @inbounds for i in r
        v = intensity(img[i])
        if !(isfinite(v) && 0f0 <= v <= 1f0)
            invalid[chunk] = true
            break
        end
        a[i] = v
    end
    return nothing
end

function grayscale_float(img::AbstractMatrix{<:Union{Real,Colorant}})
    Base.require_one_based_indexing(img)
    a = Matrix{Float32}(undef, size(img))
    n = length(a)
    n == 0 && return a
    nt = Threads.nthreads()
    if nt > 1 && n >= 65_536 && IndexStyle(img) === IndexLinear()
        nchunks = min(nt, n ÷ 16_384)
        invalid = Memory{Bool}(undef, nchunks)
        fill!(invalid, false)
        @sync for chunk in 1:nchunks
            r = _chunk_range(n, chunk, nchunks)
            Threads.@spawn _grayscale_float_chunk!(a, invalid, img, chunk, r)
        end
        any(invalid) && throw(ArgumentError(
            "image intensities must be finite and in [0,1]; unsigned integers are normalized by typemax"
        ))
    else
        @inbounds for i in eachindex(a, img)
            v = intensity(img[i])
            isfinite(v) && 0f0 <= v <= 1f0 ||
                throw(ArgumentError("image intensities must be finite and in [0,1]; unsigned integers are normalized by typemax"))
            a[i] = v
        end
    end
    return a
end

# Doubling is aligned to the original zero-based pixel grid: x -> 2x.
# At the far edge a half-pixel beyond the final sample uses edge replication.
function _double_image_cols!(b::Matrix{Float32}, a::Matrix{Float32}, cols::UnitRange{Int})
    h, w = size(a)
    @inbounds for x in cols
        x0 = (x + 1) >> 1
        if isodd(x)
            for y0 in 1:h-1
                v0 = a[y0, x0]
                v1 = a[y0 + 1, x0]
                b[2y0 - 1, x] = v0
                b[2y0,     x] = 0.5f0 * (v0 + v1)
            end
            vend = a[h, x0]
            b[2h - 1, x] = vend
            b[2h,     x] = vend
        else
            x1 = min(x0 + 1, w)
            for y0 in 1:h-1
                v0 = 0.5f0 * (a[y0, x0] + a[y0, x1])
                v1 = 0.5f0 * (a[y0 + 1, x0] + a[y0 + 1, x1])
                b[2y0 - 1, x] = v0
                b[2y0,     x] = 0.5f0 * (v0 + v1)
            end
            vend = 0.5f0 * (a[h, x0] + a[h, x1])
            b[2h - 1, x] = vend
            b[2h,     x] = vend
        end
    end
    return b
end

function double_image(a::Matrix{Float32})
    h, w = size(a)
    b = Matrix{Float32}(undef, 2h, 2w)
    (h == 0 || w == 0) && return b
    nt = Threads.nthreads()
    if nt > 1 && length(b) >= 16_384
        nchunks = min(nt, 2w)
        @sync for chunk in 1:nchunks
            cols = _chunk_range(2w, chunk, nchunks)
            Threads.@spawn _double_image_cols!(b, a, cols)
        end
    else
        _double_image_cols!(b, a, 1:2w)
    end
    return b
end

@inline function _reflect101(i::Int, n::Int)
    (1 <= i <= n) && return i
    i < 1 && (2 - i <= n) && return 2 - i
    i > n && (2n - i >= 1) && return 2n - i
    p = 2n - 2
    m = mod(i - 1, p)
    return m < n ? m + 1 : p - m + 1
end

function _blur_cols!(tmp::Matrix{Float32}, src::Matrix{Float32},
                     k::Memory{Float32}, r::Int, cols::UnitRange{Int})
    h = size(src, 1)
    k0 = @inbounds k[1]
    y_lo = r + 1
    y_hi = h - r
    @inbounds for x in cols
        if y_lo <= y_hi
            for y in 1:r
                acc = k0 * src[y, x]
                for d in 1:r
                    acc = muladd(k[d + 1], src[_reflect101(y - d, h), x] + src[y + d, x], acc)
                end
                tmp[y, x] = acc
            end
            @simd for y in y_lo:y_hi
                tmp[y, x] = k0 * src[y, x]
            end
            for d in 1:r
                kd = k[d + 1]
                @simd for y in y_lo:y_hi
                    tmp[y, x] = muladd(kd, src[y - d, x] + src[y + d, x], tmp[y, x])
                end
            end
            for y in (y_hi + 1):h
                acc = k0 * src[y, x]
                for d in 1:r
                    acc = muladd(k[d + 1], src[y - d, x] + src[_reflect101(y + d, h), x], acc)
                end
                tmp[y, x] = acc
            end
        else
            for y in 1:h
                acc = k0 * src[y, x]
                for d in 1:r
                    y1 = _reflect101(y - d, h)
                    y2 = _reflect101(y + d, h)
                    acc = muladd(k[d + 1], src[y1, x] + src[y2, x], acc)
                end
                tmp[y, x] = acc
            end
        end
    end
    return tmp
end

function _blur_rows!(dst::Matrix{Float32}, tmp::Matrix{Float32},
                     k::Memory{Float32}, r::Int, cols::UnitRange{Int})
    h, w = size(tmp)
    k0 = @inbounds k[1]
    @inbounds for x in cols
        @simd for y in 1:h
            dst[y, x] = k0 * tmp[y, x]
        end
        for d in 1:r
            kd = k[d + 1]
            x1 = _reflect101(x - d, w)
            x2 = _reflect101(x + d, w)
            @simd for y in 1:h
                dst[y, x] = muladd(kd, tmp[y, x1] + tmp[y, x2], dst[y, x])
            end
        end
    end
    return dst
end

function gaussian_blur!(dst::Matrix{Float32}, tmp::Matrix{Float32}, src::Matrix{Float32}, sigma::Real)
    size(dst) == size(src) == size(tmp) ||
        throw(DimensionMismatch("dst, tmp, and src must have the same size"))
    if sigma <= eps(Float32)
        copyto!(dst, src)
        return dst
    end
    h, w = size(src)
    (h >= 2 && w >= 2) ||
        throw(ArgumentError("gaussian_blur! requires at least two pixels along each axis"))
    # A four-sigma finite symmetric FIR kernel, with reflect-101 (no repeated endpoint).
    s64 = Float64(sigma)
    len = (round(Int, 8 * s64 + 1) | 1)
    r = len ÷ 2
    k = Memory{Float32}(undef, r + 1)
    inv2s2 = -0.5 / (s64 * s64)
    s = 1.0
    @inbounds for d in 1:r
        s += 2.0 * exp((d * d) * inv2s2)
    end
    invs = 1.0 / s
    @inbounds k[1] = Float32(invs)
    @inbounds for d in 1:r
        k[d + 1] = Float32(exp((d * d) * inv2s2) * invs)
    end
    nt = Threads.nthreads()
    if nt > 1 && length(src) >= 16_384
        nchunks = min(nt, w)
        @sync for chunk in 1:nchunks
            cols = _chunk_range(w, chunk, nchunks)
            Threads.@spawn _blur_cols!(tmp, src, k, r, cols)
        end
        @sync for chunk in 1:nchunks
            cols = _chunk_range(w, chunk, nchunks)
            Threads.@spawn _blur_rows!(dst, tmp, k, r, cols)
        end
    else
        _blur_cols!(tmp, src, k, r, 1:w)
        _blur_rows!(dst, tmp, k, r, 1:w)
    end
    return dst
end

function gaussian_blur!(dst::Matrix{Float32}, src::Matrix{Float32}, sigma::Real)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    if sigma <= eps(Float32)
        copyto!(dst, src)
        return dst
    end
    return gaussian_blur!(dst, similar(src), src, sigma)
end

gaussian_blur(a::Matrix{Float32}, sigma::Real) =
    gaussian_blur!(similar(a), a, sigma)

function downsample_2x(src::Matrix{Float32})
    h, w = size(src) .÷ 2
    dst = Matrix{Float32}(undef, h, w)
    @inbounds for x in 1:w
        sx = 2x - 1
        @simd for y in 1:h
            dst[y, x] = src[2y - 1, sx]
        end
    end
    return dst
end

function default_octaves(img, c)
    m = minimum(size(img))
    m < 2 && return 0
    return max(1, round(Int, log2(m)) - 2 - c.first_octave)
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
    raw_base = c.first_octave == -1 ? double_image(a) : a
    # Reflected filtering requires at least two pixels along each octave axis.
    n <= floor(Int, log2(minimum(size(raw_base)))) ||
        throw(ArgumentError("too many octaves: every octave must have at least two pixels along each axis"))
    blur = sqrt(max(0.0, c.sigma^2 - (c.assumed_input_sigma * 2.0^(-c.first_octave))^2))
    s = c.octave_layers
    delta_sigmas = Memory{Float64}(undef, s + 2)
    @inbounds for l in 1:s+2
        delta_sigmas[l] = sqrt((c.sigma * 2.0^(l / s))^2 - (c.sigma * 2.0^((l - 1) / s))^2)
    end
    sizehint!(pyramid, n)
    tmp = similar(raw_base)
    base = similar(raw_base)
    gaussian_blur!(base, tmp, raw_base, blur)
    for o in 1:n
        h, w = size(base)
        o > 1 && (tmp = Matrix{Float32}(undef, h, w))
        levels = Vector{Matrix{Float32}}(undef, s + 3)
        levels[1] = base
        for l in 1:s+2
            dst = (o == 1 && l == 1) ? raw_base : Matrix{Float32}(undef, h, w)
            gaussian_blur!(dst, tmp, levels[l], delta_sigmas[l])
            levels[l + 1] = dst
        end
        push!(pyramid, levels)
        if o < n
            base = downsample_2x(levels[s + 1])
        end
    end
    return pyramid
end

function _diff_matrix!(dst::Matrix{Float32}, next_lvl::Matrix{Float32}, prev_lvl::Matrix{Float32})
    @inbounds @simd for i in eachindex(dst, next_lvl, prev_lvl)
        dst[i] = next_lvl[i] - prev_lvl[i]
    end
    return dst
end

function _diff_matrix(next_lvl::Matrix{Float32}, prev_lvl::Matrix{Float32})
    return _diff_matrix!(similar(next_lvl), next_lvl, prev_lvl)
end

"""Adjacent Gaussian differences, with `octave_layers + 2` levels per octave."""
function dog_pyramid(g::AbstractVector)
    isempty(g) && return Vector{Matrix{Float32}}[]
    return [[_diff_matrix(o[l + 1], o[l]) for l in 1:length(o)-1] for o in g]
end

@inline function is_extremum(d, y::Int, x::Int, l::Int)
    @inbounds begin
        dl = d[l]
        v = dl[y, x]
        iszero(v) && return false
        if v > 0f0
            (v > dl[y - 1, x] && v > dl[y + 1, x]) || return false
            (v > dl[y - 1, x - 1] && v > dl[y, x - 1] && v > dl[y + 1, x - 1] &&
             v > dl[y - 1, x + 1] && v > dl[y, x + 1] && v > dl[y + 1, x + 1]) || return false
            dp_prev = d[l - 1]
            for j in x-1:x+1, i in y-1:y+1
                dp_prev[i, j] >= v && return false
            end
            dp_next = d[l + 1]
            for j in x-1:x+1, i in y-1:y+1
                dp_next[i, j] >= v && return false
            end
        else
            (v < dl[y - 1, x] && v < dl[y + 1, x]) || return false
            (v < dl[y - 1, x - 1] && v < dl[y, x - 1] && v < dl[y + 1, x - 1] &&
             v < dl[y - 1, x + 1] && v < dl[y, x + 1] && v < dl[y + 1, x + 1]) || return false
            dp_prev = d[l - 1]
            for j in x-1:x+1, i in y-1:y+1
                dp_prev[i, j] <= v && return false
            end
            dp_next = d[l + 1]
            for j in x-1:x+1, i in y-1:y+1
                dp_next[i, j] <= v && return false
            end
        end
    end
    return true
end

# Derivative order is x, y, scale; solve the 3-D quadratic in Float64 using StaticArrays.
@inline function derivatives(d, y::Int, x::Int, l::Int)
    @inbounds begin
        p, a, n = d[l - 1], d[l], d[l + 1]
        v = Float64(a[y, x])
        dx = (Float64(a[y, x + 1]) - a[y, x - 1]) * 0.5
        dy = (Float64(a[y + 1, x]) - a[y - 1, x]) * 0.5
        ds = (Float64(n[y, x]) - p[y, x]) * 0.5
        dxx = Float64(a[y, x + 1]) + a[y, x - 1] - 2.0 * v
        dyy = Float64(a[y + 1, x]) + a[y - 1, x] - 2.0 * v
        dss = Float64(n[y, x]) + p[y, x] - 2.0 * v
        dxy = (Float64(a[y + 1, x + 1]) - a[y + 1, x - 1] - a[y - 1, x + 1] + a[y - 1, x - 1]) * 0.25
        dxs = (Float64(n[y, x + 1]) - n[y, x - 1] - p[y, x + 1] + p[y, x - 1]) * 0.25
        dys = (Float64(n[y + 1, x]) - n[y - 1, x] - p[y + 1, x] + p[y - 1, x]) * 0.25
    end
    g = SVector{3,Float64}(dx, dy, ds)
    H = SMatrix{3,3,Float64,9}(
        dxx, dxy, dxs,
        dxy, dyy, dys,
        dxs, dys, dss
    )
    return g, H
end

function localize(d, y::Int, x::Int, l::Int, c::SIFTConfig)
    h, w = size(d[1])
    for _ in 1:5
        (6 <= y <= h - 5 && 6 <= x <= w - 5 && 2 <= l <= c.octave_layers + 1) || return nothing
        g, H = derivatives(d, y, x, l)
        iszero(det(H)) && return nothing
        offset = -(H \ g)
        all(isfinite, offset) && maximum(abs, offset) < 1e6 || return nothing
        if maximum(abs, offset) < 0.5
            contrast = @inbounds d[l][y, x] + 0.5 * dot(g, offset)
            abs(contrast) * c.octave_layers >= c.contrast_threshold || return nothing
            detxy = H[1, 1] * H[2, 2] - H[1, 2]^2
            trace = H[1, 1] + H[2, 2]
            detxy > 0.0 && trace^2 * c.edge_threshold < detxy * (c.edge_threshold + 1.0)^2 || return nothing
            sigma = c.sigma * 2.0^((l - 1 + offset[3]) / c.octave_layers)
            return (y=y + offset[2], x=x + offset[1], sigma=sigma,
                    response=abs(contrast), layer=l, iy=y, ix=x)
        end
        x += round(Int, offset[1])
        y += round(Int, offset[2])
        l += round(Int, offset[3])
    end
    return nothing
end

# Full Float32-precision branchless minimax approximation of atan2(y, x).
@inline function _atan2_f32(y::Float32, x::Float32)
    ax = abs(x)
    ay = abs(y)
    mn = min(ax, ay)
    mx = max(ax, ay)
    a = ifelse(mx == 0f0, 0f0, mn / mx)
    s = a * a
    p = muladd(s, -0.0117195f0, 0.05265332f0)
    p = muladd(p, s, -0.11643287f0)
    p = muladd(p, s, 0.19354346f0)
    p = muladd(p, s, -0.33262347f0)
    p = muladd(p, s, 0.99997726f0)
    r = p * a
    r = ifelse(ay > ax, 1.57079637f0 - r, r)
    r = ifelse(x < 0f0, 3.14159274f0 - r, r)
    return ifelse(y < 0f0, -r, r)
end

# Minimax degree-5 approximation of exp(-x) on x in [0, 1.5625] for descriptor spatial weights.
@inline function _exp_neg_f32(x::Float32)
    p = muladd(x, -0.0046062f0, 0.0293684f0)
    p = muladd(p, x, -0.1573072f0)
    p = muladd(p, x, 0.4976051f0)
    p = muladd(p, x, -0.9997653f0)
    return muladd(p, x, 0.9999968f0)
end

function _orientations!(angles::Vector{Float64}, hist::AbstractVector{Float64},
                        sm::AbstractVector{Float64}, wtab::Vector{Float32},
                        a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    fill!(hist, 0.0)
    empty!(angles)
    radius = round(Int, 4.5 * sigma)
    length(wtab) < radius + 1 && resize!(wtab, radius + 1)
    inv_2sigma2 = Float32(-1.0 / (2.0 * (1.5 * sigma)^2))
    @inbounds for k in 0:radius
        wtab[k + 1] = exp(Float32(k * k) * inv_2sigma2)
    end
    h, w = size(a)
    scale = Float32(36.0 / (2pi))
    j_min = max(-radius, 2 - x)
    j_max = min(radius, w - 1 - x)
    i_min = max(-radius, 2 - y)
    i_max = min(radius, h - 1 - y)
    @inbounds for j in j_min:j_max
        xx = x + j
        wj = wtab[abs(j) + 1]
        for i in i_min:i_max
            yy = y + i
            gx = a[yy, xx + 1] - a[yy, xx - 1]
            gy = a[yy - 1, xx] - a[yy + 1, xx]
            b = round(Int, _atan2_f32(gy, gx) * scale)
            bin = ifelse(b < 0, b + 36, ifelse(b >= 36, b - 36, b)) + 1
            mag = sqrt(muladd(gx, gx, gy * gy))
            hist[bin] += Float64(mag * (wj * wtab[abs(i) + 1]))
        end
    end
    peak = 0.0
    @inbounds for b in 1:36
        v = (hist[mod1(b - 2, 36)] + 4.0 * hist[mod1(b - 1, 36)] + 6.0 * hist[b] +
             4.0 * hist[mod1(b + 1, 36)] + hist[mod1(b + 2, 36)]) * 0.0625
        sm[b] = v
        v > peak && (peak = v)
    end
    peak > 0.0 || return angles
    thresh = 0.8 * peak
    @inbounds for b in 1:36
        left, mid, right = sm[mod1(b - 1, 36)], sm[b], sm[mod1(b + 1, 36)]
        if mid > left && mid > right && mid >= thresh
            offset = (left - right) / (2.0 * (left - 2.0 * mid + right))
            # Convert the upward-y histogram angle to clockwise image coordinates.
            push!(angles, mod(-(b - 1 + offset) * (2pi / 36.0), 2pi))
        end
    end
    return angles
end

function orientations!(angles::Vector{Float64}, hist::AbstractVector{Float64}, sm::AbstractVector{Float64},
                       a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    Base.require_one_based_indexing(angles, hist, sm)
    (length(hist) == 36 && length(sm) == 36) ||
        throw(DimensionMismatch("orientation buffers must have length 36"))
    wtab = Vector{Float32}(undef, max(0, round(Int, 4.5 * sigma)) + 1)
    return _orientations!(angles, hist, sm, wtab, a, y, x, sigma)
end

orientations(a::Matrix{Float32}, y::Int, x::Int, sigma::Real) =
    orientations!(Float64[], Memory{Float64}(undef, 36), Memory{Float64}(undef, 36), a, y, x, sigma)

struct _DescriptorWorkspace
    phist::Memory{Float32}
    rb::Memory{Float32}
    cb::Memory{Float32}
    ob::Memory{Float32}
    mag::Memory{Float32}
end

_DescriptorWorkspace(cap::Int=256) = _DescriptorWorkspace(
    Memory{Float32}(undef, 288),
    Memory{Float32}(undef, cap),
    Memory{Float32}(undef, cap),
    Memory{Float32}(undef, cap),
    Memory{Float32}(undef, cap),
)

function _descriptor!(out::AbstractVector{Float32}, ws::_DescriptorWorkspace,
                      a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real)
    phist = ws.phist
    fill!(phist, 0f0)
    # Round in zero-based coordinates, then convert back. Ties-to-even would
    # choose the opposite pixel if applied directly to a one-based coordinate.
    cy, cx = round(Int, y - 1) + 1, round(Int, x - 1) + 1
    width = 3.0 * sigma
    # Internal gradient angles use upward y, as in the standard SIFT convention.
    phi = -Float64(angle)
    cs64, sn64 = cos(phi) / width, sin(phi) / width
    cs, sn = Float32(cs64), Float32(sn64)
    h, w = size(a)
    radius = min(round(Int, width * sqrt(2.0) * 2.5), ceil(Int, hypot(h, w)))
    scale = Float32(8.0 / (2pi))
    phi_f32 = Float32(phi)
    j_min = max(-radius, 2 - cx)
    j_max = min(radius, w - 1 - cx)
    i_min_base = max(-radius, 2 - cy)
    i_max_base = min(radius, h - 1 - cy)
    rb_buf, cb_buf, ob_buf, mag_buf = ws.rb, ws.cb, ws.ob, ws.mag

    @inbounds for j in j_min:j_max
        xx = cx + j
        j_cs64 = j * cs64
        j_sn64 = j * sn64
        i_lo = i_min_base
        i_hi = i_max_base
        if abs(sn64) > 1e-12
            b1 = (j_cs64 - 2.5) / sn64
            b2 = (j_cs64 + 2.5) / sn64
            lo_s, hi_s = minmax(b1, b2)
            i_lo = max(i_lo, floor(Int, lo_s) + 1)
            i_hi = min(i_hi, ceil(Int, hi_s) - 1)
        elseif !(-2.5 < j_cs64 < 2.5)
            continue
        end
        if abs(cs64) > 1e-12
            b1 = (-2.5 - j_sn64) / cs64
            b2 = ( 2.5 - j_sn64) / cs64
            lo_c, hi_c = minmax(b1, b2)
            i_lo = max(i_lo, floor(Int, lo_c) + 1)
            i_hi = min(i_hi, ceil(Int, hi_c) - 1)
        elseif !(-2.5 < j_sn64 < 2.5)
            continue
        end
        n_i = i_hi - i_lo + 1
        n_i <= 0 && continue

        j_cs = Float32(j_cs64)
        j_sn = Float32(j_sn64)
        # Pass 1: branchless SIMD math over the contiguous column slice
        @simd for idx in 1:n_i
            i = i_lo + idx - 1
            yy = cy + i
            if32 = Float32(i)
            u = muladd(-if32, sn, j_cs)
            v = muladd( if32, cs, j_sn)
            rb_buf[idx] = v + 1.5f0
            cb_buf[idx] = u + 1.5f0
            gx = a[yy, xx + 1] - a[yy, xx - 1]
            gy = a[yy - 1, xx] - a[yy + 1, xx]
            raw_ob = (_atan2_f32(gy, gx) - phi_f32) * scale
            ob_buf[idx] = raw_ob - 8f0 * floor(raw_ob * 0.125f0)
            mag_buf[idx] = sqrt(muladd(gx, gx, gy * gy)) * _exp_neg_f32((u * u + v * v) * 0.125f0)
        end

        # Pass 2: branchless 8-way trilinear scatter into 1-cell-padded 6×6×8 histogram
        for idx in 1:n_i
            rb = rb_buf[idx]
            cb = cb_buf[idx]
            ob = ob_buf[idx]
            mag = mag_buf[idx]
            r0 = floor(Int, rb)
            c0 = floor(Int, cb)
            o0 = unsafe_trunc(Int, ob)
            fr = rb - r0
            fc = cb - c0
            fo = ob - o0
            b0 = o0 & 7
            b1 = (o0 + 1) & 7
            base00 = ((r0 + 1) * 6 + (c0 + 1)) * 8 + 1
            w_r1 = mag * fr
            w_r0 = mag - w_r1
            w_r0c1 = w_r0 * fc
            w_r0c0 = w_r0 - w_r0c1
            w_r1c1 = w_r1 * fc
            w_r1c0 = w_r1 - w_r1c1

            phist[base00 + b0]      += w_r0c0 - w_r0c0 * fo
            phist[base00 + b1]      += w_r0c0 * fo
            phist[base00 + 8 + b0]  += w_r0c1 - w_r0c1 * fo
            phist[base00 + 8 + b1]  += w_r0c1 * fo
            phist[base00 + 48 + b0] += w_r1c0 - w_r1c0 * fo
            phist[base00 + 48 + b1] += w_r1c0 * fo
            phist[base00 + 56 + b0] += w_r1c1 - w_r1c1 * fo
            phist[base00 + 56 + b1] += w_r1c1 * fo
        end
    end

    s1 = 0f0
    idx = 1
    @inbounds for pr in 1:4, pc in 1:4
        base = (pr * 6 + pc) * 8
        @simd for b in 1:8
            v = phist[base + b]
            out[idx + b - 1] = v
            s1 = muladd(v, v, s1)
        end
        idx += 8
    end
    if s1 <= 0f0
        fill!(out, 0f0)
        return out
    end
    inv_n1 = 1f0 / sqrt(s1)
    s2 = 0f0
    @inbounds for k in 1:128
        v = min(out[k] * inv_n1, 0.2f0)
        out[k] = v
        s2 = muladd(v, v, s2)
    end
    inv_n2 = 1f0 / sqrt(s2)
    @inbounds @simd for k in 1:128
        out[k] *= inv_n2
    end
    return out
end

function descriptor!(out::AbstractVector{Float32}, hist::AbstractVector{Float64},
                     a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real)
    Base.require_one_based_indexing(out, hist)
    (length(out) == 128 && length(hist) == 128) ||
        throw(DimensionMismatch("descriptor buffers must have length 128"))
    _descriptor!(out, _DescriptorWorkspace(max(256, size(a, 1))), a, y, x, sigma, angle)
    @inbounds for k in 1:128
        hist[k] = Float64(out[k])
    end
    return out
end

descriptor(a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real) =
    _descriptor!(Vector{Float32}(undef, 128), _DescriptorWorkspace(max(256, size(a, 1))), a, y, x, sigma, angle)

# Rounding a value just below 2π to Float32 can otherwise leave the interval.
@inline function angle32(angle)
    a = Float32(mod(angle, 2pi))
    return a < 2pi ? a : 0f0
end

function _scan_octave_columns!(out::Vector{LocalizedCandidate}, d::Vector{Matrix{Float32}},
                               o::Int, octave::Int, l::Int, cols::UnitRange{Int},
                               h::Int, prelim_thresh::Float32, c::SIFTConfig)
    dl = d[l]
    for x in cols, y in 6:h-5
        @inbounds abs(dl[y, x]) >= prelim_thresh || continue
        is_extremum(d, y, x, l) || continue
        p = localize(d, y, x, l, c)
        isnothing(p) && continue
        push!(out, LocalizedCandidate(
            o, octave, p.layer, p.iy, p.ix, p.y, p.x, p.sigma, p.response
        ))
    end
    return out
end

function _dedup_candidates(raw_candidates::Vector{LocalizedCandidate})
    # Converged candidates at the same lattice extremum are the same feature.
    unique_candidates = LocalizedCandidate[]
    sizehint!(unique_candidates, length(raw_candidates))
    seen = Set{UInt64}()
    sizehint!(seen, length(raw_candidates))
    for cand in raw_candidates
        id = (UInt64(cand.o) << 48) | (UInt64(cand.layer) << 36) |
             (UInt64(cand.ix) << 18) | UInt64(cand.iy)
        id in seen && continue
        push!(seen, id)
        push!(unique_candidates, cand)
    end
    return unique_candidates
end

function _scan_octave!(raw_candidates::Vector{LocalizedCandidate}, d::Vector{Matrix{Float32}},
                       o::Int, c::SIFTConfig, prelim_thresh::Float32, nt::Int)
    h, w = size(d[1])
    (h >= 11 && w >= 11) || return raw_candidates
    octave = o - 1 + c.first_octave
    ncols = w - 10
    if nt > 1 && h * ncols * c.octave_layers >= 16_384
        nchunks = min(nt, ncols)
        for l in 2:c.octave_layers+1
            chunk_lists = [LocalizedCandidate[] for _ in 1:nchunks]
            @sync for chunk in 1:nchunks
                rel_cols = _chunk_range(ncols, chunk, nchunks)
                cols = (first(rel_cols) + 5):(last(rel_cols) + 5)
                Threads.@spawn _scan_octave_columns!(
                    chunk_lists[chunk], d, o, octave, l, cols, h, prelim_thresh, c
                )
            end
            for list in chunk_lists
                append!(raw_candidates, list)
            end
        end
    else
        for l in 2:c.octave_layers+1
            _scan_octave_columns!(
                raw_candidates, d, o, octave, l, 6:w-5, h, prelim_thresh, c
            )
        end
    end
    return raw_candidates
end

function _detect_candidates(dp::Vector{Vector{Matrix{Float32}}}, c::SIFTConfig)
    prelim_thresh = Float32(0.5 * c.contrast_threshold / c.octave_layers)
    nt = Threads.nthreads()
    raw_candidates = LocalizedCandidate[]
    for o in eachindex(dp)
        _scan_octave!(raw_candidates, dp[o], o, c, prelim_thresh, nt)
    end
    return _dedup_candidates(raw_candidates)
end

function _build_pyramid_and_detect(img::AbstractMatrix, c::SIFTConfig; octaves=nothing)
    a = grayscale_float(img)
    n = isnothing(octaves) ? default_octaves(a, c) : octaves
    n isa Integer && n >= 0 || throw(ArgumentError("octaves must be a nonnegative integer"))
    pyramid = Vector{Matrix{Float32}}[]
    raw_candidates = LocalizedCandidate[]
    (isempty(a) || n == 0) && return pyramid, raw_candidates
    raw_base = c.first_octave == -1 ? double_image(a) : a
    n <= floor(Int, log2(minimum(size(raw_base)))) ||
        throw(ArgumentError("too many octaves: every octave must have at least two pixels along each axis"))
    blur = sqrt(max(0.0, c.sigma^2 - (c.assumed_input_sigma * 2.0^(-c.first_octave))^2))
    s = c.octave_layers
    delta_sigmas = Memory{Float64}(undef, s + 2)
    @inbounds for l in 1:s+2
        delta_sigmas[l] = sqrt((c.sigma * 2.0^(l / s))^2 - (c.sigma * 2.0^((l - 1) / s))^2)
    end
    prelim_thresh = Float32(0.5 * c.contrast_threshold / s)
    nt = Threads.nthreads()
    sizehint!(pyramid, n)
    tmp = similar(raw_base)
    base = similar(raw_base)
    gaussian_blur!(base, tmp, raw_base, blur)
    dogs = Vector{Matrix{Float32}}(undef, s + 2)
    for o in 1:n
        h, w = size(base)
        o > 1 && (tmp = Matrix{Float32}(undef, h, w))
        levels = Vector{Matrix{Float32}}(undef, s + 3)
        levels[1] = base
        for l in 1:s+2
            dst = (o == 1 && l == 1) ? raw_base : Matrix{Float32}(undef, h, w)
            gaussian_blur!(dst, tmp, levels[l], delta_sigmas[l])
            levels[l + 1] = dst
        end
        next_base = o < n ? downsample_2x(levels[s + 1]) : base
        # Reuse levels[1], levels[s+2], levels[s+3], and tmp in-place for DoG;
        # only levels[2..s+1] are accessed during orientation and descriptor extraction.
        dogs[s + 2] = _diff_matrix!(levels[s + 3], levels[s + 3], levels[s + 2])
        dogs[s + 1] = _diff_matrix!(levels[s + 2], levels[s + 2], levels[s + 1])
        dogs[1]     = _diff_matrix!(levels[1],     levels[2],     levels[1])
        if s >= 2
            dogs[s] = _diff_matrix!(tmp, levels[s + 1], levels[s])
            for l in 2:(s - 1)
                dogs[l] = _diff_matrix(levels[l + 1], levels[l])
            end
        end
        _scan_octave!(raw_candidates, dogs, o, c, prelim_thresh, nt)
        push!(pyramid, levels)
        if o < n
            base = next_base
        end
    end
    return pyramid, _dedup_candidates(raw_candidates)
end

function _build_keypoints_range!(out::Vector{SIFTKeypoint{Float32}},
                                 cands::Vector{LocalizedCandidate},
                                 gp::Vector{Vector{Matrix{Float32}}},
                                 r::UnitRange{Int})
    angles_buf = Float64[]
    ori_hist = Memory{Float64}(undef, 36)
    ori_sm = Memory{Float64}(undef, 36)
    ori_wtab = Vector{Float32}(undef, 64)
    for idx in r
        cand = cands[idx]
        factor = 2.0^cand.octave
        _orientations!(
            angles_buf, ori_hist, ori_sm, ori_wtab,
            gp[cand.o][cand.layer], cand.iy, cand.ix, cand.sigma
        )
        y_img = Float32(1 + (cand.y - 1) * factor)
        x_img = Float32(1 + (cand.x - 1) * factor)
        s_img = Float32(cand.sigma * factor)
        resp = Float32(cand.response)
        layer0 = cand.layer - 1
        for angle in angles_buf
            push!(out, SIFTKeypoint(
                y_img, x_img, s_img, angle32(angle), resp, cand.octave, layer0
            ))
        end
    end
    return out
end

function _build_descriptors_range!(desc::Matrix{Float32},
                                   kp::Vector{SIFTKeypoint{Float32}},
                                   gp::Vector{Vector{Matrix{Float32}}},
                                   first_octave::Int,
                                   r::UnitRange{Int})
    max_h = isempty(gp) || isempty(gp[1]) ? 256 : max(256, size(gp[1][1], 1))
    ws = _DescriptorWorkspace(max_h)
    @views for i in r
        k = kp[i]
        factor = 2.0^k.octave
        _descriptor!(
            desc[:, i], ws,
            gp[k.octave - first_octave + 1][k.layer + 1],
            1 + (k.y - 1) / factor,
            1 + (k.x - 1) / factor,
            k.sigma / factor,
            k.angle
        )
    end
    return desc
end

"""
    keypoints, descriptors = sift(img; config=SIFTConfig(), octaves=nothing)

Native SIFT detection and description. Returns floating-point keypoints and a
128×N Float32 matrix with unit-L2 columns. Multiple strong orientations create
separate features. `nfeatures=0` keeps all features; otherwise keep the strongest.
"""
function sift(img::AbstractMatrix; config::SIFTConfig=SIFTConfig(), octaves=nothing)
    c = validate(config)
    gp, unique_cands = _build_pyramid_and_detect(img, c; octaves)
    ncands = length(unique_cands)
    nt = Threads.nthreads()
    kp = SIFTKeypoint{Float32}[]
    if ncands > 0
        if nt > 1 && ncands >= 32
            nchunks = min(nt, ncands)
            chunk_kps = [SIFTKeypoint{Float32}[] for _ in 1:nchunks]
            @sync for chunk in 1:nchunks
                r = _chunk_range(ncands, chunk, nchunks)
                Threads.@spawn _build_keypoints_range!(chunk_kps[chunk], unique_cands, gp, r)
            end
            total_kp = sum(length, chunk_kps)
            sizehint!(kp, total_kp)
            for list in chunk_kps
                append!(kp, list)
            end
        else
            _build_keypoints_range!(kp, unique_cands, gp, 1:ncands)
        end
    end
    sort!(kp; by=k -> (-k.response, k.octave, k.layer, k.y, k.x, k.angle))
    c.nfeatures > 0 && resize!(kp, min(c.nfeatures, length(kp)))
    nkp = length(kp)
    desc = Matrix{Float32}(undef, 128, nkp)
    if nkp > 0
        if nt > 1 && nkp >= 16
            nchunks = min(nt, nkp)
            @sync for chunk in 1:nchunks
                r = _chunk_range(nkp, chunk, nchunks)
                Threads.@spawn _build_descriptors_range!(desc, kp, gp, c.first_octave, r)
            end
        else
            _build_descriptors_range!(desc, kp, gp, c.first_octave, 1:nkp)
        end
    end
    return kp, desc
end

@inline function sqdist128(a::AbstractMatrix{Float32}, i::Int, b::AbstractMatrix{Float32}, j::Int)
    s = 0f0
    @inbounds @simd for k in 1:128
        diff = a[k, i] - b[k, j]
        s = muladd(diff, diff, s)
    end
    return Float64(s)
end

@inline function sqdist128(a::AbstractMatrix, i::Int, b::AbstractMatrix, j::Int)
    s = 0.0
    @inbounds @simd for k in 1:128
        diff = Float64(a[k, i]) - Float64(b[k, j])
        s = muladd(diff, diff, s)
    end
    return s
end

function _match_chunk!(chunk_matches::Vector{DescriptorMatch},
                       rev_dist::AbstractVector{Float64}, rev_idx::AbstractVector{Int},
                       a::AbstractMatrix{Float32}, b::AbstractMatrix{Float32},
                       qrange::UnitRange{Int}, r2::Float64, crosscheck::Bool)
    nb = size(b, 2)
    i = first(qrange)
    i_end = last(qrange)
    @inbounds while i + 3 <= i_end
        b1, s1, idx1 = Inf32, Inf32, 0
        b2, s2, idx2 = Inf32, Inf32, 0
        b3, s3, idx3 = Inf32, Inf32, 0
        b4, s4, idx4 = Inf32, Inf32, 0
        for j in 1:nb
            d1 = 0f0; d2 = 0f0; d3 = 0f0; d4 = 0f0
            @simd for k in 1:128
                bj = b[k, j]
                df1 = a[k, i]     - bj; d1 = muladd(df1, df1, d1)
                df2 = a[k, i + 1] - bj; d2 = muladd(df2, df2, d2)
                df3 = a[k, i + 2] - bj; d3 = muladd(df3, df3, d3)
                df4 = a[k, i + 3] - bj; d4 = muladd(df4, df4, d4)
            end
            if d1 < b1; s1, b1, idx1 = b1, d1, j elseif d1 < s1; s1 = d1 end
            if d2 < b2; s2, b2, idx2 = b2, d2, j elseif d2 < s2; s2 = d2 end
            if d3 < b3; s3, b3, idx3 = b3, d3, j elseif d3 < s3; s3 = d3 end
            if d4 < b4; s4, b4, idx4 = b4, d4, j elseif d4 < s4; s4 = d4 end
            if crosscheck
                rd = rev_dist[j]
                ri = rev_idx[j]
                if Float64(d1) < rd; rd = Float64(d1); ri = i     end
                if Float64(d2) < rd; rd = Float64(d2); ri = i + 1 end
                if Float64(d3) < rd; rd = Float64(d3); ri = i + 2 end
                if Float64(d4) < rd; rd = Float64(d4); ri = i + 3 end
                rev_dist[j] = rd
                rev_idx[j] = ri
            end
        end
        if (b1 == 0f0 && s1 > 0f0) || Float64(b1) < r2 * Float64(s1)
            push!(chunk_matches, DescriptorMatch(i, idx1, sqrt(b1)))
        end
        if (b2 == 0f0 && s2 > 0f0) || Float64(b2) < r2 * Float64(s2)
            push!(chunk_matches, DescriptorMatch(i + 1, idx2, sqrt(b2)))
        end
        if (b3 == 0f0 && s3 > 0f0) || Float64(b3) < r2 * Float64(s3)
            push!(chunk_matches, DescriptorMatch(i + 2, idx3, sqrt(b3)))
        end
        if (b4 == 0f0 && s4 > 0f0) || Float64(b4) < r2 * Float64(s4)
            push!(chunk_matches, DescriptorMatch(i + 3, idx4, sqrt(b4)))
        end
        i += 4
    end
    @inbounds while i <= i_end
        b1, s1, idx1 = Inf32, Inf32, 0
        for j in 1:nb
            d1 = 0f0
            @simd for k in 1:128
                df1 = a[k, i] - b[k, j]
                d1 = muladd(df1, df1, d1)
            end
            if d1 < b1
                s1, b1, idx1 = b1, d1, j
            elseif d1 < s1
                s1 = d1
            end
            if crosscheck && Float64(d1) < rev_dist[j]
                rev_dist[j] = Float64(d1)
                rev_idx[j] = i
            end
        end
        if (b1 == 0f0 && s1 > 0f0) || Float64(b1) < r2 * Float64(s1)
            push!(chunk_matches, DescriptorMatch(i, idx1, sqrt(b1)))
        end
        i += 1
    end
    return chunk_matches
end

function _match_chunk!(chunk_matches::Vector{DescriptorMatch},
                       rev_dist::AbstractVector{Float64}, rev_idx::AbstractVector{Int},
                       a::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real},
                       qrange::UnitRange{Int}, r2::Float64, crosscheck::Bool)
    nb = size(b, 2)
    @inbounds for i in qrange
        best, second, index = Inf, Inf, 0
        for j in 1:nb
            d = sqdist128(a, i, b, j)
            if d < best
                second, best, index = best, d, j
            elseif d < second
                second = d
            end
            if crosscheck && d < rev_dist[j]
                rev_dist[j] = d
                rev_idx[j] = i
            end
        end
        # A unique exact match passes every positive ratio, including values
        # whose square underflows. Duplicate exact neighbors still fail.
        if (best == 0.0 && second > 0.0) || best < r2 * second
            push!(chunk_matches, DescriptorMatch(i, index, Float32(sqrt(best))))
        end
    end
    return chunk_matches
end

@inline function _alloc_rev_buffers(nb::Int, crosscheck::Bool)
    if crosscheck
        rdist = Memory{Float64}(undef, nb)
        fill!(rdist, Inf)
        ridx = Memory{Int}(undef, nb)
        fill!(ridx, 0)
        return rdist, ridx
    else
        return Memory{Float64}(undef, 0), Memory{Int}(undef, 0)
    end
end

function _filter_crosscheck!(matches::Vector{DescriptorMatch}, reverse_idx::Memory{Int})
    filter!(m -> @inbounds(reverse_idx[m.train]) == m.query, matches)
    return matches
end

"""
    match_descriptors(a, b; ratio=0.8, crosscheck=false)

Brute-force L2 matching for 128×N descriptors. Accept only a *strict* Lowe ratio
between the nearest and second-nearest neighbors. Fewer than two train columns
or tied nearest neighbors yield no match. `crosscheck=true` also requires a
mutual nearest neighbor (it does not apply a reverse ratio test).
"""
function match_descriptors(a::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real}; ratio::Real=0.8, crosscheck::Bool=false)
    Base.require_one_based_indexing(a, b)
    size(a, 1) == size(b, 1) == 128 || throw(DimensionMismatch("descriptors must have 128 rows"))
    isfinite(ratio) && 0 < ratio < 1 || throw(ArgumentError("ratio must be in (0,1)"))
    all(isfinite, a) && all(isfinite, b) || throw(ArgumentError("descriptors must be finite"))
    matches = DescriptorMatch[]
    na, nb = size(a, 2), size(b, 2)
    (na >= 1 && nb >= 2) || return matches
    r2 = Float64(ratio)^2
    nt = Threads.nthreads()
    nblocks = cld(na, 4)
    if nt > 1 && na * nb >= 4096 && nblocks >= 2
        nchunks = min(nt, nblocks)
        chunk_matches = [DescriptorMatch[] for _ in 1:nchunks]
        chunk_rev = [_alloc_rev_buffers(nb, crosscheck) for _ in 1:nchunks]
        @sync for chunk in 1:nchunks
            brange = _chunk_range(nblocks, chunk, nchunks)
            qrange = (4 * (first(brange) - 1) + 1):min(na, 4 * last(brange))
            rdist, ridx = chunk_rev[chunk]
            Threads.@spawn _match_chunk!(
                chunk_matches[chunk], rdist, ridx, a, b, qrange, r2, crosscheck
            )
        end
        total_matches = sum(length, chunk_matches)
        sizehint!(matches, total_matches)
        for list in chunk_matches
            append!(matches, list)
        end
        if crosscheck
            reverse_dist, reverse_idx = chunk_rev[1]
            @inbounds for chunk in 2:nchunks
                rdist, ridx = chunk_rev[chunk]
                for j in 1:nb
                    if rdist[j] < reverse_dist[j]
                        reverse_dist[j] = rdist[j]
                        reverse_idx[j] = ridx[j]
                    end
                end
            end
            _filter_crosscheck!(matches, reverse_idx)
        end
    else
        reverse_dist, reverse_idx = _alloc_rev_buffers(nb, crosscheck)
        _match_chunk!(matches, reverse_dist, reverse_idx, a, b, 1:na, r2, crosscheck)
        crosscheck && _filter_crosscheck!(matches, reverse_idx)
    end
    return matches
end

end
