module SIFTFeatures

using ColorTypes: Colorant, Gray, AbstractGray, color, gray
using Colors: Colors # registers Colorant -> Gray luminance conversions
using FixedPointNumbers: Normed
using LinearAlgebra: norm, dot, det
using StaticArrays: SVector, SMatrix

export SIFTConfig, SIFTKeypoint, DescriptorMatch, sift, match_descriptors,
       gaussian_pyramid, dog_pyramid, rootsift, rootsift!

public validate, grayscale_float, double_image, gaussian_blur, gaussian_blur!,
       default_octaves, is_extremum, derivatives, localize, orientations, orientations!,
       descriptor, descriptor!, angle32, OrientationWorkspace, DescriptorWorkspace

"""SIFT parameters. `first_octave=-1` doubles the input on a pixel-aligned grid.
Contrast uses OpenCV semantics (the threshold is divided by `octave_layers`).
Descriptors always have 4×4 spatial cells and 8 orientation bins. Set `rootsift=true`
to return L1-normalized elementwise square-root (RootSIFT / Hellinger) descriptors,
which remain unit-L2 normalized for standard Euclidean matching.
"""
@kwdef struct SIFTConfig
    nfeatures::Int = 0
    octave_layers::Int = 3
    contrast_threshold::Float64 = 0.04
    edge_threshold::Float64 = 10.0
    sigma::Float64 = 1.6
    assumed_input_sigma::Float64 = 0.5
    first_octave::Int = -1
    rootsift::Bool = false
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

"""Validate a [`SIFTConfig`](@ref), throwing `ArgumentError` on invalid parameters."""
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
# Uses ColorTypes.Gray(color(v)) luminance conversion (matching JuliaImages Gray.(img)).
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

"""Convert a one-based real, unsigned integer, or color matrix into a `Matrix{Float32}`
with intensities validated to lie in `[0, 1]`. Unsigned pixels are divided by `typemax`.
"""
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

"""Upsample `a` by `2×` using pixel-aligned bilinear interpolation (`x -> 2x`, edge replication at the far border)."""
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

@inline _load8(a::Matrix{Float32}, y::Int, x::Int) =
    unsafe_load(Ptr{SVector{8,Float32}}(pointer(a, (x - 1) * size(a, 1) + y)))
@inline _store8!(a::Matrix{Float32}, y::Int, x::Int, v::SVector{8,Float32}) =
    unsafe_store!(Ptr{SVector{8,Float32}}(pointer(a, (x - 1) * size(a, 1) + y)), v)
@inline _load2(a::Memory{Float32}, i::Int) =
    unsafe_load(Ptr{SVector{2,Float32}}(pointer(a, i)))
@inline _store2!(a::Memory{Float32}, i::Int, v::SVector{2,Float32}) =
    unsafe_store!(Ptr{SVector{2,Float32}}(pointer(a, i)), v)

@noinline function _blur_cols!(tmp::Matrix{Float32}, src::Matrix{Float32},
                               k::Memory{Float32}, r::Int, cols::UnitRange{Int})
    h = size(src, 1)
    k0 = @inbounds k[1]
    y_lo = r + 1
    y_hi = h - r
    GC.@preserve tmp src @inbounds for x in cols
        if y_lo <= y_hi
            for y in 1:r
                acc = k0 * src[y, x]
                for d in 1:r
                    acc = muladd(k[d + 1], src[_reflect101(y - d, h), x] + src[y + d, x], acc)
                end
                tmp[y, x] = acc
            end
            y = y_lo
            while y + 31 <= y_hi
                acc0 = k0 * _load8(src, y, x)
                acc1 = k0 * _load8(src, y + 8, x)
                acc2 = k0 * _load8(src, y + 16, x)
                acc3 = k0 * _load8(src, y + 24, x)
                for d in 1:r
                    kd = k[d + 1]
                    acc0 = muladd(kd, _load8(src, y - d, x)      + _load8(src, y + d, x),      acc0)
                    acc1 = muladd(kd, _load8(src, y + 8 - d, x)  + _load8(src, y + 8 + d, x),  acc1)
                    acc2 = muladd(kd, _load8(src, y + 16 - d, x) + _load8(src, y + 16 + d, x), acc2)
                    acc3 = muladd(kd, _load8(src, y + 24 - d, x) + _load8(src, y + 24 + d, x), acc3)
                end
                _store8!(tmp, y, x, acc0)
                _store8!(tmp, y + 8, x, acc1)
                _store8!(tmp, y + 16, x, acc2)
                _store8!(tmp, y + 24, x, acc3)
                y += 32
            end
            if y <= y_hi
                @simd for yi in y:y_hi
                    tmp[yi, x] = k0 * src[yi, x]
                end
                for d in 1:r
                    kd = k[d + 1]
                    @simd for yi in y:y_hi
                        tmp[yi, x] = muladd(kd, src[yi - d, x] + src[yi + d, x], tmp[yi, x])
                    end
                end
            end
            for y_tail in (y_hi + 1):h
                acc = k0 * src[y_tail, x]
                for d in 1:r
                    acc = muladd(k[d + 1], src[y_tail - d, x] + src[_reflect101(y_tail + d, h), x], acc)
                end
                tmp[y_tail, x] = acc
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

@noinline function _blur_rows!(dst::Matrix{Float32}, tmp::Matrix{Float32},
                               k::Memory{Float32}, r::Int, cols::UnitRange{Int})
    h, w = size(tmp)
    k0 = @inbounds k[1]
    GC.@preserve dst tmp @inbounds for x in cols
        y = 1
        if r < x <= w - r
            while y + 31 <= h
                acc0 = k0 * _load8(tmp, y, x)
                acc1 = k0 * _load8(tmp, y + 8, x)
                acc2 = k0 * _load8(tmp, y + 16, x)
                acc3 = k0 * _load8(tmp, y + 24, x)
                for d in 1:r
                    kd = k[d + 1]
                    x1 = x - d
                    x2 = x + d
                    acc0 = muladd(kd, _load8(tmp, y, x1)      + _load8(tmp, y, x2),      acc0)
                    acc1 = muladd(kd, _load8(tmp, y + 8, x1)  + _load8(tmp, y + 8, x2),  acc1)
                    acc2 = muladd(kd, _load8(tmp, y + 16, x1) + _load8(tmp, y + 16, x2), acc2)
                    acc3 = muladd(kd, _load8(tmp, y + 24, x1) + _load8(tmp, y + 24, x2), acc3)
                end
                _store8!(dst, y, x, acc0)
                _store8!(dst, y + 8, x, acc1)
                _store8!(dst, y + 16, x, acc2)
                _store8!(dst, y + 24, x, acc3)
                y += 32
            end
            if y <= h
                @simd for yi in y:h
                    dst[yi, x] = k0 * tmp[yi, x]
                end
                for d in 1:r
                    kd = k[d + 1]
                    @simd for yi in y:h
                        dst[yi, x] = muladd(kd, tmp[yi, x - d] + tmp[yi, x + d], dst[yi, x])
                    end
                end
            end
        else
            while y + 31 <= h
                acc0 = k0 * _load8(tmp, y, x)
                acc1 = k0 * _load8(tmp, y + 8, x)
                acc2 = k0 * _load8(tmp, y + 16, x)
                acc3 = k0 * _load8(tmp, y + 24, x)
                for d in 1:r
                    kd = k[d + 1]
                    x1 = _reflect101(x - d, w)
                    x2 = _reflect101(x + d, w)
                    acc0 = muladd(kd, _load8(tmp, y, x1)      + _load8(tmp, y, x2),      acc0)
                    acc1 = muladd(kd, _load8(tmp, y + 8, x1)  + _load8(tmp, y + 8, x2),  acc1)
                    acc2 = muladd(kd, _load8(tmp, y + 16, x1) + _load8(tmp, y + 16, x2), acc2)
                    acc3 = muladd(kd, _load8(tmp, y + 24, x1) + _load8(tmp, y + 24, x2), acc3)
                end
                _store8!(dst, y, x, acc0)
                _store8!(dst, y + 8, x, acc1)
                _store8!(dst, y + 16, x, acc2)
                _store8!(dst, y + 24, x, acc3)
                y += 32
            end
            if y <= h
                @simd for yi in y:h
                    dst[yi, x] = k0 * tmp[yi, x]
                end
                for d in 1:r
                    kd = k[d + 1]
                    x1 = _reflect101(x - d, w)
                    x2 = _reflect101(x + d, w)
                    @simd for yi in y:h
                        dst[yi, x] = muladd(kd, tmp[yi, x1] + tmp[yi, x2], dst[yi, x])
                    end
                end
            end
        end
    end
    return dst
end

"""Separable symmetric four-sigma Float32 Gaussian blur with reflect-101 borders, writing into `dst`."""
function gaussian_blur!(dst::Matrix{Float32}, tmp::Matrix{Float32}, src::Matrix{Float32}, sigma::Real)
    size(dst) == size(src) == size(tmp) ||
        throw(DimensionMismatch("dst, tmp, and src must have the same size"))
    (tmp === dst || tmp === src) &&
        throw(ArgumentError("tmp must not alias dst or src"))
    isfinite(sigma) && sigma >= 0 && 8.0 * Float64(sigma) + 1.0 < Float64(typemax(Int32)) ||
        throw(ArgumentError("sigma must be finite, nonnegative, and within kernel size limits"))
    h, w = size(src)
    (h >= 2 && w >= 2) ||
        throw(ArgumentError("gaussian_blur! requires at least two pixels along each axis"))
    if sigma <= eps(Float32)
        copyto!(dst, src)
        return dst
    end
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
    isfinite(sigma) && sigma >= 0 && 8.0 * Float64(sigma) + 1.0 < Float64(typemax(Int32)) ||
        throw(ArgumentError("sigma must be finite, nonnegative, and within kernel size limits"))
    h, w = size(src)
    (h >= 2 && w >= 2) ||
        throw(ArgumentError("gaussian_blur! requires at least two pixels along each axis"))
    if sigma <= eps(Float32)
        copyto!(dst, src)
        return dst
    end
    return gaussian_blur!(dst, similar(src), src, sigma)
end

"""Return a Gaussian-blurred copy of `a` with standard deviation `sigma`. See [`gaussian_blur!`](@ref)."""
gaussian_blur(a::Matrix{Float32}, sigma::Real) =
    gaussian_blur!(similar(a), a, sigma)

function downsample_2x!(dst::Matrix{Float32}, src::Matrix{Float32})
    h, w = size(dst)
    @inbounds for x in 1:w
        sx = 2x - 1
        @simd for y in 1:h
            dst[y, x] = src[2y - 1, sx]
        end
    end
    return dst
end

downsample_2x(src::Matrix{Float32}) =
    downsample_2x!(Matrix{Float32}(undef, size(src) .÷ 2), src)

"""Compute the default number of SIFT pyramid octaves for image `img` and configuration `c`."""
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
    n isa Integer && !(n isa Bool) && n >= 0 || throw(ArgumentError("octaves must be a nonnegative integer"))
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
    Base.require_one_based_indexing(g)
    isempty(g) && return Vector{Matrix{Float32}}[]
    for o in g
        Base.require_one_based_indexing(o)
    end
    return [[_diff_matrix(o[l + 1], o[l]) for l in 1:length(o)-1] for o in g]
end

@inline function _check_dog_neighborhood(d, y::Int, x::Int, l::Int)
    Base.require_one_based_indexing(d)
    (2 <= l <= length(d) - 1) ||
        throw(ArgumentError("scale level l must satisfy 2 <= l <= length(d) - 1"))
    p, a, n = d[l - 1], d[l], d[l + 1]
    Base.require_one_based_indexing(p, a, n)
    size(p) == size(a) == size(n) ||
        throw(DimensionMismatch("adjacent DoG levels must have the same dimensions"))
    h, w = size(a)
    (2 <= y <= h - 1 && 2 <= x <= w - 1) ||
        throw(ArgumentError("(y, x) must lie in the interior 2:h-1 × 2:w-1"))
    return nothing
end

@inline function _is_extremum(dp::Matrix{Float32}, dl::Matrix{Float32}, dn::Matrix{Float32},
                              y::Int, x::Int, v::Float32)
    @inbounds begin
        iszero(v) && return false
        if v > 0f0
            (v > dl[y - 1, x] && v > dl[y + 1, x] &&
             v > dl[y, x - 1] && v > dl[y, x + 1] &&
             v > dp[y, x]     && v > dn[y, x]) || return false
            (v > dl[y - 1, x - 1] && v > dl[y + 1, x - 1] &&
             v > dl[y - 1, x + 1] && v > dl[y + 1, x + 1]) || return false
            (v > dp[y - 1, x - 1] && v > dp[y, x - 1] && v > dp[y + 1, x - 1] &&
             v > dp[y - 1, x]                         && v > dp[y + 1, x]     &&
             v > dp[y - 1, x + 1] && v > dp[y, x + 1] && v > dp[y + 1, x + 1]) || return false
            (v > dn[y - 1, x - 1] && v > dn[y, x - 1] && v > dn[y + 1, x - 1] &&
             v > dn[y - 1, x]                         && v > dn[y + 1, x]     &&
             v > dn[y - 1, x + 1] && v > dn[y, x + 1] && v > dn[y + 1, x + 1]) || return false
        else
            (v < dl[y - 1, x] && v < dl[y + 1, x] &&
             v < dl[y, x - 1] && v < dl[y, x + 1] &&
             v < dp[y, x]     && v < dn[y, x]) || return false
            (v < dl[y - 1, x - 1] && v < dl[y + 1, x - 1] &&
             v < dl[y - 1, x + 1] && v < dl[y + 1, x + 1]) || return false
            (v < dp[y - 1, x - 1] && v < dp[y, x - 1] && v < dp[y + 1, x - 1] &&
             v < dp[y - 1, x]                         && v < dp[y + 1, x]     &&
             v < dp[y - 1, x + 1] && v < dp[y, x + 1] && v < dp[y + 1, x + 1]) || return false
            (v < dn[y - 1, x - 1] && v < dn[y, x - 1] && v < dn[y + 1, x - 1] &&
             v < dn[y - 1, x]                         && v < dn[y + 1, x]     &&
             v < dn[y - 1, x + 1] && v < dn[y, x + 1] && v < dn[y + 1, x + 1]) || return false
        end
    end
    return true
end

@inline function _is_extremum(d, y::Int, x::Int, l::Int)
    @inbounds begin
        dl = d[l]
        return _is_extremum(d[l - 1], dl, d[l + 1], y, x, dl[y, x])
    end
end

"""Test whether DoG sample `(y, x, l)` is strictly greater or strictly less than all 26 scale-space neighbors."""
@inline function is_extremum(d, y::Int, x::Int, l::Int)
    _check_dog_neighborhood(d, y, x, l)
    return _is_extremum(d, y, x, l)
end

# Derivative order is x, y, scale; solve the 3-D quadratic in Float64 using StaticArrays.
@inline function _derivatives(d, y::Int, x::Int, l::Int)
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

"""Compute the 3D finite-difference gradient and Hessian `(g, H)` (ordered `x, y, scale`) at DoG sample `(y, x, l)`."""
@inline function derivatives(d, y::Int, x::Int, l::Int)
    _check_dog_neighborhood(d, y, x, l)
    return _derivatives(d, y, x, l)
end

function _localize(d, y::Int, x::Int, l::Int, c::SIFTConfig)
    h, w = size(d[1])
    for _ in 1:5
        (6 <= y <= h - 5 && 6 <= x <= w - 5 && 2 <= l <= c.octave_layers + 1) || return nothing
        g, H = _derivatives(d, y, x, l)
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

"""Iteratively refine a discrete DoG extremum `(y, x, l)` via 3D quadratic interpolation, rejecting low-contrast and edge responses. Returns a named tuple or `nothing`."""
function localize(d, y::Int, x::Int, l::Int, c::SIFTConfig)
    validate(c)
    Base.require_one_based_indexing(d)
    length(d) >= c.octave_layers + 2 ||
        throw(ArgumentError("DoG octave must have at least octave_layers + 2 levels"))
    Base.require_one_based_indexing(d[1])
    s1 = size(d[1])
    for k in 2:(c.octave_layers + 2)
        Base.require_one_based_indexing(d[k])
        size(d[k]) == s1 || throw(DimensionMismatch("all DoG levels in an octave must have the same size"))
    end
    return _localize(d, y, x, l, c)
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

# Minimax degree-6 approximation of exp(-x) on x in [0, 1.5625] for descriptor spatial weights
# (Remez fit; max Float32 absolute error < 4e-7, max relative error < 2e-6 on [0, 1.5625]).
@inline function _exp_neg_f32(x::Float32)
    p = muladd(x, 0.0006516392f0, -0.0069679245f0)
    p = muladd(p, x, 0.04032194f0)
    p = muladd(p, x, -0.16596726f0)
    p = muladd(p, x, 0.49982446f0)
    p = muladd(p, x, -0.9999834f0)
    return muladd(p, x, 0.99999976f0)
end

"""Reusable workspace for zero-allocation calls to [`orientations!`](@ref)."""
struct OrientationWorkspace
    hist::Memory{Float64}
    sm::Memory{Float64}
    wtab::Vector{Float32}
    gx::Vector{Float32}
    gy::Vector{Float32}
    w::Vector{Float32}
    bin::Vector{Int32}
end

function OrientationWorkspace(cap::Int=64)
    wcap = max(1, cap)
    pcap = max(1024, wcap * 16)
    return OrientationWorkspace(
        Memory{Float64}(undef, 36),
        Memory{Float64}(undef, 36),
        Vector{Float32}(undef, wcap),
        Vector{Float32}(undef, pcap),
        Vector{Float32}(undef, pcap),
        Vector{Float32}(undef, pcap),
        Vector{Int32}(undef, pcap),
    )
end

@noinline function _orientations!(angles::Vector{Float64}, ws::OrientationWorkspace,
                                  a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    hist = ws.hist
    sm = ws.sm
    fill!(hist, 0.0)
    empty!(angles)
    h, w = size(a)
    radius = round(Int, 4.5 * sigma)
    j_min = max(-radius, 2 - x)
    j_max = min(radius, w - 1 - x)
    i_min = max(-radius, 2 - y)
    i_max = min(radius, h - 1 - y)
    (j_min > j_max || i_min > i_max) && return angles
    max_k = max(abs(j_min), abs(j_max), abs(i_min), abs(i_max))
    wtab = ws.wtab
    length(wtab) < max_k + 1 && resize!(wtab, max_k + 1)
    inv_2sigma2 = Float32(-1.0 / (2.0 * (1.5 * sigma)^2))
    @inbounds for k in 0:max_k
        wtab[k + 1] = exp(Float32(k * k) * inv_2sigma2)
    end
    n_i = i_max - i_min + 1
    n_j = j_max - j_min + 1
    n = n_i * n_j
    if length(ws.gx) < n
        new_cap = max(n, 2 * length(ws.gx))
        resize!(ws.gx, new_cap)
        resize!(ws.gy, new_cap)
        resize!(ws.w, new_cap)
        resize!(ws.bin, new_cap)
    end
    gx_buf = ws.gx
    gy_buf = ws.gy
    w_buf = ws.w
    bin_buf = ws.bin

    # Pack contiguous column gradients and Gaussian weights into flat L1 buffers.
    off = 0
    yy0 = y + i_min
    @inbounds for j in j_min:j_max
        xx = x + j
        wj = wtab[abs(j) + 1]
        @simd for idx in 1:n_i
            i = i_min + idx - 1
            yy = yy0 + idx - 1
            gx_buf[off + idx] = a[yy, xx + 1] - a[yy, xx - 1]
            gy_buf[off + idx] = a[yy - 1, xx] - a[yy + 1, xx]
            w_buf[off + idx]  = wj * wtab[abs(i) + 1]
        end
        off += n_i
    end

    # Single flat SIMD sweep for atan2, sqrt, and bin index; overwrites w_buf in place.
    scale = Float32(36.0 / (2pi))
    @inbounds @simd for k in 1:n
        gx = gx_buf[k]
        gy = gy_buf[k]
        b = unsafe_trunc(Int32, round(_atan2_f32(gy, gx) * scale))
        bin_buf[k] = ifelse(b < Int32(0), b + Int32(36), ifelse(b >= Int32(36), b - Int32(36), b)) + Int32(1)
        w_buf[k] = Base.sqrt_llvm(muladd(gx, gx, gy * gy)) * w_buf[k]
    end

    @inbounds for k in 1:n
        hist[bin_buf[k]] += Float64(w_buf[k])
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

@noinline function _orientations!(angles::Vector{Float64}, hist::AbstractVector{Float64},
                                  sm::AbstractVector{Float64}, wtab::Vector{Float32},
                                  a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    fill!(hist, 0.0)
    empty!(angles)
    h, w = size(a)
    radius = round(Int, 4.5 * sigma)
    j_min = max(-radius, 2 - x)
    j_max = min(radius, w - 1 - x)
    i_min = max(-radius, 2 - y)
    i_max = min(radius, h - 1 - y)
    (j_min > j_max || i_min > i_max) && return angles
    max_k = max(abs(j_min), abs(j_max), abs(i_min), abs(i_max))
    length(wtab) < max_k + 1 && resize!(wtab, max_k + 1)
    inv_2sigma2 = Float32(-1.0 / (2.0 * (1.5 * sigma)^2))
    @inbounds for k in 0:max_k
        wtab[k + 1] = exp(Float32(k * k) * inv_2sigma2)
    end
    scale = Float32(36.0 / (2pi))
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

"""Compute dominant keypoint orientations in radians `[0, 2π)` (clockwise from `+x`) into `angles` using a reusable [`OrientationWorkspace`](@ref)."""
function orientations!(angles::Vector{Float64}, ws::OrientationWorkspace,
                       a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    Base.require_one_based_indexing(angles)
    isfinite(sigma) && sigma > 0 && 4.5 * Float64(sigma) < Float64(typemax(Int32)) ||
        throw(ArgumentError("sigma must be finite, positive, and within orientation window limits"))
    return _orientations!(angles, ws, a, y, x, sigma)
end

function orientations!(angles::Vector{Float64}, hist::AbstractVector{Float64}, sm::AbstractVector{Float64},
                       a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    Base.require_one_based_indexing(angles, hist, sm)
    (length(hist) == 36 && length(sm) == 36) ||
        throw(DimensionMismatch("orientation buffers must have length 36"))
    isfinite(sigma) && sigma > 0 && 4.5 * Float64(sigma) < Float64(typemax(Int32)) ||
        throw(ArgumentError("sigma must be finite, positive, and within orientation window limits"))
    h, w = size(a)
    radius = round(Int, 4.5 * sigma)
    j_min = max(-radius, 2 - x); j_max = min(radius, w - 1 - x)
    i_min = max(-radius, 2 - y); i_max = min(radius, h - 1 - y)
    max_k = (j_min > j_max || i_min > i_max) ? 0 : max(abs(j_min), abs(j_max), abs(i_min), abs(i_max))
    wtab = Vector{Float32}(undef, max_k + 1)
    return _orientations!(angles, hist, sm, wtab, a, y, x, sigma)
end

"""Compute dominant keypoint orientations in radians `[0, 2π)` around `(y, x)` at octave scale `sigma`. See [`orientations!`](@ref)."""
orientations(a::Matrix{Float32}, y::Int, x::Int, sigma::Real) =
    orientations!(Float64[], OrientationWorkspace(), a, y, x, sigma)

"""Reusable workspace for zero-allocation calls to [`descriptor!`](@ref)."""
mutable struct DescriptorWorkspace
    const phist::Memory{Float32} # 6×6×10 padded histogram (bins 9..10 fold into 1..2)
    rb::Memory{Float32}
    cb::Memory{Float32}
    gx::Memory{Float32}
    gy::Memory{Float32}
    ew::Memory{Float32}
    idx::Memory{Int32}
end

const _DescriptorWorkspace = DescriptorWorkspace

function DescriptorWorkspace(cap::Int=4096)
    c = max(0, cap)
    return DescriptorWorkspace(
        Memory{Float32}(undef, 360),
        Memory{Float32}(undef, c),
        Memory{Float32}(undef, c),
        Memory{Float32}(undef, c),
        Memory{Float32}(undef, c),
        Memory{Float32}(undef, c),
        Memory{Int32}(undef, c),
    )
end

@noinline function _grow_descriptor_workspace!(ws::DescriptorWorkspace, cap::Int, keep::Int=0)
    new_rb = Memory{Float32}(undef, cap)
    new_cb = Memory{Float32}(undef, cap)
    new_gx = Memory{Float32}(undef, cap)
    new_gy = Memory{Float32}(undef, cap)
    new_ew = Memory{Float32}(undef, cap)
    new_idx = Memory{Int32}(undef, cap)
    if keep > 0
        old_rb, old_cb, old_gx, old_gy, old_ew = ws.rb, ws.cb, ws.gx, ws.gy, ws.ew
        @inbounds for k in 1:keep
            new_rb[k] = old_rb[k]
            new_cb[k] = old_cb[k]
            new_gx[k] = old_gx[k]
            new_gy[k] = old_gy[k]
            new_ew[k] = old_ew[k]
        end
    end
    ws.rb  = new_rb
    ws.cb  = new_cb
    ws.gx  = new_gx
    ws.gy  = new_gy
    ws.ew  = new_ew
    ws.idx = new_idx
    return ws
end

@noinline function _descriptor!(out::AbstractVector{Float32}, ws::DescriptorWorkspace,
                                a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real,
                                use_rootsift::Bool=false)
    # Round in zero-based coordinates, then convert back. Ties-to-even would
    # choose the opposite pixel if applied directly to a one-based coordinate.
    cy, cx = round(Int, y - 1) + 1, round(Int, x - 1) + 1
    width = 3.0 * sigma
    # Internal gradient angles use upward y, as in the standard SIFT convention.
    # Reduce angles outside [0, 2π) in Float64 before casting to Float32 so large
    # finite angles neither overflow to ±Inf32/NaN32 nor suffer Float32 cancellation.
    a64 = Float64(angle)
    phi = -((0.0 <= a64 < 2pi) ? a64 : rem2pi(a64, RoundDown))
    cs64, sn64 = cos(phi) / width, sin(phi) / width
    cs, sn = Float32(cs64), Float32(sn64)
    h, w = size(a)
    max_r = ceil(Int, hypot(h, w))
    r_est = width * sqrt(2.0) * 2.5
    radius = r_est < Float64(max_r) ? round(Int, r_est) : max_r
    scale = Float32(8.0 / (2pi))
    phi_f32 = Float32(phi)
    j_min = max(-radius, 2 - cx)
    j_max = min(radius, w - 1 - cx)
    i_min_base = max(-radius, 2 - cy)
    i_max_base = min(radius, h - 1 - cy)
    max_ni = i_max_base - i_min_base + 1
    if j_min > j_max || max_ni <= 0
        fill!(out, 0f0)
        return out
    end
    rb_buf, cb_buf, gx_buf, gy_buf, ew_buf = ws.rb, ws.cb, ws.gx, ws.gy, ws.ew
    cap = min(length(rb_buf), length(cb_buf), length(gx_buf), length(gy_buf), length(ew_buf), length(ws.idx))
    total = 0

    # Pass 1: pack all valid rotated-patch columns into contiguous L1-resident arrays.
    @inbounds for j in j_min:j_max
        xx = cx + j
        j_cs64 = j * cs64
        j_sn64 = j * sn64
        i_lo = i_min_base
        i_hi = i_max_base
        if abs(sn64) > 1e-12
            b1 = (j_cs64 - 2.50001) / sn64
            b2 = (j_cs64 + 2.50001) / sn64
            lo_s, hi_s = minmax(b1, b2)
            (hi_s <= i_lo || lo_s >= i_hi) && continue
            lo_s > i_lo - 1 && (i_lo = max(i_lo, floor(Int, lo_s) + 1))
            hi_s < i_hi + 1 && (i_hi = min(i_hi, ceil(Int, hi_s) - 1))
        elseif !(-2.50001 < j_cs64 < 2.50001)
            continue
        end
        if abs(cs64) > 1e-12
            b1 = (-2.50001 - j_sn64) / cs64
            b2 = ( 2.50001 - j_sn64) / cs64
            lo_c, hi_c = minmax(b1, b2)
            (hi_c <= i_lo || lo_c >= i_hi) && continue
            lo_c > i_lo - 1 && (i_lo = max(i_lo, floor(Int, lo_c) + 1))
            hi_c < i_hi + 1 && (i_hi = min(i_hi, ceil(Int, hi_c) - 1))
        elseif !(-2.50001 < j_sn64 < 2.50001)
            continue
        end
        n_i = i_hi - i_lo + 1
        n_i <= 0 && continue
        if total + n_i > cap
            new_cap = max(total + n_i, 2 * cap, 256)
            _grow_descriptor_workspace!(ws, new_cap, total)
            rb_buf, cb_buf, gx_buf, gy_buf, ew_buf = ws.rb, ws.cb, ws.gx, ws.gy, ws.ew
            cap = new_cap
        end

        j_cs = Float32(j_cs64)
        j_sn = Float32(j_sn64)
        u0 = muladd(-Float32(i_lo), sn, j_cs)
        v0 = muladd( Float32(i_lo), cs, j_sn)
        yy0 = cy + i_lo
        @simd for k in 0:(n_i - 1)
            kf = Float32(k)
            u = muladd(-kf, sn, u0)
            v = muladd( kf, cs, v0)
            yy = yy0 + k
            p = total + k + 1
            rb_buf[p] = v + 1.5f0
            cb_buf[p] = u + 1.5f0
            gx_buf[p] = a[yy, xx + 1] - a[yy, xx - 1]
            gy_buf[p] = a[yy - 1, xx] - a[yy + 1, xx]
            ew_buf[p] = (u * u + v * v) * 0.125f0
        end
        total += n_i
    end

    if total == 0
        fill!(out, 0f0)
        return out
    end

    # Pass 2: single flat branchless SIMD sweep computing bin index and trilinear weights
    # in place (rb->w00, cb->w01, gx->w10, gy->w11, ew->fo) to stay within L1 cache.
    idx_buf = ws.idx
    @inbounds @simd for k in 1:total
        rb_raw = rb_buf[k]
        cb_raw = cb_buf[k]
        valid = (-1.0f0 < rb_raw) & (rb_raw < 4.0f0) & (-1.0f0 < cb_raw) & (cb_raw < 4.0f0)
        rb = ifelse(valid, rb_raw, 0f0)
        cb = ifelse(valid, cb_raw, 0f0)
        gx = gx_buf[k]
        gy = gy_buf[k]
        mag = ifelse(valid, Base.sqrt_llvm(muladd(gx, gx, gy * gy)) * _exp_neg_f32(ew_buf[k]), 0f0)
        raw_ob = (_atan2_f32(gy, gx) - phi_f32) * scale
        # raw_ob lies in [-4, 12); branchless wrap into [0, 8) (exact 8.0f0 wraps via & 7).
        ob = raw_ob + ifelse(raw_ob < 0f0, 8f0, ifelse(raw_ob >= 8f0, -8f0, 0f0))
        r0f = floor(rb)
        c0f = floor(cb)
        o0f = floor(ob)
        fr = rb - r0f
        fc = cb - c0f
        fo = ob - o0f
        r0 = unsafe_trunc(Int32, r0f)
        c0 = unsafe_trunc(Int32, c0f)
        o0 = unsafe_trunc(Int32, o0f) & Int32(7)
        idx_buf[k] = ((r0 + Int32(1)) * Int32(6) + (c0 + Int32(1))) * Int32(10) + o0 + Int32(1)
        w_r1 = mag * fr
        w_r0 = mag - w_r1
        w01 = w_r0 * fc
        w00 = w_r0 - w01
        w11 = w_r1 * fc
        w10 = w_r1 - w11
        rb_buf[k] = w00
        cb_buf[k] = w01
        gx_buf[k] = w10
        gy_buf[k] = w11
        ew_buf[k] = fo
    end

    # Pass 3: branchless 2-wide SIMD scatter into 6×6×10 padded histogram.
    phist = ws.phist
    fill!(phist, 0f0)
    GC.@preserve phist @inbounds for k in 1:total
        base = Int(idx_buf[k])
        fo = ew_buf[k]
        vf = SVector{2,Float32}(1f0 - fo, fo)
        _store2!(phist, base,      muladd(rb_buf[k], vf, _load2(phist, base)))
        _store2!(phist, base + 10, muladd(cb_buf[k], vf, _load2(phist, base + 10)))
        _store2!(phist, base + 60, muladd(gx_buf[k], vf, _load2(phist, base + 60)))
        _store2!(phist, base + 70, muladd(gy_buf[k], vf, _load2(phist, base + 70)))
    end

    s1 = 0f0
    out_idx = 1
    @inbounds for pr in 1:4, pc in 1:4
        base = (pr * 6 + pc) * 10
        # Orientation bin 7's upper neighbor lands in padded slot 9; fold back into bin 0.
        v0 = phist[base + 1] + phist[base + 9]
        out[out_idx] = v0
        s1 = muladd(v0, v0, s1)
        @simd for b in 2:8
            v = phist[base + b]
            out[out_idx + b - 1] = v
            s1 = muladd(v, v, s1)
        end
        out_idx += 8
    end
    if s1 <= 0f0
        fill!(out, 0f0)
        return out
    end
    inv_n1 = 1f0 / Base.sqrt_llvm(s1)
    if use_rootsift
        l1 = 0f0
        @inbounds for k in 1:128
            v = min(out[k] * inv_n1, 0.2f0)
            out[k] = v
            l1 += v
        end
        inv_l1 = 1f0 / l1
        @inbounds @simd for k in 1:128
            out[k] = Base.sqrt_llvm(out[k] * inv_l1)
        end
    else
        s2 = 0f0
        @inbounds for k in 1:128
            v = min(out[k] * inv_n1, 0.2f0)
            out[k] = v
            s2 = muladd(v, v, s2)
        end
        inv_n2 = 1f0 / Base.sqrt_llvm(s2)
        @inbounds @simd for k in 1:128
            out[k] *= inv_n2
        end
    end
    return out
end

@inline function _validate_descriptor_args(y::Real, x::Real, sigma::Real, angle::Real)
    lim = Float64(typemax(Int32))
    isfinite(y) && isfinite(x) && abs(Float64(y)) < lim && abs(Float64(x)) < lim &&
        isfinite(sigma) && sigma > 0 && isfinite(angle) ||
        throw(ArgumentError("y, x must be finite within coordinate limits, angle finite, and sigma finite and positive"))
    return nothing
end

"""Compute a 128-element unit-L2 SIFT (or RootSIFT when `rootsift=true`) descriptor into `out` using a reusable [`DescriptorWorkspace`](@ref)."""
function descriptor!(out::AbstractVector{Float32}, ws::DescriptorWorkspace,
                     a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real;
                     rootsift::Bool=false)
    Base.require_one_based_indexing(out)
    length(out) == 128 || throw(DimensionMismatch("descriptor output buffer must have length 128"))
    _validate_descriptor_args(y, x, sigma, angle)
    return _descriptor!(out, ws, a, y, x, sigma, angle, rootsift)
end

function descriptor!(out::AbstractVector{Float32}, hist::AbstractVector{Float64},
                     a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real;
                     rootsift::Bool=false)
    Base.require_one_based_indexing(out, hist)
    (length(out) == 128 && length(hist) == 128) ||
        throw(DimensionMismatch("descriptor buffers must have length 128"))
    _validate_descriptor_args(y, x, sigma, angle)
    _descriptor!(out, DescriptorWorkspace(min(4096, max(256, length(a)))), a, y, x, sigma, angle, rootsift)
    @inbounds for k in 1:128
        hist[k] = Float64(out[k])
    end
    return out
end

"""Compute a 128-element unit-L2 SIFT (or RootSIFT when `rootsift=true`) descriptor around `(y, x, sigma, angle)`. See [`descriptor!`](@ref)."""
function descriptor(a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real;
                    rootsift::Bool=false)
    _validate_descriptor_args(y, x, sigma, angle)
    return _descriptor!(Vector{Float32}(undef, 128), DescriptorWorkspace(min(4096, max(256, length(a)))), a, y, x, sigma, angle, rootsift)
end

"""
    rootsift!(descriptors)

In-place RootSIFT (Hellinger) transform for a nonnegative descriptor vector or
column-wise descriptor matrix: L1-normalizes each vector/column and applies the
elementwise square root (leaving all-zero vectors unchanged). The output columns
have unit L2 norm so Euclidean matching computes the Hellinger kernel.
"""
function rootsift!(v::AbstractVector{T}) where {T<:AbstractFloat}
    Base.require_one_based_indexing(v)
    S = promote_type(T, Float64)
    peak = zero(S)
    @inbounds for i in eachindex(v)
        x = v[i]
        isfinite(x) && x >= zero(T) ||
            throw(ArgumentError("RootSIFT descriptors must be finite and nonnegative"))
        peak = max(peak, S(x))
    end
    iszero(peak) && return v

    scaled_sum = zero(S)
    @inbounds @simd for i in eachindex(v)
        scaled_sum += S(v[i]) / peak
    end
    # Taking the roots first preserves small bins even if v[i] / peak underflows.
    root_peak = sqrt(peak)
    inv_root_sum = inv(sqrt(scaled_sum))
    @inbounds @simd for i in eachindex(v)
        v[i] = (sqrt(S(v[i])) / root_peak) * inv_root_sum
    end
    return v
end

function rootsift!(m::AbstractMatrix{T}) where {T<:AbstractFloat}
    Base.require_one_based_indexing(m)
    for column in eachcol(m)
        rootsift!(column)
    end
    return m
end

"""
    rootsift(descriptors)

Return a RootSIFT-transformed copy of a nonnegative SIFT descriptor vector or
column-wise matrix. See [`rootsift!`](@ref).
"""
rootsift(v::AbstractVector{T}) where {T<:AbstractFloat} = rootsift!(copy(v))
rootsift(m::AbstractMatrix{T}) where {T<:AbstractFloat} = rootsift!(copy(m))
rootsift(x::AbstractVecOrMat{<:Real}) = Float32.(rootsift!(float.(x)))

# Rounding a value just below 2π to Float32 can otherwise leave the interval.
"""Reduce `angle` modulo `2π` and convert to `Float32` guaranteed to lie strictly in `[0, 2π)`."""
@inline function angle32(angle)
    a = Float32(mod(angle, 2pi))
    return a < 2pi ? a : 0f0
end

function _scan_octave_columns!(out::Vector{LocalizedCandidate}, d::Vector{Matrix{Float32}},
                               o::Int, octave::Int, l::Int, cols::UnitRange{Int},
                               h::Int, prelim_thresh::Float32, c::SIFTConfig)
    @inbounds begin
        dp = d[l - 1]
        dl = d[l]
        dn = d[l + 1]
    end
    neg_thresh = -prelim_thresh
    y_hi = h - 5
    GC.@preserve dp dl dn @inbounds for x in cols
        y = 6
        while y + 7 <= y_hi
            v = _load8(dl, y, x)
            # 8-wide 6-neighbor axial prefilter (exact necessary condition for 26-neighbor extremum).
            dl_u = _load8(dl, y - 1, x)
            dl_d = _load8(dl, y + 1, x)
            dl_l = _load8(dl, y, x - 1)
            dl_r = _load8(dl, y, x + 1)
            dp_c = _load8(dp, y, x)
            dn_c = _load8(dn, y, x)

            max6 = max.(max.(max.(dl_u, dl_d), max.(dl_l, dl_r)), max.(dp_c, dn_c))
            min6 = min.(min.(min.(dl_u, dl_d), min.(dl_l, dl_r)), min.(dp_c, dn_c))
            cand = ((v .>= prelim_thresh) .& (v .> max6)) .| ((v .<= neg_thresh) .& (v .< min6))
            if any(cand)
                for k in 1:8
                    if cand[k]
                        yk = y + k - 1
                        _is_extremum(dp, dl, dn, yk, x, v[k]) || continue
                        p = _localize(d, yk, x, l, c)
                        isnothing(p) && continue
                        push!(out, LocalizedCandidate(
                            o, octave, p.layer, p.iy, p.ix, p.y, p.x, p.sigma, p.response
                        ))
                    end
                end
            end
            y += 8
        end
        while y <= y_hi
            v = dl[y, x]
            if v >= prelim_thresh || v <= neg_thresh
                if _is_extremum(dp, dl, dn, y, x, v)
                    p = _localize(d, y, x, l, c)
                    if !isnothing(p)
                        push!(out, LocalizedCandidate(
                            o, octave, p.layer, p.iy, p.ix, p.y, p.x, p.sigma, p.response
                        ))
                    end
                end
            end
            y += 1
        end
    end
    return out
end

function _dedup_candidates(raw_candidates::Vector{LocalizedCandidate})
    # Converged candidates at the same lattice extremum are the same feature.
    unique_candidates = LocalizedCandidate[]
    sizehint!(unique_candidates, length(raw_candidates))
    seen = Set{NTuple{4,Int32}}()
    sizehint!(seen, length(raw_candidates))
    for cand in raw_candidates
        id = (Int32(cand.o), Int32(cand.layer), Int32(cand.iy), Int32(cand.ix))
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

function _build_pyramid_and_detect(img::AbstractMatrix, c::SIFTConfig; octaves=nothing)
    a = grayscale_float(img)
    n = isnothing(octaves) ? default_octaves(a, c) : octaves
    n isa Integer && !(n isa Bool) && n >= 0 || throw(ArgumentError("octaves must be a nonnegative integer"))
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
    empty_mat = Matrix{Float32}(undef, 0, 0)
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
        next_base = if o < n
            (o == 1 && c.first_octave == -1) ? downsample_2x!(a, levels[s + 1]) : downsample_2x(levels[s + 1])
        else
            base
        end
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
        # Release pure-DoG levels immediately so later octave allocations reuse pages without GC.
        levels[1] = empty_mat
        levels[s + 2] = empty_mat
        levels[s + 3] = empty_mat
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
    ws = OrientationWorkspace(64)
    for idx in r
        cand = cands[idx]
        factor = 2.0^cand.octave
        _orientations!(
            angles_buf, ws,
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
                                   r::UnitRange{Int},
                                   use_rootsift::Bool=false)
    ws = _DescriptorWorkspace(4096)
    @views for i in r
        k = kp[i]
        factor = 2.0^k.octave
        _descriptor!(
            desc[:, i], ws,
            gp[k.octave - first_octave + 1][k.layer + 1],
            1 + (k.y - 1) / factor,
            1 + (k.x - 1) / factor,
            k.sigma / factor,
            k.angle,
            use_rootsift
        )
    end
    return desc
end

"""
    keypoints, descriptors = sift(img; config=SIFTConfig(), octaves=nothing)

Native SIFT detection and description. Returns floating-point keypoints and a
128×N Float32 matrix with unit-L2 columns. Multiple strong orientations create
separate features. `nfeatures=0` keeps all features; otherwise keep the strongest.
Set `SIFTConfig(rootsift=true)` to compute RootSIFT (Hellinger) descriptors directly.
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
                Threads.@spawn _build_descriptors_range!(desc, kp, gp, c.first_octave, r, c.rootsift)
            end
        else
            _build_descriptors_range!(desc, kp, gp, c.first_octave, 1:nkp, c.rootsift)
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

@noinline function _match_chunk!(chunk_matches::Vector{DescriptorMatch},
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

@noinline function _match_chunk!(chunk_matches::Vector{DescriptorMatch},
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
