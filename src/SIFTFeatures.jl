module SIFTFeatures

using ColorTypes: Colorant, Gray, AbstractGray, color
using ImageCore: ImageCore, Normed, gray
using ImageFiltering: imfilter!, KernelFactors, Algorithm, CPU1, CPUThreads
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
        x0 = (x + 1) ÷ 2
        x1 = min(x0 + 1, w)
        fx = iseven(x) ? 0.5f0 : 0f0
        omfx = 1f0 - fx
        for y in 1:2h
            y0 = (y + 1) ÷ 2
            y1 = min(y0 + 1, h)
            fy = iseven(y) ? 0.5f0 : 0f0
            v0 = muladd(fx, a[y0, x1], omfx * a[y0, x0])
            v1 = muladd(fx, a[y1, x1], omfx * a[y1, x0])
            b[y, x] = muladd(fy, v1, (1f0 - fy) * v0)
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

function gaussian_blur!(dst::Matrix{Float32}, src::Matrix{Float32}, sigma::Real)
    size(dst) == size(src) || throw(DimensionMismatch("dst and src must have the same size"))
    if sigma <= eps(Float32)
        copyto!(dst, src)
        return dst
    end
    # A four-sigma finite FIR kernel, with reflect-101 (no repeated endpoint).
    len = (round(Int, 8 * sigma + 1) | 1)
    kernels = KernelFactors.gaussian((sigma, sigma), (len, len))
    use_threads = Threads.nthreads() > 1 && length(src) >= 16_384 && minimum(size(src)) > len
    resource = use_threads ? CPUThreads(Algorithm.FIR()) : CPU1(Algorithm.FIR())
    imfilter!(resource, dst, src, kernels, "reflect")
    return dst
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
    base = gaussian_blur(raw_base, blur)
    for o in 1:n
        h, w = size(base)
        levels = Vector{Matrix{Float32}}(undef, s + 3)
        levels[1] = base
        for l in 1:s+2
            dst = Matrix{Float32}(undef, h, w)
            gaussian_blur!(dst, levels[l], delta_sigmas[l])
            levels[l + 1] = dst
        end
        push!(pyramid, levels)
        if o < n
            base = downsample_2x(levels[s + 1])
        end
    end
    return pyramid
end

function _diff_matrix(next_lvl::Matrix{Float32}, prev_lvl::Matrix{Float32})
    dst = similar(next_lvl)
    @inbounds @simd for i in eachindex(dst, next_lvl, prev_lvl)
        dst[i] = next_lvl[i] - prev_lvl[i]
    end
    return dst
end

"""Adjacent Gaussian differences, with `octave_layers + 2` levels per octave."""
function dog_pyramid(g::AbstractVector)
    isempty(g) && return Vector{Matrix{Float32}}[]
    return [[_diff_matrix(o[l + 1], o[l]) for l in 1:length(o)-1] for o in g]
end

@inline function is_extremum(d, y::Int, x::Int, l::Int)
    @inbounds begin
        v = d[l][y, x]
        iszero(v) && return false
        if v > 0f0
            for k in l-1:l+1
                dk = d[k]
                for j in x-1:x+1, i in y-1:y+1
                    (k == l && j == x && i == y) && continue
                    dk[i, j] >= v && return false
                end
            end
        else
            for k in l-1:l+1
                dk = d[k]
                for j in x-1:x+1, i in y-1:y+1
                    (k == l && j == x && i == y) && continue
                    dk[i, j] <= v && return false
                end
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

function orientations!(angles::Vector{Float64}, hist::AbstractVector{Float64}, sm::AbstractVector{Float64},
                       a::Matrix{Float32}, y::Int, x::Int, sigma::Real)
    Base.require_one_based_indexing(angles, hist, sm)
    (length(hist) == 36 && length(sm) == 36) ||
        throw(DimensionMismatch("orientation buffers must have length 36"))
    fill!(hist, 0.0)
    empty!(angles)
    radius = round(Int, 4.5 * sigma)
    h, w = size(a)
    inv_2sigma2 = -1.0 / (2.0 * (1.5 * sigma)^2)
    scale = 36.0 / (2pi)
    j_min = max(-radius, 2 - x)
    j_max = min(radius, w - 1 - x)
    i_min = max(-radius, 2 - y)
    i_max = min(radius, h - 1 - y)
    @inbounds for j in j_min:j_max
        xx = x + j
        j2 = j * j
        for i in i_min:i_max
            yy = y + i
            gx = a[yy, xx + 1] - a[yy, xx - 1]
            gy = a[yy - 1, xx] - a[yy + 1, xx]
            bin = mod(round(Int, atan(gy, gx) * scale), 36) + 1
            hist[bin] += hypot(gx, gy) * exp((i * i + j2) * inv_2sigma2)
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

orientations(a::Matrix{Float32}, y::Int, x::Int, sigma::Real) =
    orientations!(Float64[], Memory{Float64}(undef, 36), Memory{Float64}(undef, 36), a, y, x, sigma)

function descriptor!(out::AbstractVector{Float32}, hist::AbstractVector{Float64},
                     a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real)
    Base.require_one_based_indexing(out, hist)
    (length(out) == 128 && length(hist) == 128) ||
        throw(DimensionMismatch("descriptor buffers must have length 128"))
    fill!(hist, 0.0)
    # Round in zero-based coordinates, then convert back. Ties-to-even would
    # choose the opposite pixel if applied directly to a one-based coordinate.
    cy, cx = round(Int, y - 1) + 1, round(Int, x - 1) + 1
    width = 3.0 * sigma
    # Internal gradient angles use upward y, as in the standard SIFT convention.
    phi = -Float64(angle)
    cs, sn = cos(phi) / width, sin(phi) / width
    h, w = size(a)
    radius = min(round(Int, width * sqrt(2.0) * 2.5), ceil(Int, hypot(h, w)))
    scale = 8.0 / (2pi)
    j_min = max(-radius, 2 - cx)
    j_max = min(radius, w - 1 - cx)
    i_min = max(-radius, 2 - cy)
    i_max = min(radius, h - 1 - cy)
    @inbounds for j in j_min:j_max
        xx = cx + j
        j_cs = j * cs
        j_sn = j * sn
        for i in i_min:i_max
            u = j_cs - i * sn
            v = j_sn + i * cs
            rb, cb = v + 1.5, u + 1.5
            (-1.0 < rb < 4.0 && -1.0 < cb < 4.0) || continue
            yy = cy + i
            gx = a[yy, xx + 1] - a[yy, xx - 1]
            gy = a[yy - 1, xx] - a[yy + 1, xx]
            ob = mod((atan(gy, gx) - phi) * scale, 8.0)
            mag = hypot(gx, gy) * exp(-(u * u + v * v) * 0.125)
            r0, c0, o0 = floor(Int, rb), floor(Int, cb), floor(Int, ob)
            fr, fc, fo = rb - r0, cb - c0, ob - o0
            b0 = o0 & 7
            b1 = (o0 + 1) & 7
            w_o0 = 1.0 - fo
            w_o1 = fo
            for dr in 0:1
                r = r0 + dr
                (0 <= r < 4) || continue
                wr = mag * (dr == 0 ? 1.0 - fr : fr)
                for dc in 0:1
                    cl = c0 + dc
                    (0 <= cl < 4) || continue
                    wrc = wr * (dc == 0 ? 1.0 - fc : fc)
                    base_idx = (r * 4 + cl) * 8 + 1
                    hist[base_idx + b0] += wrc * w_o0
                    hist[base_idx + b1] += wrc * w_o1
                end
            end
        end
    end
    s1 = 0.0
    @inbounds @simd for k in 1:128
        s1 = muladd(hist[k], hist[k], s1)
    end
    if s1 <= 0.0
        fill!(out, 0f0)
        return out
    end
    inv_n1 = 1.0 / sqrt(s1)
    s2 = 0.0
    @inbounds for k in 1:128
        v = min(hist[k] * inv_n1, 0.2)
        hist[k] = v
        s2 = muladd(v, v, s2)
    end
    inv_n2 = Float32(1.0 / sqrt(s2))
    @inbounds @simd for k in 1:128
        out[k] = Float32(hist[k]) * inv_n2
    end
    return out
end

descriptor(a::Matrix{Float32}, y::Real, x::Real, sigma::Real, angle::Real) =
    descriptor!(Vector{Float32}(undef, 128), Memory{Float64}(undef, 128), a, y, x, sigma, angle)

# Rounding a value just below 2π to Float32 can otherwise leave the interval.
@inline function angle32(angle)
    a = Float32(mod(angle, 2pi))
    return a < 2pi ? a : 0f0
end

function _scan_octave_columns!(out::Vector{LocalizedCandidate}, d::Vector{Matrix{Float32}},
                               o::Int, octave::Int, l::Int, cols::UnitRange{Int},
                               h::Int, prelim_thresh::Float64, c::SIFTConfig)
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

function _detect_candidates(dp::Vector{Vector{Matrix{Float32}}}, c::SIFTConfig)
    prelim_thresh = 0.5 * c.contrast_threshold / c.octave_layers
    nt = Threads.nthreads()
    raw_candidates = LocalizedCandidate[]
    for o in eachindex(dp)
        d = dp[o]
        h, w = size(d[1])
        (h >= 11 && w >= 11) || continue
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
    end
    # Converged candidates at the same lattice extremum are the same feature.
    unique_candidates = LocalizedCandidate[]
    sizehint!(unique_candidates, length(raw_candidates))
    seen = Set{NTuple{4,Int}}()
    sizehint!(seen, length(raw_candidates))
    for cand in raw_candidates
        id = (cand.o, cand.layer, cand.iy, cand.ix)
        id in seen && continue
        push!(seen, id)
        push!(unique_candidates, cand)
    end
    return unique_candidates
end

function _build_keypoints_range!(out::Vector{SIFTKeypoint{Float32}},
                                 cands::Vector{LocalizedCandidate},
                                 gp::Vector{Vector{Matrix{Float32}}},
                                 r::UnitRange{Int})
    angles_buf = Float64[]
    ori_hist = Memory{Float64}(undef, 36)
    ori_sm = Memory{Float64}(undef, 36)
    for idx in r
        cand = cands[idx]
        factor = 2.0^cand.octave
        orientations!(
            angles_buf, ori_hist, ori_sm,
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
    hist_buf = Memory{Float64}(undef, 128)
    @views for i in r
        k = kp[i]
        factor = 2.0^k.octave
        descriptor!(
            desc[:, i], hist_buf,
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
    gp = gaussian_pyramid(img, c; octaves)
    dp = dog_pyramid(gp)
    unique_cands = _detect_candidates(dp, c)
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
    if nt > 1 && na * nb >= 4096 && na >= 2
        nchunks = min(nt, na)
        chunk_matches = [DescriptorMatch[] for _ in 1:nchunks]
        chunk_rev = [_alloc_rev_buffers(nb, crosscheck) for _ in 1:nchunks]
        @sync for chunk in 1:nchunks
            qrange = _chunk_range(na, chunk, nchunks)
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
