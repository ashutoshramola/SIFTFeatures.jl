# Pure-Julia Normalized DLT + RANSAC Homography Estimation and Image Warping
# Uses only SIFTFeatures, LinearAlgebra, StaticArrays, Statistics, and Random.

using SIFTFeatures
using LinearAlgebra: svd, norm, det, I
using StaticArrays: SVector, SMatrix
using Statistics: mean
using Random: AbstractRNG, MersenneTwister

"""
    project_point(H::SMatrix{3,3,Float64,9}, x::Real, y::Real) -> SVector{2,Float64}

Project a 2D point `(x, y)` through a 3×3 homography `H`. Returns `(NaN, NaN)` if
the homogeneous denominator is near zero.
"""
@inline function project_point(H::SMatrix{3,3,Float64,9}, x::Real, y::Real)
    xf, yf = Float64(x), Float64(y)
    w = H[3, 1] * xf + H[3, 2] * yf + H[3, 3]
    abs(w) < 1e-12 && return SVector{2,Float64}(NaN, NaN)
    inv_w = 1.0 / w
    u = (H[1, 1] * xf + H[1, 2] * yf + H[1, 3]) * inv_w
    v = (H[2, 1] * xf + H[2, 2] * yf + H[2, 3]) * inv_w
    return SVector{2,Float64}(u, v)
end

function _normalize_points(pts::AbstractMatrix{Float64})
    n = size(pts, 1)
    mx = mean(@view pts[:, 1])
    my = mean(@view pts[:, 2])
    mean_dist = 0.0
    @inbounds for i in axes(pts, 1)
        mean_dist += hypot(pts[i, 1] - mx, pts[i, 2] - my)
    end
    mean_dist /= n
    s = mean_dist > 1e-12 ? sqrt(2.0) / mean_dist : 1.0
    T = SMatrix{3,3,Float64,9}(
        s,   0.0, 0.0,
        0.0, s,   0.0,
        -s * mx, -s * my, 1.0
    )
    normed = Matrix{Float64}(undef, n, 2)
    @inbounds for i in axes(pts, 1)
        normed[i, 1] = s * (pts[i, 1] - mx)
        normed[i, 2] = s * (pts[i, 2] - my)
    end
    return normed, T
end

"""
    fit_homography_dlt(pa::AbstractMatrix{<:Real}, pb::AbstractMatrix{<:Real}) -> SMatrix{3,3,Float64,9}

Fit a 3×3 homography mapping `N×2` points `pa` (`[x y]`) to `pb` (`[u v]`) using
Hartley-normalized Direct Linear Transform (DLT). Requires at least 4 point pairs.
"""
function fit_homography_dlt(pa::AbstractMatrix{<:Real}, pb::AbstractMatrix{<:Real})
    size(pa) == size(pb) && size(pa, 2) == 2 && size(pa, 1) >= 4 ||
        throw(DimensionMismatch("pa and pb must be N×2 matrices with N >= 4"))
    n = size(pa, 1)
    na, Ta = _normalize_points(Float64.(pa))
    nb, Tb = _normalize_points(Float64.(pb))
    A = zeros(Float64, 2n, 9)
    @inbounds for i in 1:n
        x, y = na[i, 1], na[i, 2]
        u, v = nb[i, 1], nb[i, 2]
        r1 = 2i - 1
        r2 = 2i
        A[r1, 1] = -x; A[r1, 2] = -y; A[r1, 3] = -1.0
        A[r1, 7] = u * x; A[r1, 8] = u * y; A[r1, 9] = u
        A[r2, 4] = -x; A[r2, 5] = -y; A[r2, 6] = -1.0
        A[r2, 7] = v * x; A[r2, 8] = v * y; A[r2, 9] = v
    end
    F = svd(A; full=true)
    v9 = @view F.V[:, 9]
    # Row-major 3×3 reconstruction from the 9-element nullspace vector
    Hn = SMatrix{3,3,Float64,9}(
        v9[1], v9[4], v9[7],
        v9[2], v9[5], v9[8],
        v9[3], v9[6], v9[9]
    )
    H = inv(Tb) * Hn * Ta
    return abs(H[3, 3]) > 1e-12 ? H / H[3, 3] : H
end

"""
    ransac_homography(kp1, kp2, matches; thresh_px=3.0, iters=1500, zero_based=false, rng=MersenneTwister(42))

Robustly estimate a 3×3 homography `H::SMatrix{3,3,Float64,9}` and `inliers::Vector{Bool}`
from `SIFTKeypoint` vectors `kp1`, `kp2` and `DescriptorMatch` vector `matches`.
By default uses 1-based Julia pixel coordinates `(kp.x, kp.y)`; pass `zero_based=true`
when comparing against 0-based ground-truth homography matrices (e.g. Oxford dataset).
"""
function ransac_homography(kp1::AbstractVector{<:SIFTKeypoint},
                           kp2::AbstractVector{<:SIFTKeypoint},
                           matches::AbstractVector{DescriptorMatch};
                           thresh_px::Real=3.0,
                           iters::Int=1500,
                           zero_based::Bool=false,
                           rng::AbstractRNG=MersenneTwister(42))
    nm = length(matches)
    if nm < 4
        return SMatrix{3,3,Float64,9}(I), fill(false, nm)
    end
    offset = zero_based ? 1.0 : 0.0
    pts_a = Matrix{Float64}(undef, nm, 2)
    pts_b = Matrix{Float64}(undef, nm, 2)
    @inbounds for (idx, m) in enumerate(matches)
        ka = kp1[m.query]
        kb = kp2[m.train]
        pts_a[idx, 1] = Float64(ka.x) - offset
        pts_a[idx, 2] = Float64(ka.y) - offset
        pts_b[idx, 1] = Float64(kb.x) - offset
        pts_b[idx, 2] = Float64(kb.y) - offset
    end

    t2 = Float64(thresh_px)^2
    best_H = SMatrix{3,3,Float64,9}(I)
    best_count = 0
    sub_a = Matrix{Float64}(undef, 4, 2)
    sub_b = Matrix{Float64}(undef, 4, 2)

    @inbounds for _ in 1:iters
        i1 = rand(rng, 1:nm)
        i2 = rand(rng, 1:nm)
        i3 = rand(rng, 1:nm)
        i4 = rand(rng, 1:nm)
        (i1 == i2 || i1 == i3 || i1 == i4 || i2 == i3 || i2 == i4 || i3 == i4) && continue
        idxs = (i1, i2, i3, i4)
        for k in 1:4
            r = idxs[k]
            sub_a[k, 1] = pts_a[r, 1]; sub_a[k, 2] = pts_a[r, 2]
            sub_b[k, 1] = pts_b[r, 1]; sub_b[k, 2] = pts_b[r, 2]
        end
        H = fit_homography_dlt(sub_a, sub_b)
        cnt = 0
        for i in 1:nm
            p = project_point(H, pts_a[i, 1], pts_a[i, 2])
            dx = p[1] - pts_b[i, 1]
            dy = p[2] - pts_b[i, 2]
            if dx * dx + dy * dy < t2
                cnt += 1
            end
        end
        if cnt > best_count
            best_count = cnt
            best_H = H
        end
    end

    inliers = Vector{Bool}(undef, nm)
    @inbounds for i in 1:nm
        p = project_point(best_H, pts_a[i, 1], pts_a[i, 2])
        dx = p[1] - pts_b[i, 1]
        dy = p[2] - pts_b[i, 2]
        inliers[i] = dx * dx + dy * dy < t2
    end
    if best_count >= 4
        best_H = fit_homography_dlt(pts_a[inliers, :], pts_b[inliers, :])
    end
    return best_H, inliers
end

"""
    warp_perspective!(dst::AbstractMatrix{Float32}, src::AbstractMatrix{<:Real}, H_dst_to_src::SMatrix{3,3,Float64,9})

Inverse-map `src` into `dst` using bilinear interpolation and 1-based pixel coordinates.
For each pixel `(y, x)` in `dst`, samples `src` at `project_point(H_dst_to_src, x, y)`.
Out-of-bounds pixels are filled with `0f0`.
"""
function warp_perspective!(dst::AbstractMatrix{Float32},
                           src::AbstractMatrix{<:Real},
                           H_dst_to_src::SMatrix{3,3,Float64,9})
    h_src, w_src = size(src)
    @inbounds for x in axes(dst, 2), y in axes(dst, 1)
        p = project_point(H_dst_to_src, x, y)
        sx, sy = p[1], p[2]
        if 1.0 <= sx <= w_src && 1.0 <= sy <= h_src
            x0 = clamp(floor(Int, sx), 1, w_src)
            y0 = clamp(floor(Int, sy), 1, h_src)
            x1 = min(x0 + 1, w_src)
            y1 = min(y0 + 1, h_src)
            fx = Float32(sx - x0)
            fy = Float32(sy - y0)
            v0 = (1f0 - fx) * Float32(src[y0, x0]) + fx * Float32(src[y0, x1])
            v1 = (1f0 - fx) * Float32(src[y1, x0]) + fx * Float32(src[y1, x1])
            dst[y, x] = (1f0 - fy) * v0 + fy * v1
        else
            dst[y, x] = 0f0
        end
    end
    return dst
end

"""
    homography_probe_error(H_est::SMatrix{3,3,Float64,9}, H_gt::SMatrix{3,3,Float64,9}, h::Int, w::Int; zero_based::Bool=false)

Mean Euclidean reprojection distance (in pixels) between `H_est` and `H_gt` across
5 standard probe points (the 4 image corners at 10%/90% and the image center).
"""
function homography_probe_error(H_est::SMatrix{3,3,Float64,9},
                                H_gt::SMatrix{3,3,Float64,9},
                                h::Int, w::Int;
                                zero_based::Bool=false)
    ox = zero_based ? 0.0 : 1.0
    probes = (
        (ox + 0.1 * (w - 1), ox + 0.1 * (h - 1)),
        (ox + 0.9 * (w - 1), ox + 0.1 * (h - 1)),
        (ox + 0.1 * (w - 1), ox + 0.9 * (h - 1)),
        (ox + 0.9 * (w - 1), ox + 0.9 * (h - 1)),
        (ox + 0.5 * (w - 1), ox + 0.5 * (h - 1)),
    )
    err = 0.0
    for (x, y) in probes
        p1 = project_point(H_est, x, y)
        p2 = project_point(H_gt, x, y)
        err += norm(p1 - p2)
    end
    return err / length(probes)
end

"""
    synthetic_scene(h::Int=320, w::Int=400) -> Matrix{Float32}

Generate a deterministic, multi-scale textured scene for standalone examples.
"""
function synthetic_scene(h::Int=320, w::Int=400; seed::Int=1)
    img = Matrix{Float32}(undef, h, w)
    s1 = Float64(seed)
    @inbounds for x in 1:w, y in 1:h
        u = (x - 1) / (w - 1)
        v = (y - 1) / (h - 1)
        # Multi-frequency non-periodic spatial carrier with localized Gabor-like wavepackets
        val = 0.50 +
              0.12 * sin((19.0 + 4.0 * s1) * u + (7.0 + 3.0 * s1) * v) * cos((15.0 + 4.0 * s1) * v - 7.0 * u) +
              0.10 * sin((47.0 + 6.0 * s1) * u * u + 31.0 * v) * cos((37.0 + 5.0 * s1) * v * v - 17.0 * u) +
              0.08 * cos(89.0 * u - 67.0 * v + 13.0 * s1 * u * v) * exp(-2.5 * ((u - 0.5)^2 + (v - 0.5)^2)) +
              0.09 * sin(113.0 * u + 79.0 * v + s1) * exp(-6.0 * ((u - 0.38)^2 + (v - 0.44)^2)) -
              0.09 * cos(97.0 * u - 83.0 * v - s1) * exp(-6.0 * ((u - 0.62)^2 + (v - 0.58)^2))
        for k in 1:12
            cx = 0.15 + 0.70 * mod(0.37 * k + 0.19 * k * k + 0.23 * s1, 1.0)
            cy = 0.15 + 0.70 * mod(0.61 * k + 0.29 * k * k + 0.41 * s1, 1.0)
            sigma = 0.014 + 0.004 * mod(k + seed, 5)
            amp = isodd(k + seed) ? 0.28 : -0.26
            r2 = ((u - cx)^2 + (v - cy)^2) / (2.0 * sigma^2)
            val += amp * (1.0 - 0.5 * r2) * exp(-r2)
        end
        img[y, x] = Float32(clamp(val, 0.0, 1.0))
    end
    return img
end
