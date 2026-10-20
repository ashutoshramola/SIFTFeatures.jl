# Example 3: Visual Place Recognition / Image Database Retrieval
#
# Run from the repository root:
#   julia --threads=auto --project=. examples/03_place_recognition.jl

include("homography_ransac.jl")

struct ImageDatabaseEntry
    name::String
    keypoints::Vector{SIFTKeypoint{Float32}}
    descriptors::Matrix{Float32}
end

"""
    build_image_database(named_images; config=SIFTConfig(nfeatures=500))

Extract and index SIFT descriptors for a collection of `(name, image)` pairs.
Using `nfeatures` (e.g. 500–1000) keeps each database entry compact and fast to query.
"""
function build_image_database(named_images; config::SIFTConfig=SIFTConfig(nfeatures=500))
    db = Vector{ImageDatabaseEntry}(undef, length(named_images))
    for (idx, (name, img)) in enumerate(named_images)
        kp, desc = sift(img; config)
        db[idx] = ImageDatabaseEntry(String(name), kp, desc)
    end
    return db
end

"""
    query_image_database(db, query_img; config=SIFTConfig(nfeatures=500), ratio=0.8)

Extract SIFT descriptors from `query_img` and rank all entries in `db` by the number
of mutual Lowe-ratio descriptor matches.
"""
function query_image_database(db::Vector{ImageDatabaseEntry}, query_img::AbstractMatrix;
                              config::SIFTConfig=SIFTConfig(nfeatures=500),
                              ratio::Real=0.8)
    isempty(db) && throw(ArgumentError("image database must not be empty"))
    kp_q, desc_q = sift(query_img; config)
    scores = Vector{Pair{String,Int}}(undef, length(db))
    for (i, entry) in enumerate(db)
        m = match_descriptors(desc_q, entry.descriptors; ratio, crosscheck=true)
        scores[i] = entry.name => length(m)
    end
    sort!(scores; by=last, rev=true)
    return (best_match=first(scores).first, best_count=first(scores).second,
            ranking=scores, query_keypoints=kp_q)
end

function run_place_recognition_demo()
    println("=== Example 3: Visual Place Recognition / Image Retrieval ===")
    scenes = [
        "scene_alpha" => synthetic_scene(256, 320; seed=1),
        "scene_beta"  => synthetic_scene(256, 320; seed=2),
        "scene_gamma" => synthetic_scene(256, 320; seed=3),
        "scene_delta" => synthetic_scene(256, 320; seed=4),
    ]
    cfg = SIFTConfig(nfeatures=400)
    db = build_image_database(scenes; config=cfg)

    # Create a distorted query from "scene_gamma" (90° rotated + blurred + darkened)
    query_img = clamp.(SIFTFeatures.gaussian_blur(rotl90(scenes[3].second), 1.4) .* 0.75f0 .+ 0.05f0, 0f0, 1f0)

    query_image_database(db, query_img; config=cfg)
    t_sec = @elapsed res = query_image_database(db, query_img; config=cfg)

    println("  Database size      : ", length(db), " scenes (nfeatures=400)")
    println("  Top-1 retrieved    : ", res.best_match, " (", res.best_count, " matches)")
    println("  Full ranking       : ", join(["$(k): $(v)" for (k, v) in res.ranking], ", "))
    println("  Query + search time: ", round(t_sec * 1000; digits=1), " ms")
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_place_recognition_demo()
end
