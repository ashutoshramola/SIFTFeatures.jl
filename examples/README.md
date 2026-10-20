# SIFTFeatures.jl — Practical Workflows & Examples

All scripts in `examples/` are self-contained and require **zero extra dependencies** beyond `SIFTFeatures.jl`'s standard project environment.

Run all examples at once from the repository root:

```sh
julia --threads=auto --project=. examples/run_all.jl
```

Or run any example individually:

```sh
julia --threads=auto --project=. examples/01_image_registration.jl
julia --threads=auto --project=. examples/02_object_detection.jl
julia --threads=auto --project=. examples/03_place_recognition.jl
julia --threads=auto --project=. examples/04_visual_odometry.jl
```

---

## Manual Workflow Guide by Use Case

### 0. Loading Your Own Images (PNG / JPG / TIFF)

`sift(img)` accepts any 1-based `AbstractMatrix` of:
- `Gray` or `RGB` / `RGBA` colors from `FileIO.jl` + `ImageIO.jl` (`using FileIO; img = load("photo.jpg")`)
- `UInt8` / `UInt16` raw pixel matrices (automatically normalized by `typemax`)
- `Float32` / `Float64` matrices in `[0, 1]`

```julia
using SIFTFeatures
# If loading from disk with FileIO / ImageIO in your app environment:
# using FileIO
# img1 = load("left.jpg")
# img2 = load("right.jpg")
```

---

### Use Case 1: Image Registration & Panorama Alignment (`01_image_registration.jl`)

**When to use:** Aligning two overlapping photos taken from different angles/zooms, satellite/medical image registration, or panorama stitching.

**Step-by-step workflow:**
1. Extract keypoints and descriptors from both images:
   ```julia
   kp_ref, desc_ref = sift(img_ref)
   kp_target, desc_target = sift(img_target)
   ```
2. Match descriptors with Lowe's strict ratio test (`ratio=0.8`) and mutual nearest-neighbor check (`crosscheck=true`):
   ```julia
   matches = match_descriptors(desc_ref, desc_target; ratio=0.8, crosscheck=true)
   ```
3. Estimate the $3\times 3$ homography `H` (`img_ref -> img_target`) with RANSAC and warp `img_target` onto `img_ref`'s pixel grid:
   ```julia
   include("examples/homography_ransac.jl")
   H, inliers = ransac_homography(kp_ref, kp_target, matches; thresh_px=3.0)
   aligned = zeros(Float32, size(img_ref))
   warp_perspective!(aligned, SIFTFeatures.grayscale_float(img_target), H)
   ```

---

### Use Case 2: Object / Template Localization (`02_object_detection.jl`)

**When to use:** Finding a known planar object, logo, book cover, fiducial marker, or PCB component inside a larger cluttered or partially occluded scene.

**Step-by-step workflow:**
1. Call `detect_object(template, scene)` from `examples/02_object_detection.jl`:
   ```julia
   include("examples/02_object_detection.jl")
   res = detect_object(template_img, scene_img; ratio=0.8, thresh_px=2.5)
   ```
2. Inspect `res.corners` (the 4 projected `(x, y)` corners of the template inside `scene_img`) and `count(res.inliers)` (typically `>= 10` inliers indicates a high-confidence detection).

---

### Use Case 3: Visual Place Recognition / Image Search (`03_place_recognition.jl`)

**When to use:** Searching a database of reference locations or product images to identify which scene matches a new query photo.

**Step-by-step workflow:**
1. Pre-index your reference images using a capped feature budget (`nfeatures=500` or `1000`) so each image stores a compact `128 × K` `Float32` matrix:
   ```julia
   include("examples/03_place_recognition.jl")
   cfg = SIFTConfig(nfeatures=500)
   db = build_image_database(["lobby" => img1, "hallway" => img2, "lab" => img3]; config=cfg)
   ```
2. Query the database with a new image:
   ```julia
   result = query_image_database(db, query_img; config=cfg)
   println("Best match: ", result.best_match, " with ", result.best_count, " mutual matches")
   ```

---

### Use Case 4: Multi-Frame Visual Odometry / Camera Tracking (`04_visual_odometry.jl`)

**When to use:** Tracking frame-to-frame camera motion across video frames `frames[1..N]`.

**Step-by-step workflow:**
```julia
include("examples/04_visual_odometry.jl")
# For high-FPS video tracking, use first_octave=0 and a feature cap:
cfg = SIFTConfig(first_octave=0, nfeatures=800)
vo = track_sequence(frames; config=cfg)
# vo.cumulative_H[t] maps frame 1 coordinates into frame t coordinates
```

---

### Tuning `SIFTConfig` for Different Workloads

| Workload | Recommended `SIFTConfig` | Why |
|---|---|---|
| **Offline photogrammetry / maximum recall** | `SIFTConfig()` (`first_octave=-1`, `nfeatures=0`) | `2×` upsampling detects fine sub-pixel structures (~14 FPS on `1000×700` with 8 threads). |
| **Image retrieval / database indexing** | `SIFTConfig(nfeatures=500)` or `SIFTConfig(nfeatures=1000)` | Retains the strongest features by contrast response; keeps matching $O(K^2)$ fast. |
| **Real-time video / robotics (50–60 FPS)** | `SIFTConfig(first_octave=0, nfeatures=500)` | Skips `2×` image doubling (`4×` fewer pixels in the first octave), running in ~16–19 ms per `1000×700` frame on 8 threads. |
| **Low-contrast / dark / foggy scenes** | `SIFTConfig(contrast_threshold=0.02)` | Lowers the DoG contrast rejection threshold (default `0.04`) to recover subtle features. |
