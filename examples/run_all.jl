# Run all 4 real-world SIFT workflow examples.
#
# Usage:
#   julia --threads=auto --project=. examples/run_all.jl

include("01_image_registration.jl")
include("02_object_detection.jl")
include("03_place_recognition.jl")
include("04_visual_odometry.jl")

function main()
    r1 = run_registration_demo()
    println()
    r2 = run_object_detection_demo()
    println()
    r3 = run_place_recognition_demo()
    println()
    r4 = run_visual_odometry_demo()
    return (r1, r2, r3, r4)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
