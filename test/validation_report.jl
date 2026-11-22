module ValidationReportTests

using Test
include("fixtures.jl")
include(joinpath(@__DIR__, "..", "validation", "eval_pure.jl"))

@testset "Saved evaluation reports" begin
    mktempdir() do dir
        image = joinpath(dir, "fixture.pgm")
        open(image, "w") do io
            write(io, "P5\n64 64\n255\n")
            write(io, permutedims(round.(UInt8, 255 .* fixture(64, 64))))
        end
        identity_h = "1 0 0\n0 1 0\n0 0 1\n"
        hp = joinpath(dir, "hpatches")
        scene = mkpath(joinpath(hp, "i_ajuntament"))
        mkpath(joinpath(hp, "v_unused"))  # --limit 1 must exclude this scene.
        for i in 1:6
            cp(image, joinpath(scene, "$i.ppm"))
            i > 1 && write(joinpath(scene, "H_1_$i"), identity_h)
        end

        report = joinpath(dir, "report.md")
        args = ["--hpatches", hp, "--limit", "1", "--write-report", report]
        @test main(args) == 0
        hp_only = read(report, String)
        @test occursin("HPatches 6-Photograph Ground-Truth Accuracy (5 pairs)", hp_only)
        @test occursin("| i_ajuntament |", hp_only)
        @test !occursin("Oxford", hp_only) && !occursin("RootSIFT", hp_only)

        ox = joinpath(dir, "oxford")
        for name in ("bikes", "trees", "graf", "wall", "bark", "boat", "leuven", "ubc")
            scene = mkpath(joinpath(ox, name))
            for i in 1:6
                cp(image, joinpath(scene, "img$i.pgm"))
                i > 1 && write(joinpath(scene, "H1to$(i)p"), identity_h)
            end
        end
        @test main(vcat(args, ["--oxford", ox, "--compare-rootsift"])) == 0
        combined = read(report, String)
        @test occursin("Oxford 48-Photograph Ground-Truth Accuracy (40 pairs)", combined)
        @test occursin("HPatches 6-Photograph Ground-Truth Accuracy (5 pairs)", combined)
        @test count("| RootSIFT (", combined) == 2
        @test occursin("| bikes |", combined) && occursin("| i_ajuntament |", combined)
    end
end

end
