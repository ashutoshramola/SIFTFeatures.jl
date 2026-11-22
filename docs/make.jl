using Documenter
using SIFTFeatures

DocMeta.setdocmeta!(SIFTFeatures, :DocTestSetup, :(using SIFTFeatures); recursive=true)

makedocs(;
    modules=[SIFTFeatures],
    authors="Ashutosh Ramola and contributors",
    sitename="SIFTFeatures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://ashutoshramola.github.io/SIFTFeatures.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs=:exports,
    pages=[
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "Examples" => "examples.md",
        "Implementation" => "internals.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ashutoshramola/SIFTFeatures.jl.git",
    devbranch="main",
    push_preview=true,
)
