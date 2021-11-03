using Documenter

makedocs(
    format = Documenter.HTML(prettyurls = haskey(ENV, "GITHUB_ACTIONS")), # disable for local builds
    sitename = "MetaFEM.jl",
    doctest = false,
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "Examples" => ["pikachu.md", "cantilever.md"],
        ]
)