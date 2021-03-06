using Documenter, Literate, Pkg, MetaFEM 

EXAMPLE_SRC_DIR = joinpath(@__DIR__, "src", "examples", "src")
EXAMPLE_MD_DIR = joinpath(@__DIR__, "src", "examples", "md")
rm(EXAMPLE_MD_DIR, force = true, recursive = true)
mkdir(EXAMPLE_MD_DIR)

for foldername in readdir(EXAMPLE_SRC_DIR)
    md_folder = joinpath(EXAMPLE_MD_DIR, foldername)
    src_folder = joinpath(EXAMPLE_SRC_DIR, foldername)
    mkdir(md_folder)
    for filename in readdir(src_folder)
        filename != string(foldername, ".jl") && cp(joinpath(src_folder, filename), joinpath(md_folder, filename); force=true)
    end
    input = abspath(joinpath(src_folder, string(foldername, ".jl")))
    script = Literate.script(input, md_folder)
    code = strip(read(script, String))

    # remove "hidden" lines which are not shown in the markdown
    line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
    code_clean = join(filter(x->!endswith(x,"#hide"),split(code, r"\n|\r\n")), line_ending_symbol)
    mdpost(str) = replace(str, "@__CODE__" => code_clean)
    # @example makes Documenter eval in a sandbox module instead of @__Main__ so doesn't work
    Literate.markdown(input, md_folder, postprocess = mdpost, config = Dict(:codefence => ("```julia" => "```")))
end

SEQUENCED_MD_FILES = [joinpath("examples", "md", foldername, string(foldername, ".md")) for foldername in ["pikachu", "pikachu_dynamics",
"thermal_stripe", "cantilever", "stress_concentration", "thermal_elasticity", "hyper_elasticity", "J2plasticity", "cylinderflow", "lid_driven_cavity"]]

makedocs(
    format = Documenter.HTML(prettyurls = haskey(ENV, "GITHUB_ACTIONS")), # disable for local builds
    sitename = "MetaFEM.jl",
    doctest = true,
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "Examples" => SEQUENCED_MD_FILES,
        "APIs" => "api.md",
        ],
)
##
deploydocs(
    repo = "github.com/jxx2/MetaFEM.jl.git",
    push_preview=true,
    devbranch = "main",
)