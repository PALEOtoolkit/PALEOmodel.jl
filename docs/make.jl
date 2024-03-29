using Documenter

import PALEOmodel

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "paleo_references.bib");
    style=:authoryear,
)

makedocs(;
    sitename="PALEOmodel Documentation", 
    pages = [
            "index.md",
            "Design" => [
                "MathematicalFormulation.md",
            ],
            "HOWTOs" => [
                "HOWTOshowmodelandoutput.md",
                "HOWTOsmallnegativevalues.md",
            ],
            "Reference" => [
                "PALEOmodelSolvers.md",
                "PALEOmodelOutput.md",
            ],
            "References.md",
            "indexpage.md",
        ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    plugins = [bib],
)

@info "Local html documentation is available at $(joinpath(@__DIR__, "build/index.html"))"

deploydocs(
    repo = "github.com/PALEOtoolkit/PALEOmodel.jl.git",
)
