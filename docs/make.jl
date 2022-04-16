using Documenter

import PALEOmodel

using DocumenterCitations

bib = CitationBibliography("src/paleo_references.bib")

makedocs(bib, sitename="PALEOmodel Documentation", 
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
                "PALEOmodel.md",
            ],
            "References.md",
            "indexpage.md",
        ],
        format = Documenter.HTML(
            prettyurls = get(ENV, "CI", nothing) == "true"
        ),
        repo = "https://github.com/sjdaines/PALEOdev.jl/blob/master/{path}#{line}")

@info "Local html documentation is available at $(joinpath(@__DIR__, "build/index.html"))"

deploydocs(
    repo = "github.com/PALEOtoolkit/PALEOmodel.jl.git",
)