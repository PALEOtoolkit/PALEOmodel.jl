using Documenter

import PALEOmodel

using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "src/paleo_references.bib"))

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

@info "Local html documentation is available at $(joinpath(@__DIR__, "build/index.html"))"

deploydocs(
    repo = "github.com/PALEOtoolkit/PALEOmodel.jl.git",
)