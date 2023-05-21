
"""
    Run

Container for model and output.

# Fields
- `model::Union{Nothing, PB.Model}`: The model instance.
- `output::Union{Nothing, AbstractOutputWriter}`: model output
"""
Base.@kwdef mutable struct Run
    model::Union{Nothing, PB.Model}

    output::Union{Nothing, AbstractOutputWriter}= nothing

end

function Base.show(io::IO, run::Run)
    print(io, "Run(model='", run.model, "', output='", run.output, "')")
end

function Base.show(io::IO, ::MIME"text/plain", run::Run)
    println(io, "PALEOmodel.Run")
    println(io, "  model='", run.model,"'")
    println(io, "  output='", run.output,"'")
end

"""
    [deprecated] initialize!(run::Run; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)

Call `initialize!` on `run.model`.
"""
function initialize!(run::Run; kwargs...)
    Base.depwarn("call to deprecated initialize!(run::Run; ...), please update your code to use initialize!(run.model; ...)", :initialize!, force=true)

    return initialize!(run.model; kwargs...)
end


