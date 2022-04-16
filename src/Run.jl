
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

"compact form"
function Base.show(io::IO, run::Run)
    print(io, "Run(model=", run.model,")")
end

"multiline form"
function Base.show(io::IO, ::MIME"text/plain", run::Run)
    println(io, "PALEOmodel.Run")
    println(io, "  model='", run.model,"'")
end


initialize!(run::Run; kwargs...) = initialize!(run.model; kwargs...)

"""
    initialize!(model::PB.Model; kwargs...) -> (initial_state, modeldata)

Initialize `model` and return `initial_state` Vector and `modeldata` struct

# Keywords:
- `eltype::Type=Float64`: default data type to use for model arrays
- `eltypemap=Dict{String, DataType}`: Dict of data types to look up Variable :datatype attribute
- `pickup_output=nothing`: OutputWriter with pickup data to initialise from
- `threadsafe=false`: true to create thread safe Atomic Variables where :atomic attribute = true
- `method_barrier=nothing`: thread barrier to add to dispatch lists if `threadsafe==true`
- `expect_hostdep_varnames=["global.tforce"]`: non-state-Variable host-dependent Variable names expected
"""
function initialize!(
    model::PB.Model; 
    eltype=Float64,
    eltypemap=Dict{String, DataType}(), 
    pickup_output=nothing,
    threadsafe=false,
    method_barrier=threadsafe ? 
        PB.reaction_method_thread_barrier(
            PALEOmodel.ThreadBarriers.ThreadBarrierAtomic("the barrier"),
            PALEOmodel.ThreadBarriers.wait_barrier
        ) : 
        nothing,
    expect_hostdep_varnames=["global.tforce"],
)

    modeldata = PB.create_modeldata(model, eltype, threadsafe=threadsafe)
   
    # Allocate variables
    PB.allocate_variables!(model, modeldata, eltypemap=eltypemap)

    # Create modeldata.solver_view_all for the entire model
    PB.set_default_solver_view!(model, modeldata)    

    # check all variables allocated
    PB.check_ready(model, modeldata, expect_hostdep_varnames=expect_hostdep_varnames)

    # Initialize model Reaction data arrays (calls ReactionMethod.preparefn)
    # Set modeldata.dispatchlists_all for the entire model
    PB.initialize_reactiondata!(model, modeldata, method_barrier=method_barrier)

    # check Reaction configuration
    PB.check_configuration(model)

    # Initialise state variables to norm_value
    PB.dispatch_setup(model, :norm_value, modeldata)
    PB.copy_norm!(modeldata.solver_view_all)

    # Initialise state variables etc     
    PB.dispatch_setup(model, :initial_value, modeldata)

    if !isnothing(pickup_output)
        pickup_record = length(pickup_output)
        # TODO this reads in implicit as well as explicit state vars
        # should set total corresponding to implicit state vars, and not state vars themselves ?
        all_statevars = vcat(PB.get_vars(modeldata.solver_view_all.stateexplicit), PB.get_vars(modeldata.solver_view_all.state))
        for statevar in all_statevars
            domname = statevar.domain.name
            varname = statevar.name
            if PB.has_variable(pickup_output, domname*"."*varname)
                @info "  initialising Variable $domname.$varname from pickup_output record $pickup_record"
                vardata = PB.get_data(statevar, modeldata)
                pickup_data = PB.get_data(pickup_output, domname*"."*varname, records=pickup_record)
                typeof(vardata) == typeof(pickup_data) ||
                    error("pickup_output Variable $domname.$varname has different type $(typeof(vardata)) != $(typeof(pickup_data))")
                vardata .= pickup_data
            else
                @info "  Variable $domname.$varname not present in pickup_output - leaving default initialisation"
            end
        end
    end

    initial_state = PB.get_statevar(modeldata.solver_view_all)
     
    
    return (initial_state, modeldata)
end


