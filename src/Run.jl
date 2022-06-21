
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


initialize!(run::Run; kwargs...) = initialize!(run.model; kwargs...)

"""
    initialize!(model::PB.Model; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)
    initialize!(run::Run; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)

Initialize `model` or `run.model` and return:
- an `initial_state` Vector
- a `modeldata` struct containing the model arrays.

# Initialising the state vector
With default arguments, the `model` state variables are initialised to the values defined in the `.yaml`
configuration file used to create the `model`.

The optional `pickup_output` argument can be used to provide an `OutputWriter` instance with pickup data to initialise from.
This is applied afer the default initialisation, hence can be used to (re)initialise a subset of model state variables.

# `DataType`s for model arrays
With default arguments, the `model` arrays use `Float64` as the element type. The `eltype` keyword argument can be used to
specify a different Julia `DataType`, eg for use with automatic differentiation.  Per-Variable `DataType` can be specified by using
the `:datatype` Variable attribute to specify `String`-valued tags, in combination with the `eltypemap` keyword argument to 
provide a `Dict` of tag names => `DataType`s. 

# Thread safety
A thread-safe model can be created with `threadsafe=true` (to create Atomic Variables for those Variables with attribute `:atomic==true`),
and supplying `method_barrier` (a thread barrier to add to `ReactionMethod` dispatch lists between dependency-free groups)

# Keyword summary
- `pickup_output=nothing`: OutputWriter with pickup data to initialise from
- `eltype::Type=Float64`: default data type to use for model arrays
- `eltypemap=Dict{String, DataType}()`: Dict of data types to look up Variable :datatype attribute
- `threadsafe=false`: true to create thread safe Atomic Variables where Variable attribute `:atomic==true`
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

    # check all variables allocated
    PB.check_ready(model, modeldata, expect_hostdep_varnames=expect_hostdep_varnames)

    # Create modeldata.solver_view_all for the entire model
    PB.set_default_solver_view!(model, modeldata)    

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


