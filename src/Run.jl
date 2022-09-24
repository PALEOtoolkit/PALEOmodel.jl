
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


function initialize!(run::Run; kwargs...)
    Base.depwarn("call to deprecated initialize!(run::Run; ...), please update your code to use initialize!(run.model; ...)", :initialize!, force=true)

    return initialize!(run.model; kwargs...)
end

"""
    initialize!(model::PB.Model; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)
    [deprecated] initialize!(run::Run; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)

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
- `create_solver_view_all=true`: `true` to create `modeldata.solver_view_all`
- `create_dispatchlists_all=true`: `true` to create `modeldata.dispatchlists_all`
- `generated_dispatch=true`: `true` to autogenerate code for `modeldata.dispatchlists_all` (fast dispatch, slow compile)
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
    create_solver_view_all=true,
    create_dispatchlists_all=true,
    generated_dispatch=true,
)

    modeldata = PB.create_modeldata(model, eltype; threadsafe)
   
    # Allocate variables
    @timeit "allocate_variables" PB.allocate_variables!(model, modeldata; eltypemap)

    # check all variables allocated
    PB.check_ready(model, modeldata; expect_hostdep_varnames)

    # Create modeldata.solver_view_all for the entire model
    if create_solver_view_all
        @timeit "set_default_solver_view!" set_default_solver_view!(model, modeldata)
    end

    # Initialize model Reaction data arrays (calls ReactionMethod.preparefn)
    # Set modeldata.dispatchlists_all for the entire model
    @timeit "initialize_reactiondata" PB.initialize_reactiondata!(
        model, modeldata;
        method_barrier, create_dispatchlists_all, generated_dispatch
    )

    # check Reaction configuration
    PB.check_configuration(model)

    # Initialise Reactions and non-state Variables
    @timeit "dispatch_setup :setup" PB.dispatch_setup(model, :setup, modeldata)

    # Initialise state variables to norm_value
    @timeit "dispatch_setup :norm_value" PB.dispatch_setup(model, :norm_value, modeldata)
    PALEOmodel.copy_norm!(modeldata.solver_view_all)

    # Initialise state variables etc     
    @timeit "dispatch_setup :initial_value" PB.dispatch_setup(model, :initial_value, modeldata)

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

    initial_state = get_statevar(modeldata.solver_view_all)
     
    
    return (initial_state, modeldata)
end


