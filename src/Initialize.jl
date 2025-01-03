
"""
    initialize!(model::PB.Model; kwargs...) -> (initial_state::Vector, modeldata::PB.ModelData)

Initialize `model` and return:
- an `initial_state` Vector
- a `modeldata` struct containing the model arrays.

# Initialising the state vector
With default arguments, the `model` state variables are initialised to the values defined in the `.yaml`
configuration file used to create the `model`.

The optional `pickup_output` argument can be used to provide an `OutputWriter` instance with pickup data to initialise from,
using [`set_statevar_from_output!`](@ref).
This is applied afer the default initialisation, hence can be used to (re)initialise a subset of model state variables.

# `DataType`s for model arrays
With default arguments, the `model` arrays use `Float64` as the element type. The `eltype` keyword argument can be used to
specify a different Julia `DataType`, eg for use with automatic differentiation.  Per-Variable `DataType` can be specified by using
the `:datatype` Variable attribute to specify `String`-valued tags, in combination with the `eltypemap` keyword argument to 
provide a `Dict` of tag names => `DataType`s. 

# Thread safety
A thread-safe model can be created by setting the `threadsafe=true` global parameter in the YAML config file before calling `PB.create_model_from_config` 
(used by Reactions that require eg special handling of global accumulator variables to maintain thread safety),
and supplying `method_barrier` (a thread barrier to add to `ReactionMethod` dispatch lists between dependency-free groups):

    method_barrier = PB.reaction_method_thread_barrier(
        PALEOmodel.ThreadBarriers.ThreadBarrierAtomic("the barrier"),
        PALEOmodel.ThreadBarriers.wait_barrier
    )

# Keyword summary
- `pickup_output=nothing`: OutputWriter with pickup data to initialise from
- `check_units_opt=:no`: check linked Variable units are consistent (:no to disable check, :warn to warn and continue, :error to error and stop)
- `eltype::Type=Float64`: default data type to use for model arrays
- `eltypemap=Dict{String, DataType}()`: Dict of data types to look up Variable :datatype attribute
- `method_barrier=nothing`: thread barrier to add to dispatch lists to create a thread-safe model
- `expect_hostdep_varnames=["global.tforce"]`: non-state-Variable host-dependent Variable names expected
- `SolverView_all=true`: `true` to create `modeldata.solver_view_all`
- `create_dispatchlists_all=true`: `true` to create `modeldata.dispatchlists_all`
- `generated_dispatch=true`: `true` to autogenerate code for `modeldata.dispatchlists_all` (fast dispatch, slow compile)
"""
function initialize!(
    model::PB.Model; 
    pickup_output::Union{Nothing, AbstractOutputWriter}=nothing,
    check_units_opt=:no,
    eltype=Float64,
    eltypemap=Dict{String, DataType}(),     
    method_barrier=nothing,
    expect_hostdep_varnames=["global.tforce"],
    SolverView_all=true,
    create_dispatchlists_all=true,
    generated_dispatch=true,
)
    @info """

    ================================================================================
    PALEOmodel.initialize! start
    ================================================================================
    """

    # check Variable linking early and exit if problems, so user sees appropriate error not a downstream consequence when allocating variable etc
    PB.check_variable_links(model; throw_on_error=true, expect_hostdep_varnames)

    modeldata = PB.create_modeldata(model, eltype)
   
    # Allocate variables
    @timeit "allocate_variables" PB.allocate_variables!(model, modeldata, 1; eltypemap, check_units_opt)

    # check all variables allocated
    PB.check_ready(model, modeldata)

    # Create modeldata.solver_view_all for the entire model
    if SolverView_all
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
        set_statevar_from_output!(modeldata, pickup_output)
    end

    initial_state = get_statevar(modeldata.solver_view_all)
     
    @info """

    ================================================================================
    PALEOmodel.initialize! done
    ================================================================================
    """

    return (initial_state, modeldata)
end

"""
    set_statevar_from_output!(modeldata, output::AbstractOutputWriter) -> initial_state

Initialize model state Variables from last record in `output`

NB: `modeldata` must contain `solver_view_all`
"""
function set_statevar_from_output!(modeldata, output::AbstractOutputWriter)

    !isnothing(modeldata.solver_view_all) || error("modeldata.solver_view_all is not available")

    pickup_record = length(output)
    # TODO this reads in implicit as well as explicit state vars
    # should set total corresponding to implicit state vars, and not state vars themselves ?
    all_statevars = vcat(PB.get_vars(modeldata.solver_view_all.stateexplicit), PB.get_vars(modeldata.solver_view_all.state))
    for statevar in all_statevars
        domname = statevar.domain.name
        varname = statevar.name
        if PB.has_variable(output, domname*"."*varname)
            @info "  initialising Variable $domname.$varname from output record $pickup_record"
            vardata = PB.get_data(statevar, modeldata)
            pickup_data = PB.get_data(output, domname*"."*varname, records=pickup_record)
            if typeof(vardata) != typeof(pickup_data)
                # warn and continue - will fail later if broadcast doesn't work
                @warn "output Variable $domname.$varname (type $(typeof(vardata)) size $(size(vardata)), length $(length(vardata)))"*
                    " has different type to pickup data (type $(typeof(pickup_data)) size $(size(pickup_data)) length ($(length(pickup_data))))"
            end
            vardata .= pickup_data
        else
            @info "  Variable $domname.$varname not present in output - leaving default initialisation"
        end
    end
 
    initial_state = get_statevar(modeldata.solver_view_all)
     
    return initial_state
end
