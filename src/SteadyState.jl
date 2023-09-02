module SteadyState

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
# import Infiltrator

import NLsolve
import ..SolverFunctions
import ..ODE
import ..JacobianAD
import ..SplitDAE

import TimerOutputs: @timeit, @timeit_debug

##############################################################
# Solve for steady state using NLsolve.jl
##############################################################

"""
    steadystate(run, initial_state, modeldata, tss [; kwargs...] )
    steadystateForwardDiff(run, initial_state, modeldata, tss [; kwargs...] )

Find steady-state solution (using [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) package) 
and write to `outputwriter` (two records are written, for `initial_state` and the steady-state solution).

`steadystateForwardDiff` has default keyword argument `jac_ad=:ForwardDiffSparse` to use automatic differentiation
for sparse Jacobian.

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tss`:  (yr) model tforce time for steady state solution

# Optional Keywords
- `outputwriter::PALEOmodel.AbstractOutputWriter=run.output`: container to hold output
- `initial_time=-1.0`:  tmodel to write for first output record
- `solvekwargs=NamedTuple()`: NamedTuple of keyword arguments passed through to [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl)
   (eg to set `method`, `ftol`, `iteration`, `show_trace`, `store_trace`).
- `jac_ad`: :NoJacobian, :ForwardDiffSparse, :ForwardDiff
- `use_norm=false`: not supported (must be false)
- `BLAS_num_threads=1`: number of LinearAlgebra.BLAS threads to use
- `generated_dispatch=true`: `true` to use autogenerated code (fast solve, slow compile)
"""
function steadystate(
    run, initial_state, modeldata, tss; 
    outputwriter=run.output,
    initial_time=-1.0,
    @nospecialize(solvekwargs::NamedTuple=NamedTuple{}()),
    jac_ad=:NoJacobian,
    use_norm::Bool=false,
    BLAS_num_threads=1,
    generated_dispatch=true,
)
    @info """
    
    ================================================================================
    PALEOmodel.SteadyState.steadystate
        tss=$tss
        jac_ad=$jac_ad
    ================================================================================
    """

    PB.check_modeldata(run.model, modeldata)

    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"

    !use_norm || ArgumentError("use_norm=true not supported")

    sv = modeldata.solver_view_all
    # check for implicit total variables
    iszero(PALEOmodel.num_total(sv)) || error("implicit total variables, not in constant mass matrix DAE form")
   
    iszero(PALEOmodel.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")

    # calculate residual F = dS/dt
    ssf! = SolverFunctions.ModelODE_at_t(modeldata)
    SolverFunctions.set_t!(ssf!, tss)

    if jac_ad==:NoJacobian
        @info "steadystate: no Jacobian"   
        nldf = NLsolve.OnceDifferentiable(ssf!, similar(initial_state), similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jacode, jac_prototype = JacobianAD.jac_config_ode(jac_ad, run.model, initial_state, modeldata, tss; generated_dispatch)        

        ssJ! = SolverFunctions.JacODE_at_t(jacode, tss)

        if !isnothing(jac_prototype)
            # sparse Jacobian
            nldf = NLsolve.OnceDifferentiable(ssf!, ssJ!, similar(initial_state), similar(initial_state), copy(jac_prototype)) 
        else
            nldf = NLsolve.OnceDifferentiable(ssf!, ssJ!, similar(initial_state), similar(initial_state)) 
        end

    end
    
    @info """
    
    ================================================================================
    steadystate: calling nlsolve...
    ================================================================================
    """

    @time sol = NLsolve.nlsolve(nldf, copy(initial_state); solvekwargs...);
    
    @info """
    
    ================================================================================
        * Algorithm: $(sol.method)
        * Inf-norm of residuals: $(sol.residual_norm)
        * Iterations: $(sol.iterations)
        * Convergence: $(NLsolve.converged(sol))
            * |x - x'| < $(sol.xtol): $(sol.x_converged)
            * |f(x)| < $(sol.ftol): $(sol.f_converged)
        * Function Calls (f): $(sol.f_calls)
        * Jacobian Calls (df/dx): $(sol.g_calls)
    ================================================================================
    """

    resid = similar(initial_state)
    ssf!(resid, sol.zero)
    @info "  check F inf-norm $(LinearAlgebra.norm(resid, Inf)) 2-norm $(LinearAlgebra.norm(resid, 2))"
    
    tsoln = [initial_time, tss]
    soln = [copy(initial_state), copy(sol.zero)]

    PALEOmodel.ODE.calc_output_sol!(outputwriter, run.model, tsoln, soln, modeldata)
    return sol    
end

"[`steadystate`](@ref) with argument defaults to  use ForwardDiff AD Jacobian"
function steadystateForwardDiff(
    run, initial_state, modeldata, tss; 
    jac_ad=:ForwardDiffSparse,
    kwargs...
)

    return steadystate(
        run, initial_state, modeldata, tss; 
        jac_ad=jac_ad,
        kwargs...
    )
end


##############################################################
# Solve for steady state using NLsolve.jl and naive pseudo-transient-continuation
##############################################################

"""
    steadystate_ptc(run, initial_state, modeldata, tspan, deltat_initial; kwargs...) 
    steadystate_ptcForwardDiff(run, initial_state, modeldata, tspan, deltat_initial; kwargs...) 

Find steady-state solution and write to `outputwriter`, using naive pseudo-transient-continuation
with first order implicit Euler pseudo-timesteps from `tspan[1]` to `tspan[2]`
and [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) as the non-linear solver.

`steadystate_ptcForwardDiff` has keyword argument default `jac_ad=:ForwardDiffSparse` to use automatic differentiation for 
sparse Jacobian.

Each pseudo-timestep solves the nonlinear system
S(t+Δt) = S(t) + Δt dS/dt(t+Δt)
for S(t+Δt), using a variant of Newton's method.

Initial pseudo-timestep Δt is `deltat_initial`, this is multiplied by `deltat_fac` for the next iteration
until pseudo-time `tss_max` is reached. If an iteration fails, Δt is divided by `deltat_fac` and the iteration retried.

NB: this is a _very_ naive initial implementation, there is currently no reliable error control to adapt pseudo-timesteps 
to the rate of convergence, so requires some trial-and-error to set an appropiate `deltat_fac` for each problem.

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tspan`: Vector or Tuple with `(initial_time, final_time)`
- `deltat_initial`: initial pseudo-timestep

# Keywords
- `deltat_fac=2.0`: factor to increase pseudo-timestep on success
- `tss_output=Float64[]`: Vector of model times at which to save output (empty Vector to save all output timesteps)
- `outputwriter=run.output`: output destination
- `solvekwargs=NamedTuple()`: arguments to pass through to NLsolve
- `max_iter=1000`: maximum number of PTC iterations
- `max_failed_iter=20`: maximum number of iterations that make no progress before exiting
- `jac_ad=:NoJacobian`: AD Jacobian to use
- `request_adchunksize=10`: ForwardDiff chunk size to request.
- `jac_cellranges=modeldata.cellranges_all`: CellRanges to use for Jacobian calculation
  (eg to restrict to an approximate Jacobian by using a cellrange with a non-default `operatorID`: in this case, Variables that are not calculated
  but needed for the Jacobian should set the `transfer_jacobian` attribute so that they will be copied)
- `enforce_noneg=false`: fail pseudo-timesteps that generate negative values for state variables.
- `step_callbacks=[]`: callbacks on succesful step, `[step_callback(sol.zero, tss, deltat, model, modeldata)]`
- `use_norm=false`: not supported (must be false)
- `verbose=false`: true for detailed output
- `BLAS_num_threads=1`: restrict threads used by Julia BLAS (likely irrelevant if using sparse Jacobian?)
- `generated_dispatch=true`: true to use autogenerated code (fast solve, slow compile)
- [Deprecated: `sol_min`: now has no effect, replace with `solve_kwargs=(project_region! = x->clamp!(x, sol_min, sol_max), )`]
"""
function steadystate_ptc(
    run, initial_state, modeldata, tspan, deltat_initial::Float64;
    deltat_fac=2.0,
    tss_output=Float64[],
    outputwriter=run.output,
    @nospecialize(solvekwargs::NamedTuple=NamedTuple{}()),
    max_iter=1000,
    max_failed_iter=20,
    jac_ad=:NoJacobian,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
    enforce_noneg=false,
    step_callbacks=[],
    sol_min=nothing, # deprecated
    use_norm::Bool=false,
    verbose=false,
    BLAS_num_threads=1,
    generated_dispatch=true,
)
    @info """
    
    ================================================================================
    PALEOmodel.SteadyState.steadystate_ptc:
        tspan=$tspan
        tss_output=$tss_output
        jac_ad=$jac_ad
    ================================================================================
    """

    PB.check_modeldata(run.model, modeldata)

    !use_norm || ArgumentError("use_norm=true not supported")

    isnothing(sol_min) || @warn "steadystate_ptc: 'sol_min' is deprecated and now has no effect: replace with 'solve_kwargs=(project_region! = x->clamp!(x, sol_min, sol_max), )'"

    nlsolveF = nlsolveF_PTC(
        run.model, initial_state, modeldata;
        jac_ad,
        tss_jac_sparsity=tspan[1],
        request_adchunksize,
        jac_cellranges,
        generated_dispatch,
    )

    solve_ptc(
        run, initial_state, nlsolveF, tspan, deltat_initial::Float64;
        deltat_fac,
        max_iter,
        max_failed_iter,
        tss_output,
        outputwriter,
        solvekwargs,
        enforce_noneg,
        step_callbacks,
        verbose,
        BLAS_num_threads,
    )

    return nothing    
end

function steadystate_ptcForwardDiff(
    run, initial_state, modeldata, tspan, deltat_initial::Float64;
    jac_ad=:ForwardDiffSparse,
    kwargs...
)

    return steadystate_ptc(
        run, initial_state, modeldata, tspan, deltat_initial; 
        jac_ad=jac_ad,
        kwargs...
    )
end

function steadystate_ptc(
    run, initial_state, modeldata, tss::Float64, deltat_initial::Float64, tss_max::Float64; kwargs...
) 
    Base.depwarn(
        """
        steadystate_ptc(run, initial_state, modeldata, tss, deltat_initial, tss_max; kwargs...)
            is deprecated. Please use 
        steadystate_ptc(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)""",
        :steadystate_ptc,
        force=true
    )
    return steadystate_ptc(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
end

function steadystate_ptcForwardDiff(
    run, initial_state, modeldata, tss::Float64, deltat_initial::Float64, tss_max::Float64; kwargs...
) 
    Base.depwarn(
        """
        steadystate_ptcForwardDiff(run, initial_state, modeldata, tss, deltat_initial, tss_max; kwargs...)
            is deprecated. Please use 
        steadystate_ptcForwardDiff(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
        """,
        :steadystate_ptcForwardDiff,
        force=true
    )
    return steadystate_ptcForwardDiff(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
end

"""
    steadystate_ptc_splitdae(run, initial_state, modeldata, tspan, deltat_initial; kwargs...) 
    
As [`steadystate_ptc`](@ref), with an inner Newton solve for per-cell algebraic constraints
(eg quasi-steady-state reaction intermediates).

# Keywords (in common with [`steadystate_ptc`](@ref))
- `deltat_fac`
- `tss_output`
- `outputwriter`
- `solvekwargs`
- `max_iter`
- `max_failed_iter`
- `request_adchunksize`
- `jac_cellranges`
- `enforce_noneg`
- `verbose`

# Keywords (additional to [`steadystate_ptc`](@ref))
- `operatorID_inner=3`: operatorID for Reactions to run for inner solve (typically all reservoirs and chemical reactions)
- `transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"]`: Variables not calculated by `operatorID_inner` that need to be copied for 
  inner solve (additional to those with `transfer_jacobian` set).
- `inner_jac_ad::Symbol=:ForwardDiff`: form of automatic differentiation to use for Jacobian for inner `NonlinearNewton.solve` solver (options `:ForwardDiff`, `:ForwardDiffSparse`)
- `inner_start::Symbol=:current`: start value for inner solve (options `:initial`, `:current`, `:zero`)
- `inner_kwargs::NamedTuple=(verbose=0, miniters=2, reltol=1e-12, jac_constant=true, project_region=identity)`: keywords for inner 
  `NonlinearNewton.solve` solver.
"""
function steadystate_ptc_splitdae(
    run, initial_state, modeldata, tspan, deltat_initial::Float64;
    deltat_fac=2.0,
    tss_output=Float64[],
    outputwriter=run.output,
    @nospecialize(solvekwargs::NamedTuple=NamedTuple{}()),
    max_iter=1000,
    max_failed_iter=20,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
    enforce_noneg=false,
    sol_min=nothing,
    verbose=false,
    operatorID_inner=3,
    transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"],
    inner_jac_ad=:ForwardDiff,
    inner_start=:current,
    @nospecialize(inner_kwargs::NamedTuple=(verbose=0, miniters=2, reltol=1e-12, jac_constant=true, project_region=identity)),
    step_callbacks=[],
    BLAS_num_threads=1,
    generated_dispatch=true,
)

    @info """
    
    ================================================================================
    PALEOmodel.SteadyState.steadystate_ptc_splitdae:
        tspan=$tspan
        tss_output=$tss_output
    ================================================================================
    """

    PB.check_modeldata(run.model, modeldata)

    isnothing(sol_min) || @warn "steadystate_ptc_splitdae: 'sol_min' is deprecated: replace with 'solve_kwargs=(project_region!=PALEOmodel.SolverFunctions.ClampAll!(sol_min, Inf), )'"

    if :u_min in keys(inner_kwargs)
        if ! (:project_region in keys(inner_kwargs))
            @warn "steadystate_ptc_splitdae inner_kwargs.u_min is deprecated: replace with project_region=PALEOmodel.SolverFunctions.ClampAll(u_min, Inf)"
            u_min = inner_kwargs.u_min
            inner_kwargs = NamedTuple(k=>v for (k, v) in pairs(inner_kwargs) if k != :u_min)
            inner_kwargs = merge(inner_kwargs, (project_region = x->clamp.(x, u_min, Inf),))
        else
            error("steadystate_ptc_splitdae inner_kwargs defines both project_region and u_min: remove deprecated u_min")
        end
    end

    @timeit "create_split_dae" begin
    ms, initial_state_outer, jacouter_prototype = SplitDAE.create_split_dae(
        run.model, initial_state, modeldata; 
        jac_cellranges,
        request_adchunksize,
        operatorID_inner,
        transfer_inner_vars,
        tss_jac_sparsity=tspan[1],
        inner_jac_ad,
        inner_start,
        inner_kwargs,
        generated_dispatch,
    )
    end # timeit
    nlsolveF = nlsolveF_SplitPTC(ms, initial_state_outer, jacouter_prototype)

    @timeit "solve_ptc" begin
    solve_ptc(
        run, initial_state_outer, nlsolveF, tspan, deltat_initial;
        deltat_fac,
        max_iter,
        max_failed_iter,
        tss_output,
        outputwriter,
        solvekwargs,
        enforce_noneg,
        step_callbacks,
        verbose,
        BLAS_num_threads,
    )
    end # timeit

    return nothing    
end


"""
    ConservationCallback(
        tmodel_start::Float64 # earliest model time to apply correction
        content_name::String # variable with a total of X
        flux_name::String  # variable with a corresponding boundary flux of X
        reservoir_total_name::String  # total for reservoir to apply correction to
        reservoir_statevar_name::String # state variable for reservoir to apply correction to
        reservoir_fac::Float64  # stoichiometric factor (1 / moles of quantity X per reservoir molecule)
    ) -> ccb

Provides a callback function with signature

    ccb(state, tmodel, deltat, model, modeldata)

that modifies `modeldata` arrays and `state` to enforce budget conservation.

# Example

    conservation_callback_H = Callbacks.ConservationCallback(
        tmodel_start=1e5,
        content_name="global.content_H_atmocean",
        flux_name="global.total_H",
        reservoir_total_name="atm.CH4_total", # "atm.H2_total",
        reservoir_statevar_name="atm.CH4_mr", # "atm.H2_mr",
        reservoir_fac=0.25 # 0.5, # H per reservoir molecule
    )

    then add to eg `steadystate_ptc_splitdae` with `step_callbacks` keyword argument:

        step_callbacks = [conservation_callback_H]

"""
Base.@kwdef mutable struct ConservationCallback
    tmodel_start::Float64
    last_content::Float64 = NaN
    last_flux::Float64 = NaN
    content_name::String
    flux_name::String
    reservoir_total_name::String
    reservoir_statevar_name::String
    reservoir_fac::Float64
end

function (ccb::ConservationCallback)(
    state, tss, deltat, model, modeldata
)

    content = only(PB.get_data(PB.get_variable(model, ccb.content_name), modeldata))
    flux = only(PB.get_data(PB.get_variable(model, ccb.flux_name), modeldata))
    av_flux = 0.5*(ccb.last_flux + flux)

    reservoir_total = only(PB.get_data(PB.get_variable(model, ccb.reservoir_total_name), modeldata))

    content_change = content - ccb.last_content

    expected_content_change = av_flux * deltat

    content_error = expected_content_change - content_change 

    reservoir_new_total = reservoir_total + ccb.reservoir_fac*content_error

    if tss > ccb.tmodel_start
        reservoir_multiplier = reservoir_new_total/reservoir_total
    else
        reservoir_multiplier = NaN
    end

    @info """
        ConservationCallback: content $(ccb.content_name) $content, flux $(ccb.flux_name) $flux av_flux $av_flux
                content change: actual $content_change expected $expected_content_change error $content_error
                reservoir: total $(ccb.reservoir_total_name) $reservoir_total correction multiplier $reservoir_multiplier
    """

    if !isnan(reservoir_multiplier)
        # modify state variable in modeldata arrays
        reservoir_statevar = PB.get_data(PB.get_variable(model, ccb.reservoir_statevar_name), modeldata)
        reservoir_statevar .*= reservoir_multiplier
        # copy modified aggregated state vector out from modeldata arrays
        PB.copyto!(state, modeldata.solver_view_all.stateexplicit)
    end

    ccb.last_content = content
    ccb.last_flux = flux

    return !isnan(reservoir_multiplier) # true if state modified
end


"""
    solve_ptc(run, initial_state, nlsolveF, tspan, deltat_initial::Float64; kwargs...)

Pseudo-transient continuation using NLsolve with `nlsolveF` function objects created by `nlsolveF_PTC`
"""
function solve_ptc(
    run, initial_state, @nospecialize(nlsolveF), tspan, deltat_initial::Float64;
    deltat_fac=2.0,
    tss_output=Float64[],
    max_iter=1000,
    max_failed_iter,
    outputwriter=run.output,
    @nospecialize(solvekwargs::NamedTuple=NamedTuple{}()),
    enforce_noneg=false,
    verbose=false,
    step_callbacks=[],
    BLAS_num_threads=1
)

    tss_initial, tss_max = tspan

    # workaround Julia BLAS default (mis)configuration that defaults to multi-threaded
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)

    @info """
    
    ================================================================================
    solve_ptc:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads
    ================================================================================
    """

    ############################################################
    # Vectors to accumulate solution at each pseudo-timestep
    #########################################################
  
    # remove any tss_output prior to start time
    tss_output_filtered = filter(x->x>=tss_initial, tss_output)

    # Vectors to accumulate solution at requested tss_output_filtered
    # first entry is initial state
    @info "    writing output record at initial time tmodel = $(tss_initial)"
    iout = 1                            # tss_output_filtered[iout] is model time of next record to output
    tsoln = [tss_initial]                       # output vector of pseudo-times
    soln = [copy(initial_state)]        # output vector of state vectors at each pseudo-time
    # Always write initial state as first entry (whether requested or not)
    if !isempty(tss_output_filtered) && (tss_output_filtered[1] == tss_initial)
        # don't repeat initial state if that was requested
        iout += 1
    end

    ########################################################
    # unpack callable structs from nlsolveF
    ########################################################
    ssFJ!, nldf = nlsolveF
    modelode = ssFJ!.modelode
    modeldata = modelode.modeldata
    model = modeldata.model

    #################################################
    # outer loop over pseudo-timesteps
    #################################################

    # current pseudo-timestep
    tss = tss_initial
    deltat = deltat_initial
    previous_state = copy(initial_state)
        
    # record any initial state (eg state variables for short lived species)
    initial_inner_state = get_state(ssFJ!) # so we can reset before writing output
    inner_state = get_state(ssFJ!)

    ptc_iter = 1
    failed_iter = 0
    sol = nothing
    @time while tss < tss_max && ptc_iter <= max_iter && failed_iter <= max_failed_iter
        deltat_full = deltat  # keep track of the deltat we could have used
        deltat = min(deltat, tss_max - tss) # limit last timestep to get to tss_max
        if iout <= length(tss_output_filtered)
            deltat = min(deltat, tss_output_filtered[iout] - tss) # limit timestep to get to next requested output
        end

        tss += deltat

        io = IOBuffer()
        verbose && println(io, "================================================================================")
        println(io, "solve_ptc: ptc_iter $ptc_iter tss $(tss) deltat=$(deltat) deltat_full=$deltat_full failed_iter=$failed_iter calling nlsolve...")
        verbose && println(io, "================================================================================")
        @info String(take!(io))
        
        sol_ok = true
        sol_progress = true
        try
            # solve nonlinear system for this pseudo-timestep
            set_step!(ssFJ!, tss, deltat, previous_state, inner_state)
            sol = NLsolve.nlsolve(nldf, previous_state; solvekwargs...)
            
            if verbose
                @info """
                
                ================================================================================
                 * Algorithm: $(sol.method)
                 * Inf-norm of residuals: $(sol.residual_norm)
                 * Iterations: $(sol.iterations)
                 * Convergence: $(NLsolve.converged(sol))
                   * |x - x'| < $(sol.xtol): $(sol.x_converged)
                   * |f(x)| < $(sol.ftol): $(sol.f_converged)
                 * Function Calls (f): $(sol.f_calls)
                 * Jacobian Calls (df/dx): $(sol.g_calls)
                ================================================================================
                """

                ssf!(worksp, sol.zero)
                @info "  check F inf-norm $(norm(worksp, Inf)) 2-norm $(norm(worksp, 2))"
            else
                @info "    Residual inf norm: $(sol.residual_norm) Iterations: $(sol.iterations) |f(x)| < $(sol.ftol): $(sol.f_converged)"
            end

            sol_ok = sol.f_converged
            sol_progress = sol.f_converged && (sol.iterations > 0)
            
            if sol.f_converged && enforce_noneg
                sol_hasneg = any(x -> x < 0.0, sol.zero)
                if sol_hasneg
                    @info "  solution has -ve values"
                    sol_ok = false
                end
            end
        catch e
            if isa(e, LinearAlgebra.SingularException)
                @warn "LinearAlgebra.SingularException"
                sol_ok = false # will force timestep reduction and retry
                sol_progress = false
            elseif isa(e, ErrorException)
                @warn "ErrorException: $(e.msg)"
                sol_ok = false # will force timestep reduction and retry
                sol_progress = false
            else
                throw(e) # rethrow and fail
            end
        end
       
        if sol_ok
            # write output record, if required
            # NB: test tss_output not tss_output_filtered, so we don't write every timestep if !isempty(tss_output) but all tss_output filtered out
            if isempty(tss_output) ||                       # all records requested, or ...
                (iout <= length(tss_output_filtered) &&                  # (not yet done last requested record
                    tss >= tss_output_filtered[iout])                    # and just gone past a requested record)              
                @info "    writing output record at tmodel = $(tss)"
                push!(tsoln, tss)
                push!(soln, copy(sol.zero))
                iout += 1
            end

            # record additional state (if any) from succesful step
            get_state!(inner_state, ssFJ!)

            # apply callbacks (if any). May modify state variables
            for sc in step_callbacks
                sc(sol.zero, tss, deltat, model, modeldata)
            end
            
            previous_state .= sol.zero
        
            # very crude pseudo-timestep adaptation (increase on success, reduce on failure)
            if deltat == deltat_full
                # we used the full deltat and it worked - increase deltat
                deltat *= deltat_fac
            else
                # we weren't using the full timestep as an output was requested, so go back to full
                deltat = deltat_full
            end           

        else
            @warn "iter failed, reducing deltat"
            tss -=  deltat
            deltat /= deltat_fac^2
        end

        if sol_progress
            failed_iter = 0
        else
            failed_iter += 1
        end
        
        ptc_iter += 1
    end

    ptc_iter <= max_iter || @warn("     max iterations $max_iter exceeded")
    failed_iter <= max_failed_iter || @warn("     max failed iterations $max_failed_iter exceeded")
    # always write the last record even if it wasn't explicitly requested
    if tsoln[end] != tss
        @info "    writing output record at tmodel = $(tss)"
        push!(tsoln, tss)
        push!(soln, copy(sol.zero))
    end

    set_step!(ssFJ!, tss, 0.0, previous_state, initial_inner_state) # restore initial_inner_state
    # NB: if using split dae with inner_state, this is updated in modeldata arrays as output is recalculated
    PALEOmodel.ODE.calc_output_sol!(outputwriter, run.model, tsoln, soln, modelode, modeldata)

    @info """
    
    ================================================================================
    solve_ptc:  done
    ================================================================================
    """

    return nothing    
end

"""
    FJacPTC

Function object to calculate residual for a first-order Euler implicit timestep and adapt to NLsolve interface

Given:
- ODE time derivative `du(t)/dt` supplied at construction time by function object `modelode(du, u, p, t)`
- ODE Jacobian `d(du(t)/dt)/du` supplied at construction time by function  object `jacode(J, u, p, t)`
- `t`, `delta_t`, `previous_u` set by `set_step!` before each function call.

Calculates `F(u)` and `J(u)` (Jacobian of `F`), where `F(u)` is the residual for a timestep `delta_t` to time `t`
from state `previous_u`:
- `F(u) = (u(t) - previous_u + delta_t * du(t)/dt)`
- `J(u) = I - deltat * d(du(t)/dt)/du`
"""
struct FJacPTC
    modelode #::M no specialization to minimise recompilation
    jacode #::J
    t::Base.RefValue{Float64}
    delta_t::Base.RefValue{Float64}
    previous_u::Vector{Float64}
    du_worksp::Vector{Float64}
end

"""
    set_step!(fjp::FJacPTC, t, deltat, previous_u, state::Nothing)

Set time to step to `t`, `delta_t` of this step, and `previous_u` (value of state vector at previous time step `t - delta_t`).
"""
function set_step!(fjp::FJacPTC, t, deltat, previous_u, state::Nothing)
    fjp.t[] = t
    fjp.delta_t[] = deltat
    fjp.previous_u .= previous_u

    return nothing
end

# no additional internal state
function get_state!(state::Nothing, fjp::FJacPTC)
end

function get_state(fjp::FJacPTC)
    return nothing
end

# F only
(jn::FJacPTC)(F, u) = jn(F, nothing, u)

# Jacobian only
(jn::FJacPTC)(J::SparseArrays.SparseMatrixCSC, u) = jn(nothing, J, u)

# F and J
function (jn::FJacPTC)(F, J::Union{SparseArrays.SparseMatrixCSC, Nothing}, u)
    
    jn.modelode(jn.du_worksp, u, nothing, jn.t[])

    if !isnothing(F)
        F .=  (u .- jn.previous_u - jn.delta_t[].*jn.du_worksp)
    end

    if !isnothing(J)
  
        jn.jacode(J, u, nothing, jn.t[])
        # convert J  = I - deltat * odeJac  
        for j in 1:size(J)[2]
            # idx is index in SparseMatrixCSC compressed storage, i is row index
            for idx in J.colptr[j]:(J.colptr[j+1]-1)
                i = J.rowval[idx]

                J.nzval[idx] = -jn.delta_t[]*J.nzval[idx]                

                if i == j
                    J.nzval[idx] += 1.0
                end                
            end
        end
    end

    return nothing
end

"""
    nlsolveF_PTC(
        model, initial_state, modeldata;
        jac_ad=:NoJacobian,
        request_adchunksize=10,
        jac_cellranges=modeldata.cellranges_all,
        generated_dispatch=true,
    ) -> (ssFJ!::FJacPTC, nldf::NLsolve.OnceDifferentiable)

Create a function object `nldf` to pass to NLsolve.
    
`ssFJ!` is a callable struct that holds model state, model time, and requested backwards-Euler timestep,
with methods to calculate time derivative and (optional) Jacobian.
    
`nldf` is a NLsolve wrapper containing `ssFJ!`
"""
function nlsolveF_PTC(
    model, initial_state, modeldata;
    jac_ad=:NoJacobian,
    tss_jac_sparsity=nothing,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
    generated_dispatch=true,
)
    PB.check_modeldata(model, modeldata)

    sv = modeldata.solver_view_all

    # We only support explicit ODE-like configurations (no DAE constraints or implicit variables)
    iszero(PALEOmodel.num_total(sv))                 || error("implicit total variables not supported")
    iszero(PALEOmodel.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")

    # previous_u is state at previous timestep
    previous_u = similar(initial_state)
    # workspace arrays 
    du_worksp    = similar(initial_state)

    # current time and deltat (Refs are passed into function objects, and then updated each timestep)
    tss = Ref(NaN)
    deltat = Ref(NaN)

    modelode = SolverFunctions.ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)

    if jac_ad==:NoJacobian
        @info "nlsolveF_PTC: no Jacobian"
        # Define the function we want to solve 
        ssFJ! = FJacPTC(modelode, nothing, tss, deltat, nothing, nothing, previous_u, du_worksp)
        nldf = NLsolve.OnceDifferentiable(ssFJ!, similar(initial_state), similar(initial_state)) 
    else       
        @info "nlsolveF_PTC:  using Jacobian $jac_ad"
        jacode, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
            jac_ad, model, initial_state, modeldata, tss_jac_sparsity;
            request_adchunksize,
            jac_cellranges,
            generated_dispatch,
        )
       
        ssFJ! = FJacPTC(modelode, jacode, tss, deltat, previous_u, du_worksp)
 
        # Define the function + Jacobian we want to solve
        !isnothing(jac_prototype) || error("Jacobian is not sparse")
        # function + sparse Jacobian with sparsity pattern defined by jac_prototype
        nldf = NLsolve.OnceDifferentiable(ssFJ!, ssFJ!, ssFJ!, similar(initial_state), similar(initial_state), copy(jac_prototype))         
    end

    return (ssFJ!, nldf)
end


"""
    FJacSplitPTC

Function object to calculate residual for a first-order Euler implicit timestep and adapt to NLsolve interface

Given:
- ODE time derivative `du(t)/dt` supplied at construction time by function object `modelode(du, u, p, t)`
- ODE Jacobian `d(du(t)/dt)/du` supplied at construction time by function  object `jacode(J, u, p, t)`
- `t`, `delta_t`, `previous_u` set by `set_step!` before each function call.

Calculates `F(u)` and `J(u)` (Jacobian of `F`), where `F(u)` is the residual for a timestep `delta_t` to time `t`
from state `previous_u`:
- `F(u) = (u(t) - previous_u + delta_t * du(t)/dt)`
- `J(u) = I - deltat * d(du(t)/dt)/du`
"""
struct FJacSplitPTC
    modeljacode # ::M # no specialization as this seems to cause compiler issues with Julia 1.7.3
    t::Base.RefValue{Float64}
    delta_t::Base.RefValue{Float64}
    previous_u::Vector{Float64}
    du_worksp::Vector{Float64}
end

function Base.getproperty(obj::FJacSplitPTC, sym::Symbol)
    if sym === :modelode
        return obj.modeljacode
    else
        return getfield(obj, sym)
    end
end

"""
    set_step!(fjp::FJacSplitPTC, t, deltat, previous_u, state)

Set time to step to `t`, `delta_t` of this step, `previous_u` (value of outer state vector at previous time step `t - delta_t`),
and inner state variables (corresponding to algebraic constraints) in modeldata arrays to `state`.
"""
function set_step!(fjp::FJacSplitPTC, t, deltat, previous_u, state)
    fjp.t[] = t
    fjp.delta_t[] = deltat
    fjp.previous_u .= previous_u
    # set inner state Variables from last timestep
    copyto!(fjp.modeljacode.solver_view_all.state, state)

    return nothing
end

"""
    get_state!(state, fjp::FJacSplitPTC)
    get_state(fjp::FJacSplitPTC) -> state

Record values for inner state variables from modeldata arrays to `state`
"""
function get_state!(state, fjp::FJacSplitPTC)
    # record inner state Variables from successful timestep
    copyto!(state, fjp.modeljacode.solver_view_all.state)
    return nothing
end

function get_state(fjp::FJacSplitPTC)
    return  PB.get_data(fjp.modeljacode.solver_view_all.state)
end



# F only
function (jn::FJacSplitPTC)(F::AbstractVector, u::AbstractVector)

    jn.modeljacode(jn.du_worksp, u, nothing, jn.t[])

    F .=  (u .- jn.previous_u - jn.delta_t[].*jn.du_worksp)

    return nothing
end

# Jacobian only (just discard F)
function (jn::FJacSplitPTC)(J::SparseArrays.SparseMatrixCSC, u::AbstractVector)
    jn(nothing, J, u)

    return nothing
end

# F and J
function (jn::FJacSplitPTC)(F::Union{Nothing, AbstractVector}, J::SparseArrays.SparseMatrixCSC, u::AbstractVector)
    
    jn.modeljacode(jn.du_worksp, J, u, nothing, jn.t[])

    if !isnothing(F)
        F .=  (u .- jn.previous_u - jn.delta_t[].*jn.du_worksp)
    end

    # convert J  = I - deltat * odeJac  
    for j in 1:size(J)[2]
        # idx is index in SparseMatrixCSC compressed storage, i is row index
        for idx in J.colptr[j]:(J.colptr[j+1]-1)
            i = J.rowval[idx]

            J.nzval[idx] = -jn.delta_t[]*J.nzval[idx]                

            if i == j
                J.nzval[idx] += 1.0
            end                
        end
    end

    return nothing
end

function nlsolveF_SplitPTC(ms::SplitDAE.ModelSplitDAE, initial_state_outer, jacouter_prototype)

    tss = Ref(NaN)
    deltat = Ref(NaN)
    previous_u = similar(initial_state_outer)
    du_worksp = similar(initial_state_outer)

    # SplitDAE.ModelSplitDAE provides both ODE and Jacobian functions
    ssFJ! = FJacSplitPTC(ms, tss, deltat, previous_u, du_worksp)

    # function + sparse Jacobian with sparsity pattern defined by jac_prototype
    nldf = NLsolve.OnceDifferentiable(ssFJ!, ssFJ!, ssFJ!, similar(initial_state_outer), similar(initial_state_outer), copy(jacouter_prototype))

    return (ssFJ!, nldf)
end

end # module
