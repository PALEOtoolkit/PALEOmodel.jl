module SteadyState

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import Infiltrator

import NLsolve
import ..SolverFunctions
import ..ODE
import ..JacobianAD



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
"""
function steadystate(
    run, initial_state, modeldata, tss; 
    outputwriter=run.output,
    initial_time=-1.0,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    use_norm::Bool=false,
    BLAS_num_threads=1,
)
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
        df = NLsolve.OnceDifferentiable(ssf!, similar(initial_state), similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jacode, jac_prototype = JacobianAD.jac_config_ode(jac_ad, run.model, initial_state, modeldata, tss)        

        ssJ! = SolverFunctions.JacODE_at_t(jacode, tss)

        if !isnothing(jac_prototype)
            # sparse Jacobian
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, similar(initial_state), similar(initial_state), copy(jac_prototype)) 
        else
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, similar(initial_state), similar(initial_state)) 
        end

    end
    
    @info lpad("", 80, "=")
    @info "steadystate: calling nlsolve..."
    @info lpad("", 80, "=")
    @time sol = NLsolve.nlsolve(df, copy(initial_state); solvekwargs...);
    
    @info lpad("", 80, "=")
    @info " * Algorithm: $(sol.method)"
    @info " * Inf-norm of residuals: $(sol.residual_norm)"
    @info " * Iterations: $(sol.iterations)"
    @info " * Convergence: $(NLsolve.converged(sol))"
    @info "   * |x - x'| < $(sol.xtol): $(sol.x_converged)"
    @info "   * |f(x)| < $(sol.ftol): $(sol.f_converged)"
    @info " * Function Calls (f): $(sol.f_calls)"
    @info " * Jacobian Calls (df/dx): $(sol.g_calls)"
    @info lpad("", 80, "=")

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
- `tss_output=[]`: Vector of model times at which to save output (empty Vector to save all output timesteps)
- `outputwriter=run.output`: output destination
- `solvekwargs=NamedTuple()`: arguments to pass through to NLsolve
- `jac_ad=:NoJacobian`: AD Jacobian to use
- `request_adchunksize=10`: ForwardDiff chunk size to request.
- `jac_cellranges=modeldata.cellranges_all`: CellRanges to use for Jacobian calculation
  (eg to restrict to an approximate Jacobian)
- `enforce_noneg=false`: fail pseudo-timesteps that generate negative values for state variables.
- `use_norm=false`: not supported (must be false)
- `verbose=false`: true for detailed output
- `BLAS_num_threads=1`: restrict threads used by Julia BLAS (likely irrelevant if using sparse Jacobian?)
"""
function steadystate_ptc(
    run, initial_state, modeldata, tspan, deltat_initial::Float64;
    deltat_fac=2.0,
    tss_output=[],
    outputwriter=run.output,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
    enforce_noneg=false,
    use_norm::Bool=false,
    verbose=false,
    BLAS_num_threads=1
)
    PB.check_modeldata(run.model, modeldata)

    !use_norm || ArgumentError("use_norm=true not supported")

    nlsolveF = nlsolveF_PTC(
        run.model, initial_state, modeldata;
        jac_ad=jac_ad,
        tss_jac_sparsity=tspan[1],
        request_adchunksize=request_adchunksize,
        jac_cellranges=jac_cellranges,
    )

    solve_ptc(
        run, initial_state, nlsolveF, tspan, deltat_initial::Float64;
        deltat_fac=deltat_fac,
        tss_output=tss_output,
        outputwriter=outputwriter,
        solvekwargs=solvekwargs,
        enforce_noneg=enforce_noneg,
        verbose=verbose,
        BLAS_num_threads=BLAS_num_threads,
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
    nlsolveF_PTC(
        model, initial_state, modeldata;
        jac_ad=:NoJacobian,
        request_adchunksize=10,
        jac_cellranges=modeldata.cellranges_all,
    ) -> (ssFJ!, df::NLsolve.OnceDifferentiable)

Create function object to pass to NLsolve, with function + optional Jacobian
"""
function nlsolveF_PTC(
    model, initial_state, modeldata;
    jac_ad=:NoJacobian,
    tss_jac_sparsity=nothing,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
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
        @info "steadystate: no Jacobian"
        # Define the function we want to solve 
        ssFJ! = FJacPTC(modelode, nothing, tss, deltat, nothing, nothing, previous_u, du_worksp)
        df = NLsolve.OnceDifferentiable(ssFJ!, similar(initial_state), similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jacode, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
            jac_ad, model, initial_state, modeldata, tss_jac_sparsity,
            request_adchunksize=request_adchunksize,
            jac_cellranges=jac_cellranges
        )    

        transfer_data_ad, transfer_data = PALEOmodel.JacobianAD.jac_transfer_variables(
            model,
            jacode.modeldata,
            modeldata
        )
       
        ssFJ! = FJacPTC(modelode, jacode, tss, deltat, transfer_data_ad, transfer_data, previous_u, du_worksp)
 
        # Define the function + Jacobian we want to solve
        !isnothing(jac_prototype) || error("Jacobian is not sparse")
        # function + sparse Jacobian with sparsity pattern defined by jac_prototype
        df = NLsolve.OnceDifferentiable(ssFJ!, ssFJ!, ssFJ!, similar(initial_state), similar(initial_state), copy(jac_prototype))         
    end

    return (ssFJ!, df)
end

"""
    solve_ptc(run, initial_state, nlsolveF, tspan, deltat_initial::Float64; kwargs...)

Pseudo-transient continuation using NLsolve with `nlsolveF` function objects created by `nlsolveF_PTC`
"""
function solve_ptc(
    run, initial_state, nlsolveF, tspan, deltat_initial::Float64;
    deltat_fac=2.0,
    tss_output=[],
    outputwriter=run.output,
    solvekwargs::NamedTuple=NamedTuple{}(),
    enforce_noneg=false,
    verbose=false,
    BLAS_num_threads=1
)

    tss_initial, tss_max = tspan

    # workaround Julia BLAS default (mis)configuration that defaults to multi-threaded
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate_ptc:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"
 
    
    ############################################################
    # Vectors to accumulate solution at each pseudo-timestep
    #########################################################
    
    # first entry is initial state
    tsoln = [tss_initial]                       # vector of pseudo-times
    soln = [copy(initial_state)]        # vector of state vectors at each pseudo-time

    # Vectors to accumulate solution at requested tss_output    
    iout = 1                            # tss_output[iout] is model time of next record to output
    tsoln = [tss_initial]                       # output vector of pseudo-times
    soln = [copy(initial_state)]        # output vector of state vectors at each pseudo-time
    # Always write initial state as first entry (whether requested or not)
    if !isempty(tss_output) && (tss_output[1] == tss_initial)
        # don't repeat initial state if that was requested
        iout += 1
    end

    ########################################################
    # state held in nlsolveF
    ########################################################
    ssFJ!, df = nlsolveF
    tss = ssFJ!.t
    deltat = ssFJ!.delta_t
    previous_state = ssFJ!.previous_u
    modeldata = ssFJ!.modelode.modeldata
 
    #################################################
    # outer loop over pseudo-timesteps
    #################################################
    
    # current pseudo-timestep
    tss = tss_initial
    deltat = deltat_initial
    previous_state = copy(initial_state)
        
    ptc_iter = 1
    sol = nothing
    @time while tss < tss_max
        deltat_full = deltat  # keep track of the deltat we could have used
        deltat = min(deltat, tss_max - tss) # limit last timestep to get to tss_max
        if iout < length(tss_output)
            deltat = min(deltat, tss_output[iout] - tss) # limit timestep to get to next requested output
        end

        tss += deltat

        verbose && @info lpad("", 80, "=")
        @info "steadystate: ptc_iter $ptc_iter tss $(tss) deltat=$(deltat) deltat_full=$deltat_full calling nlsolve..."
        verbose && @info lpad("", 80, "=")
        
        sol_ok = true
        try
            # solve nonlinear system for this pseudo-timestep
            set_step!(ssFJ!, tss, deltat, previous_state)
            sol = NLsolve.nlsolve(df, previous_state; solvekwargs...)
            
            if verbose
                io = IOBuffer()
                println(io, lpad("", 80, "="))
                println(io, " * Algorithm: $(sol.method)")
                println(io, " * Inf-norm of residuals: $(sol.residual_norm)")
                println(io, " * Iterations: $(sol.iterations)")
                println(io, " * Convergence: $(NLsolve.converged(sol))")
                println(io, "   * |x - x'| < $(sol.xtol): $(sol.x_converged)")
                println(io, "   * |f(x)| < $(sol.ftol): $(sol.f_converged)")
                println(io, " * Function Calls (f): $(sol.f_calls)")
                println(io, " * Jacobian Calls (df/dx): $(sol.g_calls)")
                println(io, lpad("", 80, "="))
                @info String(take!(io))

                ssf!(worksp, sol.zero)
                @info "  check F inf-norm $(norm(worksp, Inf)) 2-norm $(norm(worksp, 2))"
            else
                @info "    Residual inf norm: $(sol.residual_norm) Iterations: $(sol.iterations) |f(x)| < $(sol.ftol): $(sol.f_converged)"
            end

            sol_ok = sol.f_converged
            
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
            else
                throw(e) # rethrow and fail
            end
        end

        # very crude pseudo-timestep adaptation (increase on success, reduce on failure)
        if sol_ok          
            if deltat == deltat_full
                # we used the full deltat and it worked - increase deltat
                deltat *= deltat_fac
            else
                # we weren't using the full timestep as an output was requested, so go back to full
                deltat = deltat_full
            end
            previous_state .= sol.zero
            # write output record, if required
            if isempty(tss_output) ||                       # all records requested, or ...
                (iout <= length(tss_output) &&                  # (not yet done last requested record
                    tss >= tss_output[iout])                    # and just gone past a requested record)              
                @info "    writing output record at tmodel = $(tss)"
                push!(tsoln, tss)
                push!(soln, copy(sol.zero))
                iout += 1
            end

        else
            @warn "iter failed, reducing deltat"
            tss -=  deltat
            deltat /= deltat_fac^2
        end
        
        ptc_iter += 1
    end

    # always write the last record even if it wasn't explicitly requested
    if tsoln[end] != tss
        @info "    writing output record at tmodel = $(tss)"
        push!(tsoln, tss)
        push!(soln, copy(sol.zero))
    end

    PALEOmodel.ODE.calc_output_sol!(outputwriter, run.model, tsoln, soln, modeldata)
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
struct FJacPTC{M, J, T1, T2, W}
    modelode::M
    jacode::J
    t::Ref{Float64}
    delta_t::Ref{Float64}
    transfer_data_ad::T1
    transfer_data::T2
    previous_u::W
    du_worksp::W
end

"""
    set_step!(fjp::FJacPTC, t, deltat, previous_u)

Set time to step to `t`, `delta_t` of this step, and `previous_u` (value of state vector at previous time step `t - delta_t`).
"""
function set_step!(fjp::FJacPTC, t, deltat, previous_u)
    fjp.t[] = t
    fjp.delta_t[] = deltat
    fjp.previous_u .= previous_u

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
        # transfer Variables not recalculated by Jacobian
        for (d_ad, d) in PB.IteratorUtils.zipstrict(jn.transfer_data_ad, jn.transfer_data)                
            d_ad .= d
        end
  
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

end # module
