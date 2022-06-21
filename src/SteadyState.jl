module SteadyState

import PALEOboxes as PB

import PALEOmodel

using LinearAlgebra
import Infiltrator

import NLsolve
using ForwardDiff
using SparseArrays
using SparseDiffTools



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
- `use_norm=false`: true to normalize state variables using PALEO norm_value
- `BLAS_num_threads=1`: number of LinearAlgebra.BLAS threads to use
"""
function steadystate(
    run, initial_state, modeldata, tss; 
    outputwriter=run.output,
    initial_time=-1.0,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    use_norm=false,
    BLAS_num_threads=1,
)

    nevals = 0

    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"

    sv = modeldata.solver_view_all
    # check for implicit total variables
    iszero(PB.num_total(sv)) || error("implicit total variables, not in constant mass matrix DAE form")
   
    iszero(PB.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")
       
    # workspace array 
    worksp = similar(initial_state)

    # normalisation factors
    state_norm_factor  = ones(length(worksp))
    
    if use_norm
        @info "steadystate: using PALEO normalisation for state variables and time derivatives"
        state_norm_factor  .= PB.get_statevar_norm(sv)       
    end

    # calculate normalized residual F = dS/dt
    function ssf!(Fnorm, unorm)    
        worksp .= unorm .* state_norm_factor
       
        PB.set_tforce!(modeldata.solver_view_all, tss)
        PB.set_statevar!(modeldata.solver_view_all, worksp) 

        PB.do_deriv(modeldata.dispatchlists_all)

        PB.get_statevar_sms!(Fnorm, modeldata.solver_view_all)
        Fnorm ./= state_norm_factor
       
        nevals += 1
        return nothing
    end

    if jac_ad==:NoJacobian
        @info "steadystate: no Jacobian"   
        df = NLsolve.OnceDifferentiable(ssf!, initial_state, similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jac, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(jac_ad, run.model, initial_state, modeldata, tss)    

        function ssJ!(Jnorm, unorm)       
            worksp .= unorm .* state_norm_factor
           
            # odejac calculates un-normalized J
            jac(Jnorm, worksp, nothing, tss)
            if use_norm
                # in-place multiplication to get Jnorm
                for j in 1:size(Jnorm)[2]
                    for i in 1:size(Jnorm)[1]
                        Jnorm[i, j] *= state_norm_factor[j] ./ state_norm_factor[i]
                    end
                end
            end
            return nothing
        end

        function ssJ!(Jnorm::SparseArrays.SparseMatrixCSC, unorm)
            worksp .= unorm .* state_norm_factor
           
            # odejac calculates un-normalized J
            jac(Jnorm, worksp, nothing, tss)
            if use_norm
                # in-place multiplication to get Jnorm
                for j in 1:size(Jnorm)[2]
                    for idx in Jnorm.colptr[j]:(Jnorm.colptr[j+1]-1)
                        i = Jnorm.rowval[idx]
                        Jnorm.nzval[idx] *= state_norm_factor[j] ./ state_norm_factor[i]
                    end
                end
            end
            return nothing
        end

        if !isnothing(jac_prototype)
            # sparse Jacobian
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, initial_state, similar(initial_state), copy(jac_prototype)) 
        else
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, initial_state, similar(initial_state)) 
        end

    end
    
    @info lpad("", 80, "=")
    @info "steadystate: calling nlsolve..."
    @info lpad("", 80, "=")
    @time sol = NLsolve.nlsolve(df, initial_state; solvekwargs...);
    
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

    ssf!(worksp, sol.zero)
    @info "  check Fnorm inf-norm $(norm(worksp, Inf)) 2-norm $(norm(worksp, 2))"
    
    tsoln = [initial_time, tss]
    soln = [initial_state, sol.zero .* state_norm_factor]

    PALEOmodel.ODE.calc_output_sol(outputwriter, run.model, tsoln, soln, modeldata)
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
- `use_norm=false`: true to apply PALEO norm_value to state variables
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
    use_norm=false,
    verbose=false,
    BLAS_num_threads=1
)

    nevals = 0
  
    # start, end times
    tss, tss_max = tspan

    # workaround Julia BLAS default (mis)configuration that defaults to multi-threaded
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate_ptc:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"
 
    sv = modeldata.solver_view_all
    # We only support explicit ODE-like configurations (no DAE constraints or implicit variables)
    iszero(PB.num_total(sv))                 || error("implicit total variables not supported")
    iszero(PB.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")

    # workspace arrays 
    worksp          = similar(initial_state)
    deriv_worksp    = similar(initial_state)

    # normalisation factors
    state_norm_factor  = ones(length(worksp))
    if use_norm
        @info "steadystate: using PALEO normalisation for state variables and time derivatives"
        state_norm_factor  .= PB.get_statevar_norm(sv)       
    end

    # current pseudo-timestep
    deltat = deltat_initial

    # calculate normalized residual F = S - Sinit - deltat*dS/dt
    # (this is a 'closure', where deltat, state_norm_factor, etc refer to 
    #  captured variables defined outside the ssf! function)
    function ssf!(Fnorm, unorm)       
             
        worksp .= unorm .* state_norm_factor

        PB.set_tforce!(modeldata.solver_view_all, tss)
        PB.set_statevar!(modeldata.solver_view_all, worksp) 

        PB.do_deriv(modeldata.dispatchlists_all)

        PB.get_statevar_sms!(deriv_worksp, modeldata.solver_view_all)
        Fnorm .=  (worksp .- initial_state - deltat.*deriv_worksp) ./ state_norm_factor

        nevals += 1
        return nothing
    end

    # Define 'df' object to pass to NLsolve, with function + optional Jacobian
    if jac_ad==:NoJacobian
        @info "steadystate: no Jacobian"
        # Define the function we want to solve 
        df = NLsolve.OnceDifferentiable(ssf!, initial_state, similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jac, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
            jac_ad, run.model, initial_state, modeldata, tss,
            request_adchunksize=request_adchunksize,
            jac_cellranges=jac_cellranges
        )    

        (transfer_data_ad, transfer_data) = PALEOmodel.JacobianAD.jac_transfer_variables(
            run.model,
            jac.modeldata,
            modeldata
        )
        # Calculate dense Jacobian = dF/dS = I - deltat * odeJac
        function ssJ!(Jnorm, unorm)       
            worksp .= unorm .* state_norm_factor
        
            if !isempty(transfer_data_ad)
                PB.set_tforce!(modeldata.solver_view_all, tss)
                PB.set_statevar!(modeldata.solver_view_all, worksp)                 
                PB.do_deriv(modeldata.dispatchlists_all)
                # transfer Variables not recalculated by Jacobian
                for (d_ad, d) in zip(transfer_data_ad, transfer_data)                
                    d_ad .= d
                end
            end

            # odejac calculates un-normalized J
            jac(Jnorm, worksp, nothing, tss)
            # convert J  = I - deltat * odeJac  
            for j in 1:size(Jnorm)[2]
                for i in 1:size(Jnorm)[1]
                    Jnorm[i, j] = -deltat*Jnorm[i,j]
                    if i == j
                        Jnorm[i, j] += 1.0
                    end

                    # normalize 
                    Jnorm[i, j] *= state_norm_factor[j] ./ state_norm_factor[i]
                end
            end
           
            return nothing
        end

        # Calculate sparse Jacobian = dF/dS = I - deltat * odeJac
        function ssJ!(Jnorm::SparseArrays.SparseMatrixCSC, unorm)
          
            worksp .= unorm .* state_norm_factor
        
            if !isempty(transfer_data_ad)
                PB.set_tforce!(modeldata.solver_view_all, tss)
                PB.set_statevar!(modeldata.solver_view_all, worksp)                 
                PB.do_deriv(modeldata.dispatchlists_all)
                # transfer Variables not recalculated by Jacobian
                for (d_ad, d) in zip(transfer_data_ad, transfer_data)                
                    d_ad .= d
                end
            end

            # odejac calculates un-normalized J
            jac(Jnorm, worksp, nothing, tss)
            # convert J  = I - deltat * odeJac  
            for j in 1:size(Jnorm)[2]
                # idx is index in SparseMatrixCSC compressed storage, i is row index
                for idx in Jnorm.colptr[j]:(Jnorm.colptr[j+1]-1)
                    i = Jnorm.rowval[idx]

                    Jnorm.nzval[idx] = -deltat*Jnorm.nzval[idx]
                    if i == j
                        Jnorm.nzval[idx] += 1.0
                    end

                    # normalize
                    Jnorm.nzval[idx] *= state_norm_factor[j] ./ state_norm_factor[i]
                end
            end
    
            return nothing
        end

        function ssFJ!(Fnorm, Jnorm::SparseArrays.SparseMatrixCSC, unorm)
            # println("ssFJ! tss=", tss)
            worksp .= unorm .* state_norm_factor
        
            PB.set_tforce!(modeldata.solver_view_all, tss)
            PB.set_statevar!(modeldata.solver_view_all, worksp) 

            PB.do_deriv(modeldata.dispatchlists_all)

            PB.get_statevar_sms!(deriv_worksp, modeldata.solver_view_all)
            Fnorm .=  (worksp .- initial_state - deltat.*deriv_worksp) ./ state_norm_factor

            nevals += 1

            # transfer Variables not recalculated by Jacobian
            for (d_ad, d) in zip(transfer_data_ad, transfer_data)                
                d_ad .= d
            end

            # odejac calculates un-normalized J          
            jac(Jnorm, worksp, nothing, tss)
            # convert J  = I - deltat * odeJac  
            for j in 1:size(Jnorm)[2]
                # idx is index in SparseMatrixCSC compressed storage, i is row index
                for idx in Jnorm.colptr[j]:(Jnorm.colptr[j+1]-1)
                    i = Jnorm.rowval[idx]

                    Jnorm.nzval[idx] = -deltat*Jnorm.nzval[idx]
                    if i == j
                        Jnorm.nzval[idx] += 1.0
                    end

                    # normalize
                    Jnorm.nzval[idx] *= state_norm_factor[j] ./ state_norm_factor[i]
                end
            end
    
            return nothing
        end

        # Define the function + Jacobian we want to solve
        if !isnothing(jac_prototype)
            # function + sparse Jacobian with sparsity pattern defined by jac_prototype
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, ssFJ!, initial_state, similar(initial_state), copy(jac_prototype)) 
        else
            # function + dense Jacobian
            df = NLsolve.OnceDifferentiable(ssf!, ssJ!, ssFJ!, initial_state, similar(initial_state)) 
        end

    end

    # Vectors to accumulate solution at each pseudo-timestep
    # first entry is initial state
    tsoln = [tss]                       # vector of pseudo-times
    soln = [copy(initial_state)]        # vector of state vectors at each pseudo-time

    # Vectors to accumulate solution at requested tss_output    
    iout = 1                            # tss_output[iout] is model time of next record to output
    tsoln = [tss]                       # output vector of pseudo-times
    soln = [copy(initial_state)]        # output vector of state vectors at each pseudo-time
    # Always write initial state as first entry (whether requested or not)
    if !isempty(tss_output) && (tss_output[1] == tss)
        # don't repeat initial state if that was requested
        iout += 1
    end
 

    # outer loop over pseudo-timesteps
    initial_state = copy(initial_state)
    
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
        @info "steadystate: ptc_iter $ptc_iter tss $tss deltat=$deltat deltat_full=$deltat_full calling nlsolve..."
        verbose && @info lpad("", 80, "=")
        
        sol_ok = true
        try
            # solve nonlinear system for this pseudo-timestep
            sol = NLsolve.nlsolve(df, initial_state; solvekwargs...);
            
            if verbose
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

                ssf!(worksp, sol.zero)
                @info "  check Fnorm inf-norm $(norm(worksp, Inf)) 2-norm $(norm(worksp, 2))"
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
            initial_state .= sol.zero .* state_norm_factor
            # write output record, if required
            if isempty(tss_output) ||                       # all records requested, or ...
                (iout <= length(tss_output) &&                  # (not yet done last requested record
                    tss >= tss_output[iout])                    # and just gone past a requested record)              
                @info "    writing output record at tmodel = $(tss)"
                push!(tsoln, tss)
                push!(soln, sol.zero .* state_norm_factor)
                iout += 1
            end

        else
            @warn "iter failed, reducing deltat"
            tss -= deltat
            deltat /= deltat_fac^2
        end
        
        ptc_iter += 1
    end

    # always write the last record even if it wasn't explicitly requested
    if tsoln[end] != tss
        @info "    writing output record at tmodel = $(tss)"
        push!(tsoln, tss)
        push!(soln, sol.zero .* state_norm_factor)
    end

    PALEOmodel.ODE.calc_output_sol(outputwriter, run.model, tsoln, soln, modeldata)
    return nothing    
end

function steadystate_ptcForwardDiff(
    run, initial_state, modeldata, tspan, deltat_initial::Float64;
    jac_ad=:ForwardDiffSparse,
    kwargs...
)

    return steadystate_ptc(
        run, initial_state, modeldata, tspan, deltat_initial; 
#        (:jac_ad=>:ForwardDiffSparse, kwargs...)... 
        jac_ad=jac_ad,
        kwargs...
    )
end


steadystate_ptc(
    run, initial_state, modeldata, tss::Float64, deltat_initial::Float64, tss_max::Float64; kwargs...
) = steadystate_ptc(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
  
steadystate_ptcForwardDiff(
    run, initial_state, modeldata, tss::Float64, deltat_initial::Float64, tss_max::Float64; kwargs...
) = steadystate_ptcForwardDiff(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
  

end # module
