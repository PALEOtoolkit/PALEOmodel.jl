module SteadyState

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import Infiltrator

import NLsolve
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

    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"

    sv = modeldata.solver_view_all
    # check for implicit total variables
    iszero(PALEOmodel.num_total(sv)) || error("implicit total variables, not in constant mass matrix DAE form")
   
    iszero(PALEOmodel.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")
       
    # workspace array 
    worksp = similar(initial_state)

    state_norm_factor = ones(length(worksp))
    if use_norm
        @info "steadystate: using PALEO normalisation for state variables and time derivatives"
        state_norm_factor  .= PALEOmodel.get_statevar_norm(sv)       
    end

    # calculate normalized residual F = dS/dt
    modelode = ODE.ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)
    ssf! = FNormed(modelode, tss, worksp, state_norm_factor)

    if jac_ad==:NoJacobian
        @info "steadystate: no Jacobian"   
        df = NLsolve.OnceDifferentiable(ssf!, similar(initial_state), similar(initial_state)) 
    else       
        @info "steadystate:  using Jacobian $jac_ad"
        jacode, jac_prototype = JacobianAD.jac_config_ode(jac_ad, run.model, initial_state, modeldata, tss)        

        ssJ! = JacNormed(jacode, tss, worksp, state_norm_factor)

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
    @time sol = NLsolve.nlsolve(df, initial_state ./ state_norm_factor; solvekwargs...);
    
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
    @info "  check Fnorm inf-norm $(LinearAlgebra.norm(worksp, Inf)) 2-norm $(LinearAlgebra.norm(worksp, 2))"
    
    tsoln = [initial_time, tss]
    soln = [initial_state, sol.zero .* state_norm_factor]

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


"""
    FNormed

Function object to calculate model derivative and adapt to NLsolve interface

Calculates normalized residual F = dS/dt
"""
struct FNormed{M <: ODE.ModelODE, W, N}
    modelode::M
    tss::Float64
    worksp::W
    state_norm_factor::N
end

function (mn::FNormed)(Fnorm, unorm)
    mn.worksp .= unorm .* mn.state_norm_factor
    
    # println("FNormed: nevals=", mnl.modelode.nevals)
    mn.modelode(Fnorm, mn.worksp, nothing, mn.tss)

    Fnorm ./= mn.state_norm_factor
  
    return nothing
end

"""
    JacNormed

Function object to calculate model Jacobian and adapt to NLsolve interface
"""
struct JacNormed{J, W, N}
    jacode::J
    tss::Float64
    worksp::W
    state_norm_factor::N
end

function (jn::JacNormed)(Jnorm, unorm)
    jn.worksp .= unorm .* jn.state_norm_factor
    
    # odejac calculates un-normalized J
    jn.jacode(Jnorm, jn.worksp, nothing, jn.tss)

    # in-place multiplication to get Jnorm
    for j in 1:size(Jnorm)[2]
        for i in 1:size(Jnorm)[1]
            Jnorm[i, j] *= jn.state_norm_factor[j] / jn.state_norm_factor[i]
        end
    end

    return nothing
end

function (jn::JacNormed)(Jnorm::SparseArrays.SparseMatrixCSC, unorm)
    jn.worksp .= unorm .* jn.state_norm_factor
   
    # odejac calculates un-normalized J
    jn.jacode(Jnorm, jn.worksp, nothing, jn.tss)

    # in-place multiplication to get Jnorm
    for j in 1:size(Jnorm)[2]
        for idx in Jnorm.colptr[j]:(Jnorm.colptr[j+1]-1)
            i = Jnorm.rowval[idx]
            Jnorm.nzval[idx] *= jn.state_norm_factor[j] / jn.state_norm_factor[i]
        end
    end

    return nothing
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

    nlsolveF = nlsolveF_PTC(
        run.model, initial_state, modeldata;
        jac_ad=jac_ad,
        tss_jac_sparsity=tspan[1],
        request_adchunksize=request_adchunksize,
        jac_cellranges=jac_cellranges,
        use_norm=use_norm
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
        use_norm=false
    ) -> (ssFJ!, df::NLsolve.OnceDifferentiable)

Create function object to pass to NLsolve, with function + optional Jacobian
"""
function nlsolveF_PTC(
    model, initial_state, modeldata;
    jac_ad=:NoJacobian,
    tss_jac_sparsity=nothing,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,
    use_norm=false
)
    sv = modeldata.solver_view_all
    # We only support explicit ODE-like configurations (no DAE constraints or implicit variables)
    iszero(PALEOmodel.num_total(sv))                 || error("implicit total variables not supported")
    iszero(PALEOmodel.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")

    # previous_state is state at previous timestep
    previous_state = similar(initial_state)

    # workspace arrays 
    worksp          = similar(initial_state)
    deriv_worksp    = similar(initial_state)

    # normalisation factors
    state_norm_factor  = ones(length(worksp))
    if use_norm
        @info "steadystate: using PALEO normalisation for state variables and time derivatives"
        state_norm_factor  .= PALEOmodel.get_statevar_norm(sv)       
    end

    # current time and deltat (Refs are passed into function objects, and then updated each timestep)
    tss = Ref(NaN)
    deltat = Ref(NaN)

    modelode = ODE.ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)

    if jac_ad==:NoJacobian
        @info "steadystate: no Jacobian"
        # Define the function we want to solve 
        ssFJ! = FJacNormedPTC(modelode, nothing, tss, deltat, nothing, nothing, previous_state, worksp, deriv_worksp, state_norm_factor)
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
       
        ssFJ! = FJacNormedPTC(modelode, jacode, tss, deltat, transfer_data_ad, transfer_data, previous_state, worksp, deriv_worksp, state_norm_factor)
 
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
    previous_state = ssFJ!.previous_state
    state_norm_factor = ssFJ!.state_norm_factor
    modeldata = ssFJ!.modelode.modeldata
 
    #################################################
    # outer loop over pseudo-timesteps
    #################################################
    
    # current pseudo-timestep
    tss[] = tss_initial
    deltat[] = deltat_initial
    previous_state .= initial_state
        
    ptc_iter = 1
    sol = nothing
    @time while tss[] < tss_max
        deltat_full = deltat[]  # keep track of the deltat we could have used
        deltat[] = min(deltat[], tss_max - tss[]) # limit last timestep to get to tss_max
        if iout < length(tss_output)
            deltat[] = min(deltat[], tss_output[iout] - tss[]) # limit timestep to get to next requested output
        end

        tss[] += deltat[]

        verbose && @info lpad("", 80, "=")
        @info "steadystate: ptc_iter $ptc_iter tss $(tss[]) deltat=$(deltat[]) deltat_full=$deltat_full calling nlsolve..."
        verbose && @info lpad("", 80, "=")
        
        sol_ok = true
        try
            # solve nonlinear system for this pseudo-timestep
            sol = NLsolve.nlsolve(df, previous_state ./ state_norm_factor; solvekwargs...);
            
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
            if deltat[] == deltat_full
                # we used the full deltat and it worked - increase deltat
                deltat[] *= deltat_fac
            else
                # we weren't using the full timestep as an output was requested, so go back to full
                deltat[] = deltat_full
            end
            previous_state .= sol.zero .* state_norm_factor
            # write output record, if required
            if isempty(tss_output) ||                       # all records requested, or ...
                (iout <= length(tss_output) &&                  # (not yet done last requested record
                    tss[] >= tss_output[iout])                    # and just gone past a requested record)              
                @info "    writing output record at tmodel = $(tss[])"
                push!(tsoln, tss[])
                push!(soln, sol.zero .* state_norm_factor)
                iout += 1
            end

        else
            @warn "iter failed, reducing deltat"
            tss[] -=  deltat[]
            deltat[] /= deltat_fac^2
        end
        
        ptc_iter += 1
    end

    # always write the last record even if it wasn't explicitly requested
    if tsoln[end] != tss[]
        @info "    writing output record at tmodel = $(tss[])"
        push!(tsoln, tss[])
        push!(soln, sol.zero .* state_norm_factor)
    end

    PALEOmodel.ODE.calc_output_sol!(outputwriter, run.model, tsoln, soln, modeldata)
    return nothing    
end

"""
    FJacNormedPTC

Function object to calculate model derivative and Jacobian and adapt to NLsolve interface

Calculates sparse Jacobian = dF/dS = I - deltat * odeJac
"""
struct FJacNormedPTC{M, J, T1, T2, W, N}
    modelode::M
    jacode::J
    t::Ref{Float64}
    delta_t::Ref{Float64}
    transfer_data_ad::T1
    transfer_data::T2
    previous_state::W
    worksp::W
    deriv_worksp::W
    state_norm_factor::N
end

# F only
(jn::FJacNormedPTC)(Fnorm, unorm) = jn(Fnorm, nothing, unorm)

# Jacobian only
(jn::FJacNormedPTC)(Jnorm::SparseArrays.SparseMatrixCSC, unorm) = jn(nothing, Jnorm, unorm)

# F and J
function (jn::FJacNormedPTC)(Fnorm, Jnorm::Union{SparseArrays.SparseMatrixCSC, Nothing}, unorm)

    jn.worksp .= unorm .* jn.state_norm_factor
    
    jn.modelode(jn.deriv_worksp, jn.worksp, nothing, jn.t[])

    if !isnothing(Fnorm)
        Fnorm .=  (jn.worksp .- jn.previous_state - jn.delta_t[].*jn.deriv_worksp) ./ jn.state_norm_factor
    end

    if !isnothing(Jnorm)
        # transfer Variables not recalculated by Jacobian
        for (d_ad, d) in zip(jn.transfer_data_ad, jn.transfer_data)                
            d_ad .= d
        end

        # odejac calculates un-normalized J          
        jn.jacode(Jnorm, jn.worksp, nothing, jn.t[])
        # convert J  = I - deltat * odeJac  
        for j in 1:size(Jnorm)[2]
            # idx is index in SparseMatrixCSC compressed storage, i is row index
            for idx in Jnorm.colptr[j]:(Jnorm.colptr[j+1]-1)
                i = Jnorm.rowval[idx]

                Jnorm.nzval[idx] = -jn.delta_t[]*Jnorm.nzval[idx]
                # normalize
                Jnorm.nzval[idx] *= jn.state_norm_factor[j] ./ jn.state_norm_factor[i]

                if i == j
                    Jnorm.nzval[idx] += 1.0
                end                
            end
        end
    end

    return nothing
end

end # module
