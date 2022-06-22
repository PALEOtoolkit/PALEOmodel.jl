module ODE

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import Logging
import Infiltrator

# See http://www.stochasticlifestyle.com/differentialequations-jl-3-0-roadmap-4-0/
# for summary of state of DifferentialEquations.jl
# This is mostly useful as a wrapper around Sundials CVODE and IDA
# See https://github.com/SciML/Sundials.jl/blob/master/src/common_interface/solve.jl
# for the code that maps wrapper calls to Sundials calls.
# Import SciMLBase as a lightweight dependency.
# See https://github.com/ModiaSim/ModiaMath.jl/blob/master/src/SimulationEngine/simulate.jl
# for example of calling IDA directly (which would avoid DifferentialEquations.jl altogether)

import Sundials
import SciMLBase

##############################################################
# Wrappers around SciML function / problem / solve 
# with PALEO-specific additional etup
##############################################################

"""
    ODEfunction(model::PB.Model, modeldata [; kwargs]) -> SciMLBase.ODEFunction

Contruct SciML ODEfunction <https://diffeq.sciml.ai/latest/> with PALEO-specific setup

Keyword arguments are required to generate a Jacobian function (using automatic differentation).

# Keywords
- `jac_ad=:NoJacobian`: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)
- `initial_state::AbstractVector`: initial state vector
- `jac_ad_t_sparsity::Float64`: model time at which to calculate Jacobian sparsity pattern
- `init_logger=Logging.NullLogger()`: default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include
"""
function ODEfunction(
    model::PB.Model, modeldata;   
    jac_ad=:NoJacobian,
    initial_state=nothing,
    jac_ad_t_sparsity=nothing,    
    init_logger=Logging.NullLogger(),
)

    # check for implicit total variables
    PB.num_total(modeldata.solver_view_all) == 0 ||
        error("ODEfunction: implicit total variables, not in constant mass matrix DAE form - use DAE solver")

    # if a DAE, construct mass matrix
    num_constraints = PB.num_algebraic_constraints(modeldata.solver_view_all)
    if iszero(num_constraints)
        M = LinearAlgebra.I        
    else
        M = SparseArrays.sparse(get_massmatrix(modeldata)) # fails with M=LinearAlgebra.Diagonal
        @info "ODEfunction:  using mass matrix for DAE with $num_constraints algebraic constraints"       
    end

    m = ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)

    @info "ODEfunction: using Jacobian $jac_ad"
       
    jac, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity, 
        init_logger=init_logger,
    )    
    
    f = SciMLBase.ODEFunction{true}(m, jac=jac, jac_prototype=jac_prototype, mass_matrix=M)
       
    return f
end


"""
    DAEfunction(model::PB.Model, modeldata [; kwargs]) -> SciMLBase.DAEFunction

Contruct SciML DAEfunction <https://diffeq.sciml.ai/latest/> with PALEO-specific setup

Keyword arguments are required to generate a Jacobian function (using automatic differentation).

# Keywords
- `jac_ad=:NoJacobian`: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)
- `initial_state::AbstractVector`: initial state vector
- `jac_ad_t_sparsity::Float64`: model time at which to calculate Jacobian sparsity pattern
- `init_logger=Logging.NullLogger()`: default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include
"""
function DAEfunction(
    model::PB.Model, modeldata;   
    jac_ad=:NoJacobian,
    initial_state=nothing,
    jac_ad_t_sparsity=nothing,    
    init_logger=Logging.NullLogger(),
)

    @info "DAEfunction:  using Jacobian $jac_ad"
    
    jac, jac_prototype, odeimplicit = PALEOmodel.JacobianAD.jac_config_dae(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity,
        init_logger=init_logger,
    )
    m =  ModelDAE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, odeimplicit, 0)
    
    f = SciMLBase.DAEFunction{true}(m, jac=jac, jac_prototype=jac_prototype)  

    return f
end


"""
    integrate(run, initial_state, modeldata, tspan [; kwargs...] ) -> sol::SciMLBase.ODESolution
    integrateForwardDiff(run, initial_state, modeldata, tspan [;kwargs...]) -> sol::SciMLBase.ODESolution

Integrate run.model as an ODE or as a DAE with constant mass matrix, and write to `outputwriter`

Provides a wrapper around the Julia SciML [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl) 
package ODE solvers, with PALEO-specific additional setup. Keyword arguments `alg` and `solvekwargs` are passed through to the
`DifferentialEquations` `solve` method.

`integrateForwardDiff` sets keyword arguments `jac_ad=:ForwardDiffSparse`, `alg=Sundials.CVODE_BDF(linear_solver=:KLU)`
to use the Julia [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package to provide the Jacobian with
forward-mode automatic differentiation and automatic sparsity detection.

# Implementation
Follows the SciML standard pattern:
- Create [`ODEfunction`](@ref)
- Create `SciMLBase.ODEproblem`
- Call `SciMLBase.solve`
and then 
- Call [`print_sol_stats`](@ref)
- Call [`calc_output_sol!`](@ref) to recalculate model fields at timesteps used

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct
- `tspan`:  (tstart, tstop) integration start and stop times

# Keywords
- `alg=Sundials.CVODE_BDF()`:  ODE algorithm to use, passed through to DifferentialEquations.jl `solve` method.
  The default is appropriate for a stiff system of equations (common in biogeochemical models),
  see <https://diffeq.sciml.ai/dev/solvers/ode_solve/> for other options.
- `solvekwargs=NamedTuple()`: NamedTuple of keyword arguments passed through to DifferentialEquations.jl `solve`
   (eg to set `abstol`, `reltol`, `saveat`,  see <https://diffeq.sciml.ai/dev/basics/common_solver_opts/>)
- `outputwriter=run.output`: PALEOmodel.AbstractOutputWriter instance to hold output
- `jac_ad=:NoJacobian`: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)
- `jac_ad_t_sparsity=tspan[1]`: model time at which to calculate Jacobian sparsity pattern
- `steadystate=false`: true to use `DifferentialEquations.jl` `SteadyStateProblem`
  (not recommended, see [`PALEOmodel.SteadyState.steadystate`](@ref)).
- `BLAS_num_threads=1`: number of LinearAlgebra.BLAS threads to use
- `init_logger=Logging.NullLogger()`: default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include
"""
function integrate(
    run, initial_state, modeldata, tspan; 
    alg=Sundials.CVODE_BDF(),
    outputwriter=run.output,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    jac_ad_t_sparsity=tspan[1],
    steadystate=false,
    BLAS_num_threads=1,
    init_logger=Logging.NullLogger(),
)
  
    f = ODEfunction(
        run.model, modeldata;   
        jac_ad=jac_ad,
        initial_state=initial_state,
        jac_ad_t_sparsity=jac_ad_t_sparsity,    
        init_logger=init_logger,
    )
 
    io = IOBuffer()
    println(io, lpad("", 80, "="))
    if steadystate
        println(io, "integrate:  SteadyStateProblem using algorithm: $alg Jacobian $jac_ad")
        prob = SciMLBase.SteadyStateProblem(f, initial_state, nothing)
    else
        println(io, "integrate:  ODEProblem using algorithm: $alg Jacobian $jac_ad")
        prob = SciMLBase.ODEProblem(f, initial_state, tspan, nothing)
    end

    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    println(io, "    using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads")
    println(io, lpad("", 80, "="))
    @info String(take!(io))
   
    @time sol = SciMLBase.solve(prob, alg; solvekwargs...);

    print_sol_stats(sol)

    if !isnothing(outputwriter)
        calc_output_sol!(outputwriter, run.model, sol, tspan, initial_state, modeldata)
    end

    return sol
end

"[`integrate`](@ref) with argument defaults to  use ForwardDiff AD Jacobian"
function integrateForwardDiff(
    run, initial_state, modeldata, tspan; 
    alg=Sundials.CVODE_BDF(linear_solver=:KLU),
    jac_ad=:ForwardDiffSparse,
    jac_ad_t_sparsity=tspan[1],
    kwargs...
)

    return integrate(
        run, initial_state, modeldata, tspan;
        alg=alg,
        jac_ad=jac_ad,
        jac_ad_t_sparsity=jac_ad_t_sparsity,
        kwargs...
    )
end



"""
    integrateDAE(run, initial_state, modeldata, tspan [; kwargs...]) -> sol::SciMLBase.DAESolution
    integrateDAEForwardDiff(run, initial_state, modeldata, tspan [; kwargs...]) -> sol::SciMLBase.DAESolution

Integrate `run.model` as a DAE and copy output to `outputwriter`.

Provides a wrapper around the Julia SciML [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl) 
package DAE solvers, with PALEO-specific additional setup. Keyword arguments `alg` and `solvekwargs` are passed through to the
`DifferentialEquations` `solve` method.

`integrateDAEForwardDiff` sets keyword arguments `jac_ad=:ForwardDiffSparse`, `alg=Sundials.CVODE_BDF(linear_solver=:KLU)`
to use the Julia [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package to provide the Jacobian with
forward-mode automatic differentiation and automatic sparsity detection.

# Implementation
Follows the SciML standard pattern:
- Create [`DAEfunction`](@ref)
- Call [`get_inconsistent_initial_deriv`](@ref) -> `initial_deriv`
- Create `SciMLBase.DAEproblem`
- Call `SciMLBase.solve`
and then 
- Call [`print_sol_stats`](@ref)
- Call [`calc_output_sol!`](@ref) to recalculate model fields at timesteps used

# Keywords 
As [`integrate`](@ref), with defaults:
- `alg=Sundials.IDA()` (`integrateDAE`)
- `alg=Sundials.IDA(linear_solver=:KLU)` (`integrateDAEForwardDiff`)
"""
function integrateDAE(
    run, initial_state, modeldata, tspan; 
    alg=Sundials.IDA(),
    outputwriter=run.output,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    jac_ad_t_sparsity=tspan[1],
    BLAS_num_threads=1,
    init_logger=Logging.NullLogger(),
)
    
    func = DAEfunction(
        run.model, modeldata;   
        jac_ad=jac_ad,
        initial_state=initial_state,
        jac_ad_t_sparsity=jac_ad_t_sparsity,    
        init_logger=init_logger,
    )
   
    differential_vars = PB.state_vars_isdifferential(modeldata.solver_view_all)

    # create inconsistent initial conditions for DAE variables, rely on DAE solver to find them
    initial_deriv = get_inconsistent_initial_deriv(
        initial_state, modeldata, tspan[1], differential_vars, func.f
    )

    prob = SciMLBase.DAEProblem(
        func, initial_deriv, initial_state, tspan, nothing,
        differential_vars=differential_vars,
    )

    io = IOBuffer()
    println(io, lpad("", 80, "="))
    println(io, "integrateDAE:  DAEProblem using algorithm: $alg Jacobian $jac_ad")
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    println(io, "    using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads")
    println(io, lpad("", 80, "="))
    @info String(take!(io))
   
    @time sol = SciMLBase.solve(prob, alg; solvekwargs...);

    print_sol_stats(sol)
    
    if !isnothing(outputwriter)
        calc_output_sol!(outputwriter, run.model, sol, tspan, initial_state, modeldata)
    end

    return sol
end

"[`integrateDAE`](@ref) with argument defaults to use ForwardDiff AD Jacobian"
function integrateDAEForwardDiff(
    run, initial_state, modeldata, tspan, 
    alg=Sundials.IDA(linear_solver=:KLU),
    jac_ad=:ForwardDiffSparse,
    jac_ad_t_sparsity=tspan[1];
    kwargs...
)

    return integrateDAE(
        run, initial_state, modeldata, tspan;
        alg=alg,
        jac_ad=jac_ad,
        jac_ad_t_sparsity=jac_ad_t_sparsity,
        kwargs...
    )
end



###############################################################################
# Helper functions / adaptors for DifferentialEquations ODE / DAE integrators
###############################################################################

"""
    ModelODE

Function object to calculate model derivative and adapt to SciML ODE solver interface
"""
mutable struct ModelODE{T, S <: PB.SolverView, D}
    modeldata::PB.ModelData{T}
    solver_view::S
    dispatchlists::D
    nevals::Int
end

function ModelODE(
    modeldata; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
)
    return ModelODE(modeldata, solver_view, dispatchlists, 0)
end

function (m::ModelODE)(du,u, p, t)
   
    PB.set_statevar!(m.solver_view, u)
    PB.set_tforce!(m.solver_view, t)

    PB.do_deriv(m.dispatchlists)

    PB.get_statevar_sms!(du, m.solver_view)
   
    m.nevals += 1  

    return nothing
end


"""
    get_massmatrix(modeldata) -> LinearAlgebra.Diagonal

Return mass matrix (diagonal matrix with 1.0 for ODE variable, 0.0 for algebraic constraint)
"""
function get_massmatrix(modeldata)
    iszero(PB.num_total(modeldata.solver_view_all)) ||
        error("get_massmatrix - implicit total variables, not in mass matrix DAE form")
    return LinearAlgebra.Diagonal(
        [
            d ? 1.0 : 0.0 
            for d in PB.state_vars_isdifferential(modeldata.solver_view_all)
        ]
    )
end

"""
    ModelDAE

Function object to calculate model residual and adapt to SciML DAE solver interface
"""
mutable struct ModelDAE{T, S <: PB.SolverView, D, O}
    modeldata::PB.ModelData{T}
    solver_view::S
    dispatchlists::D
    odeimplicit::O
    nevals::Int
end

"adapt PALEO model derivative + constraints to SciML DAE problem requirements
 resid = G(dsdt,s,p,t) = -duds*dsdt + F(u(s))"
function (m::ModelDAE)(resid, dsdt, s, p, t)
    
    PB.set_statevar!(m.solver_view, s)
    PB.set_tforce!(m.solver_view, t)

    # du(s)/dt
    PB.do_deriv(m.dispatchlists)

    # get explicit deriv
    l_ts = copyto!(resid, m.solver_view.stateexplicit_deriv)
    # -dudt = -dsdt explicit variables with u(s) = s so duds = I    

    @inbounds for i in 1:l_ts
        resid[i] -= dsdt[i]
    end

    # get implicit_deriv     
    l_ti = length(m.solver_view.total)
    if l_ti > 0
        !isnothing(m.odeimplicit) ||
            error("implicit Total Variables, odeimplicit required")

        copyto!(resid, m.solver_view.total_deriv, dof=l_ts+1)

        # -dudt = -duds*dsdt implicit variables with u(s)

        # calculate duds using AD
        m.odeimplicit(m.odeimplicit.duds, s, p, t)       
        # add duds*dsdt to resid
        resid[(l_ts+1):(l_ts+l_ti)] -= m.odeimplicit.duds*dsdt
    end

    # add constraints to residual
    copyto!(resid, m.solver_view.constraints, dof=l_ts+l_ti+1)

    m.nevals += 1  

    return nothing
end


"""
    get_inconsistent_initial_deriv(
        initial_state, modeldata, initial_t, differential_vars, modeldae::ModelDAE
    ) -> initial_deriv

Create (inconsistent) `initial_deriv` for a DAE problem: ODE variables are consistent, DAE variables set to zero 
ie rely on DAE solver to find them
"""
function get_inconsistent_initial_deriv(
    initial_state, modeldata, initial_t, differential_vars, modeldae::ModelDAE
)
 
    initial_deriv = similar(initial_state)
    
    # Evaluate initial derivative
    # ODE variables will now be consistent, constraint for algebraic variables will not be satisfied 
    # implicit variables will be fixed up below
    m = ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)
    m(initial_deriv, initial_state , nothing, initial_t)

    # Find consistent initial conditions for implicit variables (if any)
    if PB.num_total(modeldata.solver_view_all) > 0
        # TODO this finds ds/dt, but doesn't yet solve for State s given Total 
        # (currently will set Total initial conditions from s, which is usually not what is wanted)
        @warn "Calculating Total variables initial conditions from State variables (calculation of State from Total not implemented)"

        l_ts = length(modeldata.solver_view_all.stateexplicit)
        l_ti = length(modeldata.solver_view_all.total)

        # Find consistent initial conditions for ds/dt
        # get dU/dS
        odeimplicit = modeldae.odeimplicit
        odeimplicit(odeimplicit.duds, initial_state, nothing, initial_t)
        # take slice to include only implicit variables (will be a square matrix)
        duds_imponly = odeimplicit.duds[:, (l_ts+1):(l_ts+l_ti)]
        # construct duds without these entries
        duds_noimp = copy(odeimplicit.duds)       
        duds_noimp[:, (l_ts+1):(l_ts+l_ti)] -= duds_imponly
        # solve for ds that gives zero residuals
        # resid = 0 = -duds*ds + initial_deriv(s)
        #           = -duds_imponly*ds -duds_noimp*initial_deriv(s) + initial_deriv(s)
        # duds_imponly * ds = (initial_deriv(s) -duds_noimp*ds)[iistrt::iiend]
        rhs = initial_deriv[(l_ts+1):(l_ts+l_ti)] - duds_noimp*initial_deriv
        ds = duds_imponly \ rhs

        # check 
        resid = similar(initial_deriv)
        modeldae(resid, initial_deriv, initial_state, nothing, initial_t)
        total_resid_norm_initial = LinearAlgebra.norm(resid[(l_ts+1):(l_ts+l_ti)])

        initial_deriv[(l_ts+1):(l_ts+l_ti)] .= ds

        # check 
        modeldae(resid, initial_deriv, initial_state, nothing, initial_t)
        total_resid_norm_solve = LinearAlgebra.norm(resid[(l_ts+1):(l_ts+l_ti)])
        @info "  Total Variables residual norm $total_resid_norm_initial -> $total_resid_norm_solve"
    end

    # Set initial_deriv to zero for algebraic variables (so these will be inconsistent)
    for i=1:length(initial_deriv)       
        if !differential_vars[i]
            initial_deriv[i] = 0.0
        end       
    end

    return initial_deriv
end  

"""
    calc_output_sol!(outputwriter, model::PB.Model, sol::SciMLBase.ODESolution, tspan, initial_state, modeldata)
    calc_output_sol!(outputwriter, model::PB.Model, sol::SciMLBase.DAESolution, tspan, initial_state, modeldata)
    calc_output_sol!(outputwriter, model::PB.Model, sol::SciMLBase.NonlinearSolution, tspan, initial_state, modeldata)
    calc_output_sol!(outputwriter, model::PB.Model, tsoln::AbstractVector, soln::AbstractVector,  modeldata)

Iterate through solution and recalculate model fields
(functions of state variables and time) and store in `outputwriter`.

# Arguments
- `outputwriter::PALEOmodel.AbstractOutputWriter`: container for output
- `model::PB.Model` used to calculate solution
- `sol`: SciML solution object
- `tspan`:  (tstart, tstop) integration start and stop times
- `initial_state::AbstractVector`: initial state vector
- `tsoln::AbstractVector`:  solution times
- `soln::AbstractVector`: solution state variables
- `modeldata::PB.Modeldata`: ModelData struct
"""
function calc_output_sol! end

function calc_output_sol!(outputwriter, model::PB.Model, sol::Union{SciMLBase.ODESolution, SciMLBase.DAESolution}, tspan, initial_state, modeldata)
    
    @info "ODE.calc_output_sol!: $(length(sol)) records"
    if length(sol.t) != length(sol)
        toffbodge = length(sol.t) - length(sol)
        @warn "ODE.calc_output_sol!: Bodging a DifferentialEquations.jl issue - "*
            "length(sol.t) $(length(sol.t)) != length(sol) $(length(sol)), omitting first $toffbodge point(s) from sol.t"
    else
        toffbodge = 0
    end

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, length(sol))

    # iterate through solution  
    for tidx in 1:length(sol)
        # call model to (re)calculate
        tmodel = sol.t[tidx+toffbodge]
        PB.set_tforce!(modeldata.solver_view_all, tmodel)   
        PB.set_statevar!(modeldata.solver_view_all, sol[:, tidx])
            
        PB.do_deriv(modeldata.dispatchlists_all)

        PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
    end

    @info "ODE.calc_output_sol!: done"
    return nothing
end

function calc_output_sol!(outputwriter, model::PB.Model, sol::SciMLBase.NonlinearSolution, tspan, initial_state, modeldata)

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, 1)

    # call model to (re)calculate
    tmodel = tspan[1]
    PB.set_tforce!(modeldata.solver_view_all, tmodel)
    PB.set_statevar!(modeldata.solver_view_all, sol.u)

    PB.do_deriv(modeldata.dispatchlists_all)

    PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
   
    return nothing
end

function calc_output_sol!(outputwriter, model::PB.Model, tsoln, soln,  modeldata)

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, length(tsoln))

    # call model to (re)calculate 
    for i in eachindex(tsoln)
        tmodel = tsoln[i]     
        PB.set_tforce!(modeldata.solver_view_all, tmodel)   
        PB.set_statevar!(modeldata.solver_view_all, soln[i])
        PB.do_deriv(modeldata.dispatchlists_all)
        PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
    end
   
    return nothing
end

"""
    print_sol_stats(sol::SciMLBase.ODESolution)
    print_sol_stats(sol::SciMLBase.DAESolution)
    print_sol_stats(sol::SciMLBase.NonlinearSolution)

Print solution statistics
"""
function print_sol_stats end

function print_sol_stats(sol::Union{SciMLBase.ODESolution, SciMLBase.DAESolution})

    io = IOBuffer()
    println(io, lpad("", 80, "="))
    println(io, "print_sol_stats:")
    println(io, "  retcode=$(sol.retcode)")
    println(io, "  nsteps $(length(sol.t))")

    # println(io, "  prob=$(sol.prob)")
    println(io, "  alg=$(sol.alg)")
    println(io, "  $(sol.destats)")
    println(io, "  length(sol.t) $(length(sol.t))")
    println(io, "  size(sol) $(size(sol))")
    println(io, lpad("", 80, "="))

    @info String(take!(io))

    return nothing
end

function print_sol_stats(sol::SciMLBase.NonlinearSolution)

    io = IOBuffer()
    println(io, lpad("", 80, "="))
    println(io, "print_sol_stats:")
    println(io, "  retcode=$(sol.retcode)")
    
    # println(io, "  prob=$(sol.prob)")
    println(io, "  alg=$(sol.alg)")
   
    println(io, "  size(sol) $(size(sol))")
    println(io, lpad("", 80, "="))

    @info String(take!(io))

    return nothing
end


end
