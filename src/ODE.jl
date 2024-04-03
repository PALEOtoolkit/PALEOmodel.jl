module ODE

import PALEOboxes as PB

import PALEOmodel
import ..SolverFunctions

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


Base.@deprecate_binding ModelODE SolverFunctions.ModelODE
# Base.@deprecate_binding ModelDAE SolverFunctions.ModelDAE


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
"""
function ODEfunction(
    model::PB.Model, modeldata;   
    jac_ad=:NoJacobian,
    initial_state=nothing,
    jac_ad_t_sparsity=nothing,    
    generated_dispatch=true,
)
    @info "ODEfunction: using Jacobian $jac_ad"

    PB.check_modeldata(model, modeldata)

    # check for implicit total variables
    PALEOmodel.num_total(modeldata.solver_view_all) == 0 ||
        error("ODEfunction: implicit total variables, not in constant mass matrix DAE form - use DAE solver")

    # if a DAE, construct mass matrix
    num_constraints = PALEOmodel.num_algebraic_constraints(modeldata.solver_view_all)
    if iszero(num_constraints)
        M = LinearAlgebra.I        
    else
        M = SparseArrays.sparse(get_massmatrix(modeldata)) # fails with M=LinearAlgebra.Diagonal
        @info "ODEfunction:  using mass matrix for DAE with $num_constraints algebraic constraints"       
    end

    m = SolverFunctions.ModelODE(modeldata; solver_view=modeldata.solver_view_all, dispatchlists=modeldata.dispatchlists_all)
       
    jac, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        generated_dispatch,
    )    
    
    f = SciMLBase.ODEFunction{true}(m; jac, jac_prototype, mass_matrix=M)
       
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
"""
function DAEfunction(
    model::PB.Model, modeldata;   
    jac_ad=:NoJacobian,
    initial_state=nothing,
    jac_ad_t_sparsity=nothing,    
    generated_dispatch=true,
)
    @info "DAEfunction:  using Jacobian $jac_ad"

    PB.check_modeldata(model, modeldata)
    
    jac, jac_prototype, odeimplicit = PALEOmodel.JacobianAD.jac_config_dae(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        generated_dispatch,
    )
    m =  SolverFunctions.ModelDAE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, odeimplicit, 0)
    
    f = SciMLBase.DAEFunction{true}(m; jac, jac_prototype)  

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
- `generated_dispatch=true`: `true` to autogenerate code (fast solve, slow compile)
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
    generated_dispatch=true,
)
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)

    @info """
    
    ================================================================================
    PALEOmodel.ODE.integrate:
        tspan: $tspan
        algorithm: $alg
        Jacobian: $jac_ad
        steadystate: $steadystate
        using $(LinearAlgebra.BLAS.get_num_threads()) BLAS threads
    ================================================================================
    """

    f = ODEfunction(
        run.model, modeldata;   
        jac_ad,
        initial_state,
        jac_ad_t_sparsity,    
        generated_dispatch,
    )
 
    if steadystate
        prob = SciMLBase.SteadyStateProblem(f, initial_state, nothing)
    else
        prob = SciMLBase.ODEProblem(f, initial_state, tspan, nothing)
    end
   
    @time sol = SciMLBase.solve(prob, alg; solvekwargs...);

    print_sol_stats(sol)

    if !isnothing(outputwriter)
        calc_output_sol!(outputwriter, run.model, sol, tspan, initial_state, modeldata)
    end

    @info """

    ================================================================================
    PALEOmodel.ODE.integrate: done
    ================================================================================
    """

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
        alg,
        jac_ad,
        jac_ad_t_sparsity,
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

`integrateDAEForwardDiff` sets keyword arguments `jac_ad=:ForwardDiffSparse`, `alg=Sundials.IDA(linear_solver=:KLU)`
to use the Julia [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package to provide the Jacobian with
forward-mode automatic differentiation and automatic sparsity detection.

# Limitations
- arbitrary combinations of implicit `total` (`T`) and algebraic `constraint` (`C`) variables are not supported by Sundials IDA 
as used here, as the IDA solver option used to find consistent initial conditions requires a partioning 
into differential and algebraic variables (see `SolverView` documentation).

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
    generated_dispatch=true,
)
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)

    num_total = PALEOmodel.num_total(modeldata.solver_view_all)
    num_constraint = PALEOmodel.num_algebraic_constraints(modeldata.solver_view_all)

    @info """

    ================================================================================
    PALEOmodel.ODE.integrateDAE:
        tspan: $tspan
        algorithm: $alg
        Jacobian: $jac_ad
        using $(LinearAlgebra.BLAS.get_num_threads()) BLAS threads
    ================================================================================
    """
    
    if !iszero(num_total) && !iszero(num_constraint)
        @warn "arbitrary combinations of total $num_total and constraint $num_constraint variables are not supported: see 'SolverView' documentation,"*
            " eg IDA initialisation requires that 'total' variables are not functions of 'constraint' variables"*
            " so that a partitioning into differential and algebraic variables is possible (this is not checked here)"
    end

    func = DAEfunction(
        run.model, modeldata;   
        jac_ad,
        initial_state,
        jac_ad_t_sparsity,    
        generated_dispatch,
    )
   
    differential_vars = PALEOmodel.state_vars_isdifferential(modeldata.solver_view_all)

    @info "calling get_inconsistent_initial_deriv"
    initial_deriv = get_inconsistent_initial_deriv(
        initial_state, modeldata, tspan[1], differential_vars, func.f
    )

    prob = SciMLBase.DAEProblem(
        func, initial_deriv, initial_state, tspan, nothing;
        differential_vars,
    )
   
    @time sol = SciMLBase.solve(prob, alg; solvekwargs...);

    print_sol_stats(sol)
    
    if !isnothing(outputwriter)
        calc_output_sol!(outputwriter, run.model, sol, tspan, initial_state, modeldata)
    end

    @info """

    ================================================================================
    PALEOmodel.ODE.integrateDAE: done
    ================================================================================
    """

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
        alg,
        jac_ad,
        jac_ad_t_sparsity,
        kwargs...
    )
end



###############################################################################
# Helper functions for DifferentialEquations ODE / DAE integrators
###############################################################################

"""
    get_massmatrix(modeldata) -> LinearAlgebra.Diagonal

Return mass matrix (diagonal matrix with 1.0 for ODE variable, 0.0 for algebraic constraint)
"""
function get_massmatrix(modeldata)
    iszero(PALEOmodel.num_total(modeldata.solver_view_all)) ||
        error("get_massmatrix - implicit total variables, not in mass matrix DAE form")
    return LinearAlgebra.Diagonal(
        [
            d ? 1.0 : 0.0 
            for d in PALEOmodel.state_vars_isdifferential(modeldata.solver_view_all)
        ]
    )
end


"""
    get_inconsistent_initial_deriv(
        initial_state, modeldata, initial_t, differential_vars, modeldae::SolverFunctions.ModelDAE
    ) -> initial_deriv

NB: IDA initialisation seems not fully understood: with Julia Sundials.IDA(init_all=false)
(the Sundials.jl default) corresponding to the IDA option `IDA_YA_YDP_INIT` to `IDACalcIC()`,
this should "direct IDACalcIC() to compute the algebraic components of y and differential components of ydot, 
given the differential components of y."
But it seems to have some sensitivity to `initial_deriv` (ydot), which shouldn't be used according to the above ?

This function takes a guess at what is needed for `initial_deriv`:
- initial_deriv for ODE variables will now be consistent
- set initial_deriv (constraint for algebraic variables) = 0, this will not be satisfied (ie will not be consistent)
- initial_deriv for implicit variables will not be consistent
"""
function get_inconsistent_initial_deriv(
    initial_state, modeldata, initial_t, differential_vars, modeldae::SolverFunctions.ModelDAE
)
 
    initial_deriv = similar(initial_state)
    
    # Evaluate initial derivative
    # ODE variable derivative will now be consistent
    # implicit (Total) derivative will not be consistent
    m = SolverFunctions.ModelODE(modeldata; solver_view=modeldata.solver_view_all, dispatchlists=modeldata.dispatchlists_all)
    m(initial_deriv, initial_state , nothing, initial_t)

    # Set initial_deriv (ie constraint) for algebraic variables - shouldn't matter according to IDA doc ?
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

calc_output_sol!(outputwriter, model::PB.Model, sol::Union{SciMLBase.ODESolution, SciMLBase.DAESolution}, tspan, initial_state, modeldata) =
    calc_output_sol!(outputwriter, model, nothing, sol, tspan, initial_state, modeldata)

function calc_output_sol!(
    outputwriter, 
    model::PB.Model,
    pa::Union{Nothing, PB.ParameterAggregator},
    sol::Union{SciMLBase.ODESolution, SciMLBase.DAESolution}, 
    tspan,
    initial_state,
    modeldata
)
    
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
        PALEOmodel.set_tforce!(modeldata.solver_view_all, tmodel)   
        PALEOmodel.set_statevar!(modeldata.solver_view_all, sol[:, tidx])
            
        if isnothing(pa)
            PB.do_deriv(modeldata.dispatchlists_all)
        else
            PB.do_deriv(modeldata.dispatchlists_all, pa)
        end

        PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
    end

    @info "ODE.calc_output_sol!: done"
    return nothing
end

function calc_output_sol!(outputwriter, model::PB.Model, sol::SciMLBase.NonlinearSolution, tspan, initial_state, modeldata)

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, 1)

    # call model to (re)calculate
    tmodel = tspan[1]
    PALEOmodel.set_tforce!(modeldata.solver_view_all, tmodel)
    PALEOmodel.set_statevar!(modeldata.solver_view_all, sol.u)

    PB.do_deriv(modeldata.dispatchlists_all)

    PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
   
    return nothing
end

function calc_output_sol!(outputwriter, model::PB.Model, tsoln, soln,  modeldata)

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, length(tsoln))

    # call model to (re)calculate 
    for i in eachindex(tsoln)
        tmodel = tsoln[i]     
        PALEOmodel.set_tforce!(modeldata.solver_view_all, tmodel)   
        PALEOmodel.set_statevar!(modeldata.solver_view_all, soln[i])
        PB.do_deriv(modeldata.dispatchlists_all)
        PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
    end
   
    return nothing
end

function calc_output_sol!(outputwriter, model::PB.Model, tsoln, soln, modelode, modeldata)

    PALEOmodel.OutputWriters.initialize!(outputwriter, model, modeldata, length(tsoln))

    # call model to (re)calculate
    du = similar(first(soln)) # not used
    for i in eachindex(tsoln)
        tmodel = tsoln[i]     
        modelode(du, soln[i], nothing, tmodel)
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

    io_stats = IOBuffer()
    try
        show(io_stats, "text/plain", sol.stats)
    catch
        @warn "Could not get sol.stats"
    end

    @info """
    
    ================================================================================
    print_sol_stats:
        retcode=$(sol.retcode)
        alg=$(sol.alg)
        stats=$(String(take!(io_stats)))
        length(sol.t) $(length(sol.t))
        size(sol) $(size(sol))
    ================================================================================
    """

   

    return nothing
end

function print_sol_stats(sol::SciMLBase.NonlinearSolution)

    @info """
    
    ================================================================================
    print_sol_stats:
        retcode=$(sol.retcode)
    
        alg=$(sol.alg)
        size(sol) $(size(sol))
    ================================================================================
    """

    return nothing
end


end
