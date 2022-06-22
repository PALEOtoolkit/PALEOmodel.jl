module SteadyStateNonlinearSolve

import PALEOboxes as PB

import PALEOmodel

import ..ODE
import ..JacobianAD

import Logging
import Infiltrator

import LinearAlgebra
import SciMLBase
import NonlinearSolve


##############################################################
# Wrappers around SciML function / problem / solve 
# with PALEO-specific additional etup
##############################################################

"""
    NonlinearFunction(model::PB.Model, modeldata, tss [; kwargs]) -> SciMLBase.ODEFunction

Contruct SciML NonlinearFunction <https://nonlinearsolve.sciml.ai/latest/> with PALEO-specific setup

Keyword arguments are required to generate a Jacobian function (using automatic differentation).

# Keywords
- `jac_ad=:NoJacobian`: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)
- `initial_state::AbstractVector`: initial state vector
- `jac_ad_t_sparsity::Float64`: model time at which to calculate Jacobian sparsity pattern
- `init_logger=Logging.NullLogger()`: default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include
"""
function NonlinearFunction(
    model::PB.Model, modeldata, tss;   
    jac_ad=:NoJacobian,
    initial_state=nothing,
    jac_ad_t_sparsity=nothing,    
    init_logger=Logging.NullLogger(),
)

    # check for implicit total variables
    iszero(PB.num_total(modeldata.solver_view_all)) ||
        error("NonlinearFunction: implicit total variables not supported")

    iszero(PB.num_algebraic_constraints(modeldata.solver_view_all)) ||
        error("NonlinearFunction: TODO algebraic constraints not supported")    

    modelode = ODE.ModelODE(modeldata, modeldata.solver_view_all, modeldata.dispatchlists_all, 0)

    @info "NonlinearFunction: using Jacobian $jac_ad"
       
    jacode, jac_prototype = JacobianAD.jac_config_ode(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity, 
        init_logger=init_logger,
    )    
    
    f = SciMLBase.NonlinearFunction{true}(
        ModelNonlinear(modelode, tss), 
        jac=JacNonlinear(jacode, tss), 
        jac_prototype=jac_prototype
    )
       
    return f
end


"""
    steadystate(run, initial_state, modeldata, tss [; kwargs...] ) -> sol::SciMLBase.ODESolution

Solve for steady-state of run.model at time `tss`, and write to `outputwriter`

Provides a wrapper around the Julia SciML [NonlinearSolve](https://github.com/SciML/NonlinearSolve.jl) 
package, with PALEO-specific additional setup. Keyword arguments `alg` and `solvekwargs` are passed through to the
`SciMLBase.solve` method.

# Implementation
Follows the SciML standard pattern:
- Create [`NonlinearFunction`](@ref)
- Create `SciMLBase.NonlinearProblem`
- Call `SciMLBase.solve`
and then 
- Call [`print_sol_stats`](@ref)
- Call [`calc_output_sol!`](@ref) to recalculate model fields at timesteps used

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct
- `tss`:  steady-state time

# Keywords
- `alg=NonlinearSolve.NewtonRaphson(autodiff=false)`:  algorithm to use, passed through to `SciMLBase.solve`
- `solvekwargs=NamedTuple()`: NamedTuple of keyword arguments passed through to DifferentialEquations.jl `solve`
   (eg to set `abstol`, `reltol`, `saveat`,  see <https://diffeq.sciml.ai/dev/basics/common_solver_opts/>)
- `outputwriter=run.output`: PALEOmodel.AbstractOutputWriter instance to hold output
- `jac_ad=:NoJacobian`: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)
- `jac_ad_t_sparsity=tss`: model time at which to calculate Jacobian sparsity pattern
- `BLAS_num_threads=1`: number of LinearAlgebra.BLAS threads to use
- `init_logger=Logging.NullLogger()`: default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include
"""
function steadystate(
    run, initial_state, modeldata, tss; 
    alg=NonlinearSolve.NewtonRaphson(autodiff=false),
    outputwriter=run.output,
    solvekwargs::NamedTuple=NamedTuple{}(),
    jac_ad=:NoJacobian,
    jac_ad_t_sparsity=tss,
    BLAS_num_threads=1,
    init_logger=Logging.NullLogger(),
)
  
    f = NonlinearFunction(
        run.model, modeldata, tss;   
        jac_ad=jac_ad,
        initial_state=initial_state,
        jac_ad_t_sparsity=jac_ad_t_sparsity,    
        init_logger=init_logger,
    )
 
    io = IOBuffer()
    println(io, lpad("", 80, "="))
    println(io, "steadystate:  NonlinearProblem using algorithm: $alg Jacobian $jac_ad")

    prob = SciMLBase.NonlinearProblem(f, initial_state, nothing)
   
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    println(io, "    using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads")
    println(io, lpad("", 80, "="))
    @info String(take!(io))
   
    @time sol = SciMLBase.solve(prob, alg; solvekwargs...);

    ODE.print_sol_stats(sol)

    if !isnothing(outputwriter)
        ODE.calc_output_sol!(outputwriter, run.model, sol, (tss, tss), initial_state, modeldata)
    end

    return sol
end


"""
    ModelNonlinear

Function object to calculate model derivative and adapt to SciML nonlinear solver interface
"""
mutable struct ModelNonlinear{M <: ODE.ModelODE}
    modelode::M
    tss::Float64
end

function (mnl::ModelNonlinear)(du, u, p) 
    # println("ModelNonlinear: nevals=", mnl.modelode.nevals)
    return mnl.modelode(du, u, p, mnl.tss) 
end
  

"""
    JacNonlinear

Function object to calculate model jacobian and adapt to SciML nonlinear solver interface
"""
mutable struct JacNonlinear{J}
    jacode::J
    tss::Float64
end

function (jnl::JacNonlinear)(J, u, p) 
    println("JacNonlinear: nevals=", jnl.jacode.nevals)
    return jnl.jacode(du, u, p, jnl.tss) 
end

end # module
