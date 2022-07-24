"""
    SolverFunctions

Function-like (callable) structs that adapt `PALEOboxes.Model` to ODE etc solvers.
"""
module SolverFunctions

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import ForwardDiff
import SparseDiffTools

# import Infiltrator # Julia debugger

"""
    ModelODE(
        modeldata; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all
    ) -> f::ModelODE

Function object to calculate model time derivative and adapt to SciML ODE solver interface

Call as `f(du,u, p, t)`
"""
mutable struct ModelODE{T, S <: PALEOmodel.SolverView, D}
    modeldata::PB.ModelData{T}
    solver_view::S
    dispatchlists::D
    nevals::Int
end

ModelODE(
    modeldata; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE(modeldata, solver_view, dispatchlists, 0)


function (m::ModelODE)(du,u, p, t)
   
    PALEOmodel.set_statevar!(m.solver_view, u)
    PALEOmodel.set_tforce!(m.solver_view, t)

    PB.do_deriv(m.dispatchlists)

    PALEOmodel.get_statevar_sms!(du, m.solver_view)
   
    m.nevals += 1  

    return nothing
end


"""
    ModelODE_at_t

Function object to calculate model derivative at `t`, eg to adapt to ForwardDiff or NLsolve interface

Calculates F = du/dt(t)
"""
mutable struct ModelODE_at_t{M <: ModelODE}
    modelode::M
    t::Float64
end

ModelODE_at_t(
    modeldata; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE_at_t(ModelODE(modeldata; solver_view=solver_view, dispatchlists=dispatchlists), NaN)

function set_t!(mt::ModelODE_at_t, t)
    mt.t = t
end

(mt::ModelODE_at_t)(F, u) = mt.modelode(F, u, nothing, mt.t)


"""
    JacODEForwardDiffDense(
        modeldata; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template, 
        jacconf,
    ) -> jac::JacODEForwardDiffDense

Function object to calculate dense Jacobian in form required for SciML ODE solver.

`solver_view`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.

Call as `jac(J, u, p, t)`
"""
mutable struct JacODEForwardDiffDense{MD <: PB.ModelData, M, W, J <: ForwardDiff.JacobianConfig}
    modeldata::MD
    deriv_at_t::M
    du_template::W
    jacconf::J    
    njacs::Int64
end

function JacODEForwardDiffDense(
    modeldata::PB.ModelData, solver_view, dispatchlists, du_template, jacconf::ForwardDiff.JacobianConfig,
)
    return JacODEForwardDiffDense(
        modeldata,
        ModelODE_at_t(modeldata, solver_view=solver_view, dispatchlists=dispatchlists), 
        du_template, 
        jacconf, 
        0,
    )
end


function (jfdd::JacODEForwardDiffDense)(J, u, p, t)
   
    set_t!(jfdd.deriv_at_t, t)
   
    ForwardDiff.jacobian!(J, jfdd.deriv_at_t,  jfdd.du_template, u, jfdd.jacconf)   
    jfdd.njacs += 1  

    return nothing
end

"""
    JacODEForwardDiffSparse(
        modeldata; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template, 
        jac_cache,
    ) -> jac::JacODEForwardDiffSparse

Function object to calculate sparse Jacobian in form required for SciML ODE solver.

`solver_view`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.

Call as `jac(J, u, p, t)`
"""
mutable struct JacODEForwardDiffSparse{MD <: PB.ModelData, M, J <: SparseDiffTools.ForwardColorJacCache}
    modeldata::MD
    deriv_at_t::M
    jac_cache::J    
    njacs::Int    
end

function JacODEForwardDiffSparse(
    modeldata::PB.ModelData, solver_view, dispatchlists, jac_cache::SparseDiffTools.ForwardColorJacCache
)
    return JacODEForwardDiffSparse(
        modeldata, 
        ModelODE_at_t(modeldata, solver_view=solver_view, dispatchlists=dispatchlists), 
        jac_cache, 
        0,
    )
end

function (jfds::JacODEForwardDiffSparse)(J, u, p, t)
    
    nnz_before = SparseArrays.nnz(J)
    
    set_t!(jfds.deriv_at_t, t)

    SparseDiffTools.forwarddiff_color_jacobian!(
        J,
        jfds.deriv_at_t,
        u,
        jfds.jac_cache
    )

    nnz_before == SparseArrays.nnz(J) || error("Jacobian sparsity changed nnz $(nnz_before) != $(SparseArrays.nnz(J))")
  
    countnan = count(isnan, J.nzval)
    iszero(countnan) || @warn "JacODEForwardDiffSparse: Jacobian contains $countnan NaN at t=$t"

    jfds.njacs += 1  
    
    return nothing
end


"""
    JacODE_at_t

Function object to calculate ODE model Jacobian at `t`, eg to adapt to NLsolve interface
"""
mutable struct JacODE_at_t{J}
    jacode::J
    t::Float64
end

function set_t!(mt::JacODE_at_t, t)
    mt.t = t
end

(jt::JacODE_at_t)(J, u) = jt.jacode(J, u, nothing, jt.t)


########################################################################
# Adaptors for SciML DAE solvers
#########################################################################

"""
    ModelDAE

Function object to calculate model residual `G` and adapt to SciML DAE solver interface.

If using Total variables, `odeimplicit` should be an [`ImplicitForwardDiffDense`](@ref) or [`ImplicitForwardDiffSparse`](@ref),
otherwise `nothing`.

Provides function signature:

    (fdae::ModelDAE)(G, dsdt, s, p, t)

where residual `G(dsdt,s,p,t)` is:
- `-dsdt + F(s)`  (for ODE-like state Variables s with time derivative F given explicitly in terms of s)
- `F(s)` (for algebraic constraints)
- `duds*dsdt + F(s, u(s))` (for Total variables u that depend implicitly on state Variables s)
"""
mutable struct ModelDAE{T, S <: PALEOmodel.SolverView, D, O}
    modeldata::PB.ModelData{T}
    solver_view::S
    dispatchlists::D
    odeimplicit::O
    nevals::Int
end

function (m::ModelDAE)(G, dsdt, s, p, t)
    
    PALEOmodel.set_statevar!(m.solver_view, s)
    PALEOmodel.set_tforce!(m.solver_view, t)

    # du(s)/dt
    PB.do_deriv(m.dispatchlists)

    # get explicit deriv
    l_ts = copyto!(G, m.solver_view.stateexplicit_deriv)
    # -dudt = -dsdt explicit variables with u(s) = s so duds = I    

    @inbounds for i in 1:l_ts
        G[i] -= dsdt[i]
    end

    # get implicit_deriv     
    l_ti = length(m.solver_view.total)
    if l_ti > 0
        !isnothing(m.odeimplicit) ||
            error("implicit Total Variables, odeimplicit required")

        copyto!(G, m.solver_view.total_deriv, dof=l_ts+1)

        # -dudt = -duds*dsdt implicit variables with u(s)

        # calculate duds using AD
        m.odeimplicit(m.odeimplicit.duds, s, p, t)       
        # add duds*dsdt to resid
        G[(l_ts+1):(l_ts+l_ti)] -= m.odeimplicit.duds*dsdt
    end

    # add constraints to residual
    copyto!(G, m.solver_view.constraints, dof=l_ts+l_ti+1)

    m.nevals += 1  

    return nothing
end



"""
    JacDAE

Function object to calculate Jacobian in form required for SciML DAE solver

`odejac` should be a [`JacODEForwardDiffDense`](@ref) or [`JacODEForwardDiffSparse`](@ref)

If using Total variables, `odeimplicit` should be an [`ImplicitForwardDiffDense`](@ref) or [`ImplicitForwardDiffSparse`](@ref),
otherwise `nothing`.

Provides function signature:

    (jdae::JacDAE)(J, dsdt, s, p, γ, t)

Calculates Jacobian `J` in the form `γ*dG/d(dsdt) + dG/ds` where `γ` is given by the solver
"""
mutable struct JacDAE{J, I}
    odejac::J
    odeimplicit::I
end

function (jdae::JacDAE)(J, dsdt, s, p, γ, t)
    # The Jacobian should be given in the form γ*dG/d(dsdt) + dG/ds where γ is given by the solver
   
    # dG/ds
    jdae.odejac(J,s, p, t)

    md = jdae.odejac.modeldata
    # γ*dG/d(dsdt) explicit variables with u(s) = s
    l_ts = length(md.solver_view_all.stateexplicit_deriv)
    for i in 1:l_ts     
        J[i, i] -= γ
    end

    # no contribution from dG/dsdt from constraints

    # γ*dG/d(dsdt) = γ*(du/ds)*dG/d(dudt) implicit variables u(s)
    l_ti = length(md.solver_view_all.total)
    if l_ti > 0             
        jdae.odeimplicit(jdae.odeimplicit.duds, s, p, t)        
        J[(l_ts+1):(l_ts+l_ti), :] .-= γ.*jdae.odeimplicit.duds
    end
    
    return nothing
end

"""
    TotalForwardDiff

Calculate Total variables, with function signature required by ForwardDiff

Calling:

    set_t!(tfd::TotalForwardDiff, t)
    tfd(T, u)
"""
mutable struct TotalForwardDiff{S, D}
    solver_view::S
    dispatchlists::D
    t::Float64
end

function set_t!(tfd::TotalForwardDiff, t)
    tfd.t = t
end

function (tfd::TotalForwardDiff)(T, u)

    PALEOmodel.set_statevar!(tfd.solver_view, u)
    PALEOmodel.set_tforce!(tfd.solver_view, tfd.t)

    PB.do_deriv(tfd.dispatchlists)

    copyto!(T, tfd.solver_view.total)      

    return nothing
end

"""
    ImplicitForwardDiffDense

Calculate dT/dS required for a model containing implicit Total variables, using ForwardDiff and dense AD
"""
mutable struct ImplicitForwardDiffDense{T, S, D, W, I, U}
    modeldata::PB.ModelData{T}
    implicitderiv::TotalForwardDiff{S, D}
    duds_template::W
    implicitconf::I
    duds::U    
end

function ImplicitForwardDiffDense(
    modeldata::PB.ModelData, solver_view, dispatchlists, duds_template, implicitconf, duds
)
    return ImplicitForwardDiffDense(
        modeldata, TotalForwardDiff(solver_view, dispatchlists, NaN), duds_template, implicitconf, duds
    )
end

function (ifdd::ImplicitForwardDiffDense)(dTdS, s, p, t)
    
    set_t!(ifdd.implicitderiv, t)

    ForwardDiff.jacobian!(dTdS, ifdd.implicitderiv, ifdd.duds_template, s, ifdd.implicitconf)  

    return  nothing
end

"""
    ImplicitForwardDiffSparse

Calculate dT/dS required for a model containing implicit Total variables, using ForwardDiff and 
sparse AD with `SparseDiffTools.forwarddiff_color_jacobian!`
"""
mutable struct ImplicitForwardDiffSparse{T, S, D, I, U}
    modeldata::PB.ModelData{T}
    implicitderiv::TotalForwardDiff{S, D}
    implicit_cache::I
    duds::U    
end

function ImplicitForwardDiffSparse(
    modeldata::PB.ModelData, solver_view, dispatchlists, implicit_cache, duds
)
    return ImplicitForwardDiffSparse(
        modeldata, TotalForwardDiff(solver_view, dispatchlists, NaN), implicit_cache, duds
    )
end

function (ifds::ImplicitForwardDiffSparse)(dTdS, s, p, t)
    
    nnz_before = SparseArrays.nnz(dTdS)

    set_t!(ifds.implicitderiv, t)
    SparseDiffTools.forwarddiff_color_jacobian!(
        dTdS,
        ifds.implicitderiv,
        s,
        ifds.implicit_cache
    )

    nnz_before == SparseArrays.nnz(dTdS) || error("dTdS sparsity changed nnz $(nnz_before) != $(SparseArrays.nnz(dTdS))")
    
    return nothing
end



end # module