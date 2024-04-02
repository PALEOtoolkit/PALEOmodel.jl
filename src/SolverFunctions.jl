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
import MultiFloats
import Sparspak

# import Infiltrator # Julia debugger

#####################################################################################
# Adaptors and function objects for NLsolve (non-linear steady-state solvers)
#####################################################################################

"""
    StepClampMultAll!(minvalue, maxvalue, minmult, maxmult) -> scma!
    StepClampMultAll!(minvalue, maxvalue, maxratio) = StepClampMultAll!(minvalue, maxvalue, 1.0/maxratio, maxratio)
    scma!(x, x_old, newton_step)

Function object to take Newton step `x .= x_old .+ newton_step` and then clamp all values in Vector `x` to specified range using:
- `clamp!(x, x_old*minmult, x_old*maxmult)`
- `clamp!(x, minvalue, maxvalue)`
"""
struct StepClampMultAll!
    minvalue::Float64
    maxvalue::Float64
    minmult::Float64
    maxmult::Float64
end

StepClampMultAll!(minvalue, maxvalue, maxratio) = StepClampMultAll!(minvalue, maxvalue, 1.0/maxratio, maxratio)

function (scma::StepClampMultAll!)(x, x_old, newton_step)
    for i in eachindex(x)
        x[i] = x_old[i] + newton_step[i]
        x[i] = clamp(x[i], x_old[i]*scma.minmult, x_old[i]*scma.maxmult)
        x[i] = clamp(x[i], scma.minvalue, scma.maxvalue)
    end
    return nothing
end

"""
    StepClampAll!(minvalue, maxvalue) -> sca!
    sca!(x, x_old, newton_step)

Function object to take Newton step `x .= x_old .+ newton_step` and then clamp all values in Vector `x` to specified range using
`clamp!(x, minvalue, maxvalue)`
"""
struct StepClampAll!
    minvalue::Float64
    maxvalue::Float64
end

function (sca::StepClampAll!)(x, x_old, newton_step)
    for i in eachindex(x)
        x[i] = x_old[i] + newton_step[i]
        x[i] = clamp(x[i], sca.minvalue, sca.maxvalue)
    end
    return nothing
end

"""
    ClampAll!(minvalue, maxvalue) -> ca!
    ca!(v)

Function object to clamp all values in Vector `v` to specified range using
`clamp!(v, minvalue, maxvalue)` (in-place, mutating version)
"""
struct ClampAll!
    minvalue::Float64
    maxvalue::Float64
end

(ca::ClampAll!)(v) = clamp!(v, ca.minvalue, ca.maxvalue)

"""
    ca = ClampAll(minvalue, maxvalue)
    ca(v) -> v

Function object to clamp all values in Vector `v` to specified range using
`clamp.(v, minvalue, maxvalue)` (out-of-place version)
"""
struct ClampAll
    minvalue::Float64
    maxvalue::Float64
end

(ca::ClampAll)(v) = clamp.(v, ca.minvalue, ca.maxvalue)

"""
    SparseLinsolveUMFPACK() -> slsu
    slsu(x, A, b)

Create solver function object to solve sparse A x = b using UMFPACK lu factorization

Reenables iterative refinement (switched off by default by Julia lu)
"""
mutable struct SparseLinsolveUMFPACK
    umfpack_control::Vector{Float64}
    lu

    function SparseLinsolveUMFPACK()
        umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64)
        # SparseArrays.UMFPACK.show_umf_ctrl(umfpack_control)
        umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement

        return new(umfpack_control, nothing)
    end
end

function (slsu::SparseLinsolveUMFPACK)(x, A, b)
    if isnothing(slsu.lu)
        slsu.lu = LinearAlgebra.lu(A; control=slsu.umfpack_control)
    else
        LinearAlgebra.lu!(slsu.lu, A; reuse_symbolic=true)
    end

    x .= slsu.lu \ b

    return nothing
end

"""
    SparseLinsolveSparspak64x2(; verbose=false) -> slsp
    slsp(x, A, b)

Create solver function object to solve sparse A x = b using Sparspak lu factorization at quad precision

Includes one step of iterative refinement
"""
mutable struct SparseLinsolveSparspak64x2
    A_mf::Union{Nothing, SparseArrays.SparseMatrixCSC{MultiFloats.MultiFloat{Float64, 2}, Int64}}
    A_mf_lu
    verbose::Bool

    function SparseLinsolveSparspak64x2(; verbose=false)
        return new(nothing, nothing, verbose)
    end
end

function (slsp::SparseLinsolveSparspak64x2)(x::Vector, A::SparseArrays.SparseMatrixCSC, b::Vector)

    if isnothing(slsp.A_mf) 
        # Julia bug - type conversion squeezes out structural nonzeros !?
        # slsp.A_mf = MultiFloats.Float64x2.(A)
        # workaround - convert type by hand, preserving structural nonzeros
        slsp.A_mf = SparseArrays.SparseMatrixCSC(size(A, 1), size(A, 2), A.colptr, A.rowval, MultiFloats.Float64x2.(A.nzval))
        slsp.A_mf_lu = Sparspak.sparspaklu(slsp.A_mf)
    else
        size(A) == size(slsp.A_mf) || error("size of A has changed")
        ((A.colptr == slsp.A_mf.colptr) && (A.rowval == slsp.A_mf.rowval)) || error("sparsity pattern of A has changed")
        slsp.A_mf.nzval .= MultiFloats.Float64x2.(A.nzval)
        newlu = Sparspak.sparspaklu!(slsp.A_mf_lu, slsp.A_mf) # reuse ordering and symbolic factorization
        newlu === slsp.A_mf_lu || error("Sparspak.sparspaklu! has not reused lu !!!")
    end
  
    # Solve with iterative refinement at Float64x4 precision
    # (high precision is not really needed as x is only returned as a Float64, but 
    # iterative refinement *is* needed to avoid strange numerical noise patterns in solution - bug in Sparspak or MultiFloats?)
    b_mf = MultiFloats.Float64x2.(b)
    x_mf4_1 = MultiFloats.Float64x4.(slsp.A_mf_lu \ b_mf)

    r_mf4_1 = b - A*x_mf4_1
    c_mf4_1 = MultiFloats.Float64x4.(slsp.A_mf_lu \ MultiFloats.Float64x2.(r_mf4_1))
    x_mf4_2 = x_mf4_1 + c_mf4_1
  
    if slsp.verbose
        r_mf4_2 = b - A*x_mf4_2

        norm2_1 = LinearAlgebra.norm(r_mf4_1)
        norminf_1 = LinearAlgebra.norm(r_mf4_1, Inf)

        norm2_2 = LinearAlgebra.norm(r_mf4_2)
        norminf_2 = LinearAlgebra.norm(r_mf4_2, Inf)

        @info """\n
        SparseLinsolveSparspak64x2      norm2=$norm2_1, norminf=$norminf_1
        SparseLinsolveSparspak64x2 ir 1 norm2=$norm2_2, norminf=$norminf_2
        """
    end
    
    x .= Float64.(x_mf4_2)
  
    return nothing
end

##############################################################################################################
# Adaptors (function objects) for SciML ODE solvers
##############################################################################################################

"""
    ModelODE(
        modeldata, [parameter_aggregator]; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all
    ) -> f::ModelODE

Function object to calculate model time derivative and adapt to SciML ODE solver interface

Call as `f(du,u, p, t)`
"""
mutable struct ModelODE{S <: PALEOmodel.SolverView, D, P <: Union{Nothing, PB.ParameterAggregator}}
    modeldata::PB.ModelData
    solver_view::S
    dispatchlists::D
    parameter_aggregator::P
    nevals::Int
end

ModelODE(
    modeldata; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE(modeldata, solver_view, dispatchlists, nothing, 0)

ModelODE(
    modeldata, parameter_aggregator; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE(modeldata, solver_view, dispatchlists, copy(parameter_aggregator), 0)

function (m::ModelODE)(du,u, p, t)
   
    PALEOmodel.set_statevar!(m.solver_view, u)
    PALEOmodel.set_tforce!(m.solver_view, t)

    if isnothing(m.parameter_aggregator)
        PB.do_deriv(m.dispatchlists)
    else
        copyto!(m.parameter_aggregator, p)
        PB.do_deriv(m.dispatchlists, m.parameter_aggregator)
    end

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
    ModelODE_p_at_t(
        modeldata, parameter_aggregator; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all
    ) -> f::ModelODE_p_at_t

Function object to calculate model derivative `du` at `t` with parameter vector `p`, eg to adapt to ForwardDiff or NLsolve interface
which require a function signature that includes state vector `u` only.

Call as:

    set_p_t!(f, p, t)
    f(du, u)  # du(u) at p, t
"""
mutable struct ModelODE_p_at_t{M <: ModelODE, P <: AbstractVector}
    modelode::M
    p::P
    t::Float64
end

ModelODE_p_at_t(
    modeldata, parameter_aggregator; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE_p_at_t(
        ModelODE(modeldata, parameter_aggregator; solver_view, dispatchlists),
        PB.get_currentvalues(parameter_aggregator),
        NaN
    )

function set_p_t!(mt::ModelODE_p_at_t, p, t)
    mt.p .= p
    mt.t = t
end

(mt::ModelODE_p_at_t)(du, u) = mt.modelode(du, u, mt.p, mt.t)



"""
    ModelODE_u_at_t(
        modeldata, u_template, parameter_aggregator; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all
    ) -> f::ModelODE_u_at_t

Function object to calculate model derivative `du` at `t` with state vector `u`, eg to adapt to ForwardDiff or NLsolve interface
which require a function signature that includes parameter vector `p` only.

Call as:

    set_u_t!(f, u, t)
    f(du, p)   # du(p) at u, t
"""
mutable struct ModelODE_u_at_t{M <: ModelODE, U <: AbstractVector}
    modelode::M
    u::U
    t::Float64
end

ModelODE_u_at_t(
    modeldata, u_template::AbstractVector, parameter_aggregator::PB.ParameterAggregator; 
    solver_view=modeldata.solver_view_all,
    dispatchlists=modeldata.dispatchlists_all
) = ModelODE_u_at_t(
        ModelODE(modeldata, parameter_aggregator; solver_view, dispatchlists),
        similar(u_template),
        NaN
    )

function set_u_t!(mt::ModelODE_u_at_t, u, t)
    mt.u .= u
    mt.t = t
end

(mt::ModelODE_u_at_t)(F, p) = mt.modelode(F, mt.u, p, mt.t)


"""
    JacODEForwardDiffDense(
        modeldata; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template, 
        jacconf,
    ) -> jac::JacODEForwardDiffDense

    JacODEForwardDiffDense_p(
        modeldata, pa::PB.ParameterAggregator; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template, 
        jacconf,
    ) -> jac::JacODEForwardDiffDense_p

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

mutable struct JacODEForwardDiffDense_p{MD <: PB.ModelData, M, W, J <: ForwardDiff.JacobianConfig}
    modeldata::MD
    deriv_at_p_t::M
    du_template::W
    jacconf::J    
    njacs::Int64
end

function JacODEForwardDiffDense_p(
    modeldata::PB.ModelData, pa::PB.ParameterAggregator, solver_view, dispatchlists, du_template, jacconf::ForwardDiff.JacobianConfig,
)
    return JacODEForwardDiffDense_p(
        modeldata,
        ModelODE_p_at_t(modeldata, pa; solver_view=solver_view, dispatchlists=dispatchlists), 
        du_template, 
        jacconf, 
        0,
    )
end


function (jfdd::JacODEForwardDiffDense_p)(J, u, p, t)
   
    set_p_t!(jfdd.deriv_at_p_t, p, t)
   
    ForwardDiff.jacobian!(J, jfdd.deriv_at_p_t,  jfdd.du_template, u, jfdd.jacconf)   
    jfdd.njacs += 1  

    return nothing
end



"""
    JacODEForwardDiffSparse(
        modeldata; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template,
        throw_on_nan, 
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
    throw_on_nan::Bool   
    njacs::Int    
end

function JacODEForwardDiffSparse(
    modeldata::PB.ModelData, solver_view, dispatchlists, jac_cache::SparseDiffTools.ForwardColorJacCache;
    throw_on_nan = false,
)
    return JacODEForwardDiffSparse(
        modeldata, 
        ModelODE_at_t(modeldata, solver_view=solver_view, dispatchlists=dispatchlists), 
        jac_cache, 
        throw_on_nan,
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
    if !iszero(countnan)
        if jfds.throw_on_nan
            @error "JacODEForwardDiffSparse: Jacobian contains $countnan NaN at t=$t"
        else
            @warn "JacODEForwardDiffSparse: Jacobian contains $countnan NaN at t=$t"
        end
    end

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


"""
    ParamJacODEForwardDiffDense(
        modeldata, PB.ParameterAggregator; 
        solver_view=modeldata.solver_view_all,
        dispatchlists=modeldata.dispatchlists_all,
        du_template, 
        paramjacconf,
    ) -> paramjac::ParamJacODEForwardDiffDense

Function object to calculate dense parameter Jacobian in form required for SciML ODE solver.

`solver_view`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.

Call as `paramjac(pJ, u, p, t)`
"""
mutable struct ParamJacODEForwardDiffDense{MD <: PB.ModelData, M, W, J <: ForwardDiff.JacobianConfig}
    modeldata::MD
    deriv_at_u_t::M
    du_template::W
    paramjacconf::J    
    njacs::Int64
end

function ParamJacODEForwardDiffDense(
    modeldata::PB.ModelData, pa::PB.ParameterAggregator, solver_view, dispatchlists, du_template, paramjacconf::ForwardDiff.JacobianConfig,
)
    return ParamJacODEForwardDiffDense(
        modeldata,
        ModelODE_u_at_t(modeldata, du_template, pa; solver_view=solver_view, dispatchlists=dispatchlists), 
        du_template, 
        paramjacconf, 
        0,
    )
end


function (pjfdd::ParamJacODEForwardDiffDense)(pJ, u, p, t)
   
    set_u_t!(pjfdd.deriv_at_u_t, u, t)

    ForwardDiff.jacobian!(pJ, pjfdd.deriv_at_u_t,  pjfdd.du_template, p, pjfdd.paramjacconf)   
    pjfdd.njacs += 1  

    return nothing
end


"""
    mutable struct ParamJacODEForwardDiffDenseT{S <: PALEOmodel.SolverView, D, P, PW, UW, F}
        modeldata::PB.ModelData
        solver_view::S
        dispatchlists::D
        parameter_aggregator::P
        p_worksp::PW
        du_worksp::UW
        filterT::F
        nevals::Int
    end

Function object to calculate dense parameter Jacobian in form required for SciML ODE solvers, optimised
for the case where `paramjac` at time t has only a small number of nonzero columns (ie only a small number of 
paramters affect `du/dt` at time `t`).

`filterT(t) -> (pidx_1, p_idx_2, ..)` should be a function that returns a Tuple of indices in the parameter vector
that affect `du/dt` at time `t` and hence have non-zero columns in `paramjac`.

`solver_view`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.

Call as `paramjac(pJ, u, p, t)`
"""
mutable struct ParamJacODEForwardDiffDenseT{S <: PALEOmodel.SolverView, D, P <: PB.ParameterAggregator, PW, UW, F}
    modeldata::PB.ModelData
    solver_view::S
    dispatchlists::D
    parameter_aggregator::P
    p_worksp::PW
    du_worksp::UW
    filterT::F
    nevals::Int
end


function (pjfddt::ParamJacODEForwardDiffDenseT)(pJ, u, p, t)
   
    PALEOmodel.set_statevar!(pjfddt.solver_view, u)
    PALEOmodel.set_tforce!(pjfddt.solver_view, t)

    active_pidx = pjfddt.filterT(t) # Tuple with indices of parameters with non-zero pJ at time t

    for j in eachindex(p)
        if j in active_pidx
            # calculate derivative wrt p[j] -> j'th column of parameter Jacobian
            # NB: low-level use of ForwardDiff.Dual, with a single partial to calculate a single derivative

            # fill p_worksp and set parameter_aggregator with Dual numbers with partials = 0.0, except for parameter j
            pjfddt.p_worksp .= p
            pjfddt.p_worksp[j] = ForwardDiff.Dual(p[j], 1.0)
            copyto!(pjfddt.parameter_aggregator, pjfddt.p_worksp)

            PB.do_deriv(pjfddt.dispatchlists, pjfddt.parameter_aggregator)
        
            PALEOmodel.get_statevar_sms!(pjfddt.du_worksp, pjfddt.solver_view)
            # fill this column in parameter Jacobian
            for i in eachindex(u)
                pJ[i, j] = only(pjfddt.du_worksp[i].partials)
            end
        else
            # zero out this column in Jacobian
            pJ[:, j] .= 0.0
        end
    end

    pjfddt.nevals += 1  

    return nothing
end


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
mutable struct ModelDAE{S <: PALEOmodel.SolverView, D, O}
    modeldata::PB.ModelData
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
mutable struct ImplicitForwardDiffDense{S, D, W, I, U}
    modeldata::PB.ModelData
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
mutable struct ImplicitForwardDiffSparse{S, D, I, U}
    modeldata::PB.ModelData
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