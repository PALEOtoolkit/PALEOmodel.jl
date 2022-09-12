module SplitDAE

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import ForwardDiff
import Infiltrator
using Logging
import StaticArrays

import NLsolve
# import ..SolverFunctions
# import ..ODE
# import ..JacobianAD

import PALEOmodel.SolverFunctions
import PALEOmodel.ODE
import PALEOmodel.JacobianAD
import PALEOmodel.NonLinearNewton
import PALEOmodel.SteadyState


"""
    split_dae(
        model, initial_state, modeldata;
        jac_ad=:ForwardDiffSparse,
        tss_jac_sparsity=nothing,
        request_adchunksize=10,
        jac_cellranges=modeldata.cellranges_all,
        operatorID_inner=0,
        init_logger=Logging.NullLogger(),
    ) -> (ModelSplitDAE, initial_state_outer)

Given a model that contains both ODE variables and algebraic constraints, creates function objects for ODE variables,
where algebraic constraints are solved by an 'inner' Newton solver.

Requires that algebraic constraints are 'local' ie per-cell.

Uses implicit function theorum to efficiently calculate Jacobian for ODE variables.
"""
function split_dae(
    model, initial_state, modeldata;
    jac_ad=:ForwardDiffSparse,
    tss_jac_sparsity=nothing,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,    
    operatorID_inner=3,
    transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"],  # additional Variables needed by 'inner' Reactions
    init_logger=Logging.NullLogger(),
    inner_reltol=1e-9,
    inner_miniters=1,
    inner_maxiters=10,
    inner_verbose=4,
)
    PB.check_modeldata(model, modeldata)

    sva = modeldata.solver_view_all
    # We only support explicit ODE + DAE constraints (no implicit variables)
    iszero(PALEOmodel.num_total(sva))                 || error("implicit total variables not supported")
    !iszero(PALEOmodel.num_algebraic_constraints(sva)) || error("no algebraic constraints present")

    # Variable aggegators for 'outer' ODE variables only
    va_outer_state = sva.stateexplicit
    va_outer_state_deriv = sva.stateexplicit_deriv
    # calculate initial_state_outer
    PALEOmodel.set_statevar!(sva, initial_state)
    initial_state_outer = Vector{eltype(modeldata)}(undef, length(va_outer_state))
    copyto!(initial_state_outer, va_outer_state)

    # full sparse jacobian jacode(Jfull, u, p, t) where Jfull has sparsity pattern jac_prototype
    @info "split_dae:  using Jacobian $jac_ad"
    jacfull, jacfull_prototype = PALEOmodel.JacobianAD.jac_config_ode(
        jac_ad, model, initial_state, modeldata, tss_jac_sparsity,
        request_adchunksize=request_adchunksize,
        jac_cellranges=jac_cellranges
    )    
    # any Variables calculated by modelode but not jacode, that need to be copied
    transfer_data_ad, transfer_data = PALEOmodel.JacobianAD.jac_transfer_variables(
        model,
        jacfull.modeldata,
        modeldata
    )

    # sparsity pattern for outer Jacobian
    ro = 1:length(va_outer_state) # indices of 'outer' variables
    ri = (length(va_outer_state)+1):length(initial_state)
    jacouter_prototype = jacfull_prototype[ro, ro]

    # include contributions from inner state Variables
    dG_dcellinner = jacfull_prototype[ri, ri] # n_inner x n_inner
    dG_douter = jacfull_prototype[ri, ro] # n_inner x n_outer
    # try and generate a non-singular matrix
    for i in eachindex(dG_dcellinner.nzval)
        dG_dcellinner.nzval[i] = i
    end
    # dcellinner_dcellouter = -dG_dcellinner \ dG_douter # n_inner x n_outer
    # \ not implemented ...
    # get an inverse with the correct sparsity    
    # find dense inverse
    dG_dcellinner_inv_dense = LinearAlgebra.inv(Matrix(dG_dcellinner))
    # reconstruct a sparse matrix
    ij = Tuple.(findall(!iszero, dG_dcellinner_inv_dense))
    I = [i for (i,j) in ij]
    J = [j for (i,j) in ij]
    dG_dcellinner_inv = SparseArrays.sparse(I, J, ones(length(I)))   
    dcellinner_dcellouter = -dG_dcellinner_inv * dG_douter # n_inner x n_outer

    # n_outer x n_outer  +=  n_outer x n_inner  * n_inner x n_outer
    jacouter_implicit = jacfull_prototype[ro, ri]*dcellinner_dcellouter
    jacouter_implicit.nzval .= 1

    # combine to get full sparsity pattern
    io = IOBuffer()
    println(io, "split_dae:")
    println(io, "    jacouter (outer variables, $(size(jacouter_prototype))): nnz=$(SparseArrays.nnz(jacouter_prototype))")
    println(io, "    jacouter_implicit (from inner variables): nnz=$(SparseArrays.nnz(jacouter_implicit))")
    jacouter_prototype += jacouter_implicit
    jacouter_prototype.nzval .= 1
    println(io, "    jacouter (all variables): nnz=$(SparseArrays.nnz(jacouter_prototype))")

    # find all (state, constraint) algebraic constraints
    vinner_names = []
    vinner_indices = []
    inner_domains = []
    va_state = sva.state
    va_constraints = sva.constraints
    
    println(io, "    quasi-steady-state Variables for inner Newton solve:")
    for (var, indices) in zip(va_state.vars, va_state.indices)
        println(io, "        $(PB.fullname(var))  indices $indices")
        push!(vinner_names, PB.fullname(var))
        push!(inner_domains, var.domain)
        push!(vinner_indices, indices)
    end
    inner_domains =unique(inner_domains)
    length(inner_domains) == 1 || error("TODO only 1 Domain with algebraic constraints supported, Domains=$vinner_domains")
    inner_domain = inner_domains[1]
    # find cellranges for Domains with algebraic constraints
    cellrange_inner = PB.Grids.create_default_cellrange(inner_domain, inner_domain.grid, operatorID=operatorID_inner)
    println(io, "    inner Newton solve Domain $inner_domain, cellrange $cellrange_inner")

    # create a modeldata_ad with Dual numbers for AD Jacobians for 'inner' Newton solve
    n_inner_solve = length(vinner_names)
    println(io, "   using ForwardDiff dense Jacobian for inner Newton solve")
    eltype_inner_ad = ForwardDiff.Dual{Nothing, eltype(modeldata), n_inner_solve}
    _, modeldata_ad_inner = Logging.with_logger(init_logger) do
        PALEOmodel.initialize!(model, eltype=eltype_inner_ad)
    end
    va_inner_stateexplicit = modeldata_ad_inner.solver_view_all.stateexplicit
    # any Variables calculated by modelode but not jacode, that need to be copied
    transfer_data_ad_inner, transfer_data_inner = PALEOmodel.JacobianAD.jac_transfer_variables(
        model,
        modeldata_ad_inner,
        modeldata;
        extra_vars=transfer_inner_vars
    )

    # construct function objects to calculate per-cell constraint and Jacobian for 'inner' Variables
    cellderivs, celljacs, cellidxfull = [], [], []
    cellworksp = fill(NaN, n_inner_solve)
    cellworksp_ad = fill(eltype_inner_ad(NaN), n_inner_solve)
   
    for i in cellrange_inner.indices
        cellrange = PB.CellRange(cellrange_inner.domain, cellrange_inner.operatorID, i)
        vars_cellranges = [cellrange for i in 1:length(va_state.vars)]
        # modeldata Arrays to calculate residual
        va_cell_state = PB.VariableAggregator(va_state.vars, vars_cellranges, modeldata)
        va_cell_constraints = PB.VariableAggregator(va_constraints.vars, vars_cellranges, modeldata)
        va_cell_dl = PB.create_dispatch_methodlists(model, modeldata, [cellrange])
        push!(cellderivs, ModelDeriv(n_inner_solve, modeldata, va_cell_state, va_cell_constraints, va_cell_dl, cellworksp))
       
        # modeldata_ad_inner Arrays to calculate Jacobian using ForwardDiff
        va_cell_state_ad = PB.VariableAggregator(va_state.vars, vars_cellranges, modeldata_ad_inner)
        va_cell_constraints_ad = PB.VariableAggregator(va_constraints.vars, vars_cellranges, modeldata_ad_inner)
        va_cell_dl_ad = PB.create_dispatch_methodlists(model, modeldata_ad_inner, [cellrange])
        push!(celljacs, ModelJacForwardDiff(ModelDeriv(n_inner_solve, modeldata_ad_inner, va_cell_state_ad, va_cell_constraints_ad, va_cell_dl_ad, cellworksp_ad)))
        # index in full model for each Variable
        va_cell_idx = [indices[i] + length(va_outer_state) for indices in va_state.indices]
        push!(cellidxfull, va_cell_idx)
    end

    @info String(take!(io))

    ms = ModelSplitDAE(
        modeldata,
        sva,
        modeldata.dispatchlists_all,
        jacfull,
        copy(jacfull_prototype),
        transfer_data_ad,
        transfer_data,
        transfer_data_ad_inner,
        transfer_data_inner,
        va_outer_state,
        va_outer_state_deriv,
        va_inner_stateexplicit,
        [c for c in cellderivs], # narrow type
        [c for c in celljacs], # narrow type
        [c for c in cellidxfull], # narrow_type
        inner_reltol,
        inner_miniters,
        inner_maxiters,
        inner_verbose,
        similar(initial_state_outer),
        similar(initial_state),
    )

    return (ms, initial_state_outer, jacouter_prototype)
end

# all Variables needed for inner and outer solve
mutable struct ModelSplitDAE{T, SVA, DLA, JF, TD1, TD2, TD3, TD4, VA1, VA2, VA3, CD, CJ}
    modeldata::PB.ModelData{T}
    solver_view_all::SVA
    dispatchlists_all::DLA
    jacfull::JF
    jacfull_ws::SparseArrays.SparseMatrixCSC{T, Int64}
    transfer_data_ad::TD1
    transfer_data::TD2
    transfer_data_ad_inner::TD3
    transfer_data_inner::TD4
    va_outer_state::VA1
    va_outer_state_deriv::VA2
    va_inner_stateexplicit::VA3
    cellderivs::CD
    celljacs::CJ
    cellidxfull::Vector{Vector{Int64}}
    inner_reltol::Float64
    inner_miniters::Int64
    inner_maxiters::Int64
    inner_verbose::Int64
    outer_worksp::Vector{T}
    full_worksp::Vector{T}
end

# calculate ODE 'outer' derivative, with Newton inner solve
mutable struct ModelODEOuter{MS <: ModelSplitDAE}
    ms::MS
end

function Base.getproperty(mode::ModelODEOuter, s::Symbol)
    if s == :modeldata
        return mode.ms.modeldata
    else
        return getfield(mode, s)
    end
end


function (mode_outer::ModelODEOuter)(du_outer, u_outer, p, t)
    ms = mode_outer.ms
    # set outer state Variables
    copyto!(ms.va_outer_state, u_outer)
    # NB: dont set inner state Variables (assumed already initialised to starting values from a previous iteration)

    # full derivative, with old values of inner state variables  
    PALEOmodel.set_tforce!(ms.solver_view_all, t)
    PB.do_deriv(ms.dispatchlists_all)

    # copy Variables to inner AD Variables 
    # explicitly requested Variables (grid geometry, photochemical rates, ...)
    for (dto, dfrom) in PB.IteratorUtils.zipstrict(ms.transfer_data_ad_inner, ms.transfer_data_inner)
        dto .= dfrom
    end
    # always set all outer state variables (inner state variables will be solved for)
    copyto!(ms.va_inner_stateexplicit, u_outer)

    # Newton solution for inner state Variables for each cell
    # NB: cell_number corresponds to 1:number_of_cells, not the actual cell indices in cellrange.indices
    for (cd, cj) in PB.IteratorUtils.zipstrict(ms.cellderivs, ms.celljacs)
        copyto!(cd.worksp, cd.state)
        initial_state = StaticArrays.SVector{ncomps(cd)}(cd.worksp)
        ms.inner_verbose > 0 && @info "initial_state: $initial_state"
        (_, Lnorm_2_cell, Lnorm_inf_cell, niter) = NonLinearNewton.solve(
            cd,
            cj, 
            initial_state;
            reltol=ms.inner_reltol,
            miniters=ms.inner_miniters,
            maxiters=ms.inner_maxiters,
            verbose=ms.inner_verbose,
        )
    end

    # reevaluate full derivative with updated inner state variables
    # TODO this could be optimized eg don't need to rerun radiative transfer
    PB.do_deriv(ms.dispatchlists_all)
    copyto!(du_outer, ms.va_outer_state_deriv)
    
    # @Infiltrator.infiltrate
    return nothing
end


# calculate ODE 'outer' derivative and Jacobian, with Newton inner solve
mutable struct ModelJacOuter{MS <: ModelSplitDAE, MO <: ModelODEOuter}
    ms::MS
    model_ode_outer::MO
end

# Jacobian only
function (jac_outer::ModelJacOuter)(J_outer, u_outer, p, t)
    ms = jac_outer.ms
    du_outer = ms.outer_worksp
    jac_outer(du_outer, J_outer, u_outer, p, t)
    return nothing
end

# derivative and Jacobian
function (jac_outer::ModelJacOuter)(du_outer, J_outer, u_outer, p, t)
    ms = jac_outer.ms
   
    # calculate derivative including solution for 'inner' state Variables
    jac_outer.model_ode_outer(du_outer, u_outer, p, t)

    # transfer Variables that are not included in Jacobian
    for (dto, dfrom) in PB.IteratorUtils.zipstrict(ms.transfer_data_ad, ms.transfer_data)
        dto .= dfrom
    end

    # calculate full Jacobian
    u_full = ms.full_worksp
    PALEOmodel.get_statevar!(u_full, ms.solver_view_all)
    J_full = ms.jacfull_ws
    J_full.nzval .= 0.0
    ms.jacfull(J_full, u_full, p, t)

    # calculate outer Jacobian using implicit function theorum
    J_outer.nzval .= 0.0
    # Jacobian of outer Variables
    ro = 1:size(J_outer, 1)
    # @Infiltrator.infiltrate
    # TODO this changes the size (stored elements) of J_outer !!
    J_outer .= J_full[ro, ro]

    # add contributions per-cell from inner Variables using implicit function theorum    
    for (cd, cj, ci) in PB.IteratorUtils.zipstrict(ms.cellderivs, ms.celljacs, ms.cellidxfull)
        dG_dcellinner = J_full[ci, ci] # n_inner x n_inner (TODO could also get this from jac used for inner solve)
        dG_douter = J_full[ci, ro] # n_inner x n_outer
        
        # dcellinner_dcellouter = -dG_dcellinner \ dG_douter # n_inner x n_outer
        # \ is not implemented
        # get an inverse with the correct sparsity    
        # find dense inverse
        dG_dcellinner_inv_dense = LinearAlgebra.inv(Matrix(dG_dcellinner))
        # reconstruct a sparse matrix
        ij = Tuple.(findall(!iszero, dG_dcellinner_inv_dense))
        I = [i for (i,j) in ij]
        J = [j for (i,j) in ij]
        V = [dG_dcellinner_inv_dense[i, j] for (i, j) in ij]
        dG_dcellinner_inv = SparseArrays.sparse(I, J, V)
        dcellinner_dcellouter = -dG_dcellinner_inv * dG_douter # n_inner x n_outer

        # @Infiltrator.infiltrate
        # n_outer x n_outer  +=  n_outer x n_inner  * n_inner x n_outer
        # TODO this changes the size (stored elements) of J_outer !!
        J_outer .+= J_full[ro, ci]*dcellinner_dcellouter 
    end

    return nothing
end

# calculate derivative to single cell
mutable struct ModelDeriv{Ncomps, T, VA1 <: PB.VariableAggregator, VA2 <: PB.VariableAggregator, D}
    modeldata::PB.ModelData{T}
    state::VA1
    deriv::VA2
    dispatchlists::D
    worksp::Vector{T}
    t::Float64
end

function ModelDeriv(Ncomps, modeldata::PB.ModelData{T}, state::VA1, deriv::VA2, dispatchlists::D, worksp::Vector{T}) where {T, VA1 <: PB.VariableAggregator, VA2 <: PB.VariableAggregator, D}
    return ModelDeriv{Ncomps, T, VA1, VA2, D}(modeldata, state, deriv, dispatchlists, worksp, NaN)
end

ncomps(md::ModelDeriv{Ncomps, T, VA1, VA2, D}) where {Ncomps, T, VA1, VA2, D} = Ncomps

function (md::ModelDeriv)(x)
    # @Infiltrator.infiltrate
    copyto!(md.state, x)
    PB.do_deriv(md.dispatchlists, 0.0)
    copyto!(md.worksp, md.deriv)

    return StaticArrays.SVector{ncomps(md)}(md.worksp)
end

# calculate Jacobian for a single cell
mutable struct ModelJacForwardDiff{MD}
    modelderiv::MD
end

function (mjfd::ModelJacForwardDiff)(x)
    return ForwardDiff.extract_jacobian(Nothing, ForwardDiff.static_dual_eval(Nothing, mjfd.modelderiv, x), x)
end

##############################################################
# Adaptor for PTC 
##############################################

function nlsolveF_PTC(ms::ModelSplitDAE, initial_state_outer, jacouter_prototype)

    # ODE function and Jacobian for 'outer' Variables, with inner Newton solve
    modelode = SplitDAE.ModelODEOuter(ms)
    jacode = SplitDAE.ModelJacOuter(ms, modelode)

    tss = Ref(NaN)
    deltat = Ref(NaN)
    previous_u = similar(initial_state_outer)
    du_worksp = similar(initial_state_outer)
 
    ssFJ! = FJacPTC(modelode, jacode, tss, deltat, previous_u, du_worksp)
        
    # function + sparse Jacobian with sparsity pattern defined by jac_prototype
    df = NLsolve.OnceDifferentiable(ssFJ!, ssFJ!, ssFJ!, similar(initial_state_outer), similar(initial_state_outer), copy(jacouter_prototype))         

    return (ssFJ!, df)
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
struct FJacPTC{M, J, W}
    modelode::M
    jacode::J
    t::Ref{Float64}
    delta_t::Ref{Float64}
    previous_u::W
    du_worksp::W
end

"""
    set_step!(fjp::FJacPTC, t, deltat, previous_u)

Set time to step to `t`, `delta_t` of this step, and `previous_u` (value of state vector at previous time step `t - delta_t`).
"""
function SteadyState.set_step!(fjp::FJacPTC, t, deltat, previous_u)
    fjp.t[] = t
    fjp.delta_t[] = deltat
    fjp.previous_u .= previous_u

    return nothing
end

# F only
function (jn::FJacPTC)(F, u)
    jn.modelode(jn.du_worksp, u, nothing, jn.t[])

    F .=  (u .- jn.previous_u - jn.delta_t[].*jn.du_worksp)

    # @Infiltrator.infiltrate

    return nothing
end

# Jacobian only (just discard F)
function (jn::FJacPTC)(J::SparseArrays.SparseMatrixCSC, u)
    jn(nothing, J, u)

    # @Infiltrator.infiltrate

    return nothing
end

# F and J
function (jn::FJacPTC)(F, J::SparseArrays.SparseMatrixCSC, u)
    
    jn.jacode(jn.du_worksp, J, u, nothing, jn.t[])

    if !isnothing(F)
        F .=  (u .- jn.previous_u - jn.delta_t[].*jn.du_worksp)
    end

    # @Infiltrator.infiltrate
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


end # module
