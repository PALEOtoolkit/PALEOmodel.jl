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
import ..SolverFunctions
import ..ODE
import ..JacobianAD
import ..NonLinearNewton

"""
    split_dae(
        model, initial_state, modeldata;
        jac_ad=:ForwardDiffSparse,
        tss_jac_sparsity=nothing,
        request_adchunksize=10,
        jac_cellranges=modeldata.cellranges_all,
        operatorID_inner=3,
        transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"],  # additional Variables needed by 'inner' Reactions
        init_logger=Logging.NullLogger(),
        inner_kwargs=(reltol=1e-9, miniters=1, maxiters=10, verbose=0),
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
    inner_kwargs=(reltol=1e-9, miniters=1, maxiters=10, verbose=0),
)
    PB.check_modeldata(model, modeldata)

    sva = modeldata.solver_view_all
    # We only support explicit ODE + DAE constraints (no implicit variables)
    iszero(PALEOmodel.num_total(sva))                 || error("implicit total variables not supported")
    !iszero(PALEOmodel.num_algebraic_constraints(sva)) || error("no algebraic constraints present")

    # dispatch list excluding costly Reactions (eg radiative transfer)
    # dispatchlists_jac = PB.create_dispatch_methodlists(model, modeldata, jac_cellranges)
    dispatchlists_jac = nothing # disable

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
    cellderivs, celljacs, cellidxfull, cellinitialstates = [], [], [], []
    cellworksp = fill(NaN, n_inner_solve)
    cellworksp_ad = fill(eltype_inner_ad(NaN), n_inner_solve)
   
    for i in cellrange_inner.indices
        cellrange = PB.CellRange(cellrange_inner.domain, cellrange_inner.operatorID, i)
        vars_cellranges = [cellrange for i in 1:length(va_state.vars)]
        # modeldata Arrays to calculate residual
        va_cell_state = PB.VariableAggregator(va_state.vars, vars_cellranges, modeldata)
        cell_initial_state = Vector{eltype(modeldata)}(undef, n_inner_solve)
        copyto!(cell_initial_state, va_cell_state)
        push!(cellinitialstates, cell_initial_state)
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
        dispatchlists_jac,
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
        [c for c in cellinitialstates], # narrow_type
        inner_kwargs,
        similar(initial_state_outer),
        similar(initial_state),
    )

    return (ms, initial_state_outer, jacouter_prototype)
end

# all Variables needed for inner and outer solve
mutable struct ModelSplitDAE{T, SVA, DLA, DLJ, JF, TD1, TD2, TD3, TD4, VA1, VA2, VA3, CD, CJ, IK}
    modeldata::PB.ModelData{T}
    solver_view_all::SVA
    dispatchlists_all::DLA
    dispatchlists_jac::DLJ
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
    cellinitialstates::Vector{Vector{Float64}}
    inner_kwargs::IK    
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
    cellidx = 1
    for (cd, cj, cs) in PB.IteratorUtils.zipstrict(ms.cellderivs, ms.celljacs, ms.cellinitialstates)
        # use current value (from previous iteration) as starting value
        # copyto!(cd.worksp, cd.state)
        # initial_state = StaticArrays.SVector{ncomps(cd)}(cd.worksp)
        # always start from initial state
        initial_state = StaticArrays.SVector{ncomps(cd)}(cs)
        ms.inner_kwargs.verbose > 0 && @info "cellidx: $cellidx initial_state: $initial_state"
        (_, Lnorm_2_cell, Lnorm_inf_cell, niter) = NonLinearNewton.solve(
            cd,
            cj, 
            initial_state;
            ms.inner_kwargs...
        )
        cellidx += 1 
    end

    # reevaluate full derivative with updated inner state variables
    # TODO this could be optimized eg don't need to rerun radiative transfer
    if !isnothing(ms.dispatchlists_jac)
        PB.do_deriv(ms.dispatchlists_jac)
    else
        PB.do_deriv(ms.dispatchlists_all)
    end
    
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
    # NB: this changes the size (stored elements) of J_outer !!
    # J_outer .= J_full[ro, ro]
    add_sparse_fixed!(J_outer, J_full[ro, ro]) # TODO use view

    # add contributions per-cell from inner Variables using implicit function theorum    
    for ci in ms.cellidxfull
        dG_dcellinner = view(J_full, ci, ci) # n_inner x n_inner (TODO could also get this from jac used for inner solve)
        dG_douter = J_full[ci, ro] # n_inner x n_outer - don't use view as view(sparse)*sparse is very slow
        
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
        # NB: this changes the size (stored elements) of J_outer !!
        # J_outer .+= J_full[ro, ci]*dcellinner_dcellouter 
        add_sparse_fixed!(J_outer, J_full[ro, ci]*dcellinner_dcellouter)
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




###############################################
# Sparse matrix additions
###############################################

"""
    add_sparse_fixed!(A, B)

Sparse matrix `A .+= B`, without modifying sparsity pattern of `A`.

Errors if `B` contains elements not in sparsity pattern of `A`

    julia> A = SparseArrays.sparse([1, 4], [2, 3], [1.0, 1.0], 4, 4)
    julia> B = SparseArrays.sparse([1], [2], [2.0], 4, 4)
    julia> SplitDAE.add_sparse_fixed!(A, B) # OK
    julia> C = SparseArrays.sparse([2], [2], [2.0], 4, 4)
    julia> SplitDAE.add_sparse_fixed!(A, C) # errors

TODO views, SubArray{Float64, 2, SparseArrays.SparseMatrixCSC{Float64, Int64}}
"""
function add_sparse_fixed!(A::SparseArrays.SparseMatrixCSC, B::SparseArrays.SparseMatrixCSC)
    size(A) == size(B) || error("A and B are not the same size")

    for j in 1:size(B, 2)
        idx_A = A.colptr[j]
        for idx_B in B.colptr[j]:(B.colptr[j+1]-1)
            i_B = B.rowval[idx_B]
            while idx_A < A.colptr[j+1] && A.rowval[idx_A] != i_B
                idx_A += 1
            end
            idx_A < A.colptr[j+1] || error("element [$i_B, $j] in B is not present in A")
            A.nzval[idx_A] += B.nzval[idx_B]
        end
    end
    return A
end



end # module
