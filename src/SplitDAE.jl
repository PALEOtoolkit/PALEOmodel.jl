module SplitDAE

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import SparseArrays
import ForwardDiff
import SparseDiffTools
import Infiltrator
using Logging
import StaticArrays
import TimerOutputs: @timeit, @timeit_debug

import NLsolve
import ..SolverFunctions
import ..ODE
import ..JacobianAD
import ..NonLinearNewton
import ..SparseUtils

"""
    create_split_dae(
        model, initial_state, modeldata;
        tss_jac_sparsity,
        request_adchunksize=10,
        jac_cellranges=modeldata.cellranges_all,
        operatorID_inner=0,
        transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"],  # additional Variables needed by 'inner' Reactions
        inner_jac_ad=:ForwardDiff: # form of automatic differentiation to use for Jacobian for inner solver (options `:ForwardDiff`, `:ForwardDiffSparse`)
        inner_kwargs=(verbose=0, miniters=2, reltol=1e-12, jac_constant=true, u_min=1e-60),
        generated_dispatch=true,
    ) -> (ms::ModelSplitDAE, initial_state_outer, jac_outer_prototype)

Given a model that contains both ODE variables and algebraic constraints, creates a callable struct `ms::`[`ModelSplitDAE`](@ref)
that calculates derivative and Jacobian for the 'outer' ODE variables, where the algebraic constraints are solved by an 'inner' Newton solver.

Requires that all algebraic constraints are 'local' ie per-cell, for a single Domain.

Uses the implicit function theorum to efficiently calculate the Jacobian for the 'outer' ODE variables from the full model Jacobian.

Subsets of Reactions can be defined for both the full model Jacobian and the 'inner' Newton solve. Two additional `modeldata` structs
are created to hold arrays with the correct ForwardDiff Dual number element type, one for the full model Jacobian and one for
the Jacobians needed for the 'inner' Newton solve. This means that it is necessary to specify any Variables calculated only by the
full model derivative that need to be copied before the Jacobian calculation or inner Newton solve:

- Jacobian: `jac_cellranges::Vector{<:PB.AbstractCellRange}` should provide Cellranges with an operatorID appropriate to calculate the full model Jacobian
  ('outer' and 'inner') variables (see [`PALEOmodel.JacobianAD.jac_config_ode`](@ref)). Variables with `transfer_jacobian` attribute set will
  be copied from `modeldata` after calculation of the full model derivative, allowing an approximate Jacobian to be defined
  by `jac_cellranges` with a non-default operatorID that omits expensive calculations.
- Inner Newton solve: `operatorID_inner` provides the operatorID for Reactions used for the 'inner' algebraic constraints. This will usually be a non-default value,
  so that only a subset of Reactions are used. Variables with `transfer_jacobian` attribute set or in the `transfer_inner_vars` list will
  be copied from `modeldata` after calculation of the full model derivative.
"""
function create_split_dae(
    model, initial_state, modeldata;
    tss_jac_sparsity,
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    jac_cellranges::Vector{<:PB.AbstractCellRange}=modeldata.cellranges_all,    
    operatorID_inner=0,
    transfer_inner_vars=["tmid", "volume", "ntotal", "Abox"],  # additional Variables needed by 'inner' Reactions
    inner_jac_ad=:ForwardDiff,
    inner_kwargs=(verbose=0, miniters=2, reltol=1e-12, jac_constant=true, u_min=1e-60),
    generated_dispatch=true,
)

    PB.check_modeldata(model, modeldata)

    # Notation:
    #
    # 'outer' Variables are PALEO stateexplicit, stateexplicit_deriv (paired ODE derivatives, with :vfunction=PB.VF_StateExplicit, PB.VF_Deriv)
    # 'inner' Variables are PALEO state, constraint (DAE state and algebraic constraints, with :vfunction=PB.VF_StateExplicit, PB.VF_Deriv)
    #
    # Three modeldata array sets are used here, with corresponding dispatchlists and variable aggregators:
    #
    # arrays_idx=1:         base Float64 elements
    # arrays_idx=2:         Dual numbers needed for full Jacobian
    # arrays_idx=3:         Dual numbers needed for per-cell Jacobian

    eltype_base = eltype(modeldata, 1)  # eg Float64, element type of base modeldata arrays

    sva = modeldata.solver_view_all
    # We only support explicit ODE + DAE constraints (no implicit variables)
    iszero(PALEOmodel.num_total(sva))                 || error("implicit total variables not supported")
    !iszero(PALEOmodel.num_algebraic_constraints(sva)) || error("no algebraic constraints present")

    # dispatch list to recalculate derivative after inner solve. Could exclude costly Reactions (eg radiative transfer)
    @timeit "dispatchlists_recalc_deriv" begin
    dispatchlists_recalc_deriv = PB.create_dispatch_methodlists(model, modeldata, 1, jac_cellranges; generated_dispatch)
    # dispatchlists_recalc_deriv = nothing # disable
    end # timeit

    # Variable aggegators for 'outer' ODE variables only
    va_stateexplicit = sva.stateexplicit
    va_stateexplicit_deriv = sva.stateexplicit_deriv
    # calculate initial_state_outer
    PALEOmodel.set_statevar!(sva, initial_state)
    initial_state_outer = Vector{eltype_base}(undef, length(va_stateexplicit))
    copyto!(initial_state_outer, va_stateexplicit)

    
    # find all (state, constraint) algebraic constraints
    io = IOBuffer()
   
    va_state = sva.state
    va_constraints = sva.constraints
    n_inner_solve = length(va_state.vars)
    println(io, "create_split_dae:  $n_inner_solve algebraic state and constraint Variables for inner Newton solve:")
    for (var, indices) in zip(va_state.vars, va_state.indices)
        println(io, "        $(PB.fullname(var))  indices $indices")
    end
    for (var, indices) in zip(va_constraints.vars, va_constraints.indices)
        println(io, "        $(PB.fullname(var))  indices $indices")
    end
    # find Domain for inner solve
    inner_domains =unique([var.domain for var in va_state.vars])
    length(inner_domains) == 1 || error("TODO only 1 Domain with algebraic constraints supported, Domains=$inner_domains")
    inner_domain = inner_domains[1]
    # find cellrange for inner Domain
    cellrange_inner = PB.Grids.create_default_cellrange(inner_domain, inner_domain.grid, operatorID=operatorID_inner)
    println(io, "    inner Newton solve Domain $inner_domain, cellrange $cellrange_inner")
    @info String(take!(io))


    ###########################################################
    # construct function objects to calculate per-cell constraint Variables
    ##################################
    cellcellranges, cellderivs, cellstateidxfull, cellconstraintsidxfull, cellinitialstates = [], [], [], [], [], []
    cellworksp = fill(NaN, n_inner_solve)
   
    @timeit "cell_functions" begin
    for i in cellrange_inner.indices
        # create a cellrange that refers to this one cell
        cellrange = PB.CellRange(cellrange_inner.domain, cellrange_inner.operatorID, i)
        push!(cellcellranges, cellrange)
        
        # VariableAggregator to calculate state and constraints for this cell
        vars_cellranges = fill(cellrange, n_inner_solve)
        va_cell_state = PB.VariableAggregator(va_state.vars, vars_cellranges, modeldata, 1)
        va_cell_constraints = PB.VariableAggregator(va_constraints.vars, vars_cellranges, modeldata, 1)
        # read out initial value for state variables
        cell_initial_state = Vector{eltype_base}(undef, n_inner_solve)
        copyto!(cell_initial_state, va_cell_state)
        push!(cellinitialstates, cell_initial_state)
        # dispatchlists for this cell
        va_cell_dl = PB.create_dispatch_methodlists(model, modeldata, 1, [cellrange]; generated_dispatch)
        push!(cellderivs, ModelDerivCell(n_inner_solve, modeldata, va_cell_state, va_cell_constraints, va_cell_dl, cellworksp))
       
        # index in full model for per-cell Variables
        push!(cellstateidxfull, [indices[i] + length(va_stateexplicit) for indices in va_state.indices])
        push!(cellconstraintsidxfull, [indices[i] + length(va_stateexplicit_deriv) for indices in va_constraints.indices])
    end
    end # timeit
    ##########################################################################################    
    # Get function object and sparsity pattern for full sparse jacobian 
    # jacfull(Jfull, u, p, t) where Jfull has sparsity pattern jacfull_prototype
    ###############################################################################

    @info "create_split_dae:  creating full Jacobian using :ForwardDiffSparse"
    @timeit "jacfull" begin
    jacfull, jacfull_prototype = PALEOmodel.JacobianAD.jac_config_ode(
        :ForwardDiffSparse, model, initial_state, modeldata, tss_jac_sparsity;
        request_adchunksize,
        jac_cellranges,
        fill_jac_diagonal=false,
        generated_dispatch,
    )
    end # timeit

    #########################################################################################
    # Calculate Jacobian sparsity patterns for outer and inner Jacobians from full Jacobian
    #########################################################################################
    @timeit "sparsity_patterns" begin
    # sparsity pattern for outer Jacobian
    ijrange_outer = 1:length(va_stateexplicit) # indices of 'outer' variables
    # ri = (length(va_stateexplicit)+1):length(initial_state)
    jacouter_prototype = jacfull_prototype[ijrange_outer, ijrange_outer]

    # include contributions from inner state Variables    
    jacouter_implicit = SparseArrays.spzeros(size(jacouter_prototype))
    dG_dcellinner_lu = nothing
    jacinner_prototype = nothing
   
    celldGdoutercols = []
    for (cidx, sci, cci) in zip(cellrange_inner.indices, cellstateidxfull, cellconstraintsidxfull)
        dG_dcellinner = jacfull_prototype[cci, sci] # n_inner x n_inner (sparsity pattern of Jacobian for inner solve)
        if isnothing(jacinner_prototype)
            jacinner_prototype = copy(dG_dcellinner)
        else
            # check sparsity pattern
            dG_dcellinner == jacinner_prototype || error("cell $cidx sparsity pattern for inner Jacobian does not match")
        end

        dG_douter = jacfull_prototype[cci, ijrange_outer] # n_inner x n_outer
        # optimization - record nonzero columns
        dG_douter_cols = Int64[j for j in ijrange_outer if SparseArrays.nnz(dG_douter[:, j-first(ijrange_outer)+1]) > 0]
        push!(celldGdoutercols, dG_douter_cols)
        # try and generate a non-singular matrix by setting values
        for i in eachindex(dG_dcellinner.nzval)
            dG_dcellinner.nzval[i] = i
        end
        # dcellinner_dcellouter = -dG_dcellinner \ dG_douter # n_inner x n_outer
        if isnothing(dG_dcellinner_lu)
            dG_dcellinner_lu = LinearAlgebra.lu(dG_dcellinner)
        else
            LinearAlgebra.lu!(dG_dcellinner_lu, dG_dcellinner)
        end
        # \ is not implemented, so as a workaround get an inverse with the correct sparsity
        dG_dcellinner_inv = SparseUtils.get_sparse_inverse(dG_dcellinner)

        @debug "cell $cidx dG_dcellinner dG_dcellinner_inv" dG_dcellinner dG_dcellinner_inv

        # @Infiltrator.infiltrate
        dcellinner_dcellouter = -dG_dcellinner_inv * dG_douter # n_inner x n_outer

        # n_outer x n_outer  +=  n_outer x n_inner  * n_inner x n_outer
        jacouter_implicit += jacfull_prototype[ijrange_outer, sci]*dcellinner_dcellouter

        
    
    end

    # combine to get full sparsity pattern
    io = IOBuffer()
    println(io, "create_split_dae:")
    println(io, "    calculating 'outer' Jacobian sparsity pattern:")
    println(io, "        jacouter (outer variables, $(size(jacouter_prototype))): nnz=$(SparseArrays.nnz(jacouter_prototype))")
    println(io, "        jacouter_implicit (from inner variables): nnz=$(SparseArrays.nnz(jacouter_implicit))")
    jacouter_implicit.nzval .= 1.0  # fill with 1.0 to avoid any risk of spurious cancellations
    jacouter_prototype += jacouter_implicit
    println(io, "        jacouter (all variables):")
    jacouter_prototype = SparseUtils.fill_sparse_jac(jacouter_prototype; val=1.0, fill_diagonal=true)

    end # timeit
    #########################################################################
    # Add array set to modeldata, with Dual numbers for AD Jacobians for 'inner' Newton solve
    #####################################################################

    if inner_jac_ad == :ForwardDiff
        println(io, "    using ForwardDiff dense Jacobian for inner Newton solve")   
        eltype_jaccell = ForwardDiff.Dual{Nothing, eltype_base, n_inner_solve}

    elseif inner_jac_ad == :ForwardDiffSparse
       
        colorvec = SparseDiffTools.matrix_colors(jacinner_prototype)
        jacinner_chunksize = ForwardDiff.pickchunksize(maximum(colorvec), request_adchunksize)
        eltype_jaccell = ForwardDiff.Dual{Nothing, eltype_base, jacinner_chunksize}
        
        println(io, "    using ForwardDiffSparse Jacobian for inner Newton solve colors $(maximum(colorvec)) chunksize $jacinner_chunksize") 

        jacinner_cache = SparseDiffTools.ForwardColorJacCache(
            nothing, cellworksp, jacinner_chunksize;
            dx = nothing, # not needed for square Jacobian
            colorvec=colorvec,
            sparsity = copy(jacinner_prototype)
        )
    else
        error("unrecognized inner_jac_ad $inner_jac_ad")
    end

    @timeit "add_arrays_data! jaccell" begin
        PB.add_arrays_data!(model, modeldata, eltype_jaccell, "jac_cell"; use_base_vars=transfer_inner_vars)
        arrays_idx_jaccell = PB.num_arrays(modeldata)
        jaccell_solverview = PALEOmodel.SolverView(model, modeldata, arrays_idx_jaccell) # Variables from whole model
        # jaccell_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, jac_cellranges; generated_dispatch)
    end # timeit
    va_stateexplicit_jaccell = jaccell_solverview.stateexplicit

    @info String(take!(io))

    ###########################################################
    # construct function objects to calculate per-cell Jacobian for 'inner' Variables
    ##################################
    jaccells = []
    cellworksp_jaccell = fill(eltype_jaccell(NaN), n_inner_solve)
    cellworksp_jacsparse = copy(jacinner_prototype)
    @timeit "cell_jacobians" begin
    for cellrange in cellcellranges
        # Variable Aggegators to calculate constraints, with appropriate Dual number element type for per-cell Jacobian
        vars_cellranges = fill(cellrange, n_inner_solve)
        va_cell_state_jaccell = PB.VariableAggregator(va_state.vars, vars_cellranges, modeldata, arrays_idx_jaccell)
        va_cell_constraints_jaccell = PB.VariableAggregator(va_constraints.vars, vars_cellranges, modeldata, arrays_idx_jaccell)
        va_cell_dl_jaccell = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jaccell, [cellrange]; generated_dispatch)

        # function object to calculate constraints, with appropriate Dual number element type for per-cell Jacobian
        constraintscell = ModelDerivCell(
            n_inner_solve, modeldata, va_cell_state_jaccell, va_cell_constraints_jaccell, va_cell_dl_jaccell, cellworksp_jaccell
        )

        # function object to calculate per-cell Jacobian
        if inner_jac_ad == :ForwardDiff
            jaccell = ModelJacForwardDiffCell(constraintscell)
        elseif inner_jac_ad == :ForwardDiffSparse
            jaccell = ModelJacForwardDiffSparseCell(
                constraintscell,
                cellworksp_jacsparse,
                jacinner_cache,
                dG_dcellinner_lu
            )
        else
            error("unrecognized inner_jac_ad $inner_jac_ad")
        end

        push!(jaccells, jaccell)
    end
    end # timeit
    
    @timeit "ms" ms = ModelSplitDAE(
        modeldata,
        sva,
        modeldata.dispatchlists_all,
        dispatchlists_recalc_deriv,
        jacfull,
        copy(jacfull_prototype),
        va_stateexplicit,
        va_stateexplicit_deriv,
        va_stateexplicit_jaccell,
        copy(collect(cellrange_inner.indices)),
        [c for c in cellderivs], # narrow type
        [c for c in jaccells], # narrow type
        [c for c in cellstateidxfull], # narrow_type
        [c for c in cellconstraintsidxfull], # narrow_type
        [c for c in cellinitialstates], # narrow_type
        [c for c in celldGdoutercols], # narrow type
        inner_kwargs,
        similar(initial_state_outer),
        similar(initial_state),
        dG_dcellinner_lu,
        cellworksp_jacsparse,
    )

    return (ms, initial_state_outer, jacouter_prototype)
end

"""
    ModelSplitDAE

Callable struct with all Variables needed for inner Newton solve, and outer derivative and Jacobian

Provides functions for outer derivative and Jacobian:

    (ms::ModelSplitDAE)(du_outer::AbstractVector, u_outer::AbstractVector, p, t)

    (ms::ModelSplitDAE)(J_outer::SparseArrays.AbstractSparseMatrixCSC, u_outer::AbstractVector, p, t)

    (ms::ModelSplitDAE)(du_outer::AbstractVector, J_outer::SparseArrays.AbstractSparseMatrixCSC, u_outer::AbstractVector, p, t)    

"""
struct ModelSplitDAE{T, SVA, DLA, DLR, JF, VA1, VA2, VA3, CD, CJ, IK, LU}
    modeldata::PB.ModelData
    solver_view_all::SVA
    dispatchlists_all::DLA
    dispatchlists_recalc_deriv::DLR
    jacfull::JF
    jacfull_ws::SparseArrays.SparseMatrixCSC{T, Int64}
    va_stateexplicit::VA1
    va_stateexplicit_deriv::VA2
    va_stateexplicit_jaccell::VA3
    cellindex::Vector{Int64}
    cellderivs::CD
    jaccells::CJ    
    cellstateidxfull::Vector{Vector{Int64}}
    cellconstraintsidxfull::Vector{Vector{Int64}}
    cellinitialstates::Vector{Vector{Float64}}
    celldGdoutercols::Vector{Vector{Int64}}
    inner_kwargs::IK    
    outer_worksp::Vector{T}
    full_worksp::Vector{T}
    dG_dcellinner_lu::LU
    jacinner_ws::SparseArrays.SparseMatrixCSC{T, Int64}
end

# calculate ODE derivative for outer Variables, with inner Newton solve for algebraic constraints
function (ms::ModelSplitDAE)(du_outer::AbstractVector, u_outer::AbstractVector, p, t)
    
    # set outer state Variables
    copyto!(ms.va_stateexplicit, u_outer)
    # NB: inner state Variables are set below

    # full derivative, with old values of inner state variables  
    PALEOmodel.set_tforce!(ms.solver_view_all, t)
    PB.do_deriv(ms.dispatchlists_all)

    # always set all outer state variables (inner state variables will be solved for)
    copyto!(ms.va_stateexplicit_jaccell, u_outer)

    # Newton solution for inner state Variables for each cell
    for (ci, cd, cj, cs) in PB.IteratorUtils.zipstrict(ms.cellindex, ms.cellderivs, ms.jaccells, ms.cellinitialstates)
        # use current value (from previous iteration) as starting value
        # copyto!(cd.worksp, cd.state)
        # initial_state = StaticArrays.SVector{ncomps(cd)}(cd.worksp)
        # always start from initial state
        initial_state = StaticArrays.SVector{ncomps(cd)}(cs)
        ms.inner_kwargs.verbose > 0 && @info "cell index: $ci initial_state: $initial_state"
        (_, Lnorm_2_cell, Lnorm_inf_cell, niter) = NonLinearNewton.solve(
            cd,
            cj, 
            initial_state;
            ms.inner_kwargs...
        )
    end

    # reevaluate full derivative with updated inner state variables
    # use dispatchlists_recalc_deriv if available (an optimization, eg don't need to rerun radiative transfer)
    if !isnothing(ms.dispatchlists_recalc_deriv)
        PB.do_deriv(ms.dispatchlists_recalc_deriv)
    else
        PB.do_deriv(ms.dispatchlists_all)
    end
    
    copyto!(du_outer, ms.va_stateexplicit_deriv)

    return nothing
end


# calculate ODE 'outer' Jacobian, with Newton inner solve
function (ms::ModelSplitDAE)(J_outer::SparseArrays.AbstractSparseMatrixCSC, u_outer::AbstractVector, p, t)
    # just forward to function that calculate derivative and Jacobian, then discard derivative
    du_outer = ms.outer_worksp # not used
    ms(du_outer, J_outer, u_outer, p, t)
    return nothing
end

# calculate ODE 'outer' derivative and Jacobian, with Newton inner solve
function (ms::ModelSplitDAE)(du_outer::AbstractVector, J_outer::SparseArrays.AbstractSparseMatrixCSC, u_outer::AbstractVector, p, t)
 
    # calculate derivative including solution for 'inner' state Variables
    ms(du_outer, u_outer, p, t)

    # calculate full Jacobian
    u_full = ms.full_worksp
    PALEOmodel.get_statevar!(u_full, ms.solver_view_all)
    J_full = ms.jacfull_ws
    J_full.nzval .= 0.0
    ms.jacfull(J_full, u_full, p, t)

    # calculate Jacobian of outer Variables 'J_outer' using implicit function theorum
    J_outer.nzval .= 0.0
    ijrange_outer = 1:size(J_outer, 1)
    # NB: .= changes the size (stored elements) of J_outer !!
    # J_outer .= J_full[ro, ro]
    SparseUtils.add_sparse_fixed!(J_outer, (@view J_full[ijrange_outer, ijrange_outer]))

    # add contributions per-cell from inner Variables using implicit function theorum
    
    # pseudocode for simple version
    # (in Julia 1.7 this hits some unimplemented operations and slow fallbacks)
    #
    # for (sci, cci) in zip(ms.cellstateidxfull, ms.cellconstraintsidxfull)
    #     dG_dcellinner = J_full[cci, sci] # n_inner x n_inner (TODO could also get this from jac used for inner solve)
    #
    #     dG_douter = J_full[cci, ijrange_outer] # n_inner x n_outer
    #    
    #     dcellinner_dcellouter = -dG_dcellinner \ dG_douter # n_inner x n_outer (not implemented in Julia 1.7)
    #
    #     n_outer x n_outer  +=  n_outer x n_inner  * n_inner x n_outer
    #     J_outer .+= J_full[ijrange_outer, sci]*dcellinner_dcellouter   # modifies sparsity pattern of J_outer !
    # end

    # per-column version
    dG_douter = zeros(length(first(ms.cellstateidxfull)))
    dcellinner_dcellouter = similar(dG_douter)
    # tmpcol = ms.outer_worksp # length(ro)
    tmpcol = SparseUtils.SparseVecAccum()
    for (sci, cci, dG_douter_cols) in zip(ms.cellstateidxfull, ms.cellconstraintsidxfull, ms.celldGdoutercols)
        dG_dcellinner = J_full[cci, sci] # n_inner x n_inner (TODO could also get this from jac used for inner solve)

        LinearAlgebra.lu!(ms.dG_dcellinner_lu, dG_dcellinner)
        # per-column loop
        for j_outer in dG_douter_cols
            # dG_douter = J_full[cci, j_outer]
            if !iszero(SparseUtils.get_column_sparse!(dG_douter, (@view J_full[cci, j_outer]))) # view is slow
            # if !iszero(SparseUtils.get_column_sparse!(dG_douter, J_full, cci, j_outer))
        
                dcellinner_dcellouter .= ms.dG_dcellinner_lu \ dG_douter # n_inner
                dcellinner_dcellouter .*= -1.0

                # nouter x 1 = nouter x ninner * ninner
                # tmpcol .= J_full[ijrange_outer, sci]*dcellinner_dcellouter
                SparseUtils.mult_sparse_vec!(tmpcol, (@view J_full[ijrange_outer, sci]), dcellinner_dcellouter)
                # J_outer[:, j_outer] += tmpcol
                SparseUtils.add_column_sparse_fixed!((@view J_outer[:, j_outer]), tmpcol)
            end
        end
    end

    return nothing
end

# calculate derivative for a single cell
mutable struct ModelDerivCell{Ncomps, T, VA1 <: PB.VariableAggregator, VA2 <: PB.VariableAggregator, D}
    modeldata::PB.ModelData
    state::VA1
    deriv::VA2
    dispatchlists::D
    worksp::Vector{T}
    t::Float64
end

function ModelDerivCell(Ncomps, modeldata::PB.ModelData, state::VA1, deriv::VA2, dispatchlists::D, worksp::Vector{T}) where {T, VA1 <: PB.VariableAggregator, VA2 <: PB.VariableAggregator, D}
    return ModelDerivCell{Ncomps, T, VA1, VA2, D}(modeldata, state, deriv, dispatchlists, worksp, NaN)
end

ncomps(md::ModelDerivCell{Ncomps, T, VA1, VA2, D}) where {Ncomps, T, VA1, VA2, D} = Ncomps

# out-of-place SVector
function (md::ModelDerivCell)(x::StaticArrays.SVector)
    # @Infiltrator.infiltrate
    copyto!(md.state, x)
    PB.do_deriv(md.dispatchlists, 0.0)
    copyto!(md.worksp, md.deriv)

    return StaticArrays.SVector{ncomps(md)}(md.worksp)
end

# in place Vector
function (md::ModelDerivCell)(y::AbstractVector, x::AbstractVector)
    # @Infiltrator.infiltrate
    copyto!(md.state, x)
    PB.do_deriv(md.dispatchlists, 0.0)
    copyto!(y, md.deriv)

    return nothing
end

# calculate Jacobian for a single cell
struct ModelJacForwardDiffCell{MD}
    modelderiv::MD
end

# TODO ForwardDiff doesn't provide an API to get jacobian without setting Dual number 'tag'
@static if isdefined(ForwardDiff, :extract_jacobian)
    const _forwarddiff_extract_jacobian = ForwardDiff.extract_jacobian # ForwardDiff v < 0.10.35
elseif isdefined(ForwardDiff, :ForwardDiffStaticArraysExt) && isdefined(ForwardDiff.ForwardDiffStaticArraysExt, :extract_jacobian) # ForwardDiff >= 0.10.35, Julia < 1.9
    const _forwarddiff_extract_jacobian = ForwardDiff.ForwardDiffStaticArraysExt.extract_jacobian
else
    # ForwardDiff >= 0.10.35, Julia >= 1.9
    const _forwarddiff_extract_jacobian = Base.get_extension(ForwardDiff, :ForwardDiffStaticArraysExt).extract_jacobian
end

function (mjfd::ModelJacForwardDiffCell)(x::StaticArrays.SVector)
    # TODO ForwardDiff doesn't provide an API to get jacobian without setting Dual number 'tag'
    return _forwarddiff_extract_jacobian(Nothing, ForwardDiff.static_dual_eval(Nothing, mjfd.modelderiv, x), x)
end

# calculate lu factorization of sparse Jacobian for a single cell
struct ModelJacForwardDiffSparseCell{MD, JWS <: SparseArrays.SparseMatrixCSC, JC <: SparseDiffTools.ForwardColorJacCache, L}
    modelderiv::MD
    jac_ws::JWS
    jac_cache::JC
    jac_lu::L
end

function (mjfd::ModelJacForwardDiffSparseCell)(x::AbstractVector)
    SparseDiffTools.forwarddiff_color_jacobian!(
        mjfd.jac_ws,
        mjfd.modelderiv,
        x,
        mjfd.jac_cache,
    )

    LinearAlgebra.lu!(mjfd.jac_lu, mjfd.jac_ws)

    return mjfd.jac_lu
end




end # module
