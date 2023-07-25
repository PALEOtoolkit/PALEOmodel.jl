module JacobianAD

import PALEOboxes as PB

import PALEOmodel
import ..SolverFunctions
import ..SparseUtils

import LinearAlgebra
import Infiltrator

import Logging

import ForwardDiff
import SparseArrays
import SparseDiffTools
import SparsityTracing

import TimerOutputs: @timeit, @timeit_debug

# moved to SolverFunctions
# Base.@deprecate_binding DerivForwardDiff  SolverFunctions.DerivForwardDiff
# Base.@deprecate_binding JacODEForwardDiffDense  .. etc
# Base.@deprecate_binding JacODEForwardDiffSparse
# Base.@deprecate_binding ImplicitDerivForwardDiff
# Base.@deprecate_binding ImplicitForwardDiffDense
# Base.@deprecate_binding ImplicitForwardDiffSparse
# Base.@deprecate_binding JacDAE SolverFunctions.JacDAE

######################################################
# ODE Jacobians using ForwardDiff
########################################################

"""
    jac_config_ode(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        kwargs...
    )-> (jac, jac_prototype::SparseMatrixCSC)

Create and return `jac` (ODE Jacobian function object), and `jac_prototype` (sparsity pattern as `SparseMatrixCSC`, or `nothing` for dense Jacobian)

`jac_ad` defines Jacobian type (:ForwardDiffSparse, :ForwardDiff)

Adds an array set to `modeldata` with appropriate datatypes for ForwardDiff AD Dual numbers,
sets up cache for ForwardDiff, calculates Jacobian sparsity (if required) at time  `jac_ad_t_sparsity`.

NB: there is a profusion of different Julia APIs here:
- ForwardDiff Sparse and dense Jacobian use different APIs and have different cache setup requirements.
- ForwardDiff requires f!(du, u) hence a closure or function object, DifferentialEquations allows context objects to be passed around.

# Keyword arguments
- `jac_cellranges=modeldata.cellranges_all`: restrict Jacobian to this subset of Domains and Reactions (via operatorID).
- `request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD`:  chunk size for `ForwardDiff` automatic differentiation
- `fill_jac_diagonal=true`: (`jac=:ForwardDiffSparse` only) true to fill diagonal of `jac_prototype`
- `generated_dispatch=true`: `true` to autogenerate code for dispatch (fast dispatch, slow compile)
- `use_base_vars=String[]`: additional Variable full names not calculated by Jacobian, which instead use arrays from `modeldata` base arrays (arrays_idx=1) instead of allocating new AD Variables
"""
function jac_config_ode(
    jac_ad::Symbol, model::PB.Model, initial_state, modeldata::PB.ModelData, jac_ad_t_sparsity;
    jac_cellranges=modeldata.cellranges_all,
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    fill_jac_diagonal=true,
    generated_dispatch=true,
    use_base_vars=String[],
)
    @info "jac_config_ode: jac_ad=$jac_ad"

    PB.check_modeldata(model, modeldata)

    iszero(PALEOmodel.num_total(modeldata.solver_view_all)) ||
        throw(ArgumentError("model contains implicit variables, solve as a DAE"))
   
    pack_domain = modeldata.solver_view_all.pack_domain

    # generate arrays with ODE layout for model Variables
    state_sms_vars_data = similar(PALEOmodel.get_statevar_sms(modeldata.solver_view_all))
    state_vars_data = similar(PALEOmodel.get_statevar(modeldata.solver_view_all))

    if jac_ad == :NoJacobian
        return (nothing, nothing)

    elseif jac_ad == :ForwardDiff       

        chunk = ForwardDiff.Chunk(length(state_sms_vars_data), request_adchunksize)
 
        jacconf = ForwardDiff.JacobianConfig(nothing, state_sms_vars_data, state_vars_data, chunk)

        PB.add_arrays_data!(model, modeldata, eltype(jacconf), "jac_ad"; use_base_vars)
        arrays_idx_jac_ad = PB.num_arrays(modeldata)
        jac_solverview = PALEOmodel.SolverView(model, modeldata, arrays_idx_jac_ad; pack_domain) # Variables from whole model
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, jac_cellranges; generated_dispatch)

        @info "  using ForwardDiff dense Jacobian chunksize=$(ForwardDiff.chunksize(chunk)))"
     
        du_template = similar(state_sms_vars_data)
        
        jac = SolverFunctions.JacODEForwardDiffDense(
            modeldata, 
            jac_solverview, # use all Variables in model
            jac_dispatchlists, # use only Reactions specified
            du_template, 
            jacconf
        )

        return (jac, nothing)
        
    elseif jac_ad == :ForwardDiffSparse
        !isnothing(initial_state) || 
            throw(ArgumentError("initial_state must be supplied for Jacobian $jac_ad"))
        !isnothing(jac_ad_t_sparsity) || 
            throw(ArgumentError("jac_ad_t_sparsity must be supplied for Jacobian $jac_ad"))
        # Use SparsityTracing to calculate sparsity pattern
        @info "  using ForwardDiff sparse Jacobian with sparsity calculated at t=$jac_ad_t_sparsity"  
        @timeit "calcJacobianSparsitySparsityTracing!" begin
        jac_proto_unfilled = calcJacobianSparsitySparsityTracing!(
            model, modeldata, initial_state, jac_ad_t_sparsity;
            jac_cellranges, use_base_vars, pack_domain,
        ) 
        end # timeit
        jac_prototype = SparseUtils.fill_sparse_jac(jac_proto_unfilled; fill_diagonal=fill_jac_diagonal)
        # println("using jac_prototype: ", jac_prototype)
       
        colorvec = SparseDiffTools.matrix_colors(jac_prototype)

        chunksize = ForwardDiff.pickchunksize(maximum(colorvec), request_adchunksize)
        @timeit "jac_ad add_arrays_data!" begin
        PB.add_arrays_data!(model, modeldata, ForwardDiff.Dual{Nothing, eltype(modeldata, 1), chunksize}, "jac_ad"; use_base_vars)
        arrays_idx_jac_ad = PB.num_arrays(modeldata)
        end # timeit
        @timeit "jac_solverview" begin
        jac_solverview = PALEOmodel.SolverView(model, modeldata, arrays_idx_jac_ad; pack_domain) # Variables from whole model
        end # timeit
        @timeit "jac_dispatchlists" begin
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, jac_cellranges; generated_dispatch)
        end # timeit
        @info "  jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec)) "*
            "chunksize=$chunksize)"    

        jac_cache = SparseDiffTools.ForwardColorJacCache(
            nothing, initial_state, chunksize;
            dx = nothing, # similar(state_sms_vars_data)
            colorvec=colorvec,
            sparsity = copy(jac_prototype)
        )

        @timeit "JacODEForwardDiffSparse" jac = SolverFunctions.JacODEForwardDiffSparse(
            modeldata, 
            jac_solverview, # use all Variables in model
            jac_dispatchlists, # use only Reactions specified
            jac_cache,
        )

        return jac, jac_prototype
    else
        error("unknown jac_ad=", jac_ad)
    end
    
    error("coding error, not reachable reached")
end




###################################################################
# DAE Jacobians using ForwardDiff
######################################################################

"""
    jac_config_dae(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        kwargs...
    ) -> (jac, jac_prototype, odeimplicit)

See [`jac_config_ode`](@ref) for keyword arguments.
"""
function jac_config_dae(
    jac_ad::Symbol, model::PB.Model, initial_state, modeldata, jac_ad_t_sparsity;
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    jac_cellranges=modeldata.cellranges_all,
    implicit_cellranges=modeldata.cellranges_all,
    generated_dispatch=true,
    use_base_vars=String[],
)
    @info "jac_config_dae: jac_ad=$jac_ad"

    PB.check_modeldata(model, modeldata)

    pack_domain = modeldata.solver_view_all.pack_domain
    
    # generate arrays with ODE layout for model Variables
    state_sms_vars_data = similar(PALEOmodel.get_statevar_sms(modeldata.solver_view_all))
    state_vars_data = similar(PALEOmodel.get_statevar(modeldata.solver_view_all))

    if jac_ad == :NoJacobian
        # check for implicit total variables
        PALEOmodel.num_total(modeldata.solver_view_all) == 0 ||
            error("implicit total variables - Jacobian required")

        return (nothing, nothing, nothing)

    elseif jac_ad == :ForwardDiff  
        chunk = ForwardDiff.Chunk(length(state_sms_vars_data), request_adchunksize)

        @info "  using ForwardDiff dense Jacobian chunksize=$(ForwardDiff.chunksize(chunk)))"
      
        jacconf = ForwardDiff.JacobianConfig(nothing, state_sms_vars_data, state_vars_data, chunk)
        PB.add_arrays_data!(model, modeldata, eltype(jacconf), "jac_ad"; use_base_vars)
        arrays_idx_jac_ad = PB.num_arrays(modeldata)
        jac_solverview = PALEOmodel.SolverView(model, modeldata, arrays_idx_jac_ad; pack_domain) # Variables from whole model
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, jac_cellranges; generated_dispatch)

        if iszero(PALEOmodel.num_total(modeldata.solver_view_all))
            odeimplicit = nothing     
        else 
            @info "  calculating dTdS for $(PALEOmodel.num_total(modeldata.solver_view_all)) Total Variables"

            duds = zeros(eltype(modeldata, 1), length(modeldata.solver_view_all.total), length(state_sms_vars_data))
            duds_template = similar(PB.get_data(modeldata.solver_view_all.total))
            implicitconf = ForwardDiff.JacobianConfig(
                nothing, duds_template, state_vars_data, chunk,
            )

            implicit_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, implicit_cellranges; generated_dispatch)
            odeimplicit = SolverFunctions.ImplicitForwardDiffDense(modeldata, jac_solverview, implicit_dispatchlists, duds_template, implicitconf, duds)
        end

        du_template = similar(state_sms_vars_data)

        jac = SolverFunctions.JacDAE(
            SolverFunctions.JacODEForwardDiffDense(
                modeldata, 
                jac_solverview, 
                jac_dispatchlists,
                du_template, 
                jacconf
            ),
            odeimplicit
        )

        return (jac, nothing, odeimplicit)

    elseif jac_ad == :ForwardDiffSparse       
        # Use SparsityTracing to calculate sparsity pattern
        !isnothing(initial_state) || 
            throw(ArgumentError("initial_state must be supplied for Jacobian $jac_ad"))
        !isnothing(jac_ad_t_sparsity) || 
            throw(ArgumentError("jac_ad_t_sparsity must be supplied for Jacobian $jac_ad"))

        @info "  using ForwardDiff sparse Jacobian with sparsity calculated at t=$jac_ad_t_sparsity"  
        jac_proto_unfilled = calcJacobianSparsitySparsityTracing!(
            model, modeldata, initial_state, jac_ad_t_sparsity;
            jac_cellranges, use_base_vars, pack_domain,
        ) 
        jac_prototype = SparseUtils.fill_sparse_jac(jac_proto_unfilled)
        # println("using jac_prototype: ", jac_prototype)
       
        colorvec = SparseDiffTools.matrix_colors(jac_prototype)
        
        @info "  jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec))"

        if !iszero(PALEOmodel.num_total(modeldata.solver_view_all))
            @info "  calculating dTdS for $(PALEOmodel.num_total(modeldata.solver_view_all)) Total Variables"

            implicit_proto_unfilled = calcImplicitSparsitySparsityTracing!(
                model, modeldata, initial_state, jac_ad_t_sparsity;
                implicit_cellranges, use_base_vars, pack_domain,
            )        
            implicit_prototype = SparseUtils.fill_sparse_jac(implicit_proto_unfilled, fill_diagonal=false)
            implicit_colorvec = SparseDiffTools.matrix_colors(implicit_prototype)
            @info "  implicit_prototype nnz=$(SparseArrays.nnz(implicit_prototype)) num colors=$(maximum(implicit_colorvec))"

            # add sparsity patterns to get DAE Jacobian sparsity
            # TODO this simplifies the DAE Jacobian code but will overestimate the sparsity pattern for the ODE part of the Jacobian calculation
            iistrt = length(modeldata.solver_view_all.stateexplicit) + 1
            iiend = iistrt+PALEOmodel.num_total(modeldata.solver_view_all)-1
            jac_prototype[iistrt:iiend, :] += implicit_prototype
            colorvec = SparseDiffTools.matrix_colors(jac_prototype)
            
            @info "    combined jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec))"
        end

        chunksize = ForwardDiff.pickchunksize(maximum(colorvec), request_adchunksize)
        PB.add_arrays_data!(model, modeldata, ForwardDiff.Dual{Nothing, eltype(modeldata, 1), chunksize}, "jac_ad"; use_base_vars)
        arrays_idx_jac_ad = PB.num_arrays(modeldata)
        jac_solverview = PALEOmodel.SolverView(model, modeldata, arrays_idx_jac_ad; pack_domain) # Variables from whole model
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_ad, jac_cellranges; generated_dispatch)

        jac_cache = SparseDiffTools.ForwardColorJacCache(
            nothing, initial_state, chunksize;
            dx = nothing,
            colorvec=colorvec,
            sparsity = copy(jac_prototype)
        )

        if iszero(PALEOmodel.num_total(modeldata.solver_view_all))            
            odeimplicit = nothing
        else
            # Calculate sparsity pattern for implicit variables
            implicit_cache = SparseDiffTools.ForwardColorJacCache(
                nothing, initial_state, chunksize;
                dx = similar(PB.get_data(modeldata.solver_view_all.total)),
                colorvec=implicit_colorvec,
                sparsity=copy(implicit_prototype)
            )
    
            duds = copy(implicit_prototype)
            
            implicit_dispatchlists = PB.create_dispatch_methodlists(
                model, modeldata, arrays_idx_jac_ad, implicit_cellranges; generated_dispatch,
            )
            odeimplicit = SolverFunctions.ImplicitForwardDiffSparse(
                modeldata, jac_solverview, implicit_dispatchlists, implicit_cache, duds
            )
    
        end

        jac = SolverFunctions.JacDAE(
            SolverFunctions.JacODEForwardDiffSparse(
                modeldata, 
                jac_solverview, 
                jac_dispatchlists,
                jac_cache,
            ),
            odeimplicit
        )

        return (jac, jac_prototype, odeimplicit)

    else
        error("unknown jac_ad=", jac_ad)
    end
    
    error("coding error, not reachable reached")
end


#####################################################################
# Utility functions to calculate sparsity etc
####################################################################

"""
    calcJacobianSparsitySparsityTracing!(model, modeldata, initial_state, tjacsparsity [; jac_cellranges=nothing] [, use_base_vars=String[]]) -> initial_jac

Configure SparsityTracing and calculate sparse Jacobian at time `tjacsparsity`.

If `jac_cellranges` is supplied, Jacobian is restricted to this subset of Domains and Reactions (via operatorID).
"""
function calcJacobianSparsitySparsityTracing!(
    model::PB.Model, modeldata::PB.ModelData, initial_state, tjacsparsity; 
    jac_cellranges=modeldata.cellranges_all,
    use_base_vars=String[],
    pack_domain="",
)

    @info "calcJacobianSparsitySparsityTracing!"
    # Jacobian setup
    initial_state_adst = SparsityTracing.create_advec(initial_state)
    @timeit "add_arrays_data!" begin
    PB.add_arrays_data!(model, modeldata, eltype(initial_state_adst), "sparsity_tracing"; use_base_vars)
    arrays_idx_adst = PB.num_arrays(modeldata)
    end # timeit
    
    solver_view_all_adst = PALEOmodel.SolverView(model, modeldata, arrays_idx_adst; pack_domain)
    PALEOmodel.set_statevar!(solver_view_all_adst, initial_state_adst) # fix up initial state, as derivative information missing
    
    PALEOmodel.set_tforce!(solver_view_all_adst, tjacsparsity)

    @timeit "jac_dispatchlists" begin
    jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_adst, jac_cellranges; generated_dispatch=false)
    end # timeit

    @info "calcJacobianSparsitySparsityTracing! do_deriv"
    @timeit "do_deriv" PB.do_deriv(jac_dispatchlists)

    state_sms_vars_data = PALEOmodel.get_statevar_sms(solver_view_all_adst)
                   
    @timeit "initial_jac" initial_jac = SparsityTracing.jacobian(state_sms_vars_data, length(initial_state))
    @info "calcJacobianSparsitySparsityTracing!  initial_jac size=$(size(initial_jac)) "*
        "nnz=$(SparseArrays.nnz(initial_jac)) non-zero=$(count(!iszero, initial_jac)) at time=$tjacsparsity"  

    PB.pop_arrays_data!(modeldata)

    return initial_jac
end


"""
    calcImplicitSparsitySparsityTracing!(
        model, modeldata, initial_state, tsparsity;
        implicit_cellranges=modeldata.cellranges_all,
        use_base_vars=String[],
    ) -> initial_dTdS

Calculate sparse dTdS for Total Variables at time `tsparsity` using SparsityTracing.
"""
function calcImplicitSparsitySparsityTracing!(
    model::PB.Model, modeldata::PB.ModelData, initial_state, tsparsity; 
    implicit_cellranges=modeldata.cellranges_all,
    use_base_vars=String[],
    pack_domain="",
)    
    # Implicit Jacobian setup
    initial_state_adst = SparsityTracing.create_advec(initial_state)
   
    PB.add_arrays_data!(model, modeldata, eltype(initial_state_adst), "sparsity_tracing_implicit"; use_base_vars)
    arrays_idx_adst = PB.num_arrays(modeldata)
    
    solver_view_all_adst = PALEOmodel.SolverView(model, modeldata, arrays_idx_adst; pack_domain)
    PALEOmodel.set_statevar!(solver_view_all_adst, initial_state_adst) # fix up initial state, as derivative information missing

    PALEOmodel.set_tforce!(solver_view_all_adst, tsparsity)

    implicit_dispatchlists = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_adst, implicit_cellranges; generated_dispatch=false)

    PB.do_deriv(implicit_dispatchlists)

    initial_dTdS = SparsityTracing.jacobian(PB.get_data(solver_view_all_adst.total), length(initial_state));
    @info "  initial_dTdS size=$(size(initial_dTdS)) nnz=$(SparseArrays.nnz(initial_dTdS)) non-zero=$(count(!iszero, initial_dTdS)) at time=$tsparsity"

    PB.pop_arrays_data!(modeldata)

    return initial_dTdS
end




#####################################################################
# Directional derivative using ForwardDiff
#####################################################################

function directional_config(
    model::PB.Model, modeldata::PB.ModelData, directional_cellranges;
    eltypestomap=String[],
    generated_dispatch=true,
    use_base_transfer_jacobian=false,
    pack_domain=modeldata.solver_view_all.pack_domain,
)

    directional_ad_eltype = ForwardDiff.Dual{Nothing, Float64, 1}
    eltypemap = Dict(e=>directional_ad_eltype for e in eltypestomap)
   
    PB.add_arrays_data!(model, modeldata, directional_ad_eltype, "directional_deriv"; eltypemap, use_base_transfer_jacobian,)
    arrays_idx_directional = PB.num_arrays(modeldata)

    directional_sv = PALEOmodel.SolverView(model, modeldata, arrays_idx_directional; pack_domain) # Variables from whole model

    directional_dispatchlists = PB.create_dispatch_methodlists(
        model, modeldata, arrays_idx_directional, directional_cellranges; generated_dispatch,
    )

    directional_workspace = similar(PALEOmodel.get_statevar_sms(directional_sv))

    return (; directional_sv, directional_cellranges, directional_dispatchlists, directional_workspace)
end

"example code for calculating directional derivative, du = J.v at u"
function directional_forwarddiff!(du, u, v, directional_context, t)

    dc = directional_context

    PALEOmodel.set_tforce!(dc.directional_sv, t)
    for i in eachindex(u)
        dc.directional_workspace[i] = ForwardDiff.Dual(u[i], v[i])
    end
    PALEOmodel.set_statevar!(dc.directional_sv, dc.directional_workspace)                       

    PB.do_deriv(dc.directional_dispatchlists)           

    PALEOmodel.get_statevar_sms!(dc.directional_workspace, dc.directional_sv)
    for i in eachindex(u)
        du[i] = ForwardDiff.partials(dc.directional_workspace[i], 1)
    end

    return nothing
end       




end # module
