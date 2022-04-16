module JacobianAD

import PALEOboxes as PB

import PALEOmodel

import LinearAlgebra
import Infiltrator

import Logging

import ForwardDiff
import SparseArrays
import SparseDiffTools
import SparsityTracing

######################################################
# ODE Jacobians using ForwardDiff
########################################################

"""
    jac_config_ode(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        [request_adchunksize] [, jac_cellranges] [, init_logger]
    )-> (jac, jac_prototype)

Create and return `jac` (ODE Jacobian function object), and `jac_prototype` (sparsity pattern, or `nothing` for dense Jacobian)

`jac_ad` defines Jacobian type (:ForwardDiffSparse, :ForwardDiff)

Sets up `modeldata_ad` with appropriate datatypes for ForwardDiff AD Dual numbers,
sets up cache for ForwardDiff, calculates Jacobian sparsity (if required) at time  `jac_ad_t_sparsity`.

If `jac_cellranges` is supplied, Jacobian is restricted to this subset of Domains and Reactions (via operatorID).

NB: there is a profusion of different Julia APIs here:
- ForwardDiff Sparse and dense Jacobian use different APIs and have different cache setup requirements.
- ForwardDiff requires f!(du, u) hence a closure or function object, DifferentialEquations allows context objects to be passed around.
"""
function jac_config_ode(
    jac_ad::Symbol, model::PB.Model, initial_state, modeldata, jac_ad_t_sparsity;
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    jac_cellranges=modeldata.cellranges_all,
    init_logger=Logging.NullLogger(),
)
    @info "jac_config_ode: jac_ad=$jac_ad"  

    iszero(PB.num_total(modeldata.solver_view_all)) ||
        throw(ArgumentError("model contains implicit variables, solve as a DAE"))
   
    # generate arrays with ODE layout for model Variables
    state_sms_vars_data = similar(PB.get_statevar_sms(modeldata.solver_view_all))
    state_vars_data = similar(PB.get_statevar(modeldata.solver_view_all))

    if jac_ad == :NoJacobian
        return (nothing, nothing)

    elseif jac_ad == :ForwardDiff       

        chunk = ForwardDiff.Chunk(length(state_sms_vars_data), request_adchunksize)
 
        jacconf = ForwardDiff.JacobianConfig(nothing, state_sms_vars_data, state_vars_data, chunk)
        _, modeldata_ad = Logging.with_logger(init_logger) do
            PALEOmodel.initialize!(model, eltype=eltype(jacconf))
        end
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, jac_cellranges)

        @info "  using ForwardDiff dense Jacobian chunksize=$(ForwardDiff.chunksize(chunk)))"
     
        du_template = similar(state_sms_vars_data)
        
        jac = JacODEForwardDiffDense(
            modeldata_ad, 
            modeldata_ad.solver_view_all, # use all Variables in model
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
        _, jac_proto_unfilled = calcJacobianSparsitySparsityTracing!(
            model, initial_state, jac_ad_t_sparsity,
            jac_cellranges=jac_cellranges, init_logger=init_logger,
        ) 
        jac_prototype = fill_sparse_jac(jac_proto_unfilled)
        # println("using jac_prototype: ", jac_prototype)
       
        colorvec = SparseDiffTools.matrix_colors(jac_prototype)

        chunksize = ForwardDiff.pickchunksize(maximum(colorvec), request_adchunksize)
        _, modeldata_ad = Logging.with_logger(init_logger) do
            PALEOmodel.initialize!(model, eltype=ForwardDiff.Dual{Nothing, eltype(modeldata), chunksize})
        end
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, jac_cellranges)
        
        @info "  jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec)) "*
            "chunksize=$chunksize)"    

        jac_cache = SparseDiffTools.ForwardColorJacCache(
            nothing, initial_state, chunksize;
            dx = nothing, # similar(modeldata_ad.state_sms_vars_data)
            colorvec=colorvec,
            sparsity = copy(jac_prototype)
        )

        jac = JacODEForwardDiffSparse(
            modeldata_ad, 
            modeldata_ad.solver_view_all, # use all Variables in model
            jac_dispatchlists, # use only Reactions specified
            jac_cache,
        )

        return jac, jac_prototype
    else
        error("unknown jac_ad=", jac_ad)
    end
    
    error("coding error, not reachable reached")
end


"""
    DerivForwardDiff
    
Function object to calculate model derivative in form required by ForwardDiff
"""
mutable struct DerivForwardDiff{S, D}
    solverview::S
    dispatchlists::D
end

"model derivative as required by ForwardDiff"
function (dfd::DerivForwardDiff)(du, u)
    PB.set_statevar!(dfd.solverview, u)
    PB.do_deriv(dfd.dispatchlists)
    PB.get_statevar_sms!(du, dfd.solverview)         

    return nothing
end

"""
    JacODEForwardDiffDense

Function object to calculate dense Jacobian in form required for SciML ODE solver.

`solverview`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.
"""
mutable struct JacODEForwardDiffDense{T, S, D, W, J}
    modeldata::PB.ModelData{T}
    deriv_forwarddiff::DerivForwardDiff{S, D}
    du_template::W
    jacconf::J    
    njacs::Int
    function JacODEForwardDiffDense(
        modeldata::PB.ModelData{T}, solverview::S, dispatchlists::D, du_template::W, jacconf::J,
    ) where {T, S, D, W, J}
        return new{T, S, D, W, J}(
            modeldata, DerivForwardDiff(solverview, dispatchlists), du_template, jacconf, 0,
        )
    end
end

"Calculate dense Jacobian for ODE solver"
function (jfdd::JacODEForwardDiffDense)(J, u, p, t)
   
    PB.set_tforce!(jfdd.deriv_forwarddiff.solverview, t) # u will be set by deriv_forwarddiff
   
    ForwardDiff.jacobian!(J, jfdd.deriv_forwarddiff,  jfdd.du_template, u, jfdd.jacconf)   
    jfdd.njacs += 1  

    return nothing
end

"""
    JacODEForwardDiffSparse

Function object to calculate sparse Jacobian in form required for SciML ODE solver.

`solverview`, `dispatchlists` should correspond to `modeldata`, which should
have the appropriate element type for ForwardDiff Dual numbers.
"""
mutable struct JacODEForwardDiffSparse{T, S, D, J}
    modeldata::PB.ModelData{T}
    deriv_forwarddiff::DerivForwardDiff{S, D}
    jac_cache::J    
    njacs::Int
    function JacODEForwardDiffSparse(
        modeldata::PB.ModelData{T}, solverview::S, dispatchlists::D, jac_cache::J
    ) where {T, S, D, J}
        return new{T, S, D, J}(
            modeldata, DerivForwardDiff(solverview, dispatchlists), jac_cache, 0
        )
    end
end

"Adapt SparseDiffTools.forwarddiff_color_jacobian! to Jacobian function required by DifferentialEquations"
function (jfds::JacODEForwardDiffSparse)(J, u, p, t)
    
    nnz_before = SparseArrays.nnz(J)
    
    PB.set_tforce!(jfds.deriv_forwarddiff.solverview, t) # set t only, u will be set by f_deriv_forwarddiff
    SparseDiffTools.forwarddiff_color_jacobian!(
        J,
        jfds.deriv_forwarddiff,
        u,
        jfds.jac_cache
    )

    nnz_before == SparseArrays.nnz(J) || error("Jacobian sparsity changed nnz $(nnz_before) != $(SparseArrays.nnz(J))")
  
    countnan = count(isnan, J.nzval)
    iszero(countnan) || @warn "JacODEForwardDiffSparse: Jacobian contains $countnan NaN at t=$t"

    jfds.njacs += 1  
    
    return nothing
end


###################################################################
# DAE Jacobians using ForwardDiff
######################################################################

"""
    jac_config_dae(
        jac_ad, model, initial_state, modeldata, jac_ad_t_sparsity;
        kwargs...
    ) -> (jac, jac_prototype, odeimplicit)

See [`jac_config_ode`](@ref) for parameters.
"""
function jac_config_dae(
    jac_ad::Symbol, model::PB.Model, initial_state, modeldata, jac_ad_t_sparsity;
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    jac_cellranges=modeldata.cellranges_all,
    implicit_cellranges=modeldata.cellranges_all,
    init_logger=Logging.NullLogger(),
)
    @info "jac_config_dae: jac_ad=$jac_ad"  
    
    # generate arrays with ODE layout for model Variables
    state_sms_vars_data = similar(PB.get_statevar_sms(modeldata.solver_view_all))
    state_vars_data = similar(PB.get_statevar(modeldata.solver_view_all))

    if jac_ad == :NoJacobian
        # check for implicit total variables
        PB.num_total(modeldata.solver_view_all) == 0 ||
            error("implicit total variables - Jacobian required")

        return (nothing, nothing, nothing)

    elseif jac_ad == :ForwardDiff  
        chunk = ForwardDiff.Chunk(length(state_sms_vars_data), request_adchunksize)

        @info "  using ForwardDiff dense Jacobian chunksize=$(ForwardDiff.chunksize(chunk)))"
      
        jacconf = ForwardDiff.JacobianConfig(nothing, state_sms_vars_data, state_vars_data, chunk)
        _, modeldata_ad = Logging.with_logger(init_logger) do 
            PALEOmodel.initialize!(model, eltype=eltype(jacconf))
        end
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, jac_cellranges)

        if iszero(PB.num_total(modeldata.solver_view_all))
            odeimplicit = nothing     
        else 
            @info "  calculating dTdS for $(PB.num_total(modeldata.solver_view_all)) Total Variables"

            sv_ad = modeldata_ad.solver_view_all
            duds = zeros(eltype(modeldata), length(sv_ad.total), length(state_sms_vars_data))
            duds_template = similar(PB.get_data(modeldata.solver_view_all.total))
            implicitconf = ForwardDiff.JacobianConfig(
                nothing, duds_template, state_vars_data, chunk,
            )

            implicit_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, implicit_cellranges)
            odeimplicit = ImplicitForwardDiffDense(modeldata_ad, sv_ad, implicit_dispatchlists, duds_template, implicitconf, duds)
        end

        du_template = similar(state_sms_vars_data)

        jac = JacDAE(
            JacODEForwardDiffDense(
                modeldata_ad, 
                modeldata_ad.solver_view_all, 
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
        modeldata_sparsitytracing, jac_proto_unfilled = calcJacobianSparsitySparsityTracing!(
            model, initial_state, jac_ad_t_sparsity,
            jac_cellranges=jac_cellranges, init_logger=init_logger,
        ) 
        jac_prototype = fill_sparse_jac(jac_proto_unfilled)
        # println("using jac_prototype: ", jac_prototype)
       
        colorvec = SparseDiffTools.matrix_colors(jac_prototype)
        
        @info "  jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec))"

        if !iszero(PB.num_total(modeldata.solver_view_all))
            @info "  calculating dTdS for $(PB.num_total(modeldata.solver_view_all)) Total Variables"

            implicit_proto_unfilled = calcImplicitSparsitySparsityTracing!(
                model, initial_state, jac_ad_t_sparsity, modeldata_sparsitytracing,
                implicit_cellranges=implicit_cellranges
            )        
            implicit_prototype = fill_sparse_jac(implicit_proto_unfilled, fill_diagonal=false)
            implicit_colorvec = SparseDiffTools.matrix_colors(implicit_prototype)
            @info "  implicit_prototype nnz=$(SparseArrays.nnz(implicit_prototype)) num colors=$(maximum(implicit_colorvec))"

            # add sparsity patterns to get DAE Jacobian sparsity
            # TODO this simplifies the DAE Jacobian code but will overestimate the sparsity pattern for the ODE part of the Jacobian calculation
            iistrt = length(modeldata.solver_view_all.stateexplicit) + 1
            iiend = iistrt+PB.num_total(modeldata.solver_view_all)-1
            jac_prototype[iistrt:iiend, :] += implicit_prototype
            colorvec = SparseDiffTools.matrix_colors(jac_prototype)
            
            @info "    combined jac_prototype nnz=$(SparseArrays.nnz(jac_prototype)) num colors=$(maximum(colorvec))"
        end

        chunksize = ForwardDiff.pickchunksize(maximum(colorvec), request_adchunksize)
        _, modeldata_ad = Logging.with_logger(init_logger) do
            PALEOmodel.initialize!(model, eltype=ForwardDiff.Dual{Nothing, eltype(modeldata), chunksize})
        end
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, jac_cellranges)
      
        if iszero(PB.num_total(modeldata.solver_view_all))
            
            odeimplicit = nothing
        else
            # Calculate sparsity pattern for implicit variables
            implicit_cache = SparseDiffTools.ForwardColorJacCache(
                nothing, initial_state, chunksize;
                dx = similar(PB.get_data(modeldata.solver_view_all.total)), # nothing, # similar(modeldata_ad.total_vars_data),
                colorvec=implicit_colorvec,
                sparsity=copy(implicit_prototype)
            )
    
            duds = copy(implicit_prototype)
            
            sv_ad = modeldata_ad.solver_view_all
            implicit_dispatchlists = PB.create_dispatch_methodlists(
                model, modeldata_ad, implicit_cellranges
            )
            odeimplicit = ImplicitForwardDiffSparse(
                modeldata_ad, sv_ad, implicit_dispatchlists, implicit_cache, duds
            )
    
        end

        jac_cache = SparseDiffTools.ForwardColorJacCache(
            nothing, initial_state, chunksize;
            dx = nothing, # similar(modeldata_ad.state_sms_vars_data)
            colorvec=colorvec,
            sparsity = copy(jac_prototype)
        )

        jac = JacDAE(
            JacODEForwardDiffSparse(
                modeldata_ad, 
                modeldata_ad.solver_view_all, 
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


mutable struct ImplicitDerivForwardDiff{S, D}
    solverview::S
    dispatchlists::D
end

"model derivative as required by ForwardDiff"
function (idfd::ImplicitDerivForwardDiff)(T, u)
    PB.set_statevar!(idfd.solverview, u)        
    PB.do_deriv(idfd.dispatchlists)
    copyto!(T, idfd.solverview.total)      

    return nothing
end

mutable struct ImplicitForwardDiffDense{T, S, D, W, I, U}
    modeldata::PB.ModelData{T}
    implicitderiv::ImplicitDerivForwardDiff{S, D}
    duds_template::W
    implicitconf::I
    duds::U
    function ImplicitForwardDiffDense(
        modeldata::PB.ModelData{T}, solverview::S, dispatchlists::D, duds_template::W, implicitconf::I, duds::U
    ) where {T, S, D, W, I, U}
        return new{T, S, D, W, I, U}(
            modeldata, ImplicitDerivForwardDiff(solverview, dispatchlists), duds_template, implicitconf, duds
        )
    end
end

"Adapt ForwardDiff.jacobian! to calculate dT/dS required for a model containing implicit variables"
function (ifdd::ImplicitForwardDiffDense)(dTdS, s, p, t)
    PB.set_tforce!(ifdd.implicitderiv.solverview, t) # set t only, u will be set by implicitderiv

    ForwardDiff.jacobian!(dTdS, ifdd.implicitderiv, ifdd.duds_template, s, ifdd.implicitconf)  

    return  nothing
end

mutable struct ImplicitForwardDiffSparse{T, S, D, I, U}
    modeldata::PB.ModelData{T}
    implicitderiv::ImplicitDerivForwardDiff{S, D}
    implicit_cache::I
    duds::U
    function ImplicitForwardDiffSparse(
        modeldata::PB.ModelData{T}, solverview::S, dispatchlists::D, implicit_cache::I, duds::U
    ) where {T, S, D, I, U}
        return new{T, S, D, I, U}(
            modeldata, ImplicitDerivForwardDiff(solverview, dispatchlists), implicit_cache, duds
        )
    end
end

"Adapt SparseDiffTools.forwarddiff_color_jacobian! to calculate dT/dS required for model containing implicit variables"
function (ifds::ImplicitForwardDiffSparse)(dTdS, s, p, t)
    
    nnz_before = SparseArrays.nnz(dTdS)

    PB.set_tforce!(ifds.implicitderiv.solverview, t) # set t only, s will be set by f_implicit_forwarddiff
    SparseDiffTools.forwarddiff_color_jacobian!(
        dTdS,
        ifds.implicitderiv,
        s,
        ifds.implicit_cache
    )

    nnz_before == SparseArrays.nnz(dTdS) || error("dTdS sparsity changed nnz $(nnz_before) != $(SparseArrays.nnz(dTdS))")
    
    return nothing
end


"""
    JacDAE

Function object to calculate Jacobian in form required for SciML DAE solver

`odejac` should be a [`JacODEForwardDiffDense`](@ref) or [`JacODEForwardDiffSparse`](@ref)

If using Total variables, `odeimplicit` should be an [`ImplicitForwardDiffDense`](@ref) or [`ImplicitForwardDiffSparse`](@ref),
otherwise `nothing`.
"""
mutable struct JacDAE{J, I}
    odejac::J
    odeimplicit::I
end

"Calculate DAE jacobian (in form required by DifferentialEquations DAE solvers) from an ODE jacobian"
function (jdae::JacDAE)(J, dsdt, s, p, γ, t)
    # The Jacobian should be given in the form γ*dG/d(dsdt) + dG/ds where γ is given by the solver
   
    # dG/ds
    jdae.odejac(J,s, p, t)

    md = jdae.odejac.modeldata
    # γ*dG/d(dsdt) explicit variables with u(s) = s
    l_ts = length(md.solver_view_all.stateexplicit_deriv)
    for i in 1:l_ts     
        J[i,i] -= γ
    end

    # γ*dG/d(dsdt) = γ*(du/ds)*dG/d(dudt) implicit variables u(s)
    l_ti = length(md.solver_view_all.total)
    if l_ti > 0             
        jdae.odeimplicit(jdae.odeimplicit.duds, s, p, t)        
        J[(l_ts+1):(l_ts+l_ti), :] -= γ.*jdae.odeimplicit.duds
    end
    
    return nothing
end

#####################################################################
# Utility functions to calculate sparsity etc
####################################################################

"""
    jac_transfer_variables(model, modeldata_ad, modeldata) -> (transfer_data_arrays_ad, transfer_data_arrays)

Build Vectors of data arrays that need to be copied from `modeldata` to `modeldata_ad` before calculating Jacobian.
(looks for Variables with :transfer_jacobian attribute set).  Only needed if the Jacobian calculation is optimised
by excluding some Reactions from (re)calculation.

`modeldata` is the whole model, `modeldata_ad` is for the Jacobian.

The copy needed is then:

    for (d_ad, d) in zip(transfer_data_ad, transfer_data)                
        d_ad .= d
    end
"""
function jac_transfer_variables(model, modeldata_ad, modeldata)

    # get list of Variables with attribute :transfer_jacobian = true
    transfer_vars = []
    for dom in model.domains
        append!(transfer_vars, PB.get_variables(dom, v -> PB.get_attribute(v, :transfer_jacobian, false)))
    end

    # build lists of data arrays to transfer
    transfer_data_arrays = []
    transfer_data_arrays_ad = []
    for v in transfer_vars
        push!(transfer_data_arrays, PB.get_data(v, modeldata))
        push!(transfer_data_arrays_ad, PB.get_data(v, modeldata_ad))
    end

    l_d = length(transfer_data_arrays)
    l_d_ad = length(transfer_data_arrays_ad) 
    l_d == l_d_ad || error("jac_transfer_variables: length mismatch transfer to ad variable components=$l_d_ad, from variable components=$l_d")

    @info "jac_transfer_variables transfer $l_d Variable components"
    return (
        [d for d in transfer_data_arrays_ad], # rebuild to get a typed Vector
        [d for d in transfer_data_arrays]     # rebuild to get a typed Vector
    )
end


"""
    calcJacobianSparsitySparsityTracing!(model, initial_state, tjacsparsity [; jac_cellranges=nothing]) -> (modeldata_ad, jac_dispatchlists, initial_jac)

Configure SparsityTracing and calculate sparse Jacobian at time `tjacsparsity`.

If `jac_cellranges` is supplied, Jacobian is restricted to this subset of Domains and Reactions (via operatorID).
"""
function calcJacobianSparsitySparsityTracing!(
    model::PB.Model, initial_state, tjacsparsity; 
    jac_cellranges=nothing,
    init_logger=Logging.NullLogger(),
)

    @info "calcJacobianSparsitySparsityTracing!"
    # Jacobian setup
    initial_state_ad = SparsityTracing.create_advec(initial_state)
    _, modeldata_ad = Logging.with_logger(init_logger) do
        PALEOmodel.initialize!(model, eltype=eltype(initial_state_ad))
    end
    
    PB.set_statevar!(modeldata_ad.solver_view_all, initial_state_ad) # fix up modeldata_ad initial_state, as derivative information missing
    
    PB.set_tforce!(modeldata_ad.solver_view_all, tjacsparsity)

    if isnothing(jac_cellranges)
        # Jacobian for whole model
        jac_cellranges = modeldata_ad.cellranges_all
        jac_dispatchlists = modeldata_ad.dispatchlists_all
    else
        jac_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_ad, jac_cellranges)
    end

    @info "calcJacobianSparsitySparsityTracing! do_deriv"
    @time PB.do_deriv(jac_dispatchlists)

    state_sms_vars_data = PB.get_statevar_sms(modeldata_ad.solver_view_all)                                  
    @time initial_jac = SparsityTracing.jacobian(state_sms_vars_data, length(initial_state))
    @info "calcJacobianSparsitySparsityTracing!  initial_jac size=$(size(initial_jac)) "*
        "nnz=$(SparseArrays.nnz(initial_jac)) non-zero=$(count(!iszero, initial_jac)) at time=$tjacsparsity"  

    return (modeldata_ad, initial_jac)
end


"""
    calcImplicitSparsitySparsityTracing!(
        model, initial_state, tsparsity, modeldata_sparsitytracing;
        implicit_cellranges=modeldata_sparsitytracing.cellranges_all
    ) -> initial_dTdS

Calculate sparse dTdS for Total Variables at time `tsparsity` using SparsityTracing.
"""
function calcImplicitSparsitySparsityTracing!(
    model::PB.Model, initial_state, tsparsity, modeldata_sparsitytracing; 
    implicit_cellranges=modeldata_sparsitytracing.cellranges_all
)    
    # Jacobian setup
    initial_state_ad = SparsityTracing.create_advec(initial_state)

    PB.set_statevar!(modeldata_sparsitytracing.solver_view_all, initial_state_ad)

    PB.set_tforce!(modeldata_sparsitytracing.solver_view_all, tsparsity)

    implicit_dispatchlists = PB.create_dispatch_methodlists(model, modeldata_sparsitytracing, implicit_cellranges)

    PB.do_deriv(implicit_dispatchlists)

    initial_dTdS = SparsityTracing.jacobian(PB.get_data(modeldata_sparsitytracing.solver_view_all.total), length(initial_state));
    @info "  initial_dTdS size=$(size(initial_dTdS)) nnz=$(SparseArrays.nnz(initial_dTdS)) non-zero=$(count(!iszero, initial_dTdS)) at time=$tsparsity"

    return initial_dTdS
end

"fill structural non-zeros: return sparse matrix with all stored elements filled with `val`"
function fill_sparse_jac(initial_jac; val=1.0, fill_diagonal=true)
    num_elements = size(initial_jac, 1)*size(initial_jac, 2)

    @info "  fill_sparse_jac: initial_jac nnz $(SparseArrays.nnz(initial_jac)) "*
        "($(round(100*SparseArrays.nnz(initial_jac)/num_elements, sigdigits=3))%) non-zero $(count(!iszero, initial_jac))"
    I, J, V = SparseArrays.findnz(initial_jac)

    jac_filled = SparseArrays.sparse(I, J, val, size(initial_jac, 1), size(initial_jac, 2) )
    if fill_diagonal
        size(initial_jac, 1) == size(initial_jac, 2) ||
            error("fill_sparse_jac: fill_diagonal == true and Jacobian is not square")
        for i = 1:size(initial_jac, 1)
            jac_filled[i, i] = val
        end
        @info "  fill_sparse_jac: after filling diagonal nnz $(SparseArrays.nnz(jac_filled))"     
    end

    return jac_filled
end


#####################################################################
# Directional derivative using ForwardDiff
#####################################################################

function directional_config(
    model::PB.Model, directional_cellranges;
    eltypestomap=String[],
    init_logger=Logging.NullLogger(),
)

    directional_ad_eltype = ForwardDiff.Dual{Nothing, Float64, 1}
    eltypemap = Dict(e=>directional_ad_eltype for e in eltypestomap)
   
    _, directional_modeldata_ad = Logging.with_logger(init_logger) do 
        PALEOmodel.initialize!(
            model,
            eltype=directional_ad_eltype,
            eltypemap=eltypemap
        )
    end

    directional_dispatchlists = PB.create_dispatch_methodlists(
        model, directional_modeldata_ad, directional_cellranges
    )

    directional_sv = directional_modeldata_ad.solver_view_all
    directional_workspace = similar(PB.get_statevar_sms(directional_sv))

    return (; directional_modeldata_ad, directional_sv, 
            directional_cellranges, directional_dispatchlists, directional_workspace)

end

"example code for calculating directional derivative, du = J.v at u"
function directional_forwarddiff!(du, u, v, directional_context, t)

    dc = directional_context

    PB.set_tforce!(dc.directional_sv, t)
    for i in eachindex(u)
        dc.directional_workspace[i] = ForwardDiff.Dual(u[i], v[i])
    end
    PB.set_statevar!(dc.directional_sv, dc.directional_workspace)                       

    PB.do_deriv(dc.directional_dispatchlists)           

    PB.get_statevar_sms!(dc.directional_workspace, dc.directional_sv)
    for i in eachindex(u)
        du[i] = ForwardDiff.partials(dc.directional_workspace[i], 1)
    end

    return nothing
end       




end # module
