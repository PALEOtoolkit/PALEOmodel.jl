module ODELocalIMEX

import Infiltrator

import PALEOboxes as PB

import PALEOmodel

import Logging
import LinearAlgebra
import ForwardDiff
import StaticArrays

"""
    integrateLocalIMEXEuler(run, initial_state, modeldata, tspan, Δt_outer [; kwargs...])

Integrate run.model representing:
```math
\\frac{dS}{dt} =  f_{outer}(t, S) + f_{inner}(t, S)
```
using first-order Euler with `Δt_outer` for `f_outer` and implicit first-order Euler for `f_inner`, where `f_inner` is
local (within-cell, ie no transport), for a single Domain, and uses only `StateExplicit` and `Deriv` variables.

`f_outer` is defined by calling `PALEOboxes.do_deriv` with `cellranges_outer` hence corresponds to those `Reactions` with `operatorID`
of `cellranges_outer`.  `f_inner` is defined by calling `PALEOboxes.do_deriv` with `cellrange_inner` hence corresponds to those `Reactions` with `operatorID`
of `cellrange_inner`.

NB: the combined time derivative is written to `outputwriter`.

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tspan`:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times
- `Δt_outer`: (yr) fixed timestep

# Keywords

- `cellranges_outer`: Vector of `CellRange` with `operatorID` defining `f_outer`.
- `cellrange_inner`: A single `CellRange` with `operatorID` defining `f_inner`.
- `exclude_var_nameroots`: State variables that are modified by Reactions in `cellrange_inner`, but not needed to find implicit solution (ie reaction rates etc don't depend on them).
- [`outputwriter=run.output`: `PALEOmodel.AbstractOutputWriter` instance to write model output to]
- [`report_interval=1000`: number of outer timesteps between progress update to console]
- [`Lnorm_inf_max=1e-3`:  normalized error tolerance for implicit solution]
- [`niter_max=10`]: maximum number of Newton iterations
- [`request_adchunksize=4`]: request ForwardDiff AD chunk size (will be restricted to an upper limit)
"""
function integrateLocalIMEXEuler(
    run, initial_state, modeldata, tspan, Δt_outer;
    cellranges_outer,
    cellrange_inner,
    exclude_var_nameroots, 
    outputwriter=run.output,
    report_interval=1000,
    Lnorm_inf_max=1e-3,
    niter_max=10,
    request_adchunksize = 4
)

    @info "integrateLocalIMEXEuler: Δt_outer=$Δt_outer (yr)"

    solver_view_outer = PB.create_solver_view(run.model, modeldata, cellranges_outer)
    @info "solver_view_outer: $(solver_view_outer)"    
    
    lictxt = create_timestep_LocalImplicit_ctxt(
        run.model, modeldata;                                   
        cellrange=cellrange_inner,
        exclude_var_nameroots=exclude_var_nameroots,
        niter_max=niter_max,
        Lnorm_inf_max=Lnorm_inf_max
    )

    timesteppers = [
        [(
            PALEOmodel.ODEfixed.timestep_Euler,
            cellranges_outer, 
            PALEOmodel.ODEfixed.create_timestep_Euler_ctxt(
                run.model,
                modeldata, 
                solver_view=solver_view_outer,
                cellranges=cellranges_outer,
            ),
        )],
        [(
            timestep_LocalImplicit,
            [cellrange_inner],
            lictxt,
        )],
    ]

    @time PALEOmodel.ODEfixed.integrateFixed(
        run, initial_state, modeldata, tspan, Δt_outer,
        timesteppers=timesteppers,
        outputwriter=outputwriter,
        report_interval=report_interval
    )

    return nothing
end         

"take a single implicit Euler timestep"
function timestep_LocalImplicit(
    model, modeldata, cellranges, 
    lictxt, 
    touter, Δt, threadid, report;
    deriv_only=false,
    integrator_barrier=nothing,
)
    length(cellranges) == 1 || error("timestep_LocalImplicit only single cellrange supported")
    cellrange = cellranges[1]

    PB.set_tforce!(modeldata.solver_view_all, touter + Δt)
    if deriv_only
        # for reporting output fluxes etc 
        for cell_idx in cellrange.indices
            cell_context = lictxt.cell_context[cell_idx]                  
            PB.do_deriv(cell_context.dispatchlists, Δt)
        end
    else
        # implicit timestep
        
        PB.set_tforce!(lictxt.modeldata_ad.solver_view_all, touter + Δt)

        S_previous = lictxt.cell_S_previous # workspace

        niter_max = 0
        niter_total = 0
        (Lnorm_inf_init, Lnorm_2_init) = (0.0, 0.0)
        (Lnorm_inf, Lnorm_2) = (0.0, 0.0)

        for cell_idx in cellrange.indices
            cell_context = lictxt.cell_context[cell_idx]

            # statevar at previous timestep
            PB.get_statevar!(S_previous, cell_context.solverview)

            (Lnorm_inf_cell, Lnorm_2_cell) = calc_residual(S_previous, lictxt, cell_idx, Δt)
            (Lnorm_inf_init_cell, Lnorm_2_init_cell) = (Lnorm_inf_cell, Lnorm_2_cell)
            
            niter = 0
            while (isnan(Lnorm_inf_cell) || Lnorm_inf_cell > lictxt.Lnorm_inf_max) && niter < lictxt.niter_max
                local_newton_update(lictxt.Valn_solve, lictxt, cell_idx, touter, Δt)          
                (Lnorm_inf_cell, Lnorm_2_cell) = calc_residual(S_previous, lictxt, cell_idx, Δt)
                niter += 1
            end
            
            niter >= lictxt.niter_max && 
                @warn "  tmodel $touter cell_idx $cell_idx implicit niter_max $(lictxt.niter_max) exceeded "*
                    "(Lnorm_inf, Lnorm_2) ($Lnorm_inf_init_cell, $Lnorm_2_init_cell) -> ($Lnorm_inf_cell, $Lnorm_2_cell)"

            # @info "    timestep_LocalImplicit: niter $niter (Lnorm_inf, Lnorm_2) "*
            #    "($Lnorm_inf_init_cell, $Lnorm_2_init_cell) -> ($Lnorm_inf_cell, $Lnorm_2_cell)"
            # update global stats
            niter_total += niter
            niter_max = max(niter, niter_max)
            Lnorm_inf = max(Lnorm_inf_cell, Lnorm_inf)
            Lnorm_inf_init = max(Lnorm_inf_init_cell, Lnorm_inf_init)
            Lnorm_2 += Lnorm_2_cell^2
            Lnorm_2_init += Lnorm_2_init_cell^2
        end

        Lnorm_2, Lnorm_2_init = sqrt(Lnorm_2), sqrt(Lnorm_2_init)

        if report
            @info "    timestep_LocalImplicit: niter_max $niter_max niter_mean $(niter_total/length(cellrange.indices)) "*
                "(Lnorm_inf, Lnorm_2) ($Lnorm_inf_init, $Lnorm_2_init) -> ($Lnorm_inf, $Lnorm_2)"
        end

        # add derivative for excluded vars (not included in implicit solution, but modified by implicit Reactions)
        PB.add_data!(lictxt.va_excluded, Δt, lictxt.va_sms_excluded)
    end

    return nothing
end

function create_timestep_LocalImplicit_ctxt(
    model, modeldata;                                   
    cellrange,
    exclude_var_nameroots,
    niter_max,
    Lnorm_inf_max
)

    lictxt = PALEOmodel.ODELocalIMEX.getLocalImplicitContext(
        model, modeldata, cellrange, exclude_var_nameroots,
    )

    Valn_solve = Val(lictxt.n_solve)    
   
    return merge(lictxt, (; Valn_solve, niter_max, Lnorm_inf_max))
end


function getLocalImplicitContext(
    model, modeldata, cellrange, exclude_var_nameroots,
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    init_logger=Logging.NullLogger(),
)

    # create SolverViews for first cell, to work out how many dof we need
    cellrange_cell = PB.CellRange(cellrange.domain, cellrange.operatorID, first(cellrange.indices) )

    # get SolverView for all Variables required for cellrange derivative
    solver_view_all_cell = PB.create_solver_view(model, modeldata, [cellrange_cell], indices_from_cellranges=true)
    @info "getLocalImplicitContext: all Variables (first cell) $solver_view_all_cell)"  
 
    # get reduced set of Variables required for nonlinear solve
    solver_view_cell = PB.create_solver_view(
        model, modeldata, [cellrange_cell], 
        exclude_var_nameroots=exclude_var_nameroots,
        indices_from_cellranges=true,
    )
    @info "getLocalImplicitContext: Variables for nonlinear solve (first cell) $solver_view_cell"
    iszero(length(solver_view_cell.total)) || 
        error("getLocalImplicitContext: total Variables not supported")

    # Find 'excluded variables' and create VariableAggregators so we can add derivative
    excluded_vars = [
        v for v in solver_view_all_cell.stateexplicit.vars if !(v in solver_view_cell.stateexplicit.vars)
    ]
    va_excluded = PB.VariableAggregator(excluded_vars, [cellrange for v in excluded_vars], modeldata)
    excluded_sms_vars = [
        v for v in solver_view_all_cell.stateexplicit_deriv.vars if !(v in solver_view_cell.stateexplicit_deriv.vars)
    ]
    va_sms_excluded = PB.VariableAggregator(excluded_sms_vars, [cellrange for v in excluded_sms_vars], modeldata)
    length(excluded_vars) == length(excluded_sms_vars) ||
        error("excluded_vars and excluded_sms_vars length mismatch")

    n_all = length(PB.get_statevar(solver_view_all_cell))
    n_solve = length(PB.get_statevar(solver_view_cell))

    cellrange_length = length(cellrange.indices)
    @info "  n_all $n_all solving for $n_solve dof x domain length $cellrange_length = $(n_solve*cellrange_length)"
    @info "  solve vars: $([PB.fullname(v) for v in solver_view_cell.stateexplicit.vars]) sms_vars: $([PB.fullname(v) for v in solver_view_cell.stateexplicit_deriv.vars])"
    @info "  excluded vars: $([PB.fullname(v) for v in excluded_vars]) sms_vars: $([PB.fullname(v) for v in excluded_sms_vars])"
   
    # create a modeldata_ad with Dual numbers for AD Jacobians
    chunksize = ForwardDiff.pickchunksize(n_solve, request_adchunksize)
    @info "  using ForwardDiff dense Jacobian chunksize=$chunksize"

    _, modeldata_ad = Logging.with_logger(init_logger) do
        PALEOmodel.initialize!(model, eltype=ForwardDiff.Dual{Nothing, eltype(modeldata), chunksize})
    end

    # allocate one cell buffers for Newton update workspace
    J = Matrix{eltype(modeldata)}(undef, n_solve, n_solve); fill!(J, NaN)
    cell_mat = similar(J); fill!(cell_mat, NaN)
    u_worksp = Vector{eltype(modeldata)}(undef, n_solve); fill!(u_worksp, NaN)
    cell_residual = similar(u_worksp); fill!(cell_residual, NaN)
    cell_S_previous = similar(u_worksp); fill!(cell_S_previous, NaN)

    # Jacobian workspace (for one cell)
    jacconf = ForwardDiff.JacobianConfig(nothing, u_worksp, u_worksp, ForwardDiff.Chunk{chunksize}())
 
    # temporarily replace modeldata with norm so can read back per-cell norms
    statevar_all_current = PB.get_statevar(modeldata.solver_view_all)
    PB.uncopy_norm!(modeldata.solver_view_all)
    # create per-cell solver_view, dispatchlists, jac
    cell_context = []
    for i in cellrange.indices
        # (ab)use that fact that Julia allows iteration over scalar i (to optimise out loop over cellrange.indices)
        cellrange = PB.CellRange(cellrange.domain, cellrange.operatorID, i) 

        solverview = PB.create_solver_view(
            model, modeldata, [cellrange], 
            exclude_var_nameroots=exclude_var_nameroots,
            indices_from_cellranges=true,
            hostdep_all=false,
        )
        PB.copy_norm!(solverview)
        statevar_norm = PB.get_statevar_norm(solverview)
        statevar_sms_norm = PB.get_statevar_sms_norm(solverview)

        solverview_ad = PB.create_solver_view(
            model, modeldata_ad, [cellrange], 
            exclude_var_nameroots=exclude_var_nameroots,
            indices_from_cellranges=true,
            hostdep_all=false,
        )
        PB.copy_norm!(solverview_ad)

        dispatchlists = PB.create_dispatch_methodlists(model, modeldata, [cellrange])
        dispatchlists_ad = PB.create_dispatch_methodlists(model, modeldata_ad, [cellrange])

        jac_cell = PALEOmodel.JacobianAD.JacODEForwardDiffDense(
            modeldata_ad, 
            solverview_ad,
            dispatchlists_ad,
            u_worksp, 
            jacconf
        )

        push!(cell_context, (;cellrange, dispatchlists, solverview, statevar_norm, statevar_sms_norm, jac_cell))
    end
    cell_context = [c for c in cell_context] # narrow type
   
    # replace modeldata statevar (temporarily was set to norm)
    PB.set_statevar!(modeldata.solver_view_all, statevar_all_current)

    return (;n_solve, modeldata_ad, J, cell_mat, u_worksp, cell_residual, cell_S_previous, cell_context, va_excluded, va_sms_excluded)
end


"Given state variable S_previous (statevar at the previous timestep), 
and a current S_next estimate in modeldata,
calculate the normalized residual = (S_next - S_previous - deltat*(dS/dt)|S_next)/S_norm
"
function calc_residual(S_previous, lictxt, cell_idx, deltat)
    cell_context = lictxt.cell_context[cell_idx]

    # get subset of state vector given by solver_view_cell
    S_next = lictxt.u_worksp # temporary workspace
    PB.get_statevar!(S_next, cell_context.solverview)

    # derivative at S_next for our subset of Variables and indices given by cellrange
    # NB: this will update all Variables including excluded Variables, not just those needed for nonlinear_solve
    PB.do_deriv(cell_context.dispatchlists, deltat)
    # get derivative
    dSdt_next = lictxt.cell_residual # temporary workspace
    PB.get_statevar_sms!(dSdt_next, cell_context.solverview)

    # calculate residual, accumulating norms as we go
    Linfnorm = 0.0
    L2norm_sq = 0.0
    @inbounds for i in eachindex(lictxt.cell_residual)
        lictxt.cell_residual[i] = 
            ((S_next[i] - S_previous[i] - deltat*dSdt_next[i])
            / cell_context.statevar_norm[i])
        rnormed = lictxt.cell_residual[i]
        Linfnorm = max(Linfnorm, abs(rnormed))
        L2norm_sq += rnormed^2        
    end
    L2norm = sqrt(L2norm_sq)

    return (Linfnorm, L2norm)
end

"Given the current residual in lictxt.residual,
and a current S_next estimate in modeldata,
do a Newton iteration to generate an improved estimate for S_next,
and save it back to modeldata.

   S_next <-- S_next - inv(I - deltat*J(S_next))*residual
"
function local_newton_update(::Val{Ncomps}, lictxt, cell_idx, t, deltat) where Ncomps

    Ncomps == length(lictxt.cell_residual) || error("Ncomps != length(cell_residual)")

    cell_context = lictxt.cell_context[cell_idx]

    # calculate Jacobian at current value of S_next 
    S_next = lictxt.u_worksp
    PB.get_statevar!(S_next, cell_context.solverview)
    cell_context.jac_cell(lictxt.J, S_next, nothing, t)
   
    # workspace to gather per-cell values
    cell_residual       = lictxt.cell_residual
    cell_var_norms      = cell_context.statevar_norm
    cell_var_sms_norms  = cell_context.statevar_sms_norm
    cell_mat            = lictxt.cell_mat
   
    # Newton update for this cell  
     
    # form cell_mat = I - deltat*(normalized Jacobian(S_next))
    fill!(cell_mat, 0.0)
    @inbounds for ij in 1:Ncomps
        cell_mat[ij, ij] = 1.0
    end
    # gather and normalize Jacobian
    @inbounds for j in 1:Ncomps, i in 1:Ncomps
        cell_mat[i, j] += -deltat * lictxt.J[i, j] / cell_var_sms_norms[i] * cell_var_norms[j]
    end
    
    # solve linear system, using StaticArrays for speed
    minusdeltaS = StaticArrays.SMatrix{Ncomps, Ncomps}(cell_mat) \ StaticArrays.SVector{Ncomps}(cell_residual)
    
    # calculate change in S_next estimate
    @inbounds for n in 1:Ncomps
        lictxt.u_worksp[n] = -(minusdeltaS[n] * cell_var_sms_norms[n])
    end
   
    # update S_next estimate
    PB.add_statevar!(cell_context.solverview, 1.0, lictxt.u_worksp)

    return nothing
end

end # module
