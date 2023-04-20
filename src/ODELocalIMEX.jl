module ODELocalIMEX

import Infiltrator

import PALEOboxes as PB

import PALEOmodel
import ..NonLinearNewton

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

    PB.check_modeldata(run.model, modeldata)

    solver_view_outer = PALEOmodel.SolverView(run.model, modeldata, 1, cellranges_outer)
    @info "solver_view_outer: $(solver_view_outer)"    
    
    lictxt = create_timestep_LocalImplicit_ctxt(
        run.model, modeldata;                                   
        cellrange=cellrange_inner,
        exclude_var_nameroots,
        niter_max,
        Lnorm_inf_max,
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
    PB.check_modeldata(model, modeldata)

    length(cellranges) == 1 || error("timestep_LocalImplicit only single cellrange supported")
    cellrange = cellranges[1]
    
    if deriv_only
        # for reporting output fluxes etc
        PALEOmodel.set_tforce!(modeldata.solver_view_all, touter + Δt)
        # NB: cell_number corresponds to 1:number_of_cells, not the actual cell indices in cellrange.indices
        for cell_number in 1:length(cellrange.indices)
            PB.do_deriv(lictxt.cell_residual.dispatchlists[cell_number], Δt)
        end
    else
        # implicit timestep

        niter_max = 0
        niter_total = 0
        (Lnorm_inf_init, Lnorm_2_init) = (0.0, 0.0)
        (Lnorm_inf, Lnorm_2) = (0.0, 0.0)

        cell_residual = lictxt.cell_residual
        cell_jacobian = lictxt.cell_jacobian

        # NB: cell_number corresponds to 1:number_of_cells, not the actual cell indices in cellrange.indices
        for cell_number in 1:length(cellrange.indices)
            
            lastS, lastSnorm = get_S(cell_residual, cell_number)

            Lnorm_inf_init = max(LinearAlgebra.norm(lastSnorm, Inf), Lnorm_inf_init)
            Lnorm_2_init += LinearAlgebra.norm(lastSnorm)^2
           
            set_cell!(cell_residual, cell_number, lastS, touter+Δt, Δt)
            set_cell!(cell_jacobian, cell_number, lastS, touter+Δt, Δt)

            (_, Lnorm_2_cell, Lnorm_inf_cell, niter) = NonLinearNewton.solve(
                cell_residual,
                cell_jacobian, 
                lastSnorm;
                reltol=lictxt.Lnorm_inf_max,
                maxiters=lictxt.niter_max,
                verbose = 0,
            )
            
            niter >= lictxt.niter_max && 
                @warn "  tmodel $touter cellindex $(cell_residual.cellindices[cell_number]) implicit niter_max $(lictxt.niter_max) exceeded "*
                    "(Lnorm_inf, Lnorm_2) ($Lnorm_inf_init_cell, $Lnorm_2_init_cell) -> ($Lnorm_inf_cell, $Lnorm_2_cell)"

            # update global stats
            niter_total += niter
            niter_max = max(niter, niter_max)
            Lnorm_inf = max(Lnorm_inf_cell, Lnorm_inf)
            Lnorm_2 += Lnorm_2_cell^2          
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
    Lnorm_inf_max,
    request_adchunksize=ForwardDiff.DEFAULT_CHUNK_THRESHOLD,
    generated_dispatch=true,
)
    PB.check_modeldata(model, modeldata)

    # create SolverViews for first cell, to work out how many dof we need
    cellrange_cell = PB.CellRange(cellrange.domain, cellrange.operatorID, first(cellrange.indices) )

    # get SolverView for all Variables required for cellrange derivative
    solver_view_all_cell = PALEOmodel.SolverView(
        model, modeldata, 1, [cellrange_cell];
        indices_from_cellranges=true
    )
    @info "getLocalImplicitContext: all Variables (first cell) $solver_view_all_cell)"  
 
    # get reduced set of Variables required for nonlinear solve
    solver_view_cell = PALEOmodel.SolverView(
        model, modeldata, 1, [cellrange_cell];
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
    va_excluded = PB.VariableAggregator(excluded_vars, [cellrange for v in excluded_vars], modeldata, 1)
    excluded_sms_vars = [
        v for v in solver_view_all_cell.stateexplicit_deriv.vars if !(v in solver_view_cell.stateexplicit_deriv.vars)
    ]
    va_sms_excluded = PB.VariableAggregator(excluded_sms_vars, [cellrange for v in excluded_sms_vars], modeldata, 1)
    length(excluded_vars) == length(excluded_sms_vars) ||
        error("excluded_vars and excluded_sms_vars length mismatch")

    n_all = length(PALEOmodel.get_statevar(solver_view_all_cell))
    n_solve = length(PALEOmodel.get_statevar(solver_view_cell))

    cellrange_length = length(cellrange.indices)
    @info "  n_all $n_all solving for $n_solve dof x domain length $cellrange_length = $(n_solve*cellrange_length)"
    @info "  solve vars: $([PB.fullname(v) for v in solver_view_cell.stateexplicit.vars]) sms_vars: $([PB.fullname(v) for v in solver_view_cell.stateexplicit_deriv.vars])"
    @info "  excluded vars: $([PB.fullname(v) for v in excluded_vars]) sms_vars: $([PB.fullname(v) for v in excluded_sms_vars])"
   
    # Add an array set with Dual numbers to modeldata, for AD Jacobians
    chunksize = ForwardDiff.pickchunksize(n_solve, request_adchunksize)
    @info "  using ForwardDiff dense Jacobian chunksize=$chunksize"
    eltype_base = eltype(modeldata, 1)
    eltype_jac_cell = ForwardDiff.Dual{Nothing, eltype_base, chunksize}
    PB.add_arrays_data!(model, modeldata, eltype_jac_cell, "jac_cell")
    arrays_idx_jac_cell = PB.num_arrays(modeldata)
 
    # temporarily replace modeldata with norm so can read back per-cell norms
    statevar_all_current = PALEOmodel.get_statevar(modeldata.solver_view_all)
    PALEOmodel.uncopy_norm!(modeldata.solver_view_all)
 
    # create per-cell solver_view, dispatchlists
    cellindices = Int64[]
    solverviews, solverviews_ad = [], []
    dispatchlists, dispatchlists_ad = [], []
    statevar_norms = []
    for i in cellrange.indices
        push!(cellindices, i)
        # (ab)use that fact that Julia allows iteration over scalar i (to optimise out loop over cellrange.indices)
        cellrange = PB.CellRange(cellrange.domain, cellrange.operatorID, i) 

        sv = PALEOmodel.SolverView(
            model, modeldata, 1, [cellrange];
            exclude_var_nameroots,
            indices_from_cellranges=true,
            hostdep_all=false,
        )
        push!(solverviews, sv)

        PALEOmodel.copy_norm!(sv)
        statevar_norm = PALEOmodel.get_statevar_norm(sv)
        push!(statevar_norms, statevar_norm)        
        
        dl = PB.create_dispatch_methodlists(model, modeldata, 1, [cellrange]; generated_dispatch)
        push!(dispatchlists, dl)

        sv_ad = PALEOmodel.SolverView(
            model, modeldata, arrays_idx_jac_cell, [cellrange]; 
            exclude_var_nameroots,
            indices_from_cellranges=true,
            hostdep_all=false,
        )
        push!(solverviews_ad, sv_ad)
       
        dl_ad = PB.create_dispatch_methodlists(model, modeldata, arrays_idx_jac_cell, [cellrange]; generated_dispatch)

        push!(dispatchlists_ad, dl_ad)

    end
       
    # replace modeldata statevar (temporarily was set to norm)
    PALEOmodel.set_statevar!(modeldata.solver_view_all, statevar_all_current)

    cell_residual = CellResidual(
        n_solve,
        eltype_base,
        cellindices,
        [sv for sv in solverviews],  # narrow type
        [dl for dl in dispatchlists], # narrow type
        [svn for svn in statevar_norms], # narrow type
    )

    cell_jacobian = CellJacobian(
        CellResidual(
            n_solve,
            eltype_jac_cell,
            cellindices,
            [sv for sv in solverviews_ad],  # narrow type
            [dl for dl in dispatchlists_ad], # narrow type
            [svn for svn in statevar_norms], # narrow type
        )
    )

    return (;niter_max, Lnorm_inf_max, n_solve, cell_residual, cell_jacobian, va_excluded, va_sms_excluded)
end

"""
    CellResidual(Ncomps, wseltype, cellindices, solverviews, dispatchlists, statevar_norms)

Callable struct to calculate residual for an implicit timestep for a set of cells, one cell at a time.

# Arguments
- `Ncomps`: number of variable components
- `wseltype`: element type of variable (eg `Float64`)
- `cellindices`: cell indices used (for diagnostic output only)
- `solverviews`: Vector of per-cell PALEOmodel.SolverView
- `dispatchlists`: Vector of per-cell dispatchlists
- `statevar_norms`: Vector of per-cell variable component norms

# Usage
To calculate the residual for one cell:
- Call `set_cell!(cr::CellResidual, celln, lastS, t, deltat)` to set cell number `celln`, state variables `lastS` at previous timestep, 
  time `t` to step to and timestep `deltat`.
  NB: `celln` corresponds to 1:length(cellindices) ie `solverviews`, `dispatchlists`, `statevar_norms` are stored as 1-based Vectors
- Call `cr(newS)` to calculate the residual for new state variables `newS`.
"""
mutable struct CellResidual{Ncomps, T, SV, DL}
    cellindices::Vector{Int64} # cell indices 
    solverviews::Vector{SV}  # vector, 1:length(cellindices)
    dispatchlists::Vector{DL}    # vector, 1:length(cellindices)
    
    statevar_norms::Vector{Vector{Float64}}

    celln::Int64
    lastSnorm::Vector{T}
    deltat::Float64
    
    newS::Vector{T}   # workspace
    residual::Vector{T}  # workspace
    dSnormdt::Vector{T}   # workspace
end

function CellResidual(Ncomps, wseltype, cellindices, solverviews, dispatchlists, statevar_norms)
    return CellResidual{Ncomps, wseltype, eltype(solverviews), eltype(dispatchlists)}(
        cellindices,
        solverviews,
        dispatchlists,
        statevar_norms,
        -1,
        Vector{wseltype}(undef, Ncomps),
        NaN,
        Vector{wseltype}(undef, Ncomps),
        Vector{wseltype}(undef, Ncomps),
        Vector{wseltype}(undef, Ncomps),
    )
end

ncomps(cr::CellResidual{Ncomps, WS, SV, DL}) where {Ncomps, WS, SV, DL} = Ncomps

function get_S(cr::CellResidual, celln)
    S, Snorm = cr.newS, cr.dSnormdt # use workspace arrays
    PALEOmodel.get_statevar!(S, cr.solverviews[celln])
    Snorm .= S ./ cr.statevar_norms[celln]
    return StaticArrays.SVector{ncomps(cr)}(S), StaticArrays.SVector{ncomps(cr)}(Snorm)
end

function set_cell!(cr::CellResidual, celln, lastS, t, deltat)
    
    cr.celln = celln
    statevar_norm = cr.statevar_norms[cr.celln]
    cr.lastSnorm .= lastS ./ statevar_norm

    PALEOmodel.set_tforce!(cr.solverviews[cr.celln], t)
    cr.deltat = deltat
end

function (cr::CellResidual)(newSnorm)
    statevar_norm = cr.statevar_norms[cr.celln]
    solverview = cr.solverviews[cr.celln]
    dispatchlists = cr.dispatchlists[cr.celln]

    cr.newS .= newSnorm .*statevar_norm
    PALEOmodel.set_statevar!(solverview, cr.newS)
    # derivative at S_next for our subset of Variables and indices given by cellrange
    # NB: this will update all Variables including excluded Variables, not just those needed for nonlinear_solve
    PB.do_deriv(dispatchlists, cr.deltat)    

    # get derivative and normalize
    PALEOmodel.get_statevar_sms!(cr.dSnormdt, solverview)
    cr.dSnormdt .= cr.dSnormdt ./ statevar_norm

    # calculate residual
    @inbounds for i in eachindex(cr.residual)
        cr.residual[i] = newSnorm[i] - cr.lastSnorm[i] - cr.deltat*cr.dSnormdt[i]
    end

    return StaticArrays.SVector{ncomps(cr)}(cr.residual)
end

mutable struct CellJacobian{CR}
    cellresidual::CR
end

set_cell!(cj::CellJacobian, celln, lastS, t, deltat) = set_cell!(cj.cellresidual, celln, lastS, t, deltat)

@inline function (cj::CellJacobian)(newSnorm)
    # return ForwardDiff.jacobian(
    #     cj.cellresidual,
    #     newSnorm,
    #     ForwardDiff.JacobianConfig(nothing, newSnorm),
    #     Val(false)
    # )

    # TODO workaround for a limitation of ForwardDiff - can't specify a jacobian with no 'tag' when using StaticArrays
    return PALEOmodel.ForwardDiffWorkarounds.vector_mode_jacobian_notag(cj.cellresidual, newSnorm)
end

end # module
