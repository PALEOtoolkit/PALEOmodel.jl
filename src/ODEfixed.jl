module ODEfixed

# import Infiltrator

import PALEOboxes as PB

import ...PALEOmodel
using ...PALEOmodel: @public

@public integrateEuler, integrateSplitEuler, integrateEulerthreads, integrateSplitEulerthreads

###########################################################################
# Fixed-timestep, first-order Euler integrators 
###########################################################################

"""
    integrateEuler(run, initial_state, modeldata, tspan, Δt [; kwargs])

Integrate `run.model` from `initial_state` using first-order Euler with fixed timestep.

Calls [`integrateFixed`](@ref)

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tspan`:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times
- `Δt`: (yr) fixed timestep

# Keywords
- `outputwriter::PALEOmodel.AbstractOutputWriter=run.output`: container to write model output to
- `report_interval=1000`: number of timesteps between progress update to console
"""
function integrateEuler(
    run, initial_state, modeldata, tspan, Δt;
    outputwriter=run.output,
    report_interval=1000
)
    PB.check_modeldata(run.model, modeldata)

    timesteppers = [
        [(
            timestep_Euler,
            modeldata.cellranges_all,
            create_timestep_Euler_ctxt(
                run.model,
                modeldata,
            ),
        )],
    ]

    return integrateFixed(
        run, initial_state, modeldata, tspan, Δt,
        timesteppers=timesteppers,
        outputwriter=outputwriter,
        report_interval=report_interval
    )
end

"""
    integrateSplitEuler(run, initial_state, modeldata, tspan, Δt_outer, n_inner;
                            cellranges_outer, cellranges_inner,
                            [,outputwriter])

Integrate run.model representing:
```math
\\frac{dS}{dt} =  f_{outer}(t, S) + f_{inner}(t, S)
```
using split first-order Euler with `Δt_outer` for `f_outer` and a shorter timestep `Δt_outer/n_inner` for `f_inner`.

`f_outer` is defined by calling `PALEOboxes.do_deriv` with `cellranges_outer` hence corresponds to those
`Reactions` with `operatorID` of `cellranges_outer`.
`f_inner` is defined by calling `PALEOboxes.do_deriv` with `cellranges_inner` hence corresponds to those
`Reactions` with `operatorID` of `cellranges_inner`.

NB: the combined time derivative is written to `outputwriter`.

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tspan`:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times
- `Δt_outer`: (yr) fixed outer timestep 
- `n_inner`: number of inner timesteps per outer timestep

# Keywords
- `cellranges_outer`: Vector of `CellRange` with `operatorID` defining `f_outer`.
- `cellranges_inner`: Vector of `CellRange` with `operatorID` defining `f_inner`.
- `outputwriter::PALEOmodel.AbstractOutputWriter=run.output`: container to write model output to
- `report_interval=1000`: number of outer timesteps between progress update to console
"""
function integrateSplitEuler(
    run, initial_state, modeldata, tspan, Δt_outer, n_inner;
    cellranges_outer, 
    cellranges_inner,
    outputwriter=run.output,
    report_interval=1000
)

    @info "integrateSplitEuler: Δt_outer=$Δt_outer (yr) n_inner=$n_inner"

    PB.check_modeldata(run.model, modeldata)

    solver_view_outer = PALEOmodel.SolverView(run.model, modeldata, 1, cellranges_outer)
    @info "solver_view_outer: $(solver_view_outer)"    
    solver_view_inner = PALEOmodel.SolverView(run.model, modeldata, 1, cellranges_inner)
    @info "solver_view_inner: $(solver_view_inner)"
    
    timesteppers = [
        [(
            timestep_Euler, 
            cellranges_outer, 
            create_timestep_Euler_ctxt(
                run.model,
                modeldata,
                solver_view=solver_view_outer,
                cellranges=cellranges_outer,
            )
        )],
        [(
            timestep_Euler,
            cellranges_inner, 
            create_timestep_Euler_ctxt(
                run.model,
                modeldata,
                solver_view=solver_view_inner,
                cellranges=cellranges_inner,
                n_substep=n_inner,
            )
        )],
    ]

    return integrateFixed(
        run, initial_state, modeldata, tspan, Δt_outer,
        timesteppers=timesteppers,
        outputwriter=outputwriter,
        report_interval=report_interval,
    )
end                      




"""
    integrateEulerthreads(run, initial_state, modeldata, cellranges, tspan, Δt;
        outputwriter=run.output, report_interval=1000)

Integrate run.model using first-order Euler with fixed timestep `Δt`, with tiling over multiple threads.

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `cellranges::Vector{Vector{AbstractCellRange}}`: Vector of Vector-of-cellranges, one per thread (so length(cellranges) == Threads.nthreads).
- `tspan`:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times
- `Δt`: (yr) fixed outer timestep

# Keywords
- `outputwriter::PALEOmodel.AbstractOutputWriter=run.output`: container to write model output to
- `report_interval=1000`: number of outer timesteps between progress update to console
"""
function integrateEulerthreads(
    run, initial_state, modeldata, cellranges, tspan, Δt;
    outputwriter=run.output,
    report_interval=1000,
)
    PB.check_modeldata(run.model, modeldata)

    nt = Threads.nthreads()
    nt == 1 || get(modeldata.model.parameters, "threadsafe", false)  || 
        error("integrateEulerthreads: Threads.nthreads() = $nt but model is not thread safe "*
            "(set 'threadsafe=true' in YAML config top-level 'parameters:')")

    lc = length(cellranges)
    lc == nt ||
        error("integrateEulerthreads: length(cellranges) $lc != nthreads $nt")

    # get solver_views for each threadid
    solver_views = [PALEOmodel.SolverView(run.model, modeldata, 1, crs) for crs in cellranges]
    @info "integrateEulerthreads: solver_views:" 
    for t in 1:Threads.nthreads()
        @info "  thread $t  $(solver_views[t])"
    end

    timesteppers = [
        [
            (
                timestep_Euler,
                cellranges[t], 
                create_timestep_Euler_ctxt(
                    run.model, modeldata,
                    cellranges=cellranges[t],
                    solver_view=solver_views[t]
                )
            )
            for t in 1:Threads.nthreads()
        ]
    ]
    
    return integrateFixedthreads(
        run, initial_state, modeldata, tspan, Δt,
        timesteppers=timesteppers,
        outputwriter=outputwriter,
        report_interval=report_interval
    )
    
end


"""
    integrateSplitEulerthreads(run, initial_state, modeldata, tspan, Δt_outer, n_inner::Int; 
                                cellranges_outer, cellranges_inner, [,outputwriter] [, report_interval])

Integrate run.model using split first-order Euler with `Δt_outer` for `f_outer` and a shorter timestep `Δt_outer/n_inner` for `f_inner`.

`f_outer` is defined by calling `PALEOboxes.do_deriv` with `cellrange_outer` hence corresponds to those `Reactions` with `operatorID`
of `cellrange_outer`.  `f_inner` is defined by calling `PALEOboxes.do_deriv` with `cellrange_inner` hence corresponds to those `Reactions` with `operatorID`
of `cellrange_inner`.
 
Uses `Threads.nthreads` threads and tiling described by `cellranges_inner` and `cellranges_outer`
(each a `Vector` of `Vector{AbstractCellRange}`, one per thread).

# Arguments
- `run::Run`: struct with `model::PB.Model` to integrate and `output` field
- `initial_state::AbstractVector`: initial state vector
- `modeldata::Modeldata`: ModelData struct with appropriate element type for forward model
- `tspan`:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times
- `Δt_outer`: (yr) fixed outer timestep 
- `n_inner`: number of inner timesteps per outer timestep (0 for non-split solver)

# Keywords
- `cellranges_outer::Vector{Vector{AbstractCellRange}}`: Vector of list-of-cellranges, one list per thread (so length(cellranges) == Threads.nthreads), with `operatorID` defining `f_outer`.
- `cellranges_inner::Vector{Vector{AbstractCellRange}}`: As `cellranges_outer`, with `operatorID` defining `f_inner`.
- `outputwriter::PALEOmodel.AbstractOutputWriter=run.output`: container to write model output to
- `report_interval=1000`: number of outer timesteps between progress update to console
"""
function integrateSplitEulerthreads(
    run, initial_state, modeldata, tspan, Δt_outer, n_inner::Int;
    cellranges_outer,
    cellranges_inner,
    outputwriter=run.output,
    report_interval=1000,
)
    PB.check_modeldata(run.model, modeldata)

    nt = Threads.nthreads()
    nt == 1 || get(modeldata.model.parameters, "threadsafe", false)  || 
        error("integrateSplitEulerthreads: Threads.nthreads() = $nt but model is not thread safe "*
            "(set 'threadsafe=true' in YAML config top-level 'parameters:')")

    lc_outer = length(cellranges_outer)
    lc_outer == nt || 
        error("integrateSplitEulerthreads: length(cellranges_outer) $lc_outer != nthreads $nt")
    lc_inner = length(cellranges_inner)
    lc_inner == nt ||
        error("integrateSplitEulerthreads: length(cellranges_inner) $lc_inner != nthreads $nt")

    # get solver_views for each threadid
    solver_views_outer = [PALEOmodel.SolverView(run.model, modeldata, 1, crs) for crs in cellranges_outer]
    solver_views_inner = [PALEOmodel.SolverView(run.model, modeldata, 1, crs) for crs in cellranges_inner]
    @info "integrateSplitEulerthreads: solver_views_outer:" 
    for t in 1:Threads.nthreads()
        @info "  thread $t $(solver_views_outer[t])"
    end
    @info "integrateSplitEulerthreads: solver_views_inner:" 
    for t in 1:Threads.nthreads()
        @info "  thread $t $(solver_views_inner[t])"
    end

    timesteppers = [
        [
            (
                timestep_Euler,
                cellranges_outer[t], 
                create_timestep_Euler_ctxt(
                    run.model, modeldata,
                    cellranges=cellranges_outer[t],
                    solver_view=solver_views_outer[t],
                )
            )
            for t in 1:Threads.nthreads()
        ],
        [
            (
                timestep_Euler,
                cellranges_inner[t], 
                create_timestep_Euler_ctxt(
                    run.model, modeldata,
                    cellranges=cellranges_inner[t],
                    solver_view=solver_views_inner[t],
                    n_substep=n_inner,
                )
            )
            for t in 1:Threads.nthreads()
        ],        
    ]

    return integrateFixedthreads(
        run, initial_state, modeldata, tspan, Δt_outer,
        timesteppers=timesteppers,
        outputwriter=outputwriter,
        report_interval=report_interval
    )
end


###############################################
# Timesteppers
##############################################

function timestep_Euler(
    model, modeldata, cellranges,
    (dispatch_lists, solver_view, n_substep), 
    touter, Δt, threadid, report;
    deriv_only=false,
    integrator_barrier=nothing,
)
 
    Δt_inner = Δt / n_substep

    if deriv_only
        # for reporting output fluxes etc
        if threadid == 1
            PALEOmodel.set_tforce!(solver_view, touter)
        end
        PB.do_deriv(dispatch_lists, Δt_inner)
    else
        # update state variables
        for n in 1:n_substep
            tinner = touter + (n-1)*Δt_inner
            if threadid == 1
                PALEOmodel.set_tforce!(solver_view, tinner)
            end
            PB.do_deriv(dispatch_lists, Δt_inner)
            PALEOmodel.ThreadBarriers.wait_barrier(integrator_barrier)
            PB.add_data!(solver_view.stateexplicit, Δt_inner, solver_view.stateexplicit_deriv)   
        end
    end
    return nothing
end

function create_timestep_Euler_ctxt(
    model, modeldata;
    cellranges=modeldata.cellranges_all,          
    solver_view=modeldata.solver_view_all,
    n_substep=1,
    verbose=false,
    generated_dispatch=true,
)
    PB.check_modeldata(model, modeldata)

    num_constraints = PALEOmodel.num_algebraic_constraints(solver_view)
    iszero(num_constraints) || error("DAE problem with $num_constraints algebraic constraints")

    dispatch_lists = PB.create_dispatch_methodlists(model, modeldata, 1, cellranges; verbose, generated_dispatch)

    return (dispatch_lists, solver_view, n_substep)
end


##############################################################
# Integrators
##############################################################

"""
    integrateFixed(run, initial_state, modeldata, tspan, Δt_outer;
                timesteppers, outputwriter=run.output, report_interval=1000)

Fixed timestep integration, with time step implemented by `timesteppers`,

    `timesteppers = [ [(timestep_function, cellranges, timestep_ctxt), ...], ...]`

Where `timestep_function(model, modeldata, cellranges, timestep_ctxt, touter, Δt, barrier)`
"""
function integrateFixed(
    run, initial_state, modeldata, tspan, Δt_outer;
    timesteppers,
    outputwriter=run.output,
    report_interval=1000
)
    PB.check_modeldata(run.model, modeldata)

    nevals = 0

    isnothing(outputwriter) || 
        PALEOmodel.OutputWriters.initialize!(outputwriter, run.model, modeldata, length(tspan))
   
    # write an output record for time tmodel
    function write_output_record(outputwriter, model, modeldata, timesteppers, tmodel, Δt_outer)
        for splitstep in timesteppers
            for (timestepper, cellranges, ctxt) in splitstep
                timestepper(
                    model, modeldata, cellranges, ctxt, tmodel, Δt_outer, 1, false,
                    deriv_only=true # don't update state variables !
                )
            end
        end

        PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, tmodel)
        return nothing
    end
    write_output_record(outputwriter::Nothing, model, modeldata, timesteppers, tmodel, Δt_outer) = nothing

    # set initial state
    touter = Float64(tspan[1])

    PALEOmodel.set_tforce!(modeldata.solver_view_all, touter)

    PALEOmodel.set_statevar!(modeldata.solver_view_all, initial_state)

    # write initial state and combined time derivative
    write_output_record(outputwriter, run.model, modeldata, timesteppers, touter, Δt_outer)
    inextoutput = 2

    while inextoutput <= length(tspan)

        report = (report_interval != 0) && (mod(nevals, report_interval) == 0)

        if report
            @info "  nevals $(nevals)    tmodel $touter"
        end

        for splitstep in timesteppers
            for (timestepper, cellranges, ctxt) in splitstep
                timestepper(run.model, modeldata, cellranges, ctxt, touter, Δt_outer, 1, report)
            end
        end
        
        nevals += 1
        # advance model time 
        touter += Δt_outer
        
        # write output record if needed
        if touter >= tspan[inextoutput]
            write_output_record(outputwriter, run.model, modeldata, timesteppers, touter, Δt_outer)            
            inextoutput += 1
        end
    end

    @info "integrateFixed: end, nevals $(nevals)    tmodel $touter"

    return nothing
end

"""
    integrateFixedthreads(run, initial_state, modeldata, tspan, Δt_outer;
                timesteppers, outputwriter=run.output, report_interval=1000)

Fixed timestep integration using `Threads.nthreads()` threads, with time step implemented by `timesteppers`,

    `timesteppers = [ [(timestep_function, cellranges, timestep_ctxt), ... (1 per thread)], ...]`

Where `timestep_function(model, modeldata, cellranges, timestep_ctxt, touter, Δt, barrier)`.
"""
function integrateFixedthreads(
    run, initial_state, modeldata, tspan, Δt_outer;
    timesteppers,
    outputwriter=run.output,
    report_interval=1000
)
    PB.check_modeldata(run.model, modeldata)

    nevals = 0
       
    isnothing(outputwriter) || 
        PALEOmodel.OutputWriters.initialize!(outputwriter, run.model, modeldata, length(tspan))


    # write an output record for time tmodel, including combined f_outer and f_inner derivative
    # NB: no synchronisation on entry and exit
    function write_output_record(outputwriter, barrier, model, modeldata, timesteppers, tmodel, Δt_outer, tid)
      
        for splitstep in timesteppers
            (timestepper, cellranges, ctxt) = splitstep[tid]
            timestepper(
                model, modeldata, cellranges, ctxt, tmodel, Δt_outer, tid, false,
                deriv_only=true, # don't update state variables !
                integrator_barrier=barrier,
            )
        end
        # no barrier at end of timesteppers
        PALEOmodel.ThreadBarriers.wait_barrier(barrier)

        if tid == 1                
            PALEOmodel.OutputWriters.add_record!(outputwriter, model, modeldata, touter)
        end
           
        return nothing
    end   
    write_output_record(outputwriter::Nothing, barrier, model, modeldata, timesteppers, tmodel, Δt_outer, tid) = nothing

    # write initial state
    touter = Float64(tspan[1])
    PALEOmodel.set_tforce!(modeldata.solver_view_all, touter)
    PALEOmodel.set_statevar!(modeldata.solver_view_all, initial_state)
    inextoutput = 2

    integrator_barrier = PALEOmodel.ThreadBarriers.ThreadBarrierAtomic("integrator barrier")

    Threads.@threads :static for t in 1:Threads.nthreads()
        tid = Threads.threadid()
        @info "start thread $tid"

        write_output_record(
            outputwriter, integrator_barrier, run.model, modeldata, timesteppers, touter, Δt_outer, tid
        ) 

        while inextoutput <= length(tspan)
           
            for splitstep in timesteppers
                 # timesteppers have barrier at start
                (timestepper, cellranges, ctxt) = splitstep[tid]
                timestepper(
                    run.model, modeldata, cellranges, ctxt, touter, Δt_outer, tid, false,
                    integrator_barrier=integrator_barrier,
                )
            end
            # no barrier at end of timestepper

            if tid == 1
                if report_interval != 0 && mod(nevals, report_interval) == 0
                    @info "  nevals $(nevals)    tmodel $touter"
                end
                nevals += 1
                # advance model time 
                touter += Δt_outer
            end
          
            # barrier so all threads see updated touter before test for output writing
            PALEOmodel.ThreadBarriers.wait_barrier(integrator_barrier) 

            # write output record if needed
            if touter >= tspan[inextoutput]               
                write_output_record(
                    outputwriter, integrator_barrier, run.model, modeldata, timesteppers, touter, Δt_outer, tid
                )
                if tid == 1
                    inextoutput += 1
                end
                # barrier so all threads see updated inextoutput before loop test
                PALEOmodel.ThreadBarriers.wait_barrier(integrator_barrier) 
            end
            
        end

        @info "complete thread $tid"
    end
    
    @info "integrateFixedthreads: nevals $(nevals)"

    return nothing
end


end
