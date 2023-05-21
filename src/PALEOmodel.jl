module PALEOmodel

import Logging
import ForwardDiff
import StaticArrays
import SparsityTracing
import PrecompileTools

import PALEOboxes as PB



import TimerOutputs: @timeit, @timeit_debug

# Get scalar value from variable x (discarding any AD derivatives)
PB.value_ad(x::SparsityTracing.ADval) = SparsityTracing.value(x)
PB.value_ad(x::ForwardDiff.Dual) = ForwardDiff.value(x)

abstract type AbstractOutputWriter
end

include("SparseUtils.jl")

include("SolverView.jl")

include("Run.jl")

include("SolverFunctions.jl")

include("JacobianAD.jl")

include("ThreadBarriers.jl")

include("NonLinearNewton.jl")

include("ODE.jl")

include("ODEfixed.jl")

include("ODELocalIMEX.jl")

include("Kinsol.jl")

include("SplitDAE.jl")

include("SteadyState.jl")

include("SteadyStateKinsol.jl")

include("FieldArray.jl")

include("FieldRecord.jl")

include("OutputWriters.jl")

include("PlotRecipes.jl")

include("ReactionNetwork.jl")

include("ForwardDiffWorkarounds.jl")

# workload for PrecompileTools
function run_example_workload()

    # Minimal model 
    model = PB.create_model_from_config(
        joinpath(@__DIR__, "../test/configreservoirs.yaml"),
        "model1",
    )

    initial_state, modeldata = PALEOmodel.initialize!(model)

    # DAE solver
    paleorun = PALEOmodel.Run(model=model, output = PALEOmodel.OutputWriters.OutputMemory())
    PALEOmodel.ODE.integrateDAEForwardDiff(
        paleorun, initial_state, modeldata, (0.0, 1.0),
        solvekwargs=(reltol=1e-5,),
    )

    # ODE solver
    paleorun = PALEOmodel.Run(model=model, output = PALEOmodel.OutputWriters.OutputMemory())
    PALEOmodel.ODE.integrateForwardDiff(
        paleorun, initial_state, modeldata, (0.0, 1.0),
        solvekwargs=(reltol=1e-5,),
    )


    # save and load
    tmpfile = tempname(; cleanup=true) 
    output = paleorun.output
    PALEOmodel.OutputWriters.save_netcdf(output, tmpfile; check_ext=false)
    load_output = PALEOmodel.OutputWriters.load_netcdf!(PALEOmodel.OutputWriters.OutputMemory(), tmpfile; check_ext=false)

    # FieldArray
    O_array = PALEOmodel.get_array(load_output, "global.O")
    T_conc_array = PALEOmodel.get_array(load_output, "ocean.T_conc", (cell=1,))

    return nothing
end


@PrecompileTools.setup_workload begin
 
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.

    logger = Logging.NullLogger()
    # logger = Logging.ConsoleLogger()

    @PrecompileTools.compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)

        try
            Logging.with_logger(logger) do
                run_example_workload()
            end

        catch ex
            @info "precompile failed with exception:" ex
        end
    end

end

end
