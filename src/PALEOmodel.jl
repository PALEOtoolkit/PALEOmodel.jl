module PALEOmodel

import PALEOboxes as PB

# autodiff setup
import ForwardDiff
import SparsityTracing

# Get scalar value from variable x (discarding any AD derivatives)
PB.value_ad(x::SparsityTracing.ADval) = SparsityTracing.value(x)
PB.value_ad(x::ForwardDiff.Dual) = ForwardDiff.value(x)

abstract type AbstractOutputWriter
end

# include("Plot.jl")

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

include("SteadyState.jl")

include("SteadyStateKinsol.jl")

include("FieldArray.jl")

include("FieldRecord.jl")

include("OutputWriters.jl")

include("PlotRecipes.jl")

include("ReactionNetwork.jl")

end
