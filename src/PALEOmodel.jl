"""
    PALEOmodel

PALEOmodel.jl is a PALEOtoolkit package that provides modules to numerically solve
a standalone model and analyse output interactively from the Julia REPL. It implements:

- Numerical solvers
- Data structures to hold model output
- Output plot recipes

It is registered as a Julia package with public github repository
[PALEOmodel.jl](https://github.com/PALEOtoolkit/PALEOmodel.jl) 
and online 
[documentation](https://paleotoolkit.github.io/PALEOmodel.jl)
"""
module PALEOmodel

import PALEOboxes as PB

# autodiff setup
import ForwardDiff
import ForwardDiff
import SparsityTracing
import OrderedCollections
using DocStringExtensions

import TimerOutputs: @timeit, @timeit_debug

# https://discourse.julialang.org/t/is-compat-jl-worth-it-for-the-public-keyword/119041/22
# define @public for backwards compatibility with Julia < v1.11
macro public(ex)
    if VERSION >= v"1.11"
        args = ex isa Symbol ? (ex,) : Base.isexpr(ex, :tuple) ? ex.args : error("@public parse error")
        esc(Expr(:public, args...))
    else
        nothing
    end
end

@public Run, initialize!, set_statevar_from_output!
@public SolverView, set_default_solver_view!, copy_norm!, set_statevar!, set_tforce, get_statevar!, get_statevar_sms!
@public SolverFunctions, JacobianAD, ODE, ThreadBarriers, ODEfixed, ODELocalIMEX, SteadyState, SteadyStateKinsol
@public AbstractOutputWriter, OutputWriters, FieldRecord, FieldArray, get_array
@public PlotPager
@public ReactionNetwork


# Get scalar value from variable x (discarding any AD derivatives)
PB.value_ad(x::SparsityTracing.ADval) = SparsityTracing.value(x)
PB.value_ad(x::ForwardDiff.Dual) = ForwardDiff.value(x)

abstract type AbstractOutputWriter
end


include("SparseUtils.jl")

include("SolverView.jl")

include("Initialize.jl")

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

include("CoordsDims.jl") 

include("FieldRecord.jl")

include("OutputWriters.jl")

include("PlotRecipes.jl")

include("ReactionNetwork.jl")

include("ForwardDiffWorkarounds.jl")


end
