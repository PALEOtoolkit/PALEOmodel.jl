

# PALEOmodel solvers

## Initialization
```@meta
CurrentModule = PALEOmodel
```
```@docs
initialize!
set_statevar_from_output!
```

## DifferentialEquations solvers

Wrappers for the Julia [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl) package ODE and DAE solvers.  These are usually appropriate for smaller biogeochemical models.

NB: see [Managing small and negative values](@ref) for best practices and common issues when using ODE or DAE solvers.

### High level wrappers
```@meta
CurrentModule = PALEOmodel.ODE
```
```@docs
integrate
integrateDAE
```

### Low level functions
```@meta
CurrentModule = PALEOmodel.ODE
```
```@docs
ODEfunction

DAEfunction
get_inconsistent_initial_deriv

print_sol_stats
calc_output_sol!
```

## Steady-state solvers (Julia [`NLsolve`](https://github.com/JuliaNLSolvers/NLsolve.jl) based)
```@meta
CurrentModule = PALEOmodel.SteadyState
```
```@docs
steadystate
steadystate_ptc
steadystate_ptc_splitdae
```

Function objects to project Newton steps into valid regions:

```@meta
CurrentModule = PALEOmodel.SolverFunctions
```
```@docs
ClampAll!
ClampAll
```

## Steady-state solvers (Sundials Kinsol based):
```@meta
CurrentModule = PALEOmodel.SteadyStateKinsol
```
```@docs
steadystate_ptc
```
```@meta
CurrentModule = PALEOmodel
```
```@docs
Kinsol
Kinsol.kin_create
Kinsol.kin_solve
```

## Fixed timestep solvers
```@meta
CurrentModule = PALEOmodel.ODEfixed
```
PALEO native fixed-timestep, first-order Euler integrators, with split-explicit and multi-threaded options.
These are usually appropriate for larger biogeochemical models (eg ocean models using GCM transport matrices).

The low-level timestepping is provided by [`integrateFixed`](@ref) and [`integrateFixedthreads`](@ref), 
with higher-level wrappers for common options provided by [`integrateEuler`](@ref) etc.

### High-level wrappers
```@meta
CurrentModule = PALEOmodel.ODEfixed
```
```@docs
integrateEuler
integrateSplitEuler
integrateEulerthreads
integrateSplitEulerthreads
```
```@meta
CurrentModule = PALEOmodel.ODELocalIMEX
```
```@docs
integrateLocalIMEXEuler
```
### Low-level timesteppers
```@meta
CurrentModule = PALEOmodel.ODEfixed
```
```@docs
integrateFixed
integrateFixedthreads
```

### Thread barriers
```@meta
CurrentModule = PALEOmodel.ThreadBarriers
```
```@docs
ThreadBarrierAtomic
ThreadBarrierCond
```

## Variable aggregation
```@meta
CurrentModule = PALEOmodel
```
A [`SolverView`](@ref) uses a collection of `PALEOboxes.VariableAggregator`s to assemble model state Variables and associated time derivatives into contiguous Vectors, for the convenience of standard numerical ODE / DAE solvers.  See [Mathematical formulation of the reaction-transport problem](@ref). 
```@docs
SolverView
set_default_solver_view!
copy_norm!
set_statevar!
get_statevar_sms!
```

## Function objects
```@meta
CurrentModule = PALEOmodel.SolverFunctions
```
Function objects are callable structs with function signatures required by DifferentialEquations or other solvers to calculate
model time derivative, Jacobian, etc.
They combine variable aggregation (using `PALEOboxes.VariableAggregator`s or [`PALEOmodel.SolverView`](@ref)) with corresponding
Reaction dispatch lists.

### ODE function objects
```@docs
ModelODE
ModelODE_at_t
JacODEForwardDiffDense
JacODEForwardDiffSparse
JacODE_at_t
```

### DAE function objects
```@docs
ModelDAE
JacDAE
TotalForwardDiff
ImplicitForwardDiffDense
ImplicitForwardDiffSparse
```