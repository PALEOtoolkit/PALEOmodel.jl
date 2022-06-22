

# PALEOmodel solvers


## Initialization
```@meta
CurrentModule = PALEOmodel
```
```@docs
initialize!
```

## [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl) solvers

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

## Steady-state solvers (Julia [`NLsolve`](https://github.com/JuliaNLSolvers/NLsolve.jl) based)
```@meta
CurrentModule = PALEOmodel.SteadyState
```
```@docs
steadystate
steadystate_ptc
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

