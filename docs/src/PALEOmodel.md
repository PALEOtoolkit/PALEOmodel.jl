

# PALEOmodel

```@meta
CurrentModule = PALEOmodel
```
```@docs
Run
```
## Create and initialize
```@docs
initialize!
```


## Integrate
```@meta
CurrentModule = PALEOmodel.ODE
```
### DifferentialEquations solvers
```@docs
integrate
integrateDAE
integrateForwardDiff
integrateDAEForwardDiff
```
### Fixed timestep solvers
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
#### Fixed timestep wrappers

```@meta
CurrentModule = PALEOmodel.ODEfixed
```
```@docs
integrateFixed
integrateFixedthreads
```

### Steady-state solvers (Julia NLsolve based)
```@meta
CurrentModule = PALEOmodel.SteadyState
```
```@docs
steadystate
steadystateForwardDiff
steadystate_ptc
steadystate_ptcForwardDiff
```

### Steady-state solvers (Sundials Kinsol based):
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

## Field Array

```@meta
CurrentModule = PALEOmodel
```
[`FieldArray`](@ref) provides a generic array type with named dimensions `PALEOboxes.NamedDimension` and optional coordinates `PALEOboxes.FixedCoord` for processing of model output.

```@docs
FieldArray
get_array
```

```@docs
FieldRecord
```

## Output
```@meta
CurrentModule = PALEOmodel
```
```@docs
OutputWriters
```
```@meta
CurrentModule = PALEOmodel.OutputWriters
```
```@docs
OutputMemory
OutputMemoryDomain
save_jld2
load_jld2!
initialize!
add_record!
```

## Plot output

```@meta
CurrentModule = PALEOmodel
```
```@docs
RecipesBase.apply_recipe(::Dict{Symbol, Any}, fa::FieldArray)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, output::AbstractOutputWriter, vars::Union{AbstractString, Vector{<:AbstractString}}, selectargs::NamedTuple)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, outputs::Vector{<:AbstractOutputWriter}, vars::Union{AbstractString, Vector{<:AbstractString}}, selectargs::NamedTuple)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, fr::FieldRecord, selectargs::NamedTuple)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, fas::Vector{<:FieldArray})

PlotPager
Plot.test_heatmap_edges

```
## Analyze reaction network
```@meta
CurrentModule = PALEOmodel
```
```@docs
ReactionNetwork
```
