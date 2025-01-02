

# PALEOmodel output

## Run

```@meta
CurrentModule = PALEOmodel
```
```@docs
Run
```

## Output
PALEO model output is accumulated into a container such as an [OutputMemory](@ref) instance that implements the [AbstractOutputWriter interface](@ref).

### AbstractOutputWriter interface

```@meta
CurrentModule = PALEOmodel
```
```@docs
AbstractOutputWriter
```

```@meta
CurrentModule = PALEOmodel.OutputWriters
```
#### Writing output
```@docs
initialize!
add_record!
```
#### Modifying output
```@docs
PB.add_field!(output::PALEOmodel.AbstractOutputWriter, fr::PALEOmodel.FieldRecord) 
```
#### Querying output
```@docs
PB.get_table(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString)
PB.show_variables(output::PALEOmodel.AbstractOutputWriter)
PB.has_variable(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString)
```

#### Accessing output data
```@docs
PALEOmodel.get_array(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString, allselectargs::NamedTuple; kwargs...)
PB.get_field(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString)
PB.get_data(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString; records=nothing)
PB.get_mesh(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString)
```
```@meta
CurrentModule = PALEOmodel
```
```@docs
FieldRecord
```

### OutputMemory

```@meta
CurrentModule = PALEOmodel.OutputWriters
```
```@docs
OutputMemory
OutputMemoryDomain
```

```@docs
save_netcdf
load_netcdf!
```

## Field Array

```@meta
CurrentModule = PALEOmodel
```
[`FieldArray`](@ref) provides a generic array type with named dimensions `PALEOboxes.NamedDimension` each with optional coordinates for processing of model output.

```@docs
FieldArray
get_array(fr::FieldRecord)
```

## Plotting output

### Plot recipes
Plotting using the Julia [Plots.jl](https://github.com/JuliaPlots/Plots.jl) package is implemented by [plot recipes](https://docs.juliaplots.org/latest/recipes/) that enable plotting of PALEO data types using the `plot` command.

The general principles are that:
- Data is extracted from model output into [`FieldArray`](@ref)s with attached coordinates
- Vector-valued arguments are "broadcast" to allow multiple line plot series to be overlaid in a single plot panel

```@meta
CurrentModule = PALEOmodel
```
```@docs
RecipesBase.apply_recipe(::Dict{Symbol, Any}, output::AbstractOutputWriter, vars::Union{AbstractString, Vector{<:AbstractString}}, selectargs::NamedTuple)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, fr::FieldRecord, selectargs::NamedTuple)
RecipesBase.apply_recipe(::Dict{Symbol, Any}, fa::FieldArray)
```
### Assembling multi-plot panels
```@docs
PlotPager
DefaultPlotPager
```

## Analyze reaction network
```@meta
CurrentModule = PALEOmodel
```
```@docs
ReactionNetwork
```
```@meta
CurrentModule = PALEOmodel.ReactionNetwork
```
```@docs
get_ratetable
get_all_species_ratevars
get_rates
get_all_species_ratesummaries
show_ratesummaries
```