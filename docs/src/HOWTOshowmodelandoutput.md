

# Displaying model configuration and output from the Julia REPL

The examples below assume the `COPSE Reloaded` example has been run from the Julia REPL.

## Displaying large tables in Julia
Several PALEO commands produce large tables.

There are several options to display these:
- Julia in VS code provides the `julia> vscodedisplay(<some table>)` command. As of Jan 2022 this is now usually the best option.
- Use `julia> show(<some table>, allcols=true, allrows=true)` to show as text in the REPL. 
- Use `julia> CSV.write("some_table.csv", <some table>)` to save as a CSV file and open in Excel etc.

## Display model configuration

### Display parameters:

Examples illustrating the use of `PALEOboxes.show_parameters`:

To show parameters for every Reaction in the model:

    julia> vscodedisplay(PB.show_parameters(run.model)) # show in VS code table viewer
    julia> show(PB.show_parameters(run.model), allrows=true) # show as text in REPL 
    julia> import CSV
    julia> CSV.write("parameters.csv", PB.show_parameters(run.model)) # save as CSV for Excel etc

This illustrates the modularised model structure, with:
- Domains global, atm, land, ocean, oceansurface, oceanfloor, sedcrust containing forcings and biogeochemical Reactions, with Parameters attached to each Reaction.
- Additional Domains fluxAtoLand, fluxLandtoSedCrust, fluxOceanBurial, fluxOceanfloor, fluxRtoOcean, fluxSedCrusttoAOcean containing flux coupler Reactions.

To show parameters for a single Reaction:
    
    julia> rct_temp_global = PB.get_reaction(run.model, "global", "temp_global")
    julia> PB.show_parameters(rct_temp_global)    # GEOCARB temperature function parameters

The Julia Type of `rct_temp_global` `PALEOreactions.Global.Temperature.ReactionGlobalTemperatureBerner` usually makes it possible to guess the location in the source code `PALEOreactions/global/Temperature.jl`.

### Display Variables:

#### Show Variables in the model:

To list all Variables in the model:

    julia> vscodedisplay(PB.show_variables(run.model)) # VS code only

To list full information for all Variables in the model (including Variable linking and current values):

    julia> vscodedisplay(PB.show_variables(run.model, modeldata=modeldata, showlinks=true))

This illustrates the modularized model structure, with:

- Domains global, atm, land, ocean, oceansurface, oceanfloor, sedcrust containing Variables linked to Reactions (either property-dependencies or target-contributors pairs).
- Additional Domains fluxAtoLand, fluxLandtoSedCrust, fluxOceanBurial, fluxOceanfloor, fluxRtoOcean, fluxSedCrusttoAOcean containing target-contributor pairs representing inter-module fluxes.

It is also possible to show Variables for a specific Domain eg:

    julia> domain_land = PB.get_domain(run.model, "land")
    julia> vscodedisplay(PB.show_variables(domain_land))

#### Show linkage for a single Domain or ReactionMethod Variable

To show linkage of a single VariableDomain in Domain "atm" with name "pO2PAL":

    julia> PB.show_links(PB.get_variable(run.model, "atm.pO2PAL"))

To show linkage of a ReactionMethod Variable with localname "pO2PAL", Reaction "ocean_copse" in Domain "ocean":

    julia> PB.show_links(PB.get_reaction_variables(run.model, "ocean", "ocean_copse", "pO2PAL"))

## Display model output

Model output is stored in a [`PALEOmodel.OutputWriters.OutputMemory`](@ref) object, which is
available as `run.output`, ie the `output` field of the default [`PALEOmodel.Run`](@ref) instance created
by the `COPSE_reloaded_reloaded.jl` script.

[`PALEOmodel.OutputWriters.OutputMemory`](@ref) stores model output by Domain:

    julia> run.output  # shows Domains

To show metadata for all Variables in the output:

    julia> vscodedisplay(PB.show_variables(run.output)) # VS code only
    julia> vscodedisplay(PB.show_variables(run.output, "land")) # just the land Domain

Output from a list of Variables or for each `Domain` can be exported to a Julia [DataFrame](https://dataframes.juliadata.org/stable/):

    julia> # display data for a list of Variables as a Table
    julia> vscodedisplay(PB.get_table(run.output, ["atm.tmodel", "atm.pCO2PAL", "fluxOceanBurial.flux_total_P"]))

    julia> # display data for every Variable in the 'atm' Domain as a Table
    julia> vscodedisplay(PB.get_table(run.output, "atm"))

    julia> # show a subset of output variables from the 'atm' Domain
    julia> PB.get_table(run.output, "atm")[!, [:tmodel, :pCO2atm, :pCO2PAL]]

Data from each Variable can be accessed as a [`PALEOmodel.FieldArray`](@ref) (a Python-xarray like struct with
named dimensions and coordinates):

    julia> pCO2atm = PALEOmodel.get_array(run.output, "atm.pCO2atm")
    julia> pCO2atm.values # raw data Array
    julia> pCO2atm.dims[1] # pCO2 is a scalar Variable with one dimension `records` which has a coordinate `tmodel`
    julia> pCO2atm.dims[1].coords[1].values # raw values for model time (`tmodel`)

Raw data arrays can also be accessed as Julia Vectors using `get_data`:

    julia> pCO2atm_raw = PB.get_data(run.output, "atm.pCO2atm")  # raw data Array
    julia> tmodel_raw = PB.get_data(run.output, "atm.tmodel") # raw data Array

(here these are the values and coordinate of the `pCO2atm` [`PALEOmodel.FieldArray`](@ref), ie `pCO2atm_raw == pCO2atm.values` and `tmodel_raw == pCO2atm.dims[1].coords[1].values`).

## Plot model output

The output can be plotted using the Julia Plots.jl package. Plot recipes are defined for [`PALEOmodel.FieldArray`](@ref), 
so output data can be plotted directly:

    julia> using Plots
    julia> plot(run.output, "atm.pCO2atm")  # plot output variable as a single command
    julia> plot(pCO2atm) # a PALEOmodel.FieldArray can be plotted
    julia> plot!(tmodel_raw, pCO2atm_raw, label="some raw data") # overlay data from standard Julia Vectors

## Spatial output

TODO - key point is that [`PALEOmodel.FieldArray`](@ref) includes coordinates to plot column and image data.

## Save and load output

Model output can be saved and loaded using the [`PALEOmodel.OutputWriters.save_jld2`](@ref) and [`PALEOmodel.OutputWriters.load_jld2!`](@ref) methods.

## Export output to a CSV file

To write Model output from a single Domain to a CSV file:

    julia> import CSV
    julia> CSV.write("copse_land.csv", PB.get_table(run.output, "land")) # all Variables from land Domain
    julia> CSV.write("copse_atm.csv", PB.get_table(run.output, "atm")[!, [:tmodel, :pCO2atm, :pO2atm]]) # subset of Variables from atm Domain
