

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

Model output is stored in a [`PALEOmodel.AbstractOutputWriter`](@ref) object, which is
available as `run.output`, ie the `output` field of the default [`PALEOmodel.Run`](@ref) instance created
by the `COPSE_reloaded_reloaded.jl` script.

The default [`PALEOmodel.OutputWriters.OutputMemory`](@ref) stores model output in memory, by Domain:

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

The output can be plotted using the Julia Plots.jl package, see [Plotting output](@ref). Plot recipes are defined for [`PALEOmodel.FieldArray`](@ref), so output data can be plotted directly using the `plot` command:

    julia> using Plots
    julia> plot(run.output, "atm.pCO2atm")  # plot output variable as a single command
    julia> plot(pCO2atm) # a PALEOmodel.FieldArray can be plotted
    julia> plot!(tmodel_raw, pCO2atm_raw, label="some raw data") # overlay data from standard Julia Vectors

## Spatial or wavelength-dependent output

To analyze spatial or eg wavelength-dependent output (eg time series from a 1D column or 3D general circulation model, or quantities that are a function of wavelength or frequency), [`PALEOmodel.get_array`](@ref) takes additional arguments to take 1D or 2D slices from the spatial, spectral and timeseries data. The [`PALEOmodel.FieldArray`](@ref) returned includes coordinates to plot column (1D) and heatmap (2D) data.

### Examples for a column-based model

Visualisation of spatial and wavelength-dependent output from the PALEOdev.jl ozone photochemistry example (a single 1D atmospheric column):

#### 1D column data
    julia> plot(title="O3 mixing ratio", output, "atm.O3_mr", (tmodel=[0.0, 0.1, 1.0, 10.0, 100.0, 1000.0], column=1),
                swap_xy=true, xaxis=:log, labelattribute=:filter_records) # plots O3 vs height

Here the `labelattribute=:filter_records` keyword argument is used to generate plot labels from the `:filter_records` FieldArray attribute, which contains the `tmodel` values used to select the timeseries records.  The plot recipe expands
the Vector-valued `tmodel` argument to overlay a sequence of plots.

This is equivalent to first creating and then plotting a sequence of `FieldArray` objects:

    julia> O3_mr = PALEOmodel.get_array(run.output, "atm.O3_mr", tmodel=0.0, column=1)
    julia> plot(title="O3 mixing ratio", O3_mr, swap_xy=true, xaxis=:log, labelattribute=:filter_records)
    julia> O3_mr = PALEOmodel.get_array(run.output, "atm.O3_mr", tmodel=0.1, column=1)
    julia> plot!(O3_mr, swap_xy=true, labelattribute=:filter_records)

#### Wavelength-dependent data
    julia> plot(title="direct transmittance", output, ["atm.direct_trans"], (tmodel=1e12, column=1, cell=[1, 80]),
                ylabel="fraction", labelattribute=:filter_region) # plots vs wavelength

Here `tmodel=1e12` selects the last model time output, and `column=1, cell=[1, 80]` selects the top and bottom cells within the first (only) 1D column. The `labelattribute=:filter_region` keyword argument is used to generate plot labels from the `:filter_region` FieldArray attribute, which contains the `column` and `cell` values used to select the spatial region.

### Examples for a 3D GCM-based model

Visualisation of spatial output from the 3D GENIE transport-matrix example (PALEOdev.jl repository)

### Horizontal slices across levels
    julia> heatmap(run.output, "ocean.O2_conc", (tmodel=1e12, k=1), swap_xy=true)

Here `k=1` selects a horizontal level corresponding to model grid cells with index k=1, which is the ocean surface in the GENIE grid.

### Vertical section at constant longitude
    julia> heatmap(run.output, "ocean.O2_conc", (tmodel=1e12, i=10), swap_xy=true, mult_y_coord=-1.0)

Here `i=10` selects a section at longitude corresponding to model grid cells with index i=10.

## Save and load output

Model output can be saved and loaded using the [`PALEOmodel.OutputWriters.save_jld2`](@ref) and [`PALEOmodel.OutputWriters.load_jld2!`](@ref) methods.

## Export output to a CSV file

To write Model output from a single Domain to a CSV file:

    julia> import CSV
    julia> CSV.write("copse_land.csv", PB.get_table(run.output, "land")) # all Variables from land Domain
    julia> CSV.write("copse_atm.csv", PB.get_table(run.output, "atm")[!, [:tmodel, :pCO2atm, :pO2atm]]) # subset of Variables from atm Domain
