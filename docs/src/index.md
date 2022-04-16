# PALEOmodel.jl

```@meta
CurrentModule = PALEOmodel
```

The [PALEOmodel](@ref) Julia package provides modules to create and solve a standalone `PALEOboxes.Model`, and to analyse output interactively from the Julia REPL. It implements:
- Numerical solvers (see [Integrate](@ref))
- Data structures in the [`OutputWriters`](@ref) submodule, eg  [`OutputWriters.OutputMemory`](@ref) to hold model output
- Output plotting (see [Plot output](@ref)).
- [`Run`](@ref) (a container for a `PALEOboxes.Model` model and output).

PALEO documentation follows the recommendations from <https://documentation.divio.com/>

