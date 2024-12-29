"""
    get_region(grid::Union{PB.AbstractMesh, Nothing}, values; coord_source, selectargs...) 
        -> values_subset, dim_subset::Vector{Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}}

Return the subset of `values` given by `selectargs` (Grid-specific keywords eg cell=, column=, ...)
and corresponding dimensions (with attached coordinates).
"""
function get_region(grid::Union{PB.AbstractMesh, Nothing}, values) end

"""
    get_region(dataset, recordidx, grid::Nothing, values) -> values[], []

Fallback for Domain with no grid, assumed 1 cell
"""
function get_region(grid::Nothing, values; coord_source=nothing)
    length(values) == 1 ||
        throw(ArgumentError("grid==Nothing and length(values) != 1"))
    return values[], Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}[]
end
   

"""
    get_region(dataset, recordidx, grid::PB.Grids.UnstructuredVectorGrid, values; cell=Nothing) -> 
        values_subset, dim_subset

# Keywords for region selection:
- `cell::Union{Nothing, Int, Symbol}`: an Int for cell number (first cell = 1), or a Symbol to look up in `cellnames`
  `cell = Nothing` is also allowed for a single-cell Domain.
"""
function get_region(
    grid::PB.Grids.UnstructuredVectorGrid, values;
    cell::Union{Nothing, Int, Symbol}=nothing,
    coord_source=nothing,
)
    if isnothing(cell)
        grid.ncells == 1 || throw(ArgumentError("'cell' argument (an Int or Symbol) is required for an UnstructuredVectorGrid with > 1 cell"))
        idx = 1
    elseif cell isa Int
        idx = cell
    else
        idx = get(grid.cellnames, cell, nothing)
        !isnothing(idx) ||
            throw(ArgumentError("cell ':$cell' not present in  grid.cellnames=$(grid.cellnames)"))
    end

    return (
        values[idx],
        Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}[],  # no dimensions (ie squeeze out a dimension length 1 for single cell)
    )
end



"""
    get_region(dataset, recordidx, grid::PB.Grids.UnstructuredColumnGrid, values; column, [cell=nothing]) -> 
        values_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `column::Union{Int, Symbol}`: (may be an Int, or a Symbol to look up in `columnames`)
- `cell::Int`: optional cell index within `column`, highest cell is cell 1
"""
function get_region(
    grid::PB.Grids.UnstructuredColumnGrid, values;
    column, 
    cell::Union{Nothing, Int}=nothing,
    coord_source=nothing
)

    if column isa Int
        column in 1:length(grid.Icolumns) ||
            throw(ArgumentError("column index $column out of range"))
        colidx = column
    else
        colidx = findfirst(isequal(column), grid.columnnames)
        !isnothing(colidx) || 
            throw(ArgumentError("columnname '$column' not present in  grid.columnnames=$(grid.columnnames)"))
    end

    if isnothing(cell)
        indices = grid.Icolumns[colidx]
        coordnames = PB.get_coordinates(grid, "cells")
        coordinates = FixedCoord[]
        for cn in coordnames
            cf, attributes = _get_field_attributes(coord_source, cn)
            push!(coordinates, FixedCoord(cn, cf.values[indices], attributes))
        end
        return (
            values[indices],
            [PB.NamedDimension("cells", length(indices)) => coordinates],
        )
    else
        # squeeze out z dimension
        idx = grid.Icolumns[colidx][cell]
        return (
            values[idx],
            Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}[],  # no dimensions (ie squeeze out a dimension length 1 for single cell)
        )
    end
    
end


"""
    get_region(dataset, recordidx, rid::Union{PB.Grids.CartesianLinearGrid{2}, PB.Grids.CartesianArrayGrid{2}} , internalvalues; [i=i_idx], [j=j_idx]) ->
        arrayvalues_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `i::Int`: optional, slice along first dimension
- `j::Int`: optional, slice along second dimension

`internalvalues` are transformed if needed from internal Field representation as a Vector length `ncells`, to
an Array (2D if neither i, j arguments present, 1D if i or j present, 0D ie one cell if both present)
"""
function get_region(
    grid::Union{PB.Grids.CartesianLinearGrid{2}, PB.Grids.CartesianArrayGrid{2}}, internalvalues; 
    i::Union{Integer, Colon}=Colon(),
    j::Union{Integer, Colon}=Colon(),
    coord_source,
)
    return _get_region(grid, internalvalues, [i, j], coord_source)
end

"""
    get_region(grid::Union{PB.Grids.CartesianLinearGrid{3}, PB.Grids.CartesianArrayGrid{3}}, internalvalues; [i=i_idx], [j=j_idx]) ->
        arrayvalues_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `i::Int`: optional, slice along first dimension
- `j::Int`: optional, slice along second dimension
- `k::Int`: optional, slice along third dimension

`internalvalues` are transformed if needed from internal Field representation as a Vector length `ncells`, to
an Array (3D if neither i, j, k arguments present, 2D if one of i, j or k present, 1D if two present,
0D ie one cell if i, j, k all specified).
"""
function get_region(
    grid::Union{PB.Grids.CartesianLinearGrid{3}, PB.Grids.CartesianArrayGrid{3}}, internalvalues;
    i::Union{Integer, Colon}=Colon(),
    j::Union{Integer, Colon}=Colon(),
    k::Union{Integer, Colon}=Colon(),
    coord_source,
)
    return _get_region(grid, internalvalues, [i, j, k], coord_source)
end

function _get_region(
    grid::Union{PB.Grids.CartesianLinearGrid, PB.Grids.CartesianArrayGrid}, internalvalues, indices, coord_source
)
    if !isempty(grid.coords) && !isempty(grid.coords_edges)
        dims = [
            PB.NamedDimension(grid.dimnames[idx], grid.coords[idx], grid.coords_edges[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    elseif !isempty(grid.coords)
        dims = [
            PB.NamedDimension(grid.dimnames[idx], grid.coords[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    else
        dims = [
            PB.NamedDimension(grid.dimnames[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    end

    values = internal_to_cartesian(grid, internalvalues)
    if !all(isequal(Colon()), indices)
        values = values[indices...]
    end

    return values, Tuple(dims)    
end

# field and attributes from a dataset that implements PB.get_field(dataset, name)::FieldRecord
function _get_field_attributes(coord_source::Tuple{Any, Int}, name)
    dataset, record_idx = coord_source
    fr = PB.get_field(dataset, name)
    return fr[record_idx], fr.attributes
end

function _get_field_attributes(coord_source::Tuple{PB.ModelData, AbstractString}, varname)
    modeldata, domainname = coord_source
     # PB.get_field(model, modeldata, varnamefull) doesn't provide attributes, so do this ourselves
     varnamefull = domainname*"."*varname
     var = PB.get_variable(modeldata.model, varnamefull)
     !isnothing(var) ||
         throw(ArgumentError("Variable $varnamefull not found"))
     f = PB.get_field(var, modeldata)
     attributes = copy(var.attributes)
     attributes[:var_name] = var.name
     attributes[:domain_name] = var.domain.name    

     return f, attributes
end