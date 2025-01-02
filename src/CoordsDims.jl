


##################################################################
# Coordinate filtering and selection
################################################################

"find indices of coord from first before range[1] to first after range[2]"
function find_indices(coord_values::AbstractVector, range)
    length(range) == 2 ||
        throw(ArgumentError("find_indices: length(range) != 2  $range"))

    idxstart = findlast(t -> t<=range[1], coord_values)
    isnothing(idxstart) && (idxstart = 1)

    idxend = findfirst(t -> t>=range[2], coord_values)
    isnothing(idxend) && (idxend = length(coord_values))

    return idxstart:idxend, (coord_values[idxstart], coord_values[idxend])
end

"find indices of coord nearest val"
function find_indices(coord_values::AbstractVector, val::Real)
    idx = 1
    for i in 1:length(coord_values)
        if abs(coord_values[i] - val) < abs(coord_values[idx] - val)
            idx = i
        end
    end

    return idx, coord_values[idx]
end

"""
    dimscoord_subset(dim::PB.NamedDimension, coords::Vector{FieldArray}, select_dimvals::AbstractString, select_filter)
        -> cidx, dim_subset, coords_subset, coords_used

Filter dimension `dim` according to key `select_dimvals` and `select_filter` (typically a single value or a range)

Filtering may be applied either to dimension indices from `dim`, or to coordinates from `coords`:
- If `select_dimvals` is of form "<dimname>_isel", use dimension indices and return `coords_used=nothing`
- Otherwise use coordinate values and return actual coordinate values used in `coords_used`

`cidx` are the filtered indices to use:
- if `cidx` is a scalar (an Int), `dim_subset=nothing` and `coords_subset=nothing` indicating this dimension should be squeezed out
- otherwise `cidx` is a Vector and `dim_subset` and `coords_subset` are the filtered subset of `dim` and `coords`
"""
function dimscoord_subset(dim::PB.NamedDimension, coords::Vector{FieldArray}, select_dimvals::AbstractString, select_filter)
    if length(select_dimvals) > 5 && select_dimvals[end-4:end] == "_isel"
        @assert select_dimvals[1:end-5] == dim.name
        cidx = select_filter
        coords_used=nothing
    else
        # find cidx corresponding to a coordinate
        # find coordinate to use in coords
        ccidx = findfirst(c -> c.name == select_dimvals, coords)
        @assert !isnothing(ccidx)
        cc = coords[ccidx]
        cidx, cvalue = find_indices(cc.values, select_filter)
        # reset to the value actually used
        coords_used = cvalue
    end

    if cidx isa AbstractVector
        dim_subset = PB.NamedDimension(dim.name, length(cidx))
        coords_subset = FieldArray[]
        for c in coords
            is_coordinate(c, dim) ||
                error("dimension $dim invalid coordinate $c")
            if is_boundary_coordinate(c)
                # c has 2 dimensions, (bounds, dim.name)
                @assert c.dims_coords[2][1] == dim
                coords_subset_dims = [ 
                    c.dims_coords[1], # bounds
                    dim_subset => FieldArray[]
                ]
                cs = FieldArray(c.name, c.values[:, cidx], coords_subset_dims, c.attributes)
            else
                # c has 1 dimension == dim
                @assert c.dims_coords[1][1] == dim
                cs = FieldArray(c.name, c.values[cidx], [dim_subset => FieldArray[]], c.attributes)
            end
            push!(coords_subset, cs)
        end
    else
        # squeeze out dimensions
        dim_subset = nothing
        coords_subset = nothing
    end

    return cidx, dim_subset, coords_subset, coords_used
end

# test whether c::FieldArray is a valid coordinate for dimension nd
function is_coordinate(c::FieldArray, nd::PB.NamedDimension)
    if length(c.dims_coords) == 1 && c.dims_coords[1][1] == nd
        return true
    elseif is_boundary_coordinate(c) && c.dims_coords[2][1] == nd
        return true
    end
    return false
end

is_boundary_coordinate(c::FieldArray) =
    length(c.dims_coords) == 2 && c.dims_coords[1][1] == PB.NamedDimension("bnds", 2)