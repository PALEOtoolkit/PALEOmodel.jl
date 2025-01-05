##################################################################
# Coordinate filtering and selection
################################################################

# test whether c::FieldArray is a valid coordinate for dimension nd
# ie c.dims_coords contains dimension nd
function is_coordinate(c::FieldArray, nd::PB.NamedDimension)
    c_has_nd = false
    for (c_nd, _) in c.dims_coords
        if c_nd == nd
            c_has_nd = true
        end
    end

    return c_has_nd
end

is_boundary_coordinate(c::FieldArray) =
    length(c.dims_coords) > 1 && c.dims_coords[1][1] == PB.NamedDimension("bnds", 2)

"find indices of coord from first before range[1] to first after range[2]"
function _find_indices(coord_values::AbstractVector, range)
    length(range) == 2 ||
        throw(ArgumentError("_find_indices: length(range) != 2  $range"))

    idxstart = findlast(t -> t<=range[1], coord_values)
    isnothing(idxstart) && (idxstart = 1)

    idxend = findfirst(t -> t>=range[2], coord_values)
    isnothing(idxend) && (idxend = length(coord_values))

    return idxstart:idxend, (coord_values[idxstart], coord_values[idxend])
end

"find indices of coord nearest val"
function _find_indices(coord_values::AbstractVector, val::Real)
    idx = 1
    for i in 1:length(coord_values)
        if abs(coord_values[i] - val) < abs(coord_values[idx] - val)
            idx = i
        end
    end

    return idx, coord_values[idx]
end

"""
    _dim_subset(dim::PB.NamedDimension, coords::Vector{FieldArray}, select_keyname::AbstractString, select_filter)
        -> indices_subset, dim_subset, coords_used

Filter indices from dimension `dim` according to key `select_keyname` and `select_filter` (typically a single value or a range)

Filtering may be applied either to dimension indices from `dim`, or to coordinates from `coords`:
- If `select_keyname` is of form "<dimname>_isel", use dimension indices and return `coords_used=nothing`
- Otherwise use coordinate values and return actual coordinate values used in `coords_used`

`indices_subset` are the filtered indices to use:
- if `indices_subset` is a scalar (an Int), then `dim_subset=nothing` indicating this dimension should be squeezed out
- otherwise `indices_subset` is a Vector and `dim_subset::PB.NamedDimension` has `dim_subset.size` corresponding to `indices_subset` used from `dim`
"""
function _dim_subset(dim::PB.NamedDimension, coords::Vector{FieldArray}, select_keyname::AbstractString, select_filter)
    if length(select_keyname) > 5 && select_keyname[end-4:end] == "_isel"
        @assert select_keyname[1:end-5] == dim.name
        indices_subset = select_filter
        coords_used=nothing
    else
        # find indices_subset corresponding to a coordinate
        # find coordinate to use in coords
        ccidx = findfirst(c -> c.name == select_keyname, coords)
        @assert !isnothing(ccidx)
        cc = coords[ccidx]
        indices_subset, cvalue = _find_indices(cc.values, select_filter)
        # reset to the value actually used
        coords_used = cvalue
    end

    if indices_subset isa AbstractVector
        dim_subset = PB.NamedDimension(dim.name, length(indices_subset))
    else
        # squeeze out dimensions
        dim_subset = nothing
    end

    return indices_subset, dim_subset, coords_used
end

# Select region from coord::FieldArray, given selectdimindices, a Dict(dimname=>dimindices, ...) of 
# indices to keep for each dimension that should be filtered.
# NB: 
# - intended for use with a `coord::FieldArray` without attached coordinates, any attached coordinates are just dropped
# - any dimensions with single index in selectdimindices are squeezed out
# - any dimension not present in selectdimindices has no filter applied
function _select_dims_indices(coord::FieldArray, selectdimindices::Dict{String, Any})
    dims = PB.get_dimensions(coord)

    select_indices = []
    select_dims = PB.NamedDimension[]
    for nd in dims
        if haskey(selectdimindices, nd.name)
            dimindices = selectdimindices[nd.name]
            push!(select_indices, dimindices)
            if length(dimindices) > 1
                # subset
                push!(select_dims, PB.NamedDimension(nd.name, length(dimindices)))
            else
                # squeeze out
            end                
        else
            # retain full dimension
            push!(select_indices, 1:nd.size)
            push!(select_dims, nd)
        end
    end

    select_dims_nocoords = Pair{PB.NamedDimension, Vector{FieldArray}}[
        d => FieldArray[] for d in select_dims
    ]

    avalues = Array{eltype(coord.values), length(select_dims)}(undef, [nd.size for nd in select_dims]...)

    if isempty(select_dims) # 0D array
        avalues[] = coord.values[select_indices...]
    else
        avalues[fill(Colon(), length(select_dims))...] .= @view coord.values[select_indices...]
    end

    return FieldArray(coord.name, avalues, select_dims_nocoords, copy(coord.attributes))
end

