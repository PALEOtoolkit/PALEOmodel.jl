
################################
# Coordinates
#################################

"""
    FixedCoord(name, values::Array{Float64, N}, attributes::Dict)

A fixed (state independent) coordinate

These are generated from coordinate variables for use in output visualisation.

N = 1: a cell-centre coordinate, size(values) = (ncells, )
N = 2: a boundary coordinate, size(values) = (2, ncells)

# Fields
$(TYPEDFIELDS)
"""
mutable struct FixedCoord{N}
    name::String
    values::Array{Float64, N}
    attributes::Dict{Symbol, Any}
end

is_boundary_coordinate(fc::FixedCoord) = ndims(fc.values) == 2 && size(fc.values, 1) == 2



##################################################################
# Coordinate filtering and selection
################################################################

"find indices of coord from first before range[1] to first after range[2]"
function find_indices(coord::AbstractVector, range)
    length(range) == 2 ||
        throw(ArgumentError("find_indices: length(range) != 2  $range"))

    idxstart = findlast(t -> t<=range[1], coord)
    isnothing(idxstart) && (idxstart = 1)

    idxend = findfirst(t -> t>=range[2], coord)
    isnothing(idxend) && (idxend = length(coord))

    return idxstart:idxend, (coord[idxstart], coord[idxend])
end

"find indices of coord nearest val"
function find_indices(coord::AbstractVector, val::Real)
    idx = 1
    for i in 1:length(coord)
        if abs(coord[i] - val) < abs(coord[idx] - val)
            idx = i
        end
    end

    return idx, coord[idx]
end

"""
    dimscoord_subset(dim::PB.NamedDimension, coords::Vector{FixedCoord}, select_dimvals::AbstractString, select_filter)
        -> cidx, dim_subset, coords_subset, coords_used

Filter dimension `dim` according to key `select_dimvals` and `select_filter` (typically a single value or a range)

Filtering may be applied either to dimension indices from `dim`, or to coordinates from `coords`:
- If `select_dimvals` is of form "<dimname>_isel", use dimension indices and return `coords_used=nothing`
- Otherwise use coordinate values and return actual coordinate values used in `coords_used`

`cidx` are the filtered indices to use:
- if `cidx` is a scalar (an Int), `dim_subset=nothing` and `coords_subset=nothing` indicating this dimension should be squeezed out
- otherwise `cidx` is a Vector and `dim_subset` and `coords_subset` are the filtered subset of `dim` and `coords`
"""
function dimscoord_subset(dim::PB.NamedDimension, coords::Vector{FixedCoord}, select_dimvals::AbstractString, select_filter)
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
        coords_subset = FixedCoord[]
        for c in coords
            if is_boundary_coordinate(c)
                cs = FixedCoord(c.name, c.values[:, cidx], c.attributes)
            else
                cs = FixedCoord(c.name, c.values[cidx], c.attributes)
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

