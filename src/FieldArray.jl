"""
    FieldArray

A generic [xarray](https://xarray.pydata.org/en/stable/index.html)-like or 
[IRIS](https://scitools-iris.readthedocs.io/en/latest/)-like 
n-dimensional Array with named dimensions and optional coordinates.

NB: this aims to be simple and generic, not efficient !!! Intended for representing model output,
not for numerically-intensive calculations.

# Fields
$(TYPEDFIELDS)
"""
struct FieldArray{T <: AbstractArray}
    "variable name"
    name::String
    "n-dimensional Array of values"
    values::T
    "Names of dimensions with optional attached coordinates"
    dims_coords::Vector{Pair{PB.NamedDimension, Vector{FieldArray}}}
    "variable attributes"
    attributes::Union{Dict{Symbol, Any}, Nothing}
end


PB.get_dimensions(f::FieldArray) = PB.NamedDimension[first(dc) for dc in f.dims_coords]

function Base.show(io::IO, fa::FieldArray)
    get(io, :typeinfo, nothing) === FieldArray || print(io, "FieldArray")
    print(io, 
        "(name=\"", fa.name, "\", eltype=", eltype(fa.values), ", size=", size(fa.values), ", dims_coords=["
    )
    for (i, (nd, coords)) in enumerate(fa.dims_coords)
        print(IOContext(io, :typeinfo=>PB.NamedDimension), nd)
        isempty(coords) || print(io, " => ", [c.name for c in coords])
        i == length(fa.dims_coords) || print(io, ", ")
    end
    print(io, "])")
end

function Base.show(io::IO, ::MIME"text/plain", fa::FieldArray)
    println(io, "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values),  ", size=", size(fa.values), ")")
    println(io, "  dims_coords:")
    for (nd, coords) in fa.dims_coords
        print(IOContext(io, :typeinfo=>PB.NamedDimension), "     ", nd)
        isempty(coords) || print(io, " => ", [c.name for c in coords])
        println()
    end
    println(io, "  attributes: ", fa.attributes)
end


# basic arithmetic operations
function Base.:*(fa_in::FieldArray, a::Real)
    fa_out = FieldArray("$a*"*fa_in.name, a.*fa_in.values, fa_in.dims_coords, copy(fa_in.attributes))
    return fa_out
end
Base.:*(a::Real, fa_in::FieldArray) = fa_in*a

# default name from attributes
default_varnamefull(attributes::Nothing) = ""

function default_varnamefull(attributes::Dict; include_selectargs=false)
    name = get(attributes, :domain_name, "")
    name *= isempty(name) ? "" : "."
    name *= get(attributes, :var_name, "")

    if include_selectargs
        selectargs_records = get(attributes, :filter_records, NamedTuple())
        selectargs_region = get(attributes, :filter_region, NamedTuple())
        if !isempty(selectargs_region) || !isempty(selectargs_records)
            name *= "(" * join(["$k=$v" for (k, v) in Dict(pairs(merge(selectargs_records, selectargs_region)))], ", ") * ")"
        end
    end

    return name
end

#############################################################
# Create from PALEO objects
#############################################################

"""
    get_array(obj, ...) -> FieldArray

Get FieldArray from PALEO object `obj`
"""
function get_array end

