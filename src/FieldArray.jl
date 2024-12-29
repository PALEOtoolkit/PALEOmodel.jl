"""
    FieldArray

A generic [xarray](https://xarray.pydata.org/en/stable/index.html)-like or 
[IRIS](https://scitools-iris.readthedocs.io/en/latest/)-like 
Array with named dimensions and optional coordinates.

NB: this aims to be simple and generic, not efficient !!! Intended for representing model output,
not for numerically-intensive calculations.
"""
struct FieldArray{T}
    name::String
    values::T
    dims_coords::Vector{Pair{PB.NamedDimension, Vector{FixedCoord}}}
    attributes::Union{Dict, Nothing}
end

PB.get_dimensions(f::FieldArray) = PB.NamedDimension[first(dc) for dc in f.dims_coords]

function Base.show(io::IO, fa::FieldArray)
    print(io, 
        "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values), ", size=", size(fa.values), ", dims_coords=", fa.dims_coords, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", fa::FieldArray)
    println(io, "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values),  ", size=", size(fa.values), ")")
    # println(io, "  dims_coords:")
    # for (nd, coords) in fa.dims_coords
    #     println(io, "    ", nd, " => ")
    #     for c in coords
    #         println(io, "        ", c)
    #     end
    # end

    println(io, "  dims_coords:")
    for (nd, coords) in fa.dims_coords
        println(io, "    ", nd, " => ", [c.name for c in coords])
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
default_fieldarray_name(attributes::Nothing) = ""

function default_fieldarray_name(attributes::Dict)
    name = get(attributes, :domain_name, "")
    name *= isempty(name) ? "" : "."
    name *= get(attributes, :var_name, "")

    selectargs_records = get(attributes, :filter_records, NamedTuple())
    selectargs_region = get(attributes, :filter_region, NamedTuple())
    if !isempty(selectargs_region) || !isempty(selectargs_records)
        name *= "(" * join(["$k=$v" for (k, v) in Dict(pairs(merge(selectargs_records, selectargs_region)))], ", ") * ")"
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

