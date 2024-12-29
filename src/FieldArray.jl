import Infiltrator
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

"""
    get_array(dataset, recordidx, f::Field [, selectargs::NamedTuple]; [attributes=nothing]) -> FieldArray

Return a [`FieldArray`](@ref) containing `f::Field` data values and
any attached coordinates, for the spatial region defined by `selectargs`.

Available `selectargs` depend on the grid `f.mesh`, and 
are passed to `get_region`.

`attributes` (if present) are added to `FieldArray`
"""
function get_array(
    f::PB.Field{FieldData, PB.ScalarSpace, V, N, Mesh}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing, coord_source,
) where {FieldData, V, N, Mesh}
    isempty(selectargs) ||
        error("get_array on Field f defined on ScalarSpace with non-empty selectargs=$selectargs")

    data_dims_coordinates = get_data_dims_coords(coord_source, f.data_dims)
    return FieldArray(default_fieldarray_name(attributes), f.values, data_dims_coordinates, attributes)
end

function get_array(
    f::PB.Field{FieldData, PB.CellSpace, V, 0, Mesh}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing, coord_source=nothing,
) where {FieldData, V, Mesh}

    values, dims_coords = get_region(f.mesh, f.values; coord_source, selectargs...)

    if !isnothing(attributes) && !isempty(selectargs)
        attributes = copy(attributes)
        attributes[:filter_region] = selectargs
    end

    return FieldArray(default_fieldarray_name(attributes), values, dims_coords, attributes)
end

# single data dimension
# TODO generalize this to arbitrary data dimensions
function get_array(
    f::PB.Field{FieldData, PB.CellSpace, V, 1, Mesh}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing, coord_source,
) where {FieldData, V, Mesh}

    f_space_dims_colons = ntuple(i->Colon(), ndims(f.values) - 1)
    f_size_datadim = size(f.values)[end]

    dvalues, dims_coords = get_region(f.mesh, f.values[f_space_dims_colons..., 1]; coord_source, selectargs...)
   
    d = (size(dvalues)..., f_size_datadim)
    values = Array{eltype(dvalues), length(d)}(undef, d...)
  
    if length(d) == 1
        # single cell - space dimension squeezed out
        for i in 1:f_size_datadim
            values[i], dims_coords = get_region(f.mesh, f.values[f_space_dims_colons..., i]; coord_source, selectargs...)
        end
    else
        dvalues_colons = ntuple(i->Colon(), ndims(dvalues))
        for i in 1:f_size_datadim
            dvalues, dims_coords = get_region(mesh, f.values[f_space_dims_colons..., i]; coord_source, selectargs...)
            values[dvalues_colons..., i] .= dvalues
        end
    end

    if !isnothing(attributes) && !isempty(selectargs)
        attributes = copy(attributes)
        attributes[:filter_region] = selectargs
    end

    data_dims_coords = get_data_dims_coords(coord_source, f.data_dims)

    return FieldArray(default_fieldarray_name(attributes), values, vcat(dims_coords, data_dims_coords), attributes)
end



"""
    get_array(modeldata, varnamefull [, selectargs::NamedTuple] [; coords::AbstractVector]) -> FieldArray
   
Get [`FieldArray`](@ref) by Variable name, for spatial region defined by `selectargs`
(which are passed to `get_region`).

Optional argument `coords` can be used to supply plot coordinates from Variable in output.
Format is a Vector of Pairs of "dim_name"=>("var_name1", "var_name2", ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>("atm.zmid", "atm.zlower", "atm.zupper")]

NB: the coordinates will be generated by applying `selectargs`,
so the supplied coordinate Variables must have the same dimensionality as `vars`.
"""
function get_array(
    modeldata::PB.AbstractModelData, varnamefull::AbstractString, selectargs::NamedTuple=NamedTuple();
    coords=nothing,
)
    varsplit = split(varnamefull, ".")
    length(varsplit) == 2 || 
        throw(ArgumentError("varnamefull $varnamefull is not of form <domainname>.<variablename>"))
    domainname, varname = varsplit
    
    coord_source = (modeldata, domainname)
    f, attributes = _get_field_attributes(coord_source, varname)

    varray = get_array(f, selectargs; attributes, coord_source)

    if isnothing(coords)
        # keep original coords (if any)
    else
        check_coords_argument(coords) ||
            error("argument coords should be a Vector of Pairs of \"dim_name\"=>(\"var_name1\", \"var_name2\", ...), eg: [\"z\"=>(\"atm.zmid\", \"atm.zlower\", \"atm.zupper\"), ...]")

        vec_coords_arrays = [
            dim_name => Tuple(get_array(modeldata, cvn, selectargs) for cvn in coord_varnames) 
            for (dim_name, coord_varnames) in coords
        ]

        varray = update_coordinates(varray, vec_coords_arrays) 
    end

    return varray
end

function get_data_dims_coords(coord_source, data_dims)
    
    data_dims_coordinates = Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}[]
    
    for dd in data_dims
        coordinates = PALEOmodel.FixedCoord[]        
        for coord_name in PB.get_coordinates(dd.name)
            ddf, attributes = _get_field_attributes(coord_source, coord_name)
            push!(coordinates, FixedCoord(coord_name, ddf.values, attributes))
        end
        push!(data_dims_coordinates, dd => coordinates)
    end
    
    return data_dims_coordinates
end

# check 'coords' of form [] or ["z"=>[ ... ], ] or ["z"=>(...),]
check_coords_argument(coords) =
    isa(coords, AbstractVector) && (
        isempty(coords) || (
            isa(coords, AbstractVector{<:Pair}) &&
            isa(first(first(coords)), AbstractString) &&
            isa(last(first(coords)), Union{AbstractVector, Tuple})
        )
    )

"""
    update_coordinates(varray::FieldArray, vec_coord_arrays::AbstractVector) -> FieldArray

Replace or add coordinates `vec_coord_arrays` to `varray`.

`new_coord_arrays` is a Vector of Pairs of "dim_name"=>(var1::FieldArray, var2::FieldArray, ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]
"""
function update_coordinates(varray::FieldArray, vec_coord_arrays::AbstractVector)

    check_coords_argument(vec_coord_arrays) || 
        error("argument vec_coords_arrays should be a Vector of Pairs of \"dim_name\"=>(var1::FieldArray, var2::FieldArray, ...), eg: [\"z\"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]")

    # generate Vector of NamedDimensions to use as new coordinates
    new_dims_coords = Pair{PB.NamedDimension, Vector{FixedCoord}}[nd => copy(coords) for (nd, coords) in varray.dims_coords]
    for (dim_name, coord_arrays) in vec_coord_arrays
        dc_idx = findfirst(dc->first(dc).name == dim_name, new_dims_coords)
        !isnothing(dc_idx) || error("FieldArray $varray has no dimension $dim_name")
        nd, coords = new_dims_coords[dc_idx]
        empty!(coords)
        for coord_array in coord_arrays
            coord_array_name = get(coord_array.attributes, :var_name, "")
            nd.size == length(coord_array.values) || 
                error("FieldArray $varray dimension $dim_name size $(nd.size) != new coordinate $coord_array_name length(values) $(coord_array.values)")
            push!(coords, PB.FixedCoord(coord_array_name, coord_array.values, coord_array.attributes))
        end
    end

    # replace coordinates
    varray_newcoords = FieldArray(varray.name, varray.values, new_dims_coords, varray.attributes)

    return varray_newcoords
end