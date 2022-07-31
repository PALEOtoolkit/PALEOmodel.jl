"""
    FieldArray

A generic [xarray](https://xarray.pydata.org/en/stable/index.html)-like or 
[IRIS](https://scitools-iris.readthedocs.io/en/latest/)-like 
Array with named dimensions and optional coordinates.

NB: this aims to be simple and generic, not efficient !!! Intended for representing model output,
not for numerically-intensive calculations.
"""
struct FieldArray{T, N}
    name::String
    values::T
    dims::NTuple{N, PB.NamedDimension}
    attributes::Union{Dict, Nothing}
end


function Base.show(io::IO, fa::FieldArray)
    print(io, 
        "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values),", size=", size(fa.values), ", dims=", fa.dims, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", fa::FieldArray)
    println(io, "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values),")")
    println(io, "  dims: ", fa.dims)
    println(io, "  attributes: ", fa.attributes)
end

# basic arithmetic operations
function Base.:*(fa_in::FieldArray, a::Real)
    fa_out = FieldArray("$a*"*fa_in.name, a.*fa_in.values, fa_in.dims, copy(fa_in.attributes))
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
    get_array(f::Field [, selectargs::NamedTuple]; [attributes=nothing]) -> FieldArray

Return a [`FieldArray`](@ref) containing `f::Field` data values and
any attached coordinates, for the spatial region defined by `selectargs`.

Available `selectargs` depend on the grid `f.mesh`, and 
are passed to `PB.Grids.get_region`.

`attributes` (if present) are added to `FieldArray`
"""
function get_array(
    f::PB.Field{D, PB.ScalarSpace, V, N, M};
    attributes=nothing,
) where {D, V, N, M}

    return FieldArray(default_fieldarray_name(attributes), f.values, f.data_dims, attributes)
end

function get_array(
    f::PB.Field{D, PB.CellSpace, V, 0, M}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing,    
) where {D, V, M}

    values, dims = PB.Grids.get_region(f.mesh, f.values; selectargs...)

    if !isempty(selectargs)
        attributes = isnothing(attributes) ? Dict{Symbol, Any}() : copy(attributes)
        attributes[:filter_region] = selectargs
    end

    return FieldArray(default_fieldarray_name(attributes), values, dims, attributes)
end

# single data dimension
# TODO generalize this to arbitrary data dimensions
function get_array(
    f::PB.Field{D, PB.CellSpace, V, 1, M}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing,
) where {D, V, M}

    f_space_dims_colons = ntuple(i->Colon(), ndims(f.values) - 1)
    f_size_datadim = size(f.values)[end]

    dvalues, dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., 1]; selectargs...)
   
    d = (size(dvalues)..., f_size_datadim)
    values = Array{eltype(dvalues), length(d)}(undef, d...)
  
    if length(d) == 1
        # single cell - space dimension squeezed out
        for i in 1:f_size_datadim
            values[i], dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., i]; selectargs...)
        end
    else
        dvalues_colons = ntuple(i->Colon(), ndims(dvalues))
        for i in 1:f_size_datadim
            dvalues, dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., i]; selectargs...)
            values[dvalues_colons..., i] .= dvalues
        end
    end

    if !isempty(selectargs)
        attributes = isnothing(attributes) ? Dict{Symbol, Any}() : copy(attributes)
        attributes[:filter_region] = selectargs
    end

    return FieldArray(default_fieldarray_name(attributes), values, (dims..., f.data_dims...), attributes)
end



"""
    get_array(modeldata, varnamefull [, selectargs::NamedTuple] [; coords::AbstractVector]) -> FieldArray
   
Get [`FieldArray`](@ref) by Variable name, for spatial region defined by `selectargs`
(which are passed to `PB.Grids.get_region`).

Optional argument `coords` can be used to supply plot coordinates from Variable in output.
Format is a Vector of Pairs of "coord_name"=>("var_name1", "var_name2", ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>("atm.zmid", "atm.zlower", "atm.zupper")]

NB: the coordinates will be generated by applying `selectargs`,
so the supplied coordinate Variables must have the same dimensionality as `vars`.
"""
function get_array(
    modeldata::PB.AbstractModelData, varnamefull::AbstractString, selectargs::NamedTuple=NamedTuple();
    coords=nothing,
)
    var = PB.get_variable(modeldata.model, varnamefull)
    !isnothing(var) ||
        throw(ArgumentError("Variable $varnamefull not found"))
    f = PB.get_field(var, modeldata)
    attributes = copy(var.attributes)
    attributes[:var_name] = var.name
    attributes[:domain_name] = var.domain.name    

    varray = get_array(f, selectargs; attributes=attributes)

    if isnothing(coords)
        # keep original coords (if any)
    else
        check_coords_argument(coords) ||
            error("argument coords should be a Vector of Pairs of \"coord_name\"=>(\"var_name1\", \"var_name2\", ...), eg: [\"z\"=>(\"atm.zmid\", \"atm.zlower\", \"atm.zupper\"), ...]")

        vec_coords_arrays = [
            coord_name => Tuple(get_array(modeldata, cvn, selectargs) for cvn in coord_varnames) 
            for (coord_name, coord_varnames) in coords
        ]

        varray = update_coordinates(varray, vec_coords_arrays) 
    end

    return varray
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

`new_coord_arrays` is a Vector of Pairs of "coord_name"=>(var1::FieldArray, var2::FieldArray, ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]
"""
function update_coordinates(varray::FieldArray, vec_coord_arrays::AbstractVector)

    check_coords_argument(vec_coord_arrays) || 
        error("argument vec_coords_arrays should be a Vector of Pairs of \"coord_name\"=>(var1::FieldArray, var2::FieldArray, ...), eg: [\"z\"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]")

    # generate Vector of NamedDimensions to use as new coordinates
    named_dimensions = PB.NamedDimension[]
    for (coord_name, coord_arrays) in vec_coord_arrays
        fixed_coords = []
        for coord_array in coord_arrays
            push!(fixed_coords, PB.FixedCoord(get(coord_array.attributes, :var_name, ""), coord_array.values, coord_array.attributes))
        end
        push!(
            named_dimensions, PB.NamedDimension(
                coord_name,
                length(first(fixed_coords).values),
                fixed_coords,
            )
        )
    end

    # replace coordinates
    varray_newcoords = FieldArray(varray.name, varray.values, Tuple(named_dimensions), varray.attributes)

    return varray_newcoords
end