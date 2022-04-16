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

#############################################################
# Create from PALEO objects
#############################################################

"""
    get_array(obj, ...) -> FieldArray

Get FieldArray from PALEO object `obj`
"""
function get_array end

"""
    get_array(f::Field; [attributes=nothing], [selectargs...]) -> FieldArray

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

    return FieldArray("", f.values, f.data_dims, attributes)
end

function get_array(
    f::PB.Field{D, PB.CellSpace, V, 0, M};
    attributes=nothing,
    selectargs...
) where {D, V, M}

    values, dims = PB.Grids.get_region(f.mesh, f.values; selectargs...)

    return FieldArray("", values, dims, attributes)
end

# single data dimension
# TODO generalize this to arbitrary data dimensions
function get_array(
    f::PB.Field{D, PB.CellSpace, V, 1, M};
    attributes=nothing,
    selectargs...
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

    return FieldArray("", values, (dims..., f.data_dims...), attributes)
end


"""
    get_array(model::PB.Model, modeldata, varnamefull; selectargs...) -> FieldArray

Get [`FieldArray`](@ref) by Variable name, for spatial region defined by `selectargs`
(which are passed to `PB.Grids.get_region`).
"""
function get_array(model::PB.Model, modeldata::PB.AbstractModelData, varnamefull::AbstractString; selectargs...)
    var = PB.get_variable(model, varnamefull)
    !isnothing(var) ||
        throw(ArgumentError("Variable $varnamefull not found"))
    f = PB.get_field(var, modeldata)
    attrbs = deepcopy(var.attributes)
    attrbs[:var_name] = var.name
    attrbs[:domain_name] = var.domain.name
    return get_array(f; attributes=attrbs, selectargs...)
end