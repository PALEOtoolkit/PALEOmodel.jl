
"""
    FieldRecord{FieldData <: AbstractData, Space <: AbstractSpace, V, N, Mesh <: AbstractMeshOrNothing, R}
    FieldRecord(dataset, f::PB.Field, attributes; [sizehint=nothing])

A series of `records::R` each containing values from a `PALEOboxes.Field{FieldData, Space, N, V, Mesh}`.

Access stored values:
- As a [`FieldArray`](@ref), using [`FieldArray(fr::FieldRecord)`](@ref) or [`get_array`](@ref).
- For scalar data only, as a Vector using `PALEOboxes.get_data`.

# Implementation
Storage in `records::R` is an internal format that may have:
- An optimisation to store Fields with single `values` as a Vector of `eltype(Field.values)`,
  cf Fields with array `values` are stored in `records` as a Vector of arrays.
- A spatial cartesian grid stored as a linear Vector (eg `mesh` isa `PALEOboxes.Grids.CartesianLinearGrid`)
"""
struct FieldRecord{FieldData <: PB.AbstractData, Space <: PB.AbstractSpace, V, N, Mesh <: PB.AbstractMeshOrNothing, R}
    "parent dataset containing this FieldRecord"
    dataset 
    "series of records in internal format"
    records::Vector{R}
    "data dimensions of stored records"
    data_dims::NTuple{N, PB.NamedDimension}
    "grid defining spatial dimensions of stored records"
    mesh::Mesh
    "variable attributes"
    attributes::Dict{Symbol, Any}
end

function Base.getproperty(fr::FieldRecord, p::Symbol)
    if p == :name
        return get(fr.attributes, :var_name, "")        
    else
        return getfield(fr, p)
    end
end

Base.propertynames(fr::FieldRecord) = (:name, fieldnames(typeof(fr))...)

# create empty FieldRecord
function FieldRecord(
    dataset, f::PB.Field{FieldData, Space, V, N, Mesh}, attributes;
    sizehint::Union{Nothing, Int}=nothing
) where {FieldData, Space, V, N, Mesh}
    if _field_single_element(f)
        # if Field contains single elements, store as a Vector of elements
        records = Vector{eltype(f.values)}()
    else
        # if Field contains something else, store as a Vector of those things
        records = Vector{typeof(f.values)}()
    end
    if !isnothing(sizehint)
        sizehint!(records, sizehint)
    end
    return FieldRecord{FieldData, Space, V, N, Mesh, eltype(records)}(
        dataset, 
        records, 
        f.data_dims, 
        f.mesh, 
        attributes,
    )
end


# create a new FieldRecord, containing supplied `existing_records::Vector` data arrays
function FieldRecord(
    dataset,
    existing_records::Vector, 
    FieldData::Type, 
    data_dims::NTuple{N, PB.NamedDimension},
    Space::Type{<:PB.AbstractSpace}, 
    mesh::Mesh,
    attributes;
) where {N, Mesh}

    if _field_single_element(FieldData, N, Space, Mesh)
        # assume existing_records is a Vector, with each element to be stored in Field values::V as a length 1 Vector
        V = Vector{eltype(existing_records)}
    else
        # assume existing_records is a Vector of Field values::V
        V = eltype(existing_records)
    end

    return FieldRecord{FieldData, Space, V, N, typeof(mesh), eltype(existing_records)}(
        dataset, existing_records, data_dims, mesh, attributes,
    )
end

# create a new copy of a FieldRecord, with new dataset, copying records and attributes and updating mesh, attributes[:domain]
function FieldRecord(
    fr::FieldRecord{FieldData, Space, V, N, Mesh, R}, dataset;
    mesh = dataset.grid,
    domain::AbstractString = dataset.name,
) where {FieldData, Space, V, N, Mesh, R}
    new_attributes = deepcopy(fr.attributes)
    new_attributes[:domain_name] = domain
    return FieldRecord{FieldData, Space, V, N, Mesh, R}(
        dataset,
        deepcopy(fr.records),
        fr.data_dims,
        mesh,
        new_attributes,
    )
end

space(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}) where {FieldData, Space, V, N, Mesh, R} = Space

field_data(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}) where {FieldData, Space, V, N, Mesh, R} = FieldData

function PB.get_dimensions(fr::FieldRecord; expand_cartesian=false)
    dims_spatial = PB.get_dimensions(fr.mesh, space(fr); expand_cartesian)
    dims = [dims_spatial..., fr.data_dims..., PB.NamedDimension(fr.dataset.record_dim.name, length(fr))]
    return dims
end

function PB.get_dimension(fr::FieldRecord, dimname::AbstractString; expand_cartesian=false)
    dims = PB.get_dimensions(fr; expand_cartesian)
    idx = findfirst(d -> d.name==dimname, dims)
    !isnothing(idx) ||
            error("no dimension='$dimname' (available dimensions: $dims")
    return dims[idx]
end

function PB.get_coordinates(fr::FieldRecord, dimname::AbstractString; expand_cartesian=false)
    if !isnothing(findfirst(nd -> nd.name == dimname, PB.get_dimensions(fr.mesh, space(fr); expand_cartesian)))
        # no grid (fr.mesh == nothing) defines a "cell" dimension, but not get_coordinates
        return isnothing(fr.mesh) ? String[] : PB.get_coordinates(fr.mesh, dimname)
    elseif !isnothing(findfirst(nd -> nd.name == dimname, fr.data_dims))
        return hasproperty(fr.dataset, :data_dims_coordinates) ? get(fr.dataset.data_dims_coordinates, dimname, String[]) : String[]
    elseif dimname in ("records", fr.dataset.record_dim.name)
        return hasproperty(fr.dataset, :record_dim_coordinates) ? fr.dataset.record_dim_coordinates : String[]
    end

    error("FieldRecord $fr has no dimension $dimname")
end

function Base.show(io::IO, fr::FieldRecord)
    print(io, "FieldRecord(name='", fr.name, "', eltype=", eltype(fr),", length=", length(fr), ")",)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(fr::FieldRecord))
    show(io, fr)
    println()
    # println(io, "  records: ", fr.records)
    println(io, "  data_dims: ", fr.data_dims)
    println(io, "  mesh: ", fr.mesh)
    println(io, "  attributes: ", fr.attributes)
    if PB.has_internal_cartesian(fr.mesh, space(fr))
        dimlist = [ "  dimensions (internal): " => false, "  dimensions (cartesian): " => true]
    else
        dimlist = ["  dimensions: " => false]
    end
    for (str, expand_cartesian) in dimlist
        println(io, str)
        dims = PB.get_dimensions(fr; expand_cartesian)
        for nd in dims
            print(io, "    ", nd)
            coord_names = PB.get_coordinates(fr, nd.name; expand_cartesian)
            if isempty(coord_names)
                println()
            else
                println(" => ", coord_names)
            end
        end
    end
    return nothing
end

"""
    _field_single_element(fr::FieldRecord)::Bool
    _field_single_element(f::Field)::Bool
    _field_single_element(FieldData, N, Space, Mesh) 
 
Test whether FieldRecord contains Fields with single elements stored as a Vector instead of a  Vector of records.

- `_field_single_element == false`: Fields contain array `values`, these are stored in FieldRecord `records` as a Vector of arrays.
- `_field_single_element == true` Fields contain a single value, stored in FieldRecord `records` as Vector of `eltype(Field.values)`. 

NB: this works on Types, and will return false for a field with Space == CellSpace with ncells=1, even though this actually contains a single
value. TODO might be clearer to directly use PB.internal_size(Space, mesh) == (1,) which would also handle the CellSpace with ncells=1 case,
but this wouldn't work in the type domain (needs an instance of mesh::Mesh)
"""
function _field_single_element(::Type{FieldData}, N, ::Type{Space}, ::Type{Mesh}) where {FieldData <: PB.AbstractData, Space <: PB.AbstractSpace, Mesh}
    if PB.field_single_element(FieldData, N) && (Space == PB.ScalarSpace || (Space == PB.CellSpace && Mesh == Nothing))
        return true
    else
        return false
    end
end

_field_single_element(f::PB.Field{FieldData, Space, V, N, Mesh}) where {FieldData, Space, V, N, Mesh} = 
    _field_single_element(FieldData, N, Space, Mesh)
_field_single_element(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}) where {FieldData, Space, V, N, Mesh, R} = 
    _field_single_element(FieldData, N, Space, Mesh)


function Base.push!(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}, f::PB.Field{FieldData, Space, V, N, Mesh}) where {FieldData, Space, V, N, Mesh, R}
    if _field_single_element(fr)
        # if Field contains single elements, store as a Vector of elements
        push!(fr.records, f.values[])
    else
        # if Field contains something else, store as a Vector of those things
        push!(fr.records, copy(f.values))
    end
    return fr
end

function Base.push!(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}, record::R) where {FieldData, Space, V, N, Mesh, R}
    push!(fr.records, record)
    return fr
end

Base.length(fr::FieldRecord) = length(fr.records)

Base.eltype(::Type{FieldRecord{FieldData, Space, V, N, Mesh, R}}) where {FieldData, Space, V, N, Mesh, R} = PB.Field{FieldData, Space, V, N, Mesh}

function Base.getindex(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}, i::Int) where {FieldData, Space, V, N, Mesh, R}

    if _field_single_element(fr)
        # if Field contains single elements, FieldRecord stores as a Vector of elements
        return PB.Field([fr.records[i]], FieldData, fr.data_dims, missing, Space, fr.mesh)
    else
        # if Field contains something else, FieldRecord stores as a Vector of those things
        return PB.Field(fr.records[i], FieldData, fr.data_dims, missing, Space, fr.mesh)       
    end
end

Base.lastindex(fr::FieldRecord) = lastindex(fr.records)

function Base.copy(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}) where {FieldData, Space, V, N, Mesh, R}
    return FieldRecord{FieldData, Space, V, N, Mesh, R}(
        fr.dataset,
        deepcopy(fr.records),
        fr.data_dims,
        fr.mesh,
        deepcopy(fr.attributes),
    )
end

"""
    PB.get_data(fr::FieldRecord; records=nothing, squeeze_all_single_dims=true)

Get data records in raw format. Only recommended for variables with scalar data ie one value per record.

`records` may be `nothing` to get all records, 
an `Int` to select a single record, or a range to select multiple records.

If `squeeze_all_single_dims=true` (the default), if each record represents a scalar
(eg a PALEO Variable with Space PB.ScalarSpace, or a PB.CellSpace variable in a Domain with
a single cell), then data is returned as a Vector. NB: if `records` is an Int,
the single record requested is returned as a length-1 Vector.

Non-scalar data (eg a non-ScalarSpace variable from a Domain with > 1 cell) 
is returned in internal format as a Vector-of-Vectors.
"""    
function PB.get_data(@nospecialize(fr::FieldRecord); records=nothing, squeeze_all_single_dims=true)
    
    # Optionally squeeze out single cell stored internally as a Vector-of-Vectors, length 1 
    # (eg a CellSpace Variable in a Domain with 1 cell)
    squeeze_vecvec = squeeze_all_single_dims && !isempty(fr.records) && length(first(fr.records)) == 1
    if _field_single_element(fr) || squeeze_vecvec
        if _field_single_element(fr)
            # internal format already is a Vector
            records_vec = fr.records
        else
            # combine Vector of length 1 Vectors into a Vector
            records_vec = [only(r) for r in fr.records]
        end        
        if isnothing(records)
            data_output = records_vec
        else        
            # bodge - fix scalar data
            # if isa(records, Integer) && !isa(data_output, AbstractVector)
            #     data_output =[data_output]
            # 
            if isa(records, Integer)
                # represent a scalar as a length 1 Vector
                # (a 0D Array might be more logical, but for backwards compatibility keep as a Vector)
                data_output =[records_vec[records]]
            else
                data_output = records_vec[records]
            end
        end
    else
        # Vector-of-Vectors - return raw data
        if isnothing(records)
            data_output = fr.records
        else
            data_output = fr.records[records]
        end
    end

    return data_output
end  

"""    
    get_array(
        fr::FieldRecord [, allselectargs::NamedTuple];
        coords=nothing,
        lookup_coords=true,
        add_attributes=true,
        update_name=true,
        omit_recorddim_if_constant=false, 
    ) -> fa_out::FieldArray

    [deprecated] get_array(fr::FieldRecord [; kwargs...] [; allselectargs...]) -> fa_out::FieldArray

Return a [`FieldArray`](@ref) containing an Array of values and
any attached coordinates, for records and spatial region defined by `allselectargs`.

Combines [`FieldArray(fr::FieldRecord)`](@ref) and [`select(fa::FieldArray)`](@ref)

Optional `allselectargs` field `expand_cartesian` (default `false`) is only needed for spatially resolved output using a `CartesianLinearGrid`, 
set to `true` to expand compressed internal data (with spatial dimension `cells`) to a cartesian array  (with spatial dimensions eg `lon`, `lat`, `zt`)

# Keyword arguments
See  [`FieldArray(fr::FieldRecord)`](@ref) and [`select(fa::FieldArray)`](@ref)

# Examples
- select a timeseries for single cell index 3 :

      get_array(fr, (cell=3, ))
- select first column at nearest available time to model time 10.0 :

      get_array(fr, (column=1, tmodel=10.0))
- set first column at nearest available time to model time 10.0, replacing atmosphere model pressure coordinate with z coordinate:

      get_array(
          fr, (column=1, tmodel=10.0);
          coords=["cells"=>("atm.zmid", "atm.zlower", "atm.zupper")]
      )
- select surface 2D array (dimension `zt`, index 1) from 3D output at nearest available 
    time to model time 10.0, expanding `CartesianLinearGrid`:

      get_array(fr, (zt_isel=1, tmodel=10.0, expand_cartesian=true))
- select section 2D array at nearest index to longitude 200 degrees from 3D output at nearest available 
  time to model time 10.0, expanding `CartesianLinearGrid`:

      get_array(fr, (lon=200.0, tmodel=10.0, expand_cartesian=true))

- get full data cube as used for netcdf output, omitting coordinates and attributes, retaining singleton dimensions, and omitting 
  record dimension if variable is a constant:

      get_array(
          fr, (expand_cartesian=true, squeeze_all_single_dims=false);
          lookup_coordinates=false, add_attributes=false, omit_recorddim_if_constant=true
      )

# Limitations
- it is only possible to select either a single slice or a contiguous range for each dimension, not a set of slices for a Vector of index
  or coordinate values.
"""
function get_array(
    @nospecialize(fr::FieldRecord); 
    coords=nothing,
    lookup_coords::Bool=true, 
    add_attributes::Bool=true,
    update_name=true,
    omit_recorddim_if_constant::Bool=false,
    verbose::Bool=false,
    allselectargs...
)
    isempty(allselectargs) ||
        Base.depwarn(
            "allselectargs... will be deprecated in a future release.  Please use allselectargs::NamedTuple instead",
            :get_array,
        )

    return get_array(
        fr, NamedTuple(allselectargs); 
        coords, lookup_coords, add_attributes, update_name, omit_recorddim_if_constant, verbose,
    )
end

function get_array(
    @nospecialize(fr::FieldRecord), @nospecialize(allselectargs::NamedTuple); # allselectargs::NamedTuple=NamedTuple() creates a method ambiguity with deprecated form above
    coords=nothing,
    lookup_coords::Bool=true,
    add_attributes::Bool=true,
    update_name=true,
    omit_recorddim_if_constant::Bool=false, 
    verbose::Bool=false,
)
    frname = _fieldarray_default_varnamefull(fr.attributes; include_selectargs=true)
    fa_select = nothing

    try
        verbose && println("get_array (begin): $frname, allselectargs $allselectargs lookup_coords $lookup_coords")
    
        # override expand_cartesian
        if :expand_cartesian in keys(allselectargs)
            expand_cartesian = allselectargs.expand_cartesian
            # remove :expand_cartesian from allselectargs
            allselectargs = Base.structdiff(allselectargs, NamedTuple{(:expand_cartesian,)}((nothing,)))
        else
            expand_cartesian = false
        end
        
        fa_full = FieldArray(
            fr;
            expand_cartesian,
            omit_recorddim_if_constant,
            lookup_coords,
            coords,
        )

        fa_select = select(
            fa_full, allselectargs;
            record_dim_name=fr.dataset.record_dim.name,
            mesh=fr.mesh,
            update_name,
            add_attributes,
            verbose,
        )

    catch
        @error "get_array exception: $frname, allselectargs $allselectargs"
        rethrow()
    end

    return fa_select
end


"""
    FieldRecordValues <: AbstractArray

Internal use by FieldArray: provides an n-dimensional Array view on FieldArray records
"""
struct FieldRecordValues{E, R, LI, NumSpatialDims, NumDataDims, HasRecordDim, FieldSingleElement, N} <: AbstractArray{E, N}
    records::Vector{R}
    linear_index::LI # Nothing or Array{Union{Missing, Int32}, NumSpatialDims}
end

Base.eltype(::Type{FieldRecordValues{E}}) where {E} = E

# HasRecordDim true, FieldSingleElement false
Base.size(rv::FieldRecordValues{E, R, Nothing, NumSpatialDims, NumDataDims, true, false, N}) where {
    E, R, NumSpatialDims, NumDataDims, N,
} = (size(first(rv.records))..., length(rv.records))
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 0, true, false, N}, i, ridx) where {E, R, N} = rv.records[ridx][i]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 2, 0, true, false, N}, i, j, ridx) where {E, R, N} = rv.records[ridx][i, j]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 3, 0, true, false, N}, i, j, k, ridx) where {E, R, N} = rv.records[ridx][i, j, k]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 1, true, false, N}, i, ridx) where {E, R, N} = rv.records[ridx][i]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 2, true, false, N}, i, j, ridx) where {E, R, N} = rv.records[ridx][i, j]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 1, true, false, N}, i, j, ridx) where {E, R, N} = rv.records[ridx][i, j]

Base.size(rv::FieldRecordValues{E, R, LI, NumSpatialDims, NumDataDims, true, false, N}) where {
    E, R, LI, NumSpatialDims, NumDataDims, N,
} = (size(rv.linear_index)..., size(first(rv.records))[2:end]..., length(rv.records))
function Base.getindex(rv::FieldRecordValues{E, R, LI, 2, 0, true, false, N}, i, j, ridx) where {E, R, LI, N}
    linear_idx = rv.linear_index[i, j]
    return ismissing(linear_idx) ? missing : rv.records[ridx][linear_idx]
end
function Base.getindex(rv::FieldRecordValues{E, R, LI, 3, 0, true, false, N}, i, j, k, ridx) where {E, R, LI, N}
    linear_idx = rv.linear_index[i, j, k]
    return ismissing(linear_idx) ? missing : rv.records[ridx][linear_idx]
end

# HasRecordDim true, FieldSingleElement true
Base.size(rv::FieldRecordValues{E, R, Nothing, 0, 0, true, true, N}) where {E, R,  N} = (length(rv.records),)
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 0, true, true, N}, ridx) where {E, R, N } = rv.records[ridx]

Base.size(rv::FieldRecordValues{E, R, Nothing, 1, 0, true, true, N}) where {E, R,  N} = (1, length(rv.records),)
function Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 0, true, true, N}, i, ridx) where {E, R, N}
    i == 1 || throw(BoundsError(rv, i))
    return rv.records[ridx]
end

# HasRecordDim false, FieldSingleElement false
Base.size(rv::FieldRecordValues{E, R, Nothing, NumSpatialDims, NumDataDims, false, false, N}) where {
    E, R, NumSpatialDims, NumDataDims, N,
} = size(first(rv.records))
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 0, false, false, N}, i) where {E, R, N} = rv.records[1][i]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 2, 0, false, false, N}, i, j) where {E, R, N} = rv.records[1][i, j]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 3, 0, false, false, N}, i, j, k) where {E, R, N} = rv.records[1][i, j, k]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 1, false, false, N}, i) where {E, R, N} = rv.records[1][i]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 2, false, false, N}, i, j) where {E, R, N} = rv.records[1][i, j]
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 1, false, false, N}, i, j) where {E, R, N} = rv.records[1][i, j]

Base.size(rv::FieldRecordValues{E, R, LI, NumSpatialDims, NumDataDims, false, false, N}) where {
    E, R, LI, NumSpatialDims, NumDataDims, N,
} = (size(rv.linear_index)..., size(first(rv.records))[2:end]...)
function Base.getindex(rv::FieldRecordValues{E, R, LI, 2, 0, false, false, N}, i, j) where {E, R, LI, N}
    linear_idx = rv.linear_index[i, j]
    return ismissing(linear_idx) ? missing : rv.records[1][linear_idx]
end
function Base.getindex(rv::FieldRecordValues{E, R, LI, 3, 0, false, false, N}, i, j, k) where {E, R, LI, N}
    linear_idx = rv.linear_index[i, j, k]
    return ismissing(linear_idx) ? missing : rv.records[1][linear_idx]
end

# HasRecordDim false, FieldSingleElement true
Base.size(rv::FieldRecordValues{E, R, Nothing, 0, 0, false, true, N}) where {E, R, N} = ()
Base.getindex(rv::FieldRecordValues{E, R, Nothing, 0, 0, false, true, N}) where {E, R, N} = rv.records[1] 

Base.size(rv::FieldRecordValues{E, R, Nothing, 1, 0, false, true, N}) where {E, R, N} = (1, )
function Base.getindex(rv::FieldRecordValues{E, R, Nothing, 1, 0, false, true, N}, i) where {E, R, N}
    i == 1 || throw(BoundsError(rv, i))
    return rv.records[1]
end

"""
    FieldArray(
        fr::FieldRecord; 
        expand_cartesian=true, 
        omit_recorddim_if_constant=true,
        lookup_coords=true,
        coords=nothing,
    )

Construct [`FieldArray`](@ref) from all records in `fr::FieldRecord`

Provides a view of internal storage format of `FieldRecord` as an n-dimensional Array.

# Keyword arguments
- `expand_cartesian`: (spatially resolved output using a `CartesianLinearGrid` only), `true` to expand compressed internal data
  (with spatial dimension `cells`) to a cartesian array  (with spatial dimensions eg `lon`, `lat`, `zt`)
- `omit_recorddim_if_constant`: Specify whether to include multiple (identical) records and record dimension for constant variables
   (with attribute `is_constant = true`). PALEO currently always stores these records in the input `fr::FieldRecord`; 
   set `false` include them in `FieldArray` output, `true` to omit duplicate records and record dimension from output.
- `lookup_coords`: `true` to include coordinates, `false` to omit coordinates.
- `coords`: replace the attached coordinates from the input `fr::FieldRecord` for one or more dimensions.
  Format is a Vector of Pairs of `"<dim_name>"=>("<var_name1>", "<var_name2>", ...)`, 
  eg to replace an nD column atmosphere model default pressure coordinate with a z coordinate:

      coords=["cells"=>("atm.zmid", "atm.zlower", "atm.zupper")]
"""
function FieldArray(
    @nospecialize(fr::FieldRecord);
    expand_cartesian::Bool=false,
    omit_recorddim_if_constant::Bool=true,
    lookup_coords::Bool=true,
    coords=nothing,
)

    is_constant = omit_recorddim_if_constant ? _variable_is_constant(fr) : false
    _field_single_element(fr)

    dims_spatial = PB.get_dimensions(fr.mesh, space(fr); expand_cartesian)
    E = _field_single_element(fr) ? eltype(fr.records) : eltype(eltype(fr.records))  # eltype of values 
    if expand_cartesian && PB.has_internal_cartesian(fr.mesh, space(fr))
        linear_index = fr.mesh.linear_index
        E = Union{Missing, E}
    else
        linear_index = nothing
    end

    values = FieldRecordValues{
        E,
        eltype(fr.records),
        typeof(linear_index),
        length(dims_spatial),
        length(fr.data_dims),
        !is_constant, 
        _field_single_element(fr), 
        length(dims_spatial) + length(fr.data_dims) + !is_constant,
    }(fr.records, linear_index)

    dims = [dims_spatial..., fr.data_dims...]
    if !is_constant
        push!(dims, PB.NamedDimension(fr.dataset.record_dim.name, length(fr)))
    end

    dims_coords = Vector{Pair{PB.NamedDimension, Vector{FieldArray}}}(undef, length(dims))
    for i in 1:(length(dims))
        dims_coords[i] = dims[i] => lookup_coords ? _read_coordinates(
            fr, dims[i]; expand_cartesian, coords,
        ) : FieldArray[]
    end

    fa = FieldArray(
        fr.name,
        values,
        dims_coords,
        fr.attributes,
    )

    return fa
end

function _read_coordinates(
    fr::FieldRecord, dim::PB.NamedDimension;
    expand_cartesian=true,
    coords=nothing,
)
    coord_names = nothing
    if !isnothing(coords)
        _check_coordinates_argument(coords) ||
            error("argument coords should be a Vector of Pairs of \"dim_name\"=>(\"var_name1\", \"var_name2\", ...), eg: [\"z\"=>(\"atm.zmid\", \"atm.zlower\", \"atm.zupper\"), ...]")
 
        for (sdim_name, scoord_names) in coords
            if sdim_name == dim.name
                coord_names = scoord_names
                break
            end
        end
    end
    
    if isnothing(coord_names)
        coord_names = PB.get_coordinates(fr, dim.name; expand_cartesian)
    end

    coord_values = FieldArray[]
    for cn in coord_names
        cfr = PB.get_field(fr.dataset, cn)
        cv = FieldArray(cfr; lookup_coords=false, expand_cartesian, omit_recorddim_if_constant=true)
        is_coordinate(cv, dim) ||
            error("dimension $dim coord name $cn read invalid coordinate $cv")
        push!(coord_values, cv)
    end
    return coord_values
end

# check 'coords' of form [] or ["z"=>[ ... ], ] or ["z"=>(...),]
_check_coordinates_argument(coords) =
    isa(coords, AbstractVector) && (
        isempty(coords) || (
            isa(coords, AbstractVector{<:Pair}) &&
            isa(first(first(coords)), AbstractString) &&
            isa(last(first(coords)), Union{AbstractVector, Tuple})
        )
    )




# TODO time-independent variables are indicated by setting :is_constant attribute,
# but this is only recently added and FieldRecord still stores a record for every timestep
# This uses :is_constant to guess if a variable is constant, and then checks the values really are constant
function _variable_is_constant(fr::FieldRecord)

    is_constant = false
    # PALEO indicates time-independent variables by setting :is_constant attribute,
    # but currently FieldRecord still stores all the records
    if get(fr.attributes, :is_constant, false) == true
        data_identical = false
        for rcd in fr.records
            if rcd == first(fr.records) || (all(isnan, rcd) && all(isnan, first(fr.records)))
                data_identical = true
            end
        end
        if data_identical
            is_constant = true
        else
            @warn "variable $(fr.name) has :is_constant set but data is not constant !"
        end
    end

    return is_constant
end


"""
    FieldRecord(dataset, avalues::Array, avalues_dimnames, attributes::Dict{Symbol, Any}; [expand_cartesian=false])

Create from an Array of data values `avalues`.

FieldRecord type and dimensions are set from a combination of `attributes` and `dataset` dimensions and grid, where
`Space = attributes[:space]`, `FieldData = attributes[:field_data]`, `data_dims` are set from names in `attributes[:data_dims]`.

Dimension names and size are cross-checked against supplied names in `avalues_dimnames`

This is effectively the inverse of:

    a = FieldArray(
        fr::FieldRecord;
        lookup_coords=false, omit_recorddim_if_constant=true, expand_cartesian=true,
    )
    avalues = a.values
    avalues_dimnames = [nd.name for (nd, _) in a.dims_coords]
    attributes = a.attributes
"""
function FieldRecord(
    dataset,
    avalues::AbstractArray,
    avalues_dimnames::Union{Vector{String}, Tuple{Vararg{String}}},
    attributes::Dict{Symbol, Any};
    expand_cartesian::Bool=false,
)
    FieldData = attributes[:field_data]
    Space = attributes[:space]
    data_dim_names = attributes[:data_dims]

    dims_spatial_expected = PB.get_dimensions(dataset.grid, Space; expand_cartesian)
    
    dataset_dims_all = PB.get_dimensions(dataset)
 
    data_dims_nd_vec = PB.NamedDimension[]
    for dd_name in data_dim_names
        dd_idx = findfirst(nd -> nd.name == dd_name, dataset_dims_all)
        push!(data_dims_nd_vec, dataset_dims_all[dd_idx])
    end
    data_dims = Tuple(data_dims_nd_vec)
   
    record_dim = dataset_dims_all[end]
    is_constant = !(record_dim.name in avalues_dimnames)

    # check dimension names and sizes
    dims_expected = [dims_spatial_expected..., data_dims...]
    if !is_constant
        push!(dims_expected, record_dim)
    end
    if length(avalues_dimnames) == length(dims_expected)   
        for i in 1:length(avalues_dimnames)
            @assert avalues_dimnames[i] == dims_expected[i].name
            @assert size(avalues, i) == dims_expected[i].size
        end
    else
        length_expected = prod(nd->nd.size, dims_expected)
        errmsg = "var $(attributes[:domain_name]).$(attributes[:var_name]) netcdf dimensions $avalues_dimnames $(size(avalues)) != dims_expected $dims_expected"
        if length(avalues) == length_expected
            @warn "$errmsg but length is the same - continuing"
        else
            error(errmsg)
        end
    end

    if _field_single_element(FieldData, length(data_dims), Space, typeof(dataset.grid))
        if is_constant
            @assert length(avalues) == 1
            #  avalues is 0D Array if a scalar variable
            #  avalues is a Vector length 1 if a CellSpace variable in a Domain with no grid
            records = fill(avalues[], record_dim.size)  
        else
            @assert length(dims_spatial_expected) <= 1
            if isempty(dims_spatial_expected)
                # scalar variable
                records = avalues
            elseif length(dims_spatial_expected) == 1
                # eg a CellSpace variable in a Domain with no grid still has a spatial dimension size 1
                # avalues will be a Matrix size(1, nrecs)
                @assert only(dims_spatial_expected).size == 1
                records = vec(avalues)
            end
        end
    else
        if expand_cartesian && PB.has_internal_cartesian(dataset.grid, Space)
            pack_fn = x -> PB.Grids.cartesian_to_internal(dataset.grid, x)
        else
            pack_fn = identity
        end
        
        if is_constant
            first_record = pack_fn(avalues)
            records = [first_record for i in 1:record_dim.size]
        else
            records = _create_records_from_array!(avalues, pack_fn)
        end
    end
    
    vfr = PALEOmodel.FieldRecord(
        dataset,
        records, 
        FieldData, 
        data_dims,
        Space, 
        dataset.grid,
        attributes,
    ) 

    return vfr
end

# function barrier optimisation
function _create_records_from_array!(avalues, pack_fn)
    acolons_no_recorddim = ntuple(x->Colon(), ndims(avalues)-1)

    first_record = pack_fn(avalues[acolons_no_recorddim..., 1])
    records = Vector{typeof(first_record)}(undef, last(size(avalues)))
    records[1] = first_record

    for ri in 2:last(size(avalues))
        records[ri] = pack_fn(avalues[acolons_no_recorddim..., ri])
    end

    return records
end
