"""
    FieldRecord{D <: AbstractData, S <: AbstractSpace, V, N, M, R}
    FieldRecord(
        f::PB.Field{D, S, V, N, M}, attributes; 
        coords_record, 
        sizehint::Union{Nothing, Int}=nothing
    ) -> fr

A series of `records::R` each containing the `values` from a `PALEOboxes.Field{D, S, N, V, M}`.

A `coords_record` may be attached to provide a coordinate (eg model time) corresponding to `records`.

# Implementation
Fields with array `values` are stored in `records` as a Vector of arrays.
Fields with single `values` (`field_single_element` true) are stored as a Vector of `eltype(Field.values)`. 
"""
struct FieldRecord{D <: PB.AbstractData, S <: PB.AbstractSpace, V, N, M, R}
    records::Vector{R}
    data_dims::NTuple{N, PB.NamedDimension}
    mesh::M
    attributes::Dict{Symbol, Any}
    coords_record::Vector{PB.FixedCoord} # coordinates attached to record dimension
end

function Base.show(io::IO, fr::FieldRecord)
    print(io, 
        "FieldRecord(eltype=", eltype(fr),", length=", length(fr), 
        ", attributes=", fr.attributes, ", coords_record=", fr.coords_record, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", fr::FieldRecord)
    println(io, "FieldRecord(eltype=", eltype(fr),", length=", length(fr), ")") 
    println(io, "  data_dims: ", fr.data_dims)
    println(io, "  mesh: ", fr.mesh)
    println(io, "  attributes: ", fr.attributes)
    println(io, "  coords_record: ", fr.coords_record)
    return nothing
end

"test whether Field contains single elements"
function field_single_element(::Type{field_data}, N, ::Type{S}, ::Type{M}) where {field_data <: PB.AbstractData, S <: PB.AbstractSpace, M}
    if PB.field_single_element(field_data, N) && (S == PB.ScalarSpace || (S == PB.CellSpace && M == Nothing))
        return true
    else
        return false
    end
end

field_single_element(::Type{PB.Field{D, S, V, N, M}}) where {D, S, V, N, M} = field_single_element(D, N, S, M)
field_single_element(::Type{FR}) where {FR <: FieldRecord} = field_single_element(eltype(FR))
field_single_element(f::T) where {T} = field_single_element(T)


function FieldRecord(
    f::PB.Field{D, S, V, N, M}, attributes;
    coords_record, 
    sizehint::Union{Nothing, Int}=nothing
) where {D, S, V, N, M}
    if field_single_element(f)
        # if Field contains single elements, store as a Vector of elements
        records = Vector{eltype(f.values)}()
    else
        # if Field contains something else, store as a Vector of those things
        records = Vector{typeof(f.values)}()
    end
    if !isnothing(sizehint)
        sizehint!(records, sizehint)
    end
    return FieldRecord{D, S, V, N, M, eltype(records)}(records, f.data_dims, f.mesh, attributes, coords_record)
end

"create a new FieldRecord, containing supplied `existing_values::Vector` data arrays"
function wrap_fieldrecord(
    existing_values::Vector, 
    field_data::Type, 
    data_dims::NTuple{N, PB.NamedDimension},
    data_type::Union{DataType, Missing},
    space::Type{<:PB.AbstractSpace}, 
    mesh::M,
    attributes;
    coords_record
) where {N, M}
    # check_values(
    #    existing_values, field_data, data_dims, data_type, space, spatial_size(space, mesh), 
    # )
    if field_single_element(field_data, N, space, M)
        # assume existing_values is a Vector, with each element to be stored in Field values::V as a length 1 Vector
        V = Vector{eltype(existing_values)}
    else
        # assume existing_values is a Vector of Field values::V
        V = eltype(existing_values)
    end

    return FieldRecord{field_data, space, V, N, typeof(mesh), eltype(existing_values)}(
        existing_values, data_dims, mesh, attributes, coords_record
    )
end

function Base.push!(fr::FieldRecord{D, S, V, N, M, R}, f::PB.Field{D, S, V, N, M}) where {D, S, V, N, M, R}
    if field_single_element(fr)
        # if Field contains single elements, store as a Vector of elements
        push!(fr.records, f.values[])
    else
        # if Field contains something else, store as a Vector of those things
        push!(fr.records, copy(f.values))
    end
    return fr
end

Base.length(fr::FieldRecord) = length(fr.records)

Base.eltype(::Type{FieldRecord{D, S, V, N, M, R}}) where {D, S, V, N, M, R} = PB.Field{D, S, V, N, M}

function Base.getindex(fr::FieldRecord{D, S, V, N, M, R}, i::Int) where {D, S, V, N, M, R}

    if field_single_element(fr)
        # if Field contains single elements, FieldRecord stores as a Vector of elements
        return PB.wrap_field([fr.records[i]], D, fr.data_dims, missing, S, fr.mesh)
    else
        # if Field contains something else, FieldRecord stores as a Vector of those things
        return PB.wrap_field(fr.records[i], D, fr.data_dims, missing, S, fr.mesh)       
    end
end

Base.lastindex(fr::FieldRecord) = lastindex(fr.records)

function Base.copy(fr::FieldRecord{D, S, V, N, M, R}) where {D, S, V, N, M, R}
    return FieldRecord{D, S, V, N, M, R}(
        deepcopy(fr.records),
        fr.data_dims,
        fr.mesh,
        deepcopy(fr.attributes),
        copy(fr.coords_record),
    )
end

"""    
    get_array(fr::FieldRecord [, allselectargs::NamedTuple] [; coords::AbstractVector]) -> FieldArray
    [deprecated] get_array(fr::FieldRecord [; coords::AbstractVector] [; allselectargs...]) -> FieldArray

Return a [`FieldArray`](@ref) containing `fr::FieldRecord` data values and
any attached coordinates, for records and spatial region defined by `allselectargs`.

`allselectarg` may include `recordarg` to define records, `selectargs` to define a spatial region.

`recordarg` can be one of:
- `records=r::Int` to select a single record, or `records = first:last` to select a range.
- `<record coord>`: (where eg `<record coord>=tmodel`), `<record_coord>=t::Float64` to select a single record
  with the nearest value of `fr.coords_record`, or `<record_coord>=(first::Float64, last::Float64)` (a Tuple) to select a range
  starting at the record with the nearest value of `fr.coords_record` before `first` and ending at the nearest record after
  `last`.

Available `selectargs` depend on the grid `fr.mesh`, and 
are passed to `PB.Grids.get_region`.

Optional argument `coords` can be used to supply coordinates from additional `FieldRecords`, replacing any coordinates
attached to `fr`. Format is a Vector of Pairs of "coord_name"=>(cr1::FieldRecord, cr2::FieldRecord, ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:

    coords=["z"=>(zmid::FieldRecord, zlower::FieldRecord, zupper::FieldRecord)]

NB: the coordinates will be generated by applying `allselectargs`,
so the supplied coordinate FieldRecords must have the same dimensionality as `fr`.
"""
function get_array(
    fr::FieldRecord; 
    coords=nothing,
    allselectargs...
)
    isempty(allselectargs) ||
        Base.depwarn(
            "allselectargs... will be deprecated in a future release.  Please use allselectargs::NamedTuple instead",
            :get_array,
        )

    return get_array(fr, NamedTuple(allselectargs); coords=coords)
end

function get_array(
    fr::FieldRecord, allselectargs::NamedTuple; # allselectargs::NamedTuple=NamedTuple() creates a method ambiguity with deprecated form above
    coords::Union{Nothing, AbstractVector}=nothing
)

    varray = nothing

    try
        # get data NB: with original coords or no coords
        varray = _get_array(fr, allselectargs)

        if !isnothing(coords)
            check_coords_argument(coords) ||
                error("coords argument should be of form 'coords=[\"z\"=>zmid::FieldRecord, zlower::FieldRecord, zupper::FieldRecord), ...]")

            # get arrays for replacement coordinates
            vec_coord_arrays = [
                coord_name => Tuple(_get_array(coord_fr, allselectargs) for coord_fr in coord_fieldrecords) 
                for (coord_name, coord_fieldrecords) in coords
            ]

            # replace coordinates
            varray = update_coordinates(varray, vec_coord_arrays)
        end
    catch
        frname = get(fr.attributes, :domain_name, "<none>")*"."*get(fr.attributes, :var_name, "<none>")
        @warn "PALEOmodel.get_array(fr::FieldRecord) failed for variable $frname"
        rethrow()
    end

    return varray
end

function _get_array(
    fr::FieldRecord{D, S, V, N, M, R}, allselectargs::NamedTuple, 
) where {D, S, V, N, M, R}

    # select records to use and create PB.NamedDimension
    ridx = 1:length(fr)
    selectargs_region = NamedTuple()
    selectargs_records = NamedTuple()
    for (k, v) in zip(keys(allselectargs), allselectargs)
        if k==:records
            if v isa Integer
                ridx = [v]                
            else
                ridx = v
            end
            selectargs_records = (records=v,)
        elseif String(k) in getfield.(fr.coords_record, :name)
            # find ridx corresponding to a coordinate
            for cr in fr.coords_record
                if String(k) == cr.name
                    ridx, cvalue = PB.find_indices(cr.values, v)
                    selectargs_records=NamedTuple((k=>cvalue,))
                end
            end            
        else
            selectargs_region = merge(selectargs_region, NamedTuple((k=>v,)))
        end
    end
    records_dim = PB.NamedDimension("records", length(ridx), PB.get_region(fr.coords_record, ridx))

    # add attributes for selection used
    attributes = copy(fr.attributes)
    isempty(selectargs_records) || (attributes[:filter_records] = selectargs_records;)
    isempty(selectargs_region) || (attributes[:filter_region] = selectargs_region;)

    # Generate name from attributes
    name = default_fieldarray_name(attributes)

    # Select selectargs_region 
    if field_single_element(fr)
        # create FieldArray directly from FieldRecord
        isempty(selectargs_region) || 
            throw(ArgumentError("invalid index $selectargs_region"))
        if length(ridx) > 1
            return FieldArray(
                name,
                fr.records[ridx], 
                (fr.data_dims..., records_dim),
                attributes
            )
        else
            # squeeze out records dimension
            return FieldArray(
                name,
                fr.records[first(ridx)], 
                fr.data_dims,
                attributes
            )
        end
    else
        # pass through to Field

        # get FieldArray from first Field record and use this to figure out array shapes etc
        far = get_array(fr[first(ridx)], selectargs_region)
        # TODO - Julia bug ? length(far.dims) returning wrong value, apparently triggered by this line
        #        attributes = isnothing(attributes) ? Dict{Symbol, Any}() : copy(attributes)
        # in get_array
        if length(ridx) > 1
            # add additional (last) dimension for records
            favalues = Array{eltype(far.values), length(far.dims)+1}(undef, size(far.values)..., length(ridx))
            fa = FieldArray(
                name,
                favalues, 
                (far.dims..., records_dim), 
                attributes,
            )
            # copy values for first record
            if isempty(far.dims)
                fa.values[1] = far.values
            else
                fa.values[fill(Colon(), length(far.dims))..., 1] .= far.values
            end
        else
            # squeeze out record dimension so this is just fa with attributes added
            fa = FieldArray(
                name, 
                far.values,
                far.dims,
                attributes
            ) 
        end
        
        # fill with values from FieldArrays for Fields for remaining records
        if length(ridx) > 1
            for (i, r) in enumerate(ridx[2:end])
                far = get_array(fr[r], selectargs_region)
                if isempty(far.dims)
                    fa.values[i+1] = far.values
                else
                    fa.values[fill(Colon(), length(far.dims))..., i+1] .= far.values
                end
            end
        end

        return fa
    end

end

