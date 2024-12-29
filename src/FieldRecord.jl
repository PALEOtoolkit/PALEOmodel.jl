"""
    FieldRecord{FieldData <: AbstractData, Space <: AbstractSpace, V, N, Mesh, R}
    FieldRecord(dataset, f::PB.Field, attributes; [sizehint=nothing])

A series of `records::R` each containing the `values` from a `PALEOboxes.Field{FieldData, Space, N, V, Mesh}`.

# Implementation
Fields with array `values` are stored in `records` as a Vector of arrays.
Fields with single `values` (`field_single_element` true) are stored as a Vector of `eltype(Field.values)`. 
"""
struct FieldRecord{FieldData <: PB.AbstractData, Space <: PB.AbstractSpace, V, N, Mesh, R}
    dataset 
    records::Vector{R}
    data_dims::NTuple{N, PB.NamedDimension}
    mesh::Mesh
    attributes::Dict{Symbol, Any}
end

# create empty FieldRecord
function FieldRecord(
    dataset, f::PB.Field{FieldData, Space, V, N, Mesh}, attributes;
    sizehint::Union{Nothing, Int}=nothing
) where {FieldData, Space, V, N, Mesh}
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
    return FieldRecord{FieldData, Space, V, N, Mesh, eltype(records)}(
        dataset, 
        records, 
        f.data_dims, 
        f.mesh, 
        attributes,
    )
end


# create a new FieldRecord, containing supplied `existing_values::Vector` data arrays
function FieldRecord(
    dataset,
    existing_values::Vector, 
    FieldData::Type, 
    data_dims::NTuple{N, PB.NamedDimension},
    Space::Type{<:PB.AbstractSpace}, 
    mesh::Mesh,
    attributes;
) where {N, Mesh}
    # check_values(
    #    existing_values, FieldData, data_dims, data_type, Space, spatial_size(space, mesh), 
    # )
    if field_single_element(FieldData, N, Space, Mesh)
        # assume existing_values is a Vector, with each element to be stored in Field values::V as a length 1 Vector
        V = Vector{eltype(existing_values)}
    else
        # assume existing_values is a Vector of Field values::V
        V = eltype(existing_values)
    end

    return FieldRecord{FieldData, Space, V, N, typeof(mesh), eltype(existing_values)}(
        dataset, existing_values, data_dims, mesh, attributes,
    )
end

space(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}) where {FieldData, Space, V, N, Mesh, R} = Space

function PB.get_dimensions(fr::FieldRecord; expand_cartesian=false)
    dims_spatial = PB.get_dimensions(fr.mesh, space(fr); expand_cartesian)
    dims = [dims_spatial..., fr.data_dims..., PB.NamedDimension(fr.dataset.record_dim.name, length(fr))]
    return dims
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
    print(io, 
        "FieldRecord(eltype=", eltype(fr),", length=", length(fr), 
        ", attributes=", fr.attributes, 
        ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", fr::FieldRecord)
    println(io, "FieldRecord(eltype=", eltype(fr),", length=", length(fr), ")")
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

"test whether Field contains single elements
TODO this will currently return false for CellSpace with ncells=1, which could be changed to true ?
TODO would be clearer to directly use PB.internal_size(Space, mesh) == (1,) which would also handle the CellSpace with ncells=1 case,
but this wouldn't work in the type domain (needs an instance of mesh::Mesh)
"
function field_single_element(::Type{FieldData}, N, ::Type{Space}, ::Type{Mesh}) where {FieldData <: PB.AbstractData, Space <: PB.AbstractSpace, Mesh}
    if PB.field_single_element(FieldData, N) && (Space == PB.ScalarSpace || (Space == PB.CellSpace && Mesh == Nothing))
        return true
    else
        return false
    end
end

field_single_element(::Type{PB.Field{FieldData, Space, V, N, Mesh}}) where {FieldData, Space, V, N, Mesh} = field_single_element(FieldData, N, Space, Mesh)
field_single_element(::Type{FR}) where {FR <: FieldRecord} = field_single_element(eltype(FR))
field_single_element(f::T) where {T} = field_single_element(T)


function Base.push!(fr::FieldRecord{FieldData, Space, V, N, Mesh, R}, f::PB.Field{FieldData, Space, V, N, Mesh}) where {FieldData, Space, V, N, Mesh, R}
    if field_single_element(fr)
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

    if field_single_element(fr)
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

Available `selectargs` depend on the FieldRecord dimensions (as returned by `get_dimensions`, which will be a subset
of grid spatial dimensions and Domain data dimensions) and corresponding attached coordinates (as returned by `get_coordinates`.

Optional argument `coords` can be used to replace the attached coordinates for one or more dimensions.
Format is a Vector of Pairs of `"<dim_name>"=>("<var_name1>", "<var_name2>", ...)`, 
eg to replace a 1D column default pressure coordinate with a z coordinate:

    coords=["cells"=>("atm.zmid", "atm.zlower", "atm.zupper")]

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
    coords=nothing,
    expand_cartesian=false, # can be overridden in allselectargs
    squeeze_all_single_dims=true, # can be overridden in allselectargs
    verbose=false,
)
    frname = default_fieldarray_name(fr.attributes)

    verbose && @info "get_array (begin): $frname, allselectargs $allselectargs"
   
    if !isnothing(coords)
        check_coords_argument(coords) ||
            error("argument coords should be a Vector of Pairs of \"dim_name\"=>(\"var_name1\", \"var_name2\", ...), eg: [\"z\"=>(\"atm.zmid\", \"atm.zlower\", \"atm.zupper\"), ...]")
    end

    ##########################################################################
    # preprocess allselectargs to strip non-selection arguments and fix quirks
    ########################################################################
    allselectargs_sort = [String(k) => v for (k, v) in pairs(allselectargs)]
    
    # override expand_cartesian
    idx_expand_cartesian = findfirst(x-> x[1] == "expand_cartesian", allselectargs_sort)
    if !isnothing(idx_expand_cartesian)
        _, expand_cartesian = popat!(allselectargs_sort, idx_expand_cartesian)
    end
    # override squeeze_all_single_dims
    idx_squeeze_all_single_dims = findfirst(x-> x[1] == "squeeze_all_single_dims", allselectargs_sort)
    if !isnothing(idx_squeeze_all_single_dims)
        _, squeeze_all_single_dims = popat!(allselectargs_sort, idx_squeeze_all_single_dims)
    end

    # quirk: column must precede cell selection, so move to front if present
    idx_column = findfirst(x-> x[1] == "column", allselectargs_sort)
    if !isnothing(idx_column)
        column_select = popat!(allselectargs_sort, idx_column)
        pushfirst!(allselectargs_sort, column_select)
    end

    selectargs_used = fill(false, length(allselectargs_sort))

    ################################################################
    # read dimensions 
    # (TODO reproduce code from get_dimensions so we can also get dims_spatial)
    ##########################################################
    # order is spatial (from grid), data_dims, then record dimension
    dims_spatial = PB.get_dimensions(fr.mesh, space(fr); expand_cartesian)
    dims = [dims_spatial..., fr.data_dims..., PB.NamedDimension(fr.dataset.record_dim.name, length(fr))]
    recorddimidx = length(dims)

    ##################################################################################
    # Read coordinates and apply selection: record dimension first, followed by non-record dimensions
    # so we can handle the case where a single record is selected and coordinates are not constant.
    # TODO this doesn't handle cases where multiple records are used with non-constant coordinates
    ##########################################################################################

    # vector of filtered dim => coords corresponding to select_indices, nothing if dimension squeezed out
    dims_coords = Vector{Any}(undef, length(dims))
    # indices in full array to use. Start with everything, then apply selections from 'allselectargs'
    select_indices = Any[1:(nd.size) for nd in dims]

    # Record dimension

    # read record coordinates
    dims_coords[recorddimidx] = dims[recorddimidx] => _read_coordinates(
        fr, dims[recorddimidx], nothing, expand_cartesian; substitute_coords=coords
    )
    
    # keep track of selection used, to provide as attributes in FieldArray
    selectargs_records = OrderedCollections.OrderedDict()
    
    _filter_dims_coords(
        select_indices, dims_coords, recorddimidx, 
        allselectargs_sort, selectargs_used,  selectargs_records,
        fr,
    )

    # get record indices to use
    ridx_to_use = select_indices[recorddimidx]
    have_recorddim = !isnothing(dims_coords[recorddimidx])

    # Non-record dimensions

    # read non-record coordinates, from first record selected
    for i in 1:(length(dims)-1)
        dims_coords[i] = dims[i] => _read_coordinates(fr, dims[i], first(ridx_to_use), expand_cartesian; substitute_coords=coords)
    end

    selectargs_region = OrderedCollections.OrderedDict()

    _filter_dims_coords(
        select_indices, dims_coords, 1:length(dims)-1, 
        allselectargs_sort, selectargs_used,  selectargs_region,
        fr,
    )    

    unused_selectargs = [a for (a, u) in zip(allselectargs_sort, selectargs_used) if !u]
    isempty(unused_selectargs) ||
        error("allselectargs contains select filter(s) $unused_selectargs that do not match any dimensions or coordinates !")

    #############################################
    # squeeze out dimensions and coordinates
    #############################################

    # selections that produce a dimension with a single index are already squeezed out,
    # but not dimensions that started with a single index and haven't had a selection applied
    if squeeze_all_single_dims
        for i in eachindex(dims_coords, select_indices)
            if !isnothing(dims_coords[i]) && dims_coords[i][1].size == 1
                @info "get_array: $frname squeezing out dimension $(dims_coords[i][1].name)"
                @assert !isa(select_indices[i], Number)
                @assert length(select_indices[i]) == 1
                dims_coords[i] = nothing
                select_indices[i] = first(select_indices[i])
            end
        end
    end

    dims_coords_sq = Pair{PB.NamedDimension, Vector{PALEOmodel.FixedCoord}}[dc for dc in dims_coords if !isnothing(dc)]
    dims_sq = [d for (d, c) in dims_coords_sq]

    # get non-record dimensions indices to use
    nonrecordindicies = select_indices[1:end-1]
    # squeeze out single dimensions by filtering out scalars in nonrecordindicies
    # NB: lhs (assignment to output array) is contiguous !
    nonrecordindicies_sq = [1:length(nri) for nri in nonrecordindicies if nri isa AbstractVector]
    # consistency check between dimensions and indices of output array
    @assert length(dims_sq) == length(nonrecordindicies_sq) + have_recorddim
    for (d_sq, nri_sq) in zip(dims_sq, nonrecordindicies_sq) # will stop after shortest iter and so skip recorddim if present 
        @assert d_sq.size == length(nri_sq)
    end

    ###############################################
    # create output FieldArray
    ###############################################

    # create values array
    if field_single_element(fr)
        if have_recorddim
            avalues = fr.records[ridx_to_use]
        else
            # represent a scalar as a 0D Array
            avalues = Array{eltype(fr.records), 0}(undef)
            avalues[] = fr.records[ridx_to_use]
        end
    else        
        if expand_cartesian && !isempty(dims_spatial)
            expand_fn = x -> PB.Grids.internal_to_cartesian(fr.mesh, x)
            aeltype = Union{Missing, eltype(first(fr.records))}
        else
            expand_fn = identity
            aeltype = eltype(first(fr.records))
        end
        avalues = Array{aeltype, length(dims_sq)}(undef, [nd.size for nd in dims_sq]...)
        if have_recorddim            
            for (riselect, ri) in enumerate(ridx_to_use)
                if isempty(nonrecordindicies_sq)
                    avalues[riselect] = expand_fn(fr.records[ri])[nonrecordindicies...]
                else
                    avalues[nonrecordindicies_sq..., riselect] .= expand_fn(fr.records[ri])[nonrecordindicies...]
                end
            end
        else
            if isempty(nonrecordindicies_sq)
                avalues[] = expand_fn(fr.records[ridx_to_use])[nonrecordindicies...]
            else
                avalues[nonrecordindicies_sq...] .= expand_fn(fr.records[ridx_to_use])[nonrecordindicies...]
            end
        end
    end
    
    # add attributes for selection used
    attributes = copy(fr.attributes)
    if !isempty(selectargs_records)
        attributes[:filter_records] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_records)
    end
    if !isempty(selectargs_region)
        attributes[:filter_region] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_region)
    end

    # Generate name from attributes, including selection suffixes
    name = default_fieldarray_name(attributes)

    verbose && @info "get_array (end): $frname -> $name, allselectargs $allselectargs"

    return FieldArray(
        name,
        avalues,
        dims_coords_sq,
        attributes,
    )
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

# Read coordinates attached to dimension 'dim'
# For record coordinate, ridx_to_use = nothing
# For other coordinates, ridx_to_use = record index to use for coordinate (ie limitation only one record in use, or coordinate is constant)
function _read_coordinates(
    fr::FieldRecord, dim::PB.NamedDimension, ridx_to_use::Union{Int, Nothing}, expand_cartesian::Bool; 
    substitute_coords=nothing,
)
    coord_names = nothing
    if !isnothing(substitute_coords)
        for (sdim_name, scoord_names) in substitute_coords
            if sdim_name == dim.name
                coord_names = scoord_names
                break
            end
        end
    end
    
    if isnothing(coord_names)
        coord_names = PB.get_coordinates(fr, dim.name; expand_cartesian)
    end

    coords = FixedCoord[]
    for cn in coord_names
        cfr = PB.get_field(fr.dataset, cn)
        coord_values = isnothing(ridx_to_use) ? cfr.records : cfr.records[ridx_to_use]
        push!(coords, FixedCoord(cn, coord_values, cfr.attributes))
    end
    return coords
end

# Apply filters in allselectargs to dimension indices dim_indices_to_filter,
# returning indices to use in select_indices and filtered dimension and coords in dims_coords
function _filter_dims_coords(
    select_indices::Vector, dims_coords::Vector, dims_indices_to_filter, 
    allselectargs_sort::Vector, selectargs_used::Vector,  selectargs_applied::AbstractDict,
    fr::FieldRecord, 
)

    for k in eachindex(allselectargs_sort, selectargs_used)
        select_dimcoordname_orig, select_filter_orig = allselectargs_sort[k]
        select_dimcoordname, select_filter = select_dimcoordname_orig, select_filter_orig

        # name substitutions
        if select_dimcoordname == "records"
            select_dimcoordname = fr.dataset.record_dim.name*"_isel"
        end
        if select_dimcoordname in ("cell", "cells")
            select_dimcoordname = "cells_isel"
        end
        if select_dimcoordname == "cells_isel"
            select_filter = PB.Grids.substitute_cell_names(fr.mesh, select_filter)
        end
        if select_dimcoordname == "column"
            select_filter = PB.Grids.column_indices(fr.mesh, select_filter)
            select_dimcoordname = "cells_isel"
        end

        # try each dimension in turn to see if select_dimcoordname applies to either the dimension or attached coordinates
        for dimidx in dims_indices_to_filter
            if !isnothing(dims_coords[dimidx]) # dimension may have been squeezed out
                dim, coords = dims_coords[dimidx]
                if select_dimcoordname == dim.name*"_isel" || !isnothing(findfirst(c->c.name==select_dimcoordname, coords))
                    !selectargs_used[k] ||
                        error("select  $select_dimcoordname_orig  => $select_filter_orig matches multiple dimensions or coordinates !")
                    sidx, dim_subset, coords_subset, coords_filter_used = dimscoord_subset(dim, coords, select_dimcoordname, select_filter)
                    select_indices[dimidx] = select_indices[dimidx][sidx] # sidx is relative to current select_indices[dimidx]
                    if isnothing(dim_subset)
                        # squeeze out this dimension
                        dims_coords[dimidx] = nothing
                    else
                        dims_coords[dimidx] = dim_subset => coords_subset                    
                    end

                    # keep track of selection used, to provide as attributes in FieldArray
                    select_filter_used = isnothing(coords_filter_used) ? select_filter_orig : coords_filter_used
                    selectargs_applied[select_dimcoordname_orig] = select_filter_used

                    selectargs_used[k] = true
                end
            end
        end
    end

    return nothing
end