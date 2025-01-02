import Infiltrator
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
    field_single_element(fr::FieldRecord)::Bool
    field_single_element(f::Field)::Bool
    field_single_element(::Type{FieldRecord})::Bool
    field_single_element(::Type{Field})::Bool

Test whether FieldRecord contains Fields with single elements stored as a Vector instead of a  Vector of records.

- `field_single_element == false`: Fields contain array `values`, these are stored in FieldRecord `records` as a Vector of arrays.
- `field_single_element == true` Fields contain a single value, stored in FieldRecord `records` as Vector of `eltype(Field.values)`. 

NB: this works on Types, and will return false for a field with Space == CellSpace with ncells=1, even though this actually contains a single
value. TODO might be clearer to directly use PB.internal_size(Space, mesh) == (1,) which would also handle the CellSpace with ncells=1 case,
but this wouldn't work in the type domain (needs an instance of mesh::Mesh)
"""
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
        fr.dataset,
        deepcopy(fr.records),
        fr.data_dims,
        fr.mesh,
        deepcopy(fr.attributes),
    )
end

"""
    PB.get_data(fr::FieldRecord; records=nothing)

Get data records in raw format.

`records` may be `nothing` to get all records, 
an `Int` to select a single record, or a range to select multiple records.
"""    
function PB.get_data(fr::FieldRecord; records=nothing)
    
    if isnothing(records)
        data_output = fr.records
    else        
        # bodge - fix scalar data
        # if isa(records, Integer) && !isa(data_output, AbstractVector)
        #     data_output =[data_output]
        # 
        if isa(records, Integer) && field_single_element(fr)
            # represent a scalar as a length 1 Vector
            # (a 0D Array might be more logical, but for backwards compatibility keep as a Vector)
            data_output =[fr.records[records]]
        else
            data_output = fr.records[records]
        end
    end

    return data_output
end  

"""    
    get_array(fr::FieldRecord [, allselectargs::NamedTuple] ; kwargs...) -> fa::FieldArray
    [deprecated] get_array(fr::FieldRecord [; kwargs...] [; allselectargs...]) -> fa::FieldArray

Return a [`FieldArray`](@ref) containing an Array of values and
any attached coordinates, for records and spatial region defined by `allselectargs`.

# Selecting records and regions
`allselectargs` is a `NamedTuple` of form:

    (<dimcoordname> = <filter>, <dimcoordname> = <filter>,  ... [,expand_cartesian=false] [, squeeze_all_single_dims=true])

where `<dimcoordname>` is of form:
- `<dimname>_isel` to select by array indices: `<filter>` may then be a single `Int` to select a single index, or a range `first:last`
  to select a range of indices.
- `<coordname>` to select by coordinate values using the coordinates attached to each dimension: `<filter>` may then be a single number
  to select a single index corresponding to the nearest value of the corresponding coordinate, or `(first::Float64, last::Float64)` 
  (a Tuple) to select a range starting at the index with the nearest value of `fr.coords_record` before `first` and ending at 
  the nearest index after `last`.

Available dimensions and coordinates `<dimcoordname>` depend on the FieldRecord dimensions (as returned by `get_dimensions`, which will be a subset
of grid spatial dimensions and Domain data dimensions) and corresponding attached coordinates (as returned by `get_coordinates`).

Some synonyms are defined for commonly used `<dimnamecoordname>`:

|synonyms     | dimcoordname            | comment                                           |
|:------------| :---------------------- |:--------------------------------------------------|
| records     | <recorddim>_isel        | <recorddim> is usually tmodel                     |
| cells, cell | cells_isel              |                                                   |
| column=<n>  | cells_isel = first:last | select range of cells corresponding to column n   |  


NB: Dimensions corresponding to a selection for a single index or coordinate value are always squeezed out from the returned [`FieldArray`](@ref).
Optional argument `squeeze_all_single_dims` (default `true`) controls whether *all* output dimensions that contain a single index are
squeezed out (including eg a selection for a range that results in a dimension with one index, or where the input `FieldRecord` contains a dimension 
with a single index).

Optional argument `expand_cartesian` (default `false`) is only needed for spatially resolved output using a `CartesianLinearGrid`, 
set to `true` to expand compressed internal data (with spatial dimension `cells`) to a cartesian array  (with spatial dimensions eg `lon`, `lat`, `zt`)

Selection arguments used are returned as strings in `fa.attributes` `filter_records` and `filter_region` for use in plot labelling etc.

# Keyword arguments
- `coords=nothing`: replace the attached coordinates from the input `fr::FieldRecord` for one or more dimensions.
  Format is a Vector of Pairs of `"<dim_name>"=>("<var_name1>", "<var_name2>", ...)`, 
  eg to replace an nD column atmosphere model default pressure coordinate with a z coordinate:

      coords=["cells"=>("atm.zmid", "atm.zlower", "atm.zupper")]
- `lookup_coordinates=true`: `true` to include coordinates, `false` to omit coordinates (both as selection options and in output `FieldArray`).
- `add_attributes=true`: `true` to transfer attributes from input `fr::FieldRecord` to output `FieldArray`, `false` to omit.
- `update_name=true`: `true` to update output `FieldArray` name to add Domain name prefix and suffix generated from `allselectargs` (NB: requires `add_attributes=true`), 
  `false` to use name from input FieldRecord.
- `omit_recorddim_if_constant=false`: Specify whether to include multiple (identical) records and record dimension for constant variables
   (with attribute `is_constant = true`). PALEO currently always stores these records in the input `fr::FieldRecord`; 
   set `false` include them in `FieldArray` output, `true` to omit duplicate records and record dimension from output.
  

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
- time-varying coordinates (eg a z coordinate in an atmosphere climate model) are not fully handled. If a single record is selected, the correct
  values for the coordinate at that time will be returned, but if multiple records are selected, the coordinate will be incorrectly fixed to the
  values at the time corresponding to the first record.
"""
function get_array(
    @nospecialize(fr::FieldRecord); 
    coords=nothing,
    expand_cartesian::Bool=false, # can be overridden in allselectargs
    squeeze_all_single_dims::Bool=true, # can be overridden in allselectargs
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
        coords, expand_cartesian, squeeze_all_single_dims, lookup_coords, add_attributes, update_name, omit_recorddim_if_constant, verbose,
    )
end


function get_array(
    @nospecialize(fr::FieldRecord), @nospecialize(allselectargs::NamedTuple); # allselectargs::NamedTuple=NamedTuple() creates a method ambiguity with deprecated form above
    coords=nothing,
    expand_cartesian::Bool=false, # can be overridden in allselectargs
    squeeze_all_single_dims::Bool=true, # can be overridden in allselectargs
    lookup_coords::Bool=true,
    add_attributes::Bool=true,
    update_name=true,
    omit_recorddim_if_constant::Bool=false, 
    verbose::Bool=false,
)
    frname = default_varnamefull(fr.attributes; include_selectargs=true)
    fa = nothing
    # @Infiltrator.infiltrate
    try
        verbose && println("get_array (begin): $frname, allselectargs $allselectargs lookup_coords $lookup_coords")
    
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
        selectargs_records = OrderedCollections.OrderedDict()  # keep track of selection used, to provide as attributes in FieldArray
        is_constant = omit_recorddim_if_constant ? variable_is_constant(fr) : false
        if !is_constant
            # read record coordinates and apply selection
            dims_coords[recorddimidx] = dims[recorddimidx] => lookup_coords ? _read_coordinates(
                fr, dims[recorddimidx], nothing, expand_cartesian; substitute_coords=coords
            ) : FieldArray[]
            
            _filter_dims_coords(
                select_indices, dims_coords, recorddimidx, 
                allselectargs_sort, selectargs_used,  selectargs_records,
                fr,
            )
        else
            # use first record and squeeze out record dimension
            select_indices[recorddimidx] = 1 # use first records (all records will be identical)
            dims_coords[recorddimidx] = nothing  # dimension will be squeezed out
            selectargs_records["is_constant"] = "true"
        end

        # get record indices to use
        ridx_to_use = select_indices[recorddimidx]
        have_recorddim = !isnothing(dims_coords[recorddimidx])

        # Non-record dimensions

        # read non-record coordinates, from first record selected
        for i in 1:(length(dims)-1)
            dims_coords[i] = dims[i] => lookup_coords ? _read_coordinates(
                fr, dims[i], first(ridx_to_use), expand_cartesian; substitute_coords=coords
            ) : FieldArray[]
        end

        selectargs_region = OrderedCollections.OrderedDict() # keep track of selection used, to provide as attributes in FieldArray

        _filter_dims_coords(
            select_indices, dims_coords, 1:length(dims)-1, 
            allselectargs_sort, selectargs_used,  selectargs_region,
            fr,
        )    

        unused_selectargs = [a for (a, u) in zip(allselectargs_sort, selectargs_used) if !u]
        isempty(unused_selectargs) ||
            error(
                "allselectargs contains select filter(s) ", unused_selectargs, " that do not match any dimensions or coordinates !\n",
                "allselectargs_sort: ", allselectargs_sort, "\n",
                "selectargs_used: ", selectargs_used
            )

        #############################################
        # squeeze out dimensions and coordinates
        #############################################

        # selections that produce a dimension with a single index are already squeezed out,
        # but not dimensions that started with a single index and haven't had a selection applied
        if squeeze_all_single_dims
            for i in eachindex(dims_coords, select_indices)
                if !isnothing(dims_coords[i]) && dims_coords[i][1].size == 1
                    verbose && println("get_array: $frname squeezing out dimension $(dims_coords[i][1].name)")
                    @assert !isa(select_indices[i], Number)
                    @assert length(select_indices[i]) == 1
                    dims_coords[i] = nothing
                    select_indices[i] = first(select_indices[i])
                end
            end
        end

        dims_coords_sq = Pair{PB.NamedDimension, Vector{FieldArray}}[dc for dc in dims_coords if !isnothing(dc)]
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
            @assert length(dims_coords) <= 2
            if have_recorddim                
                if length(dims_coords) == 1 || isnothing(dims_coords[1])
                    # no spatial dimension
                    avalues = fr.records[ridx_to_use]
                else
                    # eg a CellSpace variable in a Domain with no grid still has a spatial dimension size 1
                    avalues = reshape(fr.records[ridx_to_use], 1, :)
                end
            else
                if length(dims_coords) == 1 || isnothing(dims_coords[1])
                    # no spatial dimension
                    # represent a scalar as a 0D Array
                    avalues = Array{eltype(fr.records), 0}(undef)
                    avalues[] = fr.records[ridx_to_use]
                else
                    # eg a CellSpace variable in a Domain with no grid still has a spatial dimension size 1
                    # represent as a 1D Array 
                    avalues = [fr.records[ridx_to_use]]
                end
            end
        else        
            if expand_cartesian && PB.has_internal_cartesian(fr.mesh, space(fr)) # !isempty(dims_spatial)
                expand_fn = x -> PB.Grids.internal_to_cartesian(fr.mesh, x)
                aeltype = Union{Missing, eltype(first(fr.records))}
            else
                expand_fn = identity
                aeltype = eltype(first(fr.records))
            end
            avalues = Array{aeltype, length(dims_sq)}(undef, [nd.size for nd in dims_sq]...)
            if have_recorddim
                _fill_array_from_records(avalues, Tuple(nonrecordindicies_sq), fr.records, expand_fn, ridx_to_use, Tuple(nonrecordindicies))
            else
                if isempty(nonrecordindicies_sq)
                    avalues[] .= @view expand_fn(fr.records[ridx_to_use])[nonrecordindicies...]
                else
                    avalues[nonrecordindicies_sq...] .= @view expand_fn(fr.records[ridx_to_use])[nonrecordindicies...]
                end
            end
        end
        
        if add_attributes
            # add attributes for selection used
            attributes = copy(fr.attributes)
            if !isempty(selectargs_records)
                attributes[:filter_records] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_records)
            end
            if !isempty(selectargs_region)
                attributes[:filter_region] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_region)
            end
            if update_name # Generate name from attributes, including selection suffixes
                name = default_varnamefull(attributes; include_selectargs=true)
            else
                name = fr.name
            end
        else
            attributes = nothing
            name = fr.name
        end

       

        verbose && println("get_array (end): $frname -> $name, allselectargs $allselectargs")

        fa = FieldArray(
            name,
            avalues,
            dims_coords_sq,
            attributes,
        )
    catch
        @error "get_array exception: $frname, allselectargs $allselectargs"
        rethrow()
    end

    return fa
end

# function barrier optimisation
function _fill_array_from_records(avalues, nonrecordindicies_sq, records, expand_fn, ridx_to_use, nonrecordindicies)
    for (riselect, ri) in enumerate(ridx_to_use)
        if isempty(nonrecordindicies_sq)
            avalues[riselect] = expand_fn(records[ri])[nonrecordindicies...]
        else
            avalues[nonrecordindicies_sq..., riselect] .= @view expand_fn(records[ri])[nonrecordindicies...]
        end
    end
    return nothing
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

    coords = FieldArray[]
    for cn in coord_names
        cfr = PB.get_field(fr.dataset, cn)
        if isnothing(ridx_to_use)
            coord = get_array(cfr; lookup_coords=false, update_name=false)
        else
            coord = get_array(cfr, (records=ridx_to_use, ); lookup_coords=false, update_name=false)
        end
        is_coordinate(coord, dim) ||
            error("dimension $dim coord name $cn read invalid coordinate $coord")
        push!(coords, coord)
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


# TODO time-independent variables are indicated by setting :is_constant attribute,
# but this is only recently added and FieldRecord still stores a record for every timestep
# This uses :is_constant to guess if a variable is constant, and then checks the values really are constant
function variable_is_constant(fr::FieldRecord)

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

    a = get_array(
        fr::FieldRecord, (expand_cartesian, squeeze_all_single_dims=false);
        lookup_coordinates=false, add_attributes=false, omit_recorddim_if_constant=true,
    )
    avalues = a.values
    avalues_dimnames = [nd.name for (nd, _) in a.dims_coords]
    attributes = fr.attributes
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

    if field_single_element(FieldData, length(data_dims), Space, typeof(dataset.grid))
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
