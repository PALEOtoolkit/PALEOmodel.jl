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
struct FieldArray
    "variable name"
    name::String
    "n-dimensional Array of values"
    values::AbstractArray # an abstract type to minimise compilation latency
    "Names of dimensions with optional attached coordinates"
    dims_coords::Vector{Pair{PB.NamedDimension, Vector{FieldArray}}}
    "variable attributes"
    attributes::Union{Dict{Symbol, Any}, Nothing}
end

Base.copy(fa::FieldArray) = FieldArray(fa.name, copy(fa.values), copy(fa.dims_coords), isnothing(fa.attributes) ? nothing : copy(fa.attributes))

PB.get_dimensions(fa::FieldArray) = PB.NamedDimension[first(dc) for dc in fa.dims_coords]

function PB.get_dimension(fa::FieldArray, dimname::AbstractString)
    dimidx = findfirst(dc -> dc[1].name == dimname, fa.dims_coords)
    !isnothing(dimidx) || error("FieldArray $(fa.name) has no dimension $dimname")
    return fa.dims_coords[dimidx][1]
end

function PB.get_coordinates(fa::FieldArray, dimname::AbstractString)
    dimidx = findfirst(dc -> dc[1].name == dimname, fa.dims_coords)
    !isnothing(dimidx) || error("FieldArray $(fa.name) has no dimension $dimname")
    return fa.dims_coords[dimidx][2]
end

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
_fieldarray_default_varnamefull(attributes::Nothing) = ""

function _fieldarray_default_varnamefull(attributes::Dict; include_selectargs=false)
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

################################################################
# Select regions 
##############################################################

"""    
    select(fa::FieldArray [, allselectargs::NamedTuple] ; kwargs...) -> fa_out::FieldArray
  
Select a region of a [`FieldArray`](@ref) defined by `allselectargs`.

# Selecting records and regions
`allselectargs` is a `NamedTuple` of form:

    (<dimcoordname> = <filter>, <dimcoordname> = <filter>,  ... [, squeeze_all_single_dims=true])

where `<dimcoordname>` is of form:
- `<dimname>_isel` to select by array indices: `<filter>` may then be a single `Int` to select a single index, or a range `first:last`
  to select a range of indices.
- `<coordname>` to select by coordinate values using the coordinates attached to each dimension: `<filter>` may then be a single number
  to select a single index corresponding to the nearest value of the corresponding coordinate, or `(first::Float64, last::Float64)` 
  (a Tuple) to select a range starting at the index with the nearest value of `fr.coords_record` before `first` and ending at 
  the nearest index after `last`.

Available dimensions and coordinates `<dimcoordname>` depend on the FieldArray dimensions (as returned by `get_dimensions`, which will be a subset
of grid spatial dimensions and Domain data dimensions) and corresponding attached coordinates (as returned by `get_coordinates`).

Some synonyms are defined for commonly used `<dimnamecoordname>`, these may require optional `record_dim_name` and `mesh` to be supplied in `kwargs`:

|synonyms     | dimcoordname            | comment                                                                         |
|:------------| :---------------------- |:--------------------------------------------------------------------------------|
| records     | <`record_dim_name`>_isel  | requires `record_dim_name` to be suppliied, usually tmodel                      |
| cells, cell | cells_isel              | substituting named cells requires `mesh` to be supplied                         | 
| column=<n>  | cells_isel = first:last | requires `mesh` to be supplied, select range of cells corresponding to column n |  


NB: Dimensions corresponding to a selection for a single index or coordinate value are always squeezed out from the returned [`FieldArray`](@ref).
Optional argument `squeeze_all_single_dims` (default `true`) controls whether *all* output dimensions that contain a single index are
squeezed out (including eg a selection for a range that results in a dimension with one index, or where the input `FieldArray` contains a dimension 
with a single index).

Selection arguments used are optionally returned as strings in `fa_out.attributes[:filter_records]`
and `fa_out.attributes[:filter_region]` for use in plot labelling etc.

# Keyword arguments
- `record_dim_name=nothing`: optionally supply name of record dimension as a String, to substitute for `<dimcoordname>` `records`
   and to define dimension to use to populate `filter_records` attribute.
- `mesh=nothing`: optionally supply a grid from `PALEOboxes.Grids`, to use for substituting cell names, looking up column indices.
- `add_attributes=true`: `true` to transfer attributes from input `fr::FieldRecord` to output `FieldArray` 
   adding `:filter_region` and `:filter_records`, `false` to omit.
- `update_name=true`: `true` to update output `FieldArray` name to add Domain name prefix, and a suffix generated from `allselectargs` 
   (NB: requires `add_attributes=true`), `false` to use name from input FieldRecord.
  
# Limitations
- it is only possible to select either a single slice or a contiguous range for each dimension, not a set of slices for a Vector of index
  or coordinate values.
"""
function select(
    @nospecialize(fa::FieldArray), @nospecialize(allselectargs::NamedTuple)=NamedTuple(); # creates a method ambiguity with deprecated form above
    mesh=PB.AbstractMeshOrNothing=nothing, # for substitution of cell names etc
    add_attributes=true,
    record_dim_name::Union{Nothing, AbstractString}=nothing, # for reporting selection filters used
    update_name=false,
    verbose::Bool=false,
)
    faname = fa.name
    fa_out = nothing

    try
        verbose && println("select (begin): $faname, allselectargs $allselectargs")

        ##########################################################################
        # preprocess allselectargs to strip non-selection arguments and fix quirks
        ########################################################################
        allselectargs_sort = [String(k) => v for (k, v) in pairs(allselectargs)]
        
        # get squeeze_all_single_dims from allselectargs_sort
        idx_squeeze_all_single_dims = findfirst(x-> x[1] == "squeeze_all_single_dims", allselectargs_sort)
        if isnothing(idx_squeeze_all_single_dims)
            squeeze_all_single_dims = true
        else
            _, squeeze_all_single_dims = popat!(allselectargs_sort, idx_squeeze_all_single_dims)
        end

        # quirk: column must precede cell selection, so move to front if present
        idx_column = findfirst(x-> x[1] == "column", allselectargs_sort)
        if !isnothing(idx_column)
            column_select = popat!(allselectargs_sort, idx_column)
            pushfirst!(allselectargs_sort, column_select)
        end

        selectargs_used = fill(false, length(allselectargs_sort))

        # vector of filtered dim => coords corresponding to select_indices, nothing if dimension squeezed out
        dims_coords = Any[dc for dc in fa.dims_coords]
        dims = [nd for (nd, c) in fa.dims_coords]
        # indices in full array to use. Start with everything, then apply selections from 'allselectargs'
        select_indices = Any[1:(nd.size) for nd in dims]

        selectargs_records = OrderedCollections.OrderedDict()  # keep track of selection used, to provide as attributes in FieldArray
        selectargs_region = OrderedCollections.OrderedDict() # keep track of selection used, to provide as attributes in FieldArray
        idx_record_dim = findfirst(nd->nd.name == record_dim_name, dims)
        selectargs_applied = [i == idx_record_dim ? selectargs_records : selectargs_region for i in 1:length(dims)]

        _filter_dims_coords!(
            select_indices, dims_coords, selectargs_applied,
            allselectargs_sort, selectargs_used;
            mesh, record_dim_name,
        )    

        unused_selectargs = [a for (a, u) in zip(allselectargs_sort, selectargs_used) if !u]
        isempty(unused_selectargs) ||
            error(
                "allselectargs contains select filter(s) ", unused_selectargs, " that do not match any dimensions or coordinates !\n",
                "allselectargs_sort: ", allselectargs_sort, "\n",
                "selectargs_used: ", selectargs_used
            )

        ##############################################################
        # filter coordinates
        ################################################################
        
        alldimindices = Dict{String, Any}(nd.name => indices for (nd, indices) in zip(dims, select_indices))
        for id in 1:length(dims_coords)
            if !isnothing(dims_coords[id])
                nd, unfiltered_coords = dims_coords[id]
                filtered_coords = FieldArray[_select_dims_indices(fc, alldimindices) for fc in unfiltered_coords]
                dims_coords[id] = nd => filtered_coords
            end
        end

        #############################################
        # squeeze out dimensions and coordinates
        #############################################

        # selections that produce a dimension with a single index are already squeezed out,
        # but not dimensions that started with a single index and haven't had a selection applied
        if squeeze_all_single_dims
            for i in eachindex(dims_coords, select_indices)
                if !isnothing(dims_coords[i]) && dims_coords[i][1].size == 1
                    verbose && println("get_array: $faname squeezing out dimension $(dims_coords[i][1].name)")
                    @assert !isa(select_indices[i], Number)
                    @assert length(select_indices[i]) == 1
                    dims_coords[i] = nothing
                    select_indices[i] = first(select_indices[i])
                end
            end
        end

        dims_coords_sq = Pair{PB.NamedDimension, Vector{FieldArray}}[dc for dc in dims_coords if !isnothing(dc)]
        dims_sq = [d for (d, c) in dims_coords_sq]

        # squeeze out single dimensions by filtering out scalars in select_indices
        # NB: lhs (assignment to output array) is contiguous !
        select_indices_sq = [1:length(si) for si in select_indices if si isa AbstractVector]
        # consistency check between dimensions and indices of output array
        @assert length(dims_sq) == length(select_indices_sq)
        for (d_sq, si_sq) in zip(dims_sq, select_indices_sq) 
            @assert d_sq.size == length(si_sq)
        end

        ###############################################
        # create output FieldArray
        ###############################################

        # create values array
        avalues = Array{eltype(fa.values), length(dims_sq)}(undef, [nd.size for nd in dims_sq]...)
        
        if isempty(select_indices_sq) # 0D array
            avalues[] = fa.values[select_indices...]
        else
            # TODO will need to define similar() for fa.values  to do anything other than @view
            avalues[fill(Colon(), length(dims_sq))...] .= @view fa.values[select_indices...]
        end
        
        if add_attributes
            # add attributes for selection used
            attributes = copy(fa.attributes)
            if !isempty(selectargs_records)
                attributes[:filter_records] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_records)
            end
            if !isempty(selectargs_region)
                attributes[:filter_region] = NamedTuple(Symbol(k)=>v for (k, v) in selectargs_region)
            end
            if update_name # Generate name from attributes, including selection suffixes
                name = _fieldarray_default_varnamefull(attributes; include_selectargs=true)
            else
                name = fa.name
            end
        else
            attributes = nothing
            name = fa.name
        end

        verbose && println("select (end): $faname -> $name, allselectargs $allselectargs")

        fa_out = FieldArray(
            name,
            avalues,
            dims_coords_sq,
            attributes,
        )
    catch
        @error "select exception: $faname, allselectargs $allselectargs"
        rethrow()
    end

    return fa_out
end



# Apply filters in allselectargs, setting selectargs_used true when filter used.
# Update dimension indices to use in select_indices, filtered dimension (but unfiltered coords) in dims_coords, 
# and add corresponding filter to selectargs_applied.
function _filter_dims_coords!(
    select_indices::Vector, dims_coords::Vector, selectargs_applied::Vector{<:AbstractDict},
    allselectargs_sort::Vector, selectargs_used::Vector{Bool};
    mesh, record_dim_name,
)

    for k in eachindex(allselectargs_sort, selectargs_used)
        select_dimcoordname_orig, select_filter_orig = allselectargs_sort[k]
        select_dimcoordname, select_filter = select_dimcoordname_orig, select_filter_orig

        # name substitutions
        if select_dimcoordname == "records" && !isnothing(record_dim_name)
            select_dimcoordname = record_dim_name*"_isel"
        end
        if select_dimcoordname in ("cell", "cells")
            select_dimcoordname = "cells_isel"
        end
        if select_dimcoordname == "cells_isel" && !isnothing(mesh)
            select_filter = PB.Grids.substitute_cell_names(mesh, select_filter)
        end
        if select_dimcoordname == "column" && !isnothing(mesh)
            select_filter = PB.Grids.column_indices(mesh, select_filter)
            select_dimcoordname = "cells_isel"
        end

        # try each dimension in turn to see if select_dimcoordname applies to either the dimension or attached coordinates
        for dimidx in eachindex(dims_coords, select_indices, selectargs_applied)
            if !isnothing(dims_coords[dimidx]) # dimension may have been squeezed out
                dim, coords = dims_coords[dimidx]
                if select_dimcoordname == dim.name*"_isel" || !isnothing(findfirst(c->c.name==select_dimcoordname, coords))
                    !selectargs_used[k] ||
                        error("select  $select_dimcoordname_orig  => $select_filter_orig matches multiple dimensions or coordinates !")
                    sidx, dim_subset, coords_filter_used = _dim_subset(dim, coords, select_dimcoordname, select_filter)
                    select_indices[dimidx] = select_indices[dimidx][sidx] # sidx is relative to current select_indices[dimidx]
                    if isnothing(dim_subset)
                        # squeeze out this dimension
                        dims_coords[dimidx] = nothing
                    else
                        dims_coords[dimidx] = dim_subset => coords # no select on coords !! (this is applied later)
                    end

                    # keep track of selection used, to provide as attributes in FieldArray
                    select_filter_used = isnothing(coords_filter_used) ? select_filter_orig : coords_filter_used
                    selectargs_applied[dimidx][select_dimcoordname_orig] = select_filter_used

                    selectargs_used[k] = true
                end
            end
        end
    end

    return nothing
end
