"""
    OutputWriters

Data structures and methods to hold and manage model output.
"""
module OutputWriters

import PALEOboxes as PB

import PALEOmodel

import OrderedCollections
import DataFrames
import FileIO
import NCDatasets

import Infiltrator # Julia debugger


##################################
# AbstractOutputWriter interface
###################################

"""
    AbstractOutputWriter

Interface implemented by containers for PALEO model output.

Implementations should define methods for:

# Writing output
- [`initialize!`](@ref)
- [`add_record!`](@ref)

# Modifying output
- [`PB.add_field!`](@ref)

# Querying output
- [`PB.get_table`](@ref)
- [`PB.show_variables`](@ref)
- [`PB.has_variable`](@ref)

# Accessing output data
- [`PALEOmodel.get_array`](@ref)
- [`PB.get_field`](@ref)
- [`PB.get_mesh`](@ref)
- [`PB.get_data`](@ref)

"""
PALEOmodel.AbstractOutputWriter

"""
    initialize!(
        output::PALEOmodel.AbstractOutputWriter, model, modeldata, [nrecords] 
        [;record_dim_name=:tmodel] [record_coord_units="yr"]
    )

Initialize from a PALEOboxes::Model, optionally reserving memory for an assumed output dataset of `nrecords`.

The default for `record_dim_name` is `:tmodel`, for a sequence of records following the time evolution
of the model.
"""
function initialize!(
    output::PALEOmodel.AbstractOutputWriter, model::PB.Model, modeldata::PB.ModelData, nrecords
)
end

"""
    add_record!(output::PALEOmodel.AbstractOutputWriter, model, modeldata, rec_coord)

Add an output record for current state of `model` at record coordinate `rec_coord`.
The usual case (set by [`initialize!`](@ref)) is that the record coordinate is model time `tmodel`.
"""
function add_record!(output::PALEOmodel.AbstractOutputWriter, model, modeldata, rec_coord) end

"""
    add_field!(output::PALEOmodel.AbstractOutputWriter, fr::PALEOmodel.FieldRecord) 

Add [`PALEOmodel.FieldRecord`](@ref) `fr` to `output`, with Domain and name defined by `fr.attributes[:var_domain]` and
`fr.attributes[:var_name]`.
"""
function PB.add_field!(output::PALEOmodel.AbstractOutputWriter, fr::PALEOmodel.FieldRecord) end

"""
    get_table(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString) -> Table
    get_table(output::PALEOmodel.AbstractOutputWriter, varnames::Vector{<:AbstractString}) -> Table

Return a `DataFrame` with raw model `output` data for Domain `domainname`, or for Variables `varnames`
"""
function PB.get_table(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString) end

"""
    has_variable(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString)  -> Bool

True if model `output` contains Variable `varname`.
"""
function PB.has_variable(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString) end


"""
    show_variables(output::PALEOmodel.AbstractOutputWriter; kwargs...) -> Table
    show_variables(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString; kwargs...) -> Table

# Keywords:
- `attributes=[:units, :vfunction, :space, :field_data, :description]`: Variable attributes to include
- `filter = attrb->true`: function to filter by Variable attributes.
  Example: `filter=attrb->attrb[:vfunction]!=PB.VF_Undefined` to show state Variables and derivatives.

# Examples:
    julia> vscodedisplay(PB.show_variables(run.output))  # show all output Variables in VS code table viewer
    julia> vscodedisplay(PB.show_variables(run.output, ["atm.pCO2PAL", "fluxOceanBurial.flux_P_total"]))  # show subset of output Variables in VS code table viewer
"""
function PB.show_variables(output::PALEOmodel.AbstractOutputWriter) end

"""
    get_array(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString [, allselectargs::NamedTuple]; kwargs...) -> FieldArray
    [deprecated] get_array(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString; allselectargs...) -> FieldArray

Return a [`PALEOmodel.FieldArray`](@ref) containing data values and any attached coordinates.

Equivalent to `PALEOmodel.get_array(PB.get_field(output, varname), allselectargs; kwargs)`,
see [`PALEOmodel.get_array(fr::PALEOmodel.FieldRecord)`](@ref).
"""
function PALEOmodel.get_array(
    output::PALEOmodel.AbstractOutputWriter, varname::AbstractString, @nospecialize(allselectargs::NamedTuple); # allselectargs::NamedTuple=NamedTuple() creates a method ambiguity with deprecated form above
    kwargs...
)
    fr = PB.get_field(output, varname)   

    return PALEOmodel.get_array(fr, allselectargs; kwargs...)
end

function PALEOmodel.get_array(
    output::PALEOmodel.AbstractOutputWriter, varname::AbstractString;
    kwargs...
) 
    fr = PB.get_field(output, varname)

    return PALEOmodel.get_array(fr; kwargs...)
end


"""
    get_field(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString) -> FieldRecord

Return the [`PALEOmodel.FieldRecord`](@ref) for `varname`.
"""
function PB.get_field(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString) end

"""
    get_data(output::PALEOmodel.AbstractOutputWriter, varname; records=nothing) -> values

Get Variable `varname` raw data array, optionally restricting to `records`
"""
function PB.get_data(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString; records=nothing) end

"""
    get_mesh(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString) -> grid::Union{AbstractMesh, Nothing}

Return `grid` for `output` Domain `domainname`.
"""
function PB.get_mesh(output::PALEOmodel.AbstractOutputWriter, domainname::AbstractString) end


##########################
# OutputMemoryDomain
###########################

"""
    OutputMemoryDomain

In-memory model output, for one model Domain.

Includes an additional record coordinate variable not present in Domain (usually `:tmodel`, when storing output vs time).
"""
mutable struct OutputMemoryDomain
    parentdataset::Union{PALEOmodel.AbstractOutputWriter, Nothing}
    "Domain name"
    name::String
    "Model output for this Domain"
    data::OrderedCollections.OrderedDict{Symbol, PALEOmodel.FieldRecord}
    "record dimension"
    record_dim::PB.NamedDimension
    "record coordinates"
    record_dim_coordinates::Vector{String}
    "Domain data_dims"
    data_dims::Vector{PB.NamedDimension}
    "data_dims coordinates"
    data_dims_coordinates::Dict{String, Vector{String}}
    "Domain Grid (if any)"
    grid::Union{PB.AbstractMesh, Nothing}

    # internal use only: all Variables in sorted order
    _all_vars::Vector{Union{Nothing, PB.VariableDomain}}
end

"""
    OutputMemoryDomain(
        name::AbstractString; 
        [record_dim_name::Symbol=:tmodel], [record_coord_name::Symbol=record_dim_name]
        [grid=nothing]
        [data_dims::Vector{PB.NamedDimension} = Vector{PB.NamedDimension}()]
    )

Create empty OutputMemoryDomain. Add additional Fields with
`add_field!`.
"""
function OutputMemoryDomain(
    name::AbstractString;
    parentdataset=nothing,
    data_dims::Vector{PB.NamedDimension} = Vector{PB.NamedDimension}(),
    data_dims_coordinates::Dict{String, Vector{String}} = Dict{String, Vector{String}}(),
    grid = nothing,
    record_dim_name::Symbol=:tmodel,
    record_coord_name::Symbol=record_dim_name,
    coords_record::Union{Symbol, Nothing}=nothing, # deprecated
)
    if !isnothing(coords_record)
        @warn "coords_record argument is deprecated, use record_dim_name"
        record_dim_name = coords_record
        record_coord_name = coords_record
    end

    return OutputMemoryDomain(
        parentdataset,
        name,
        OrderedCollections.OrderedDict{Symbol, PALEOmodel.FieldRecord}(),        
        PB.NamedDimension(string(record_dim_name), 0),
        [string(record_coord_name)],
        deepcopy(data_dims),
        deepcopy(data_dims_coordinates),
        deepcopy(grid),
        PB.VariableDomain[],      
    )

end

"create from a PALEOboxes::Domain"
function OutputMemoryDomain(
    dom::PB.Domain, modeldata::PB.ModelData, nrecords=nothing; 
    parentdataset=nothing,
    record_dim_name::Symbol=:tmodel,
    record_coord_name::Symbol=record_dim_name,
    record_coord_units::AbstractString="yr",
    coords_record::Union{Symbol, Nothing}=nothing, # deprecated
    coords_record_units::Union{AbstractString, Nothing}=nothing, # deprecated
)
    if !isnothing(coords_record)
        @warn "coords_record is deprecated, use record_dim_name"
        record_dim_name = coords_record
        record_coord_name = coords_record
    end
    if !isnothing(coords_record_units)
        @warn "coords_record_units is deprecated, use record_coord_units"
        record_coord_units = coords_record_units
    end

    odom =  OutputMemoryDomain(
        dom.name;
        parentdataset,
        record_dim_name,
        record_coord_name,
        data_dims = dom.data_dims,
        data_dims_coordinates = dom.data_dims_coordinates,
        grid = dom.grid,
    )

    # add record coodinate variable
    odom.data[record_coord_name] = PALEOmodel.FieldRecord(
        odom,
        Float64[], 
        PB.ScalarData, 
        (),
        PB.ScalarSpace, 
        nothing,
        Dict{Symbol, Any}(
            :var_name => string(record_coord_name),
            :domain_name => dom.name,
            :field_data => PB.ScalarData,
            :space => PB.ScalarSpace,
            :data_dims => (),
            :units => record_coord_units,
        ),
    )

    # create list of variables sorted by host dependent type, then by name
    odom._all_vars = vcat(
        nothing, # coords_record has no actual variable
        sort(
            PB.get_variables(
                dom, v->PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_StateExplicit, PB.VF_Total, PB.VF_Constraint)
            ), 
            by=var->var.name
        ),
        sort(
            PB.get_variables(dom, v->PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_Deriv,)),
            by=var->var.name
        ),
        sort(
            PB.get_variables(dom, v->PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_State, PB.VF_StateTotal)),
            by=var->var.name
        ),
        sort(
            PB.get_variables(
                dom, v-> !(PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_StateExplicit, PB.VF_Total, PB.VF_StateTotal, PB.VF_Constraint, PB.VF_Deriv, PB.VF_State))),
            by=var->var.name
        ),
    )

    # add variables
    for var in odom._all_vars
        if !isnothing(var)
            # records storage type is that of FieldRecord.records
            field = PB.get_field(var, modeldata)
            attrbs = deepcopy(var.attributes)
            attrbs[:var_name] = var.name
            attrbs[:domain_name] = dom.name
        
            odom.data[Symbol(var.name)] = PALEOmodel.FieldRecord(odom, field, attrbs)
        end
    end

    if !isnothing(nrecords)
        for (_, fr) in odom.data
            sizehint!(fr.records, nrecords)
        end
    end

    return odom
end

# create from a record coordinate FieldRecord
function OutputMemoryDomain(
    name::AbstractString, record_coordinate::PALEOmodel.FieldRecord;
    parentdataset=nothing,
    data_dims::Vector{PB.NamedDimension} = Vector{PB.NamedDimension}(),
    data_dims_coordinates::Dict{String, Vector{String}} = Dict{String, Vector{String}}(),
    grid = nothing,
)
    record_dim_name = Symbol(record_coordinate.attributes[:var_name])
    record_coord_name = record_dim_name
    odom = OutputMemoryDomain(
        name; 
        record_dim_name,
        record_coord_name,
        parentdataset, data_dims, data_dims_coordinates, grid
    )

    odom.record_dim = PB.NamedDimension(string(record_dim_name), length(record_coordinate))
    PB.add_field!(odom, record_coordinate)
    
    return odom
end

"create from a DataFrames DataFrame containing scalar data"
function OutputMemoryDomain(
    name::AbstractString, data::DataFrames.DataFrame;
    record_dim_name::Symbol=:tmodel,
    record_coord_name::Symbol=record_dim_name, 
    record_coord_units::AbstractString="yr",
    metadata::Union{Dict{String, Dict{Symbol, Any}}, Nothing}=nothing,
    coords_record::Union{Symbol, Nothing}=nothing, # deprecated
    coords_record_units::Union{AbstractString, Nothing}=nothing, # deprecated
)

    if !isnothing(coords_record)
        @warn "coords_record is deprecated, use record_dim_name"
        record_dim_name = coords_record
        record_coord_name = coords_record
    end
    if !isnothing(coords_record_units)
        @warn "coords_record_units is deprecated, use record_coord_units"
        record_coord_units = coords_record_units
    end

    String(record_coord_name) in DataFrames.names(data) ||
        @warn "record_coord_name $record_coord_name is not present in supplied data DataFrame"

    if isnothing(metadata)
        metadata = Dict(String(record_coord_name)=>Dict{Symbol, Any}(:units=>record_coord_units))
    end

    odom =  OutputMemoryDomain(name; record_dim_name, record_coord_name)
    odom.record_dim = PB.NamedDimension(string(record_dim_name), DataFrames.nrow(data))

    # create minimal metadata for scalar Variables
    for vname in DataFrames.names(data)
        vmeta = get!(metadata, vname, Dict{Symbol, Any}())
        vmeta[:var_name] = vname
        vmeta[:domain_name] = name
        vmeta[:field_data] = PB.ScalarData
        vmeta[:space] = PB.ScalarSpace
        vmeta[:data_dims] = ()

        odom.data[Symbol(vname)] = PALEOmodel.FieldRecord(
            odom,
            data[!, Symbol(vname)], 
            PB.ScalarData, 
            (),
            PB.ScalarSpace, 
            nothing,
            vmeta;
            # coords_record
        ) 
    end

    return odom
end

Base.length(output::OutputMemoryDomain) = output.record_dim.size

function add_record!(odom::OutputMemoryDomain, modeldata, rec_coord)
        
    odom.record_dim = PB.NamedDimension(odom.record_dim.name, odom.record_dim.size + 1)
  
    rec_dim_fr = odom.data[Symbol(odom.record_dim.name)]
    push!(rec_dim_fr, rec_coord)

    for (fr, var) in PB.IteratorUtils.zipstrict(values(odom.data), odom._all_vars)
        if !isnothing(var)
            field = PB.get_field(var, modeldata)
            push!(fr, field)
        end    
    end

    return nothing
end

function Base.append!(output1::OutputMemoryDomain, output2::OutputMemoryDomain)
    error("Not implemented")
end

function PB.add_field!(odom::OutputMemoryDomain, fr::PALEOmodel.FieldRecord)
    
    length(fr) == length(odom) ||
        throw(ArgumentError("FieldRecord length $(length(fr)) != OutputMemoryDomain length $(length(odom))"))

    !isempty(fr.name) ||
        throw(ArgumentError("FieldRecord has empty name = \"\""))
    !(Symbol(fr.name) in keys(odom.data)) ||
        throw(ArgumentError("Variable $(fr.name) already exists"))

    odom.data[Symbol(fr.name)] = PALEOmodel.FieldRecord(fr, odom)

    return nothing
end

function PB.has_variable(odom::OutputMemoryDomain, varname::AbstractString)
    return haskey(odom.data, Symbol(varname))
end

function PB.get_field(odom::OutputMemoryDomain, varname_or_varnamefull::AbstractString)

    is_varnamefull = contains(varname_or_varnamefull, ".")

    if is_varnamefull
        fr = PB.get_field(odom.parentdataset, varname_or_varnamefull)
    else
        fr = get(odom.data, Symbol(varname_or_varnamefull), nothing)
        !isnothing(fr) || 
            error("Variable $varname_or_varnamefull not found in output (no key '$varname_or_varnamefull' in OutputMemoryDomain output.domains[\"$(odom.name)\"].data)")
    end

    return fr
end

function PB.get_data(output::OutputMemoryDomain, varname::AbstractString; records=nothing)

    fr = PB.get_field(output, varname)

    return PB.get_data(fr; records)
end    

function PB.show_variables(
    odom::OutputMemoryDomain; 
    attributes=[:units, :vfunction, :space, :field_data, :description],
    filter = attrb->true, 
)
    shownames = []
    for (vnamesym, vfr) in odom.data
        if filter(vfr.attributes)
            push!(shownames, string(vnamesym))
        end
    end
    sort!(shownames)
    
    df = DataFrames.DataFrame()
    df.name = shownames
    for att in attributes
        DataFrames.insertcols!(df, att=>[get(odom.data[Symbol(vn)].attributes, att, missing) for vn in shownames])
    end   

    return df
end


function PB.get_table(
    odom::OutputMemoryDomain, varnames::Vector{<:AbstractString} = AbstractString[],
)
    df = DataFrames.DataFrame(
        [k => v.records for (k, v) in odom.data if (isempty(varnames) || string(k) in varnames)]
    )

    return df    
end

function PB.get_dimensions(odom::OutputMemoryDomain)
    spatial_dims = isnothing(odom.grid) ? PB.NamedDimension[] : PB.get_dimensions(odom.grid)
    return vcat(spatial_dims, odom.data_dims, odom.record_dim)
end

function PB.get_dimension(odom::OutputMemoryDomain, dimname::AbstractString)
    # special case records to always return record_dim
    if dimname == "records"
        dimname = odom.record_dim.name
    end
    #
    ad = PB.get_dimensions(odom)
    idx = findfirst(d -> d.name==dimname, ad)
    !isnothing(idx) ||
            error("OutputMemoryDomain $(odom.name) has no dimension='$dimname' (available dimensions: $ad")
    return ad[idx]
end

function PB.get_coordinates(odom::OutputMemoryDomain, dimname::AbstractString)
    coord_names = String[]
    if !isnothing(odom.grid)
        coord_names = PB.get_coordinates(odom.grid, dimname)
    end
    if !isnothing(findfirst(dd -> dd.name == dimname, odom.data_dims))
        coord_names = get(odom.data_dims_coordinates, dimname, String[])
    end
    if dimname in ("records", odom.record_dim.name)
        coord_names = odom.record_dim_coordinates
    end

    return coord_names
end

########################################
# OutputMemory
##########################################

const UserDataTypes = Union{Float64, Int64, String, Vector{Float64}, Vector{Int64}, Vector{String}}

"""
    OutputMemory(; user_data=Dict{String, UserDataTypes}())

In-memory container for model output, organized by model Domains.

Implements the [`PALEOmodel.AbstractOutputWriter`](@ref) interface, with additional methods
[`save_netcdf`](@ref) and [`load_netcdf!`](@ref) to save and load data.

# Implementation
- Field `domains::Dict{String, OutputMemoryDomain}` contains per-Domain model output.
- Field `user_data::Dict{String, UserDataTypes}` contains optional user data
  NB:
  - available types are restricted to those that are compatible with NetCDF attribute types,
    ie Float64, Int64, String, Vector{Float64}, Vector{Int64}, Vector{String}
  - Vectors with a single element are read back from netcdf as scalars,
    see https://alexander-barth.github.io/NCDatasets.jl/dev/issues/#Corner-cases
"""
struct OutputMemory <: PALEOmodel.AbstractOutputWriter
    domains::Dict{String, OutputMemoryDomain}
    user_data::Dict{String, UserDataTypes}
end


const default_user_data=Dict{String, UserDataTypes}(
    "title"=>"PALEO (exo)Earth system model output",
    "source"=>"PALEOmodel https://github.com/PALEOtoolkit/PALEOmodel.jl",
)

function OutputMemory(; user_data=default_user_data)
    return OutputMemory(Dict{String, OutputMemoryDomain}(), user_data)
end

"create from collection of OutputMemoryDomain"
function OutputMemory(output_memory_domains::Union{Vector, Tuple}; user_data=default_user_data)
    om = OutputMemory(Dict(om.name => om for om in output_memory_domains), user_data)
    return om
end

function Base.length(output::OutputMemory)
    lengths = unique([length(omd) for (k, omd) in output.domains])
    length(lengths) == 1 ||
        error("output $output has Domains of different lengths")

    return lengths[]
end


function PB.get_table(output::OutputMemory, domainname::AbstractString, varnames::Vector{<:AbstractString} = AbstractString[])
    haskey(output.domains, domainname) ||
        throw(ArgumentError("no Domain $domainname"))

    return PB.get_table(output.domains[domainname], varnames)
end

function PB.get_table(output::OutputMemory, varnamefulls::Vector{<:AbstractString})
    df = DataFrames.DataFrame()

    for varnamefull in varnamefulls
        vdom, varname = domain_variable_name(varnamefull)
        if haskey(output.domains, vdom)
            if PB.has_variable(output.domains[vdom], varname)
                vardata = PB.get_data(output.domains[vdom], varname)
                df = DataFrames.insertcols!(df, varnamefull=>vardata)
            else
                @warn "no Variable found for $varnamefull"
            end
        else
            @warn "no Domain found for $varnamefull"
        end
    end

    return df
end

function PB.get_mesh(output::OutputMemory, domainname::AbstractString)
    haskey(output.domains, domainname) ||
        throw(ArgumentError("no Domain $domainname"))

    return output.domains[domainname].grid
end

function PB.show_variables(
    output::OutputMemory, domainname::AbstractString; kwargs...
)
    haskey(output.domains, domainname) ||
        throw(ArgumentError("no Domain $domainname"))

    return PB.show_variables(output.domains[domainname]; kwargs...)
end

function PB.show_variables(output::OutputMemory; kwargs...)
    df = DataFrames.DataFrame()
    for (domname, odom) in output.domains
        dfdom = PB.show_variables(odom; kwargs...)
        # prepend domain name
        DataFrames.insertcols!(dfdom, 1, :domain=>fill(domname, size(dfdom,1)))
        # append to df
        df = vcat(df, dfdom)
    end
    DataFrames.sort!(df, [:domain, :name])
    return df
end


"""
    save_jld2(output::OutputMemory, filename)

Removed in PALEOmodel v0.16 - use [`save_netcdf`](@ref)

Save to `filename` in JLD2 format (NB: filename must either have no extension or have extension `.jld2`)
"""
function save_jld2(output::OutputMemory, filename)

    @error """save_jld2 has been removed in PALEOmodel v0.16
              Please use save_netcdf instead"""

    return nothing
end

"""
    load_jld2!(output::OutputMemory, filename)

Removed in PALEOmodel v0.16 - use [`save_netcdf`](@ref), [`load_netcdf!`](@ref) for new output,
or use an earlier version of PALEOmodel to load jld2 output and save to netcdf.

Load from `filename` in JLD2 format, replacing any existing content in `output`.
(NB: filename must either have no extension or have extension `.jld2`).
"""
function load_jld2!(output::OutputMemory, filename)

   @error """load_jld2! has been removed in PALEOmodel v0.16.
             Please use save_netcdf, load_netcdf! for new output,
             or use an earlier version of PALEOmodel to load jld2 output and save to netcdf."""

    return nothing
end



"append output2 to the end of output1"
function Base.append!(output1::OutputMemory, output2::OutputMemory)
    for domname in keys(output1.domains)
        o1dom, o2dom = output1.domains[domname], output2.domains[domaname]
        append!(o1dom, o2dom)
    end
  
    return output1
end


function initialize!(
    output::OutputMemory, model::PB.Model, modeldata::PB.ModelData, nrecords=nothing;
    record_dim_name::Symbol=:tmodel,
    record_coord_name::Symbol=record_dim_name,
    record_coord_units::AbstractString="yr",
    coords_record::Union{Symbol, Nothing}=nothing, # deprecated
    coords_record_units::Union{AbstractString, Nothing}=nothing, # deprecated
    rec_coord::Union{Symbol, Nothing}=nothing, # deprecated
)
    if !isnothing(coords_record)
        @warn "coords_record is deprecated, use record_dim_name"
        record_dim_name = coords_record
        record_coord_name = coords_record
    end
    if !isnothing(rec_coord)
        @warn "rec_coord is deprecated, use record_dim_name"
        record_dim_name = rec_coord
        record_coord_name = rec_coord
    end
    if !isnothing(coords_record_units)
        @warn "coords_record_units is deprecated, use record_coord_units"
        record_coord_units = coords_record_units
    end

    empty!(output.domains)
    for dom in model.domains
        output.domains[dom.name] = OutputMemoryDomain(
            dom, modeldata, nrecords;
            parentdataset=output,
            record_dim_name,
            record_coord_name,
            record_coord_units,
        )
    end
  
    return nothing
end


function add_record!(output::OutputMemory, model, modeldata, rec_coord)

    for dom in model.domains
        odom = output.domains[dom.name]
        add_record!(odom, modeldata, rec_coord)
    end
end

function PB.has_variable(output::OutputMemory, varname::AbstractString)

    domainname, varname = domain_variable_name(varname)

    return (
        haskey(output.domains, domainname) 
        && PB.has_variable(output.domains[domainname], varname)
    )
end

function PB.get_field(output::OutputMemory, varname::AbstractString)
  
    domainname, varname = domain_variable_name(varname)
    
    haskey(output.domains, domainname) || 
        error("Variable $varname not found in output: domain $(domainname) not present")

    odom = output.domains[domainname]

    return PB.get_field(odom, varname)
end


function PB.add_field!(output::OutputMemory, fr::PALEOmodel.FieldRecord)
    domainname = PB.get(fr.attributes, :var_domain, nothing)

    haskey(output.domains) ||
        throw(ArgumentError("no Domain $domainname in output $output"))
    return PB.add_field!(output.domains[domainname], fr)
end

function PB.get_data(output::OutputMemory, varnamefull::AbstractString; records=nothing)

    domainname, varname = domain_variable_name(varnamefull, defaultdomainname=nothing)
    
    haskey(output.domains, domainname) || 
        error("Variable $varnamefull not found in output: domain $(domainname) not present")

    odom = output.domains[domainname]

    return PB.get_data(odom, varname; records)
end    


###########################
# Pretty printing
############################

# compact form
function Base.show(io::IO, odom::OutputMemoryDomain)
    print(io, 
        "OutputMemoryDomain(name=", odom.name,
        ", record_dim=", odom.record_dim,
        ", data_dims=", odom.data_dims, 
    ")")
end

# multiline form
function Base.show(io::IO, ::MIME"text/plain", odom::OutputMemoryDomain)
    println(io, "OutputMemoryDomain")
    println(io, "  name: $(odom.name)")
    println(io, "  record_dim: ", odom.record_dim)
    if !isempty(odom.record_dim_coordinates)
        println(io, "    coordinates: ", odom.record_dim_coordinates)
    end
    println(io, "  data_dims:")
    for nd in odom.data_dims
        println(io, "    ", nd)
        if haskey(odom.data_dims_coordinates, nd.name)
            println(io, "      coordinates: ", odom.data_dims_coordinates[nd.name])
        end
    end
    println(io, "  grid:")
    if !isnothing(odom.grid)
        iogrid = IOBuffer()
        show(iogrid, MIME"text/plain"(), odom.grid)
        seekstart(iogrid)
        for line in eachline(iogrid)
            println(io, "    ", line)
        end
    end
    # TODO - show variables ?
end


# compact form
function Base.show(io::IO, output::OutputMemory)
    print(io, "OutputMemory(domains=", keys(output.domains), ", user_data=", output.user_data, ")")
end

# multiline form
function Base.show(io::IO, ::MIME"text/plain", output::OutputMemory)
    println(io, "OutputMemory")
    println(io, "  user_data:")
    for (k, v) in output.user_data
        println(io, "    ", k, " => ", v)
    end
    println(io, "  domains:")
    for (name, odom) in output.domains
        iodom = IOBuffer()
        show(iodom, MIME"text/plain"(), odom)
        seekstart(iodom)
        for line in eachline(iodom)
            println(io, "    ", line)
        end
    end

    return nothing
end


############################
# Utility functions
############################

function domain_variable_name(varnamefull; defaultdomainname=nothing)
    varsplit = split(varnamefull, '.')
    if length(varsplit) == 1
        !isnothing(defaultdomainname) ||
            error("domain_variable_name: \"$varnamefull\" is not of form <domainname>.<varname>")
        domainname = defaultdomainname
        varname = varnamefull       
    elseif length(varsplit) == 2
        domainname = varsplit[1]
        varname = varsplit[2]
    else
        error("domain_variable_name: invalid 'varnamefull' = \"$varnamefull\" is not of form <domainname>.<varname>")
    end

    return domainname, varname
end

###############################################
# netCDF i/o
###############################################


"""
    save_netcdf(output::OutputMemory, filename; kwargs...)

Save to `filename` in netcdf4 format (NB: filename must either have no extension or have extension `.nc`)

# Notes on structure of netcdf output
- Each PALEO Domain is written to a netcdf4 group. These can be read into a Python xarray using the `group=<domainname>` argument to `open_dataset`.
- Isotope-valued variables (`field_data = PB.IsotopeLinear`) are written with an extra `isotopelinear` netCDF dimension, containing the variable `total` and `delta`.
- Any '/' characters in PALEO variables are substited for '%' in the netcdf name.

# Keyword arguments
- `check_ext::Bool = true`: check that filename ends in ".nc"
"""
function save_netcdf(
    output::OutputMemory, filename;
    check_ext::Bool=true,
)
    if check_ext
        filename = _check_filename_ext(filename, ".nc")
    end

    # Fails with variables with missing values eg that are  Union{Missing, Float64}
    # appears to be a NCDatasets.jl limitation (at least in v0.12.17) - the logic to map these to netcdf is 
    # combined with that to write the data, and the alternate form with just the type fails
    # TODO may not work with NCDatasets v0.13 and later due to variable indexing changes ?
    define_all_first = false 
    # define_all_first = true

    @info "saving to $filename ..."

    NCDatasets.NCDataset(filename, "c") do nc_dataset
        nc_dataset.attrib["PALEO_netcdf_version"] = "0.2.0"
        nc_dataset.attrib["PALEO_domains"] =  join([k for (k, v) in output.domains], " ")

        for (k, v) in output.user_data
            if k in ("PALEO_netcdf_version", "PALEO_domains")
                @warn "ignoring reserved user_data key $k"
                continue
            end
            nc_dataset.attrib[k] = v
        end

        for odom in values(output.domains)

            dsg = NCDatasets.defGroup(nc_dataset, odom.name; attrib=[])
            record_dim_name = odom.record_dim.name # record dimension (eg tmodel)
            dsg.attrib["record_dim_name"] =  record_dim_name
            NCDatasets.defDim(dsg, record_dim_name, odom.record_dim.size)
            dsg.attrib["record_dim_coordinates"] = odom.record_dim_coordinates

            grid_to_netcdf!(dsg, odom.grid)

            # data_dims
            # TODO these are NamedDimension with attached FixedCoord, where
            # the FixedCoord may not be present as a variable in the data,
            # and may also not have the correct name or even a unique name !
            # As a workaround, we generate a unique name from dim_name * coord_name, and add the Variable
            data_dim_names = String[d.name for d in odom.data_dims]
            dsg.attrib["data_dims"] = data_dim_names
            coordnames_to_netcdf(dsg, "data_dims_coords", odom.data_dims_coordinates)            

            varnames = sort(String.(keys(odom.data)))
            nc_all_vdata = []
            for vname in varnames
                varfr = PB.get_field(odom, vname)

                @debug "writing Variable $(odom.name).$vname records eltype $(eltype(varfr.records)) space $(space(varfr))"

                nc_v, nc_vdata = variable_to_netcdf!(
                    dsg,
                    vname, 
                    varfr;
                    define_only=define_all_first,  # just define the variable, don't write the data
                )
                
                if define_all_first
                    # keep netcdf variable and modified data to write in separate loop
                    push!(nc_all_vdata, (nc_v, nc_vdata))
                end
            end

            # write data (only used if define_all_first==true)
            for (nc_v, nc_vdata) in nc_all_vdata
                # TODO may not work with NCDatasets v0.13 and later ?
                nc_v[:] = nc_vdata
            end
        end
    end
   
    @info "done"

    return nothing
end


"""
    load_netcdf!(output::OutputMemory, filename)

Load from `filename` in netCDF format, replacing any existing content in `output`.
(NB: filename must either have no extension or have extension `.nc`).

# Example
```julia
julia> output = PALEOmodel.OutputWriters.load_netcdf!(PALEOmodel.OutputWriters.OutputMemory(), "savedoutput.nc")
```
"""
function load_netcdf!(output::OutputMemory, filename; check_ext=true)

    if check_ext
        filename = _check_filename_ext(filename, ".nc")
    end

    @info "loading from $filename ..."

    NCDatasets.NCDataset(filename, "r") do nc_dataset
        empty!(output.domains)

        paleo_netcdf_version = get(nc_dataset.attrib, "PALEO_netcdf_version", missing)
        !ismissing(paleo_netcdf_version) || error("not a PALEO netcdf output file ? (key PALEO_netcdf_version not present)")
        paleo_netcdf_version in ("0.1.0", "0.2.0") || 
            @warn "unsupported PALEO_netcdf_version $paleo_netcdf_version : supported versions 0.1, 0.2"

        for (k, v) in nc_dataset.attrib
            if k in ("PALEO_netcdf_version", "PALEO_domains")
                continue
            end
            # workaround for https://github.com/Alexander-Barth/NCDatasets.jl/issues/258
            if v isa Int32
                v = Int64(v)
            elseif v isa Vector{Int32}
                v = Int64.(v)
            end
            output.user_data[k] = v
        end

        for (domainname, dsg) in nc_dataset.group
            if haskey(dsg.attrib, "record_dim_name")
                record_dim_name = dsg.attrib["record_dim_name"]
            else
                # PALEO_netcdf_version 0.1.0 backwards compatibility
                record_dim_name = dsg.attrib["coords_record"]
            end
            nrecs = dsg.dim[record_dim_name]

            if haskey(dsg.attrib, "record_dim_coordinates")
                record_dim_coordinates = ncattrib_as_vector(dsg, "record_dim_coordinates")
            else
                # PALEO_netcdf_version 0.1.0 backwards compatibility
                record_dim_coordinates = String[coords_record]
            end

            # reading out variables is slow, so do this once
            dsgvars = Dict(varname=>var for (varname, var) in dsg)

            grid = netcdf_to_grid(dsg, dsgvars)

            data_dim_names = ncattrib_as_vector(dsg, "data_dims")
            data_dim_sizes = [dsg.dim[ddn] for ddn in data_dim_names]
            data_dims = [PB.NamedDimension(ddn, dds) for (ddn, dds) in zip(data_dim_names, data_dim_sizes)]
            data_dims_coordinates = netcdf_to_coordnames(dsg, "data_dims_coords")

            odom = OutputMemoryDomain(
                output,
                domainname,
                OrderedCollections.OrderedDict{Symbol, PALEOmodel.FieldRecord}(),
                PB.NamedDimension(record_dim_name, nrecs),
                record_dim_coordinates,
                data_dims,
                data_dims_coordinates,
                grid,
                [],  # omit _allvars
            )

            for (vnamenetcdf, var) in dsgvars
                attributes = netcdf_to_attributes(var)
                if haskey(attributes, :var_name) # a PALEO variable, not eg a grid variable
                    vname = netcdf_to_name(vnamenetcdf, attributes)
                    vfr = netcdf_to_variable(
                        odom,
                        var,
                        attributes,
                    )

                    odom.data[Symbol(vname)] = vfr
                end
            end

            output.domains[domainname] = odom

        end
    end

    return output
end


# write a variable to netcdf. cf get_field, FieldRecord
function variable_to_netcdf!(
    ds,
    vname::AbstractString, 
    @nospecialize(varfr::PALEOmodel.FieldRecord);
    define_only=true,
) 
    vname = name_to_netcdf(vname, varfr.attributes)

    varray = PALEOmodel.get_array(
        varfr, (expand_cartesian=true, squeeze_all_single_dims=false);
        lookup_coords=false, add_attributes=false, omit_recorddim_if_constant=true
    )
    vdata = varray.values
    vdata_dims = [nd.name for (nd, _) in varray.dims_coords]

    field_data = PALEOmodel.field_data(varfr)
    add_field_data_netcdf_dimensions(ds, field_data)
    field_data_dims = field_data_netcdf_dimensions(field_data)
    vdata = field_data_to_netcdf(field_data, vdata)
    vdata_dims = (field_data_dims..., vdata_dims...)

    # if define_only = true, only define the variable, don't actually write the data
    # (allows optimisation as netcdf is slow to swap between 'define' and 'data' mode)
    # TODO fails if vdata contains missing (so eltype is eg Union{Missing, Float64}) at least with NCDatsets v0.12.17
    # https://github.com/Alexander-Barth/NCDatasets.jl/issues/223
    vdt = define_only ? eltype(vdata) : vdata
    v = NCDatasets.defVar(ds, vname, vdt, vdata_dims)
   
    attributes_to_netcdf!(v, varfr.attributes)
   
    return (v, vdata)
end

# read a variable from netcdf
function netcdf_to_variable(
    odom,
    var,
    attributes,
)
    field_data = attributes[:field_data]

    vdata = Array(var) # convert to Julia Array

    vdata = netcdf_to_field_data(vdata, field_data)

    fielddata_dim_names = field_data_netcdf_dimensions(field_data)
    vdata_dimnames = filter(dn -> !(dn in fielddata_dim_names), NCDatasets.dimnames(var))
  
    vfr = PALEOmodel.FieldRecord(
        odom,
        vdata,
        vdata_dimnames,
        attributes;
        expand_cartesian=true,
    )
    
    return vfr
end

function coordnames_to_netcdf(ds, rootname::AbstractString, coordinates::Dict{String, Vector{String}})
    for (dimname, coordnames) in coordinates
        ds.attrib[rootname*"_"*dimname] = coordnames
    end

    return nothing
end

function netcdf_to_coordnames(ds, rootname::AbstractString)
    coordinates = Dict{String, Vector{String}}()

    for attname in keys(ds.attrib)
        if (length(attname) > length(rootname) + 1) && attname[1:length(rootname)] == rootname
            dimname = attname[(length(rootname)+2):end]
            coordinates[dimname] = ncattrib_as_vector(ds, attname)
        end
    end

    return coordinates
end

# ScalarData no additional dimensions
field_data_netcdf_dimensions(field_data::Type{PB.ScalarData}) = ()
add_field_data_netcdf_dimensions(ds, field_data::Type{PB.ScalarData}) = nothing
field_data_netcdf_dimensions(field_data::Type{PB.ArrayScalarData}) = ()
add_field_data_netcdf_dimensions(ds, field_data::Type{PB.ArrayScalarData}) = nothing
field_data_to_netcdf(field_data::Type, x) = x # fallback for ScalarData, ArrayScalarData

# serialize IsotopeLinear as (total, delta), NB: internal representation is (total, total*delta)
field_data_netcdf_dimensions(field_data::Type{PB.IsotopeLinear}) = ("isotopelinear",)
function add_field_data_netcdf_dimensions(ds, field_data::Type{PB.IsotopeLinear})
    if !haskey(ds.dim, "isotopelinear")
        NCDatasets.defDim(ds, "isotopelinear", 2)
        v = NCDatasets.defVar(ds, "isotopelinear", ["total", "delta"], ("isotopelinear",))
        v.attrib["comment"] = "components of an isotope variable"
    end
    return nothing
end
field_data_to_netcdf(field_data::Type{PB.IsotopeLinear}, x) = (x.v, x.v_delta)
field_data_to_netcdf(field_data::Type{PB.IsotopeLinear}, ::Missing) = (missing, missing)
function field_data_to_netcdf(field_data::Type{PB.IsotopeLinear}, x::Array{T}) where {T}

    
    isotopelinear_datatype(x::Type{PB.IsotopeLinear{ILT, ILT}}) where {ILT} = ILT

    # strip Missing from x, find out datatype, replace Missing for xout
    nelt = nonmissingtype(T)
    ondt = isotopelinear_datatype(nelt)
    if nelt == T # no missing in x
        OutEltype = ondt
    else # x contained missing
        OutEltype = Union{Missing, ondt}
    end

    # add extra first dimension
    xout = Array{OutEltype}(undef, (2, size(x)...))
    for i in CartesianIndices(x)
        xout[:, i] .= field_data_to_netcdf(field_data, x[i])
    end
    return xout
end

netcdf_to_field_data(x, field_data::Type{<:PB.AbstractData}) = x # fallback

# julia> PALEOmodel.OutputWriters.netcdf_to_field_data([1.0, 2.0], PB.IsotopeLinear)
# (v=1.0, v_moldelta=2.0, ‰=2.0)
#
# julia> x = [1.0 3.0 missing; 2.0 4.0 missing]
# julia> xout = PALEOmodel.OutputWriters.netcdf_to_field_data(x, PB.IsotopeLinear)
# 3-element Vector{Union{Missing, PALEOboxes.IsotopeLinear{Float64, Float64}}}:
# (v=1.0, v_moldelta=2.0, ‰=2.0)
# (v=3.0, v_moldelta=12.0, ‰=4.0)
# missing
function netcdf_to_field_data(x, field_data::Type{PB.IsotopeLinear})
    # first dimension is two components of IsotopeLinear
    first(size(x)) == 2 || error("netcdf_to_field_data IsotopeLinear has wrong first dimension (should be 2)")
    if length(size(x)) == 1
        # scalar
        xout = PB.IsotopeLinear(x[1], x[1]*x[2])
    else
        sz = size(x)[2:end] # strip first dimension
        # x may have missing values - recreate these "outside" IsotopeLinear type
        nelt = nonmissingtype(eltype(x))
        if nelt == eltype(x)
            xout_eltype = PB.IsotopeLinear{nelt, nelt}
        else
            xout_eltype = Union{Missing, PB.IsotopeLinear{nelt, nelt}}
        end
        xout = Array{xout_eltype}(undef, sz...)
        for i in CartesianIndices(xout)
            if any(ismissing.(x[:, i]))
                xout[i] = missing
            else
                xout[i] = PB.IsotopeLinear(x[1, i], x[1, i]*x[2, i])
            end
        end
    end

    return xout
end

function subdomain_to_netcdf!(ds, name::AbstractString, subdom::PB.Grids.BoundarySubdomain)
    NCDatasets.defDim(ds, "subdomain_"*name, length(subdom.indices))

    v = NCDatasets.defVar(ds, "subdomain_"*name, subdom.indices .- 1 , ("subdomain_"*name,)) # convert to zero based
    v.attrib["subdomain_type"] = "BoundarySubdomain"
end

function subdomain_to_netcdf!(ds, name::AbstractString, subdom::PB.Grids.InteriorSubdomain)
    NCDatasets.defDim(ds, "subdomain_"*name, length(subdom.indices))

    # NB: issue in NCDatasets v0.13 (probably) - v0.14.1 causes failure, fixed in v0.14.2 
    # https://github.com/Alexander-Barth/NCDatasets.jl/issues/246
    # "v0.14 cannot create variable from an Int64 array with missing values"

    v = NCDatasets.defVar(ds, "subdomain_"*name, subdom.indices .- 1, ("subdomain_"*name,)) # convert to zero based
    v.attrib["subdomain_type"] = "InteriorSubdomain"
end

function netcdf_to_subdomains(dsvars)
    subdomains = Dict{String, PB.AbstractSubdomain}()

    for (vname, v) in dsvars
        if haskey(v.attrib, "subdomain_type")
            subdomain_type = v.attrib["subdomain_type"]
            if subdomain_type == "BoundarySubdomain"
                subdom = PB.Grids.BoundarySubdomain(Array(v))
            elseif subdomain_type == "InteriorSubdomain"
                subdom = PB.Grids.InteriorSubdomain(Array(v))
            else
                error("invalid subdomain_type = $subdomain_type")
            end
            subdom_name = vname[11:end] # strip "subdomain_"
            subdomains[subdom_name] = subdom
        end
    end

    return subdomains
end

function grid_to_netcdf!(ds, grid::Nothing)
    ds.attrib["PALEO_GridType"] = "Nothing"

    return nothing
end


function grid_to_netcdf!(ds, grid::PB.Grids.UnstructuredVectorGrid)

    ds.attrib["PALEO_GridType"] = "UnstructuredVectorGrid"

    NCDatasets.defDim(ds, "cells", grid.ncells)

    coordnames_to_netcdf(ds, "PALEO_grid_coords", grid.coordinates)

    # named cells
    cellnames = [String(k) for (k, v) in grid.cellnames]
    cellnames_indices = [v for (k, v) in grid.cellnames]
    ds.attrib["PALEO_cellnames"] = cellnames
    ds.attrib["PALEO_cellnames_indices"] = cellnames_indices .- 1 # zero offset for netcdf
    
    # subdomains
    for (name, subdom) in grid.subdomains
        subdomain_to_netcdf!(ds, name, subdom)
    end

    return nothing
end

function grid_to_netcdf!(ds, grid::PB.Grids.UnstructuredColumnGrid)

    ds.attrib["PALEO_GridType"] = "UnstructuredColumnGrid"

    NCDatasets.defDim(ds, "cells", grid.ncells)
    NCDatasets.defDim(ds, "columns", length(grid.Icolumns))
   
    # similar to netCDF CF contiguous ragged array representation
    v = NCDatasets.defVar(ds, "cells_in_column", [length(ic) for ic in grid.Icolumns], ("columns",)) 
    # v.attrib["sample_dimension"] = "cells"  # similar to, but not the same as, netCDF CF contiguous ragged array
    v.attrib["comment"] = "number of cells in each column"
    Icolumns = reduce(vcat, grid.Icolumns)
    v = NCDatasets.defVar(ds, "Icolumns", Icolumns .- 1, ("cells",)) # NB: zero-based
    v.attrib["comment"] = "zero-based indices of cells from top to bottom ordered by columns"

    coordnames_to_netcdf(ds, "PALEO_grid_coords", grid.coordinates)

    # optional column labels
    if !isempty(grid.columnnames)    
        v = NCDatasets.defVar(ds, "columnnames", String.(grid.columnnames), ("columns",))
    end

    # subdomains
    for (name, subdom) in grid.subdomains
        subdomain_to_netcdf!(ds, name, subdom)
    end

    return nothing
end

function grid_to_netcdf!(ds, grid::PB.Grids.CartesianArrayGrid{N}) where {N}

    ds.attrib["PALEO_GridType"] = "CartesianArrayGrid"

    _cartesiandimscoords_to_netcdf(ds, grid)

    # subdomains
    for (name, subdom) in grid.subdomains
        subdomain_to_netcdf!(ds, name, subdom)
    end

    coordnames_to_netcdf(ds, "PALEO_grid_coords", grid.coordinates)

    return nothing
end

function grid_to_netcdf!(ds, grid::PB.Grids.CartesianLinearGrid{N}) where {N}

    ds.attrib["PALEO_GridType"] = "CartesianLinearGrid"

    ds.attrib["PALEO_columns"] = grid.ncolumns

    _cartesiandimscoords_to_netcdf(ds, grid)

    dimnames = [nd.name for nd in grid.dimensions]
    nc_linear_index = NCDatasets.defVar(ds, "linear_index", grid.linear_index .- 1, dimnames) # netcdf zero indexed
    nc_linear_index.attrib["coordinates"] = join(dimnames, " ")
 
    # # CF conventions 'lossless compression by gathering'
    # poorly supported ? (doesn't work with xarray or iris)
    # nc_cells = NCDatasets.defVar(ds, "cells", Int, ("cells",))
    # # rightmost entry in the compress list varies most rapidly
    # # (C like array convention is used by netCDF CF)
    # # we reverse dimnames so leftmost entry in dimnames varies most rapidly
    # # (so we have a Julia/Fortran/Matlab like array convention for compress)
    # nc_cells.attrib["compress"] = join(reverse(grid.dimnames), " ") 
    # cells = Int[]
    # for ci in grid.cartesian_index
    #     cit = Tuple(ci)
    #     cell = cit[1] - 1
    #     cell += (cit[2]-1)*grid.dims[1]
    #     if N == 3
    #         cell += (cit[3])*grid.dims[1]*grid.dims[2]
    #     end
    #     push!(cells, cell)
    # end
    # nc_cells[:] .= cells

    # subdomains
    for (name, subdom) in grid.subdomains
        subdomain_to_netcdf!(ds, name, subdom)
    end

    coordnames_to_netcdf(ds, "PALEO_grid_coords", grid.coordinates)

    return nothing
end

function _cartesiandimscoords_to_netcdf(
    ds::NCDatasets.Dataset,
    grid::Union{PB.Grids.CartesianLinearGrid{N}, PB.Grids.CartesianArrayGrid{N}}
) where {N}
  
    # dimensions
    NCDatasets.defDim(ds, "cells", grid.ncells)
   
    ds.attrib["PALEO_dimnames"] = [nd.name for nd in grid.dimensions]
    for nd in grid.dimensions
        NCDatasets.defDim(ds, nd.name, nd.size)
    end
    ds.attrib["PALEO_dimnames_extra"] = [nd.name for nd in grid.dimensions_extra]
    for nd in grid.dimensions_extra
        NCDatasets.defDim(ds, nd.name, nd.size)
    end
    
    ds.attrib["PALEO_londim"] = grid.londim
    ds.attrib["PALEO_latdim"] = grid.latdim
    ds.attrib["PALEO_zdim"] = grid.zdim
    ds.attrib["PALEO_zidxsurface"] = grid.zidxsurface
    ds.attrib["PALEO_display_mult"] = grid.display_mult

    
    return nothing
    
end
    


function netcdf_to_grid(ds::NCDatasets.Dataset, dsvars::Dict)
    gridtypes = Dict(
        "Nothing" => Nothing,
        "UnstructuredVectorGrid" => PB.Grids.UnstructuredVectorGrid,
        "UnstructuredColumnGrid" => PB.Grids.UnstructuredColumnGrid,
        "CartesianArrayGrid" => PB.Grids.CartesianArrayGrid,
        "CartesianLinearGrid" => PB.Grids.CartesianLinearGrid,
    )

    gridtypestring = ds.attrib["PALEO_GridType"]
    if haskey(gridtypes, gridtypestring)
        return netcdf_to_grid(gridtypes[gridtypestring], ds, dsvars)
    else
        error("invalid PALEO_GridType $gridtypestring")
    end
end

netcdf_to_grid(::Type{Nothing}, ds::NCDatasets.Dataset, dsvars::Dict) = nothing

function netcdf_to_grid(::Type{PB.Grids.UnstructuredVectorGrid}, ds::NCDatasets.Dataset, dsvars::Dict)
    ncells = ds.dim["cells"]
    subdomains = netcdf_to_subdomains(dsvars)

    vec_cellnames = Symbol.(ncattrib_as_vector(ds, "PALEO_cellnames"))
    vec_cellnames_indices = ncattrib_as_vector(ds, "PALEO_cellnames_indices") .+ 1 # netcdf is zero offset
    cellnames = Dict{Symbol, Int}(k=>v for (k, v) in zip(vec_cellnames, vec_cellnames_indices))
    
    coordinates = netcdf_to_coordnames(ds, "PALEO_grid_coords")

    return PB.Grids.UnstructuredVectorGrid(ncells, cellnames, subdomains, coordinates)
end

function netcdf_to_grid(::Type{PB.Grids.UnstructuredColumnGrid}, ds::NCDatasets.Dataset, dsvars::Dict)
    ncells = ds.dim["cells"]
    subdomains = netcdf_to_subdomains(dsvars)  

    # convert contiguous ragged array representation
    # back to vector-of-vectors 
    cells_in_column = Array(dsvars["cells_in_column"]) # number of cells in each column
    Icolumns_indices = Array(dsvars["Icolumns"]) .+ 1  # netcdf is zero based
    Icolumns = Vector{Vector{Int}}()
    colstart = 1
    for cells_this_column in cells_in_column
        push!(Icolumns, Icolumns_indices[colstart:(colstart+cells_this_column-1)])
        colstart += cells_this_column
    end

    coordinates = netcdf_to_coordnames(ds, "PALEO_grid_coords")
    # backwards compatibility
    if haskey(ds.attrib, "PALEO_z_coords") && !haskey(coordinates, "cells")
        coordinates["cells"] = ncattrib_as_vector(ds, "PALEO_z_coords")
    end
  
    # optional column labels
    if haskey(dsvars, "columnnames")
        columnnames = Symbol.(Array(dsvars["columnnames"]))
    else
        columnnames = Symbol[]
    end
    
    return PB.Grids.UnstructuredColumnGrid(ncells, Icolumns, columnnames, subdomains, coordinates)
end

function netcdf_to_grid(::Type{PB.Grids.CartesianArrayGrid}, ds::NCDatasets.Dataset, dsvars::Dict)
    
    subdomains = netcdf_to_subdomains(dsvars)  

    ncells, dimensions, dimensions_extra, zidxsurface, display_mult, londim, latdim, zdim =
        _netcdf_to_cartesiandimscoords(PB.Grids.CartesianArrayGrid, ds, dsvars)

    coordinates = netcdf_to_coordnames(ds, "PALEO_grid_coords")

    grid = PB.Grids.CartesianArrayGrid{length(dims)}(
        ncells,
        dimensions, dimensions_extra,
        londim, latdim, zdim,
        zidxsurface, display_mult,
        subdomains,
        coordinates,
    )
    
    return grid
end

function netcdf_to_grid(::Type{PB.Grids.CartesianLinearGrid}, ds::NCDatasets.Dataset, dsvars::Dict)
    
    subdomains = netcdf_to_subdomains(dsvars)  

    ncells, dimensions, dimensions_extra, zidxsurface, display_mult, londim, latdim, zdim =
        _netcdf_to_cartesiandimscoords(PB.Grids.CartesianLinearGrid, ds, dsvars)
    ncolumns = ds.attrib["PALEO_columns"]

    coordinates = netcdf_to_coordnames(ds, "PALEO_grid_coords")

    # convert back to linear vectors
    linear_index = Array(dsvars["linear_index"]) .+ 1 # netcdf zero based
    # reconstruct cartesian index 
    cartesian_index = Vector{CartesianIndex{length(dimensions)}}()
    lin_cart_index = Int[]
    for ci in CartesianIndices(linear_index)
        i = linear_index[ci]
        if !ismissing(i)
            push!(lin_cart_index, i)
            push!(cartesian_index, ci)
        end
    end
    cartesian_index = [cartesian_index[i] for i in sortperm(lin_cart_index)]
    l = length(cartesian_index)
    l == ncells || error("cartesian <-> linear index invalid, length(cartesian_index) $l != ncells $ncells")

    grid = PB.Grids.CartesianLinearGrid{length(dimensions)}(
        ncells,
        ncolumns,    
        dimensions, dimensions_extra,
        londim, latdim, zdim,
        zidxsurface, display_mult,
        subdomains,
        coordinates,
        linear_index,
        cartesian_index,
    )

    return grid
end

function _netcdf_to_cartesiandimscoords(
    ::Type{GT}, 
    ds::NCDatasets.Dataset,
    dsvars::Dict,
) where {GT <: Union{PB.Grids.CartesianArrayGrid, PB.Grids.CartesianLinearGrid}}

    ncells = ds.dim["cells"]
    
    # dimensions
    dimnames = ncattrib_as_vector(ds, "PALEO_dimnames")
    dimensions = [PB.NamedDimension(dn, ds.dim[dn]) for dn in dimnames]
    dimnames_extra = ncattrib_as_vector(ds, "PALEO_dimnames_extra")
    dimensions_extra = [PB.NamedDimension(dn, ds.dim[dn]) for dn in dimnames_extra]
    londim = ds.attrib["PALEO_londim"]
    latdim = ds.attrib["PALEO_latdim"]
    zdim = ds.attrib["PALEO_zdim"]
    zidxsurface = ds.attrib["PALEO_zidxsurface"]
    display_mult = ncattrib_as_vector(ds, "PALEO_display_mult")

    return ncells, dimensions, dimensions_extra, zidxsurface, display_mult, londim, latdim, zdim 
end

function name_to_netcdf(vname, attributes)

    vnamenetcdf = replace(vname, "/"=>"%")
    if vnamenetcdf != vname
        @debug "  replaced / with % in variable name $vname -> $vnamenetcdf" 
    end

    return vnamenetcdf
end

function netcdf_to_name(vnamenetcdf, attributes)
    vname = get(attributes, :var_name, vnamenetcdf)
    if vname != vnamenetcdf
        @debug "  replaced % with / in variable name $vnamenetcdf -> $vname" 
    end

    return vname
end

function attributes_to_netcdf!(v, attributes)

    for (aname, aval) in attributes
        # TODO serialize/deserialize attributes with type conversion
        
        if aname == :_FillValue
            # shouldn't happen as this will never be set by PALEO, and should be filtered out when netcdf file is read
            @warn "attributes_to_netcdf! netcdf variable $v ignoring netcdf reserved attribute $aname = $aval"
            continue
        elseif aname == :data_dims
            aval = String[v for v in aval] # Tuple to vector        
        else
            if (typeof(aval) in (Float64, Int64, String, Vector{Float64}, Vector{Int64}, Vector{String})) #  Bool))
                # supported netCDF type, no conversion
            else
                # anything else - convert to string
                aval = string(aval)
            end
        end

        v.attrib[String(aname)] = aval # string(aval)
    end

    return nothing
end

function netcdf_to_attributes(v)

    # explict type conversion for known attribute names
    known_attrib_to_typed = Dict(
        :space => v->parse(PB.AbstractSpace, last(split(v, "."))), # "PALEOboxes.CellSpace" fails, "CellSpace" works
        :field_data => v->parse(PB.AbstractData, v),
        :vfunction => v->parse(PB.VariableFunction, v),
        :vphase => v->parse(PB.VariablePhase, v),
        :datatype => v->isdefined(Base, Symbol(v)) ? getfield(Base, Symbol(v)) : v,  # look for a type eg Float64, fallback to String if not found
    )
    # convert string value for other attributes (currently just bools)
    attrib_val_to_typed = Dict(
        "false" => false,
        "true" => true,
    )

    attributes = Dict{Symbol, Any}()

    for (aname, avalnc) in v.attrib
        anamesym = Symbol(aname)
        # try known attribute then generic, then guess boolean conversions, then just leave as string
        if anamesym == :_FillValue
            # ignore reserved netcdf attributes
            continue
        elseif haskey(known_attrib_to_typed, anamesym)
            aval = known_attrib_to_typed[anamesym](avalnc)
        elseif PB.is_standard_attribute(anamesym) && PB.standard_attribute_type(anamesym) <: AbstractVector
            aval = isa(avalnc, Vector) ? avalnc : [avalnc] # stored as a vector but returned as a scalar if length 1
        elseif PB.is_standard_attribute(anamesym) && PB.standard_attribute_type(anamesym) <: Tuple
            aval = isa(avalnc, Vector) ? Tuple(avalnc) : (avalnc, ) # stored as a vector but returned as a scalar if length 1
        else
            # guess at how to convert values
            aval = get(attrib_val_to_typed, avalnc, avalnc)
        end
        attributes[anamesym] = aval
    end

    return attributes
end


# NCDatasets will return a scalar even if the attribute was written as a vector with 1 element
function ncattrib_as_vector(d, name) 
    x = d.attrib[name]
    if !isa(x, Vector)
        x = [x]
    end
    return x
end

function _check_filename_ext(filename, requiredext)
    froot, fext = splitext(filename)
    if isempty(fext)
        fext = requiredext
    elseif fext != requiredext
        error("filename '$filename' must have extension $requiredext")
    end
    filename = froot*fext

    return filename
end


end # module
