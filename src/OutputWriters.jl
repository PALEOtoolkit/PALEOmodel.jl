"""
    OutputWriters

Data structures and methods to hold and manage model output.
"""
module OutputWriters

import PALEOboxes as PB

import PALEOmodel

import DataFrames
import FileIO
import JLD2

# using Infiltrator # Julia debugger


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
    initialize!(output::PALEOmodel.AbstractOutputWriter, model, modeldata, nrecords [;rec_coord=:tmodel])

Initialize from a PALEOboxes::Model, reserving memory for an assumed output dataset of `nrecords`.

The default for `rec_coord` is `:tmodel`, for a sequence of records following the time evolution
of the model.
"""
function initialize!(
    output::PALEOmodel.AbstractOutputWriter, model::PB.Model, modeldata::PB.ModelData, nrecords;
    rec_coord::Symbol=:tmodel
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
    get_array(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString; kwargs...) -> FieldArray

Return a [`PALEOmodel.FieldArray`](@ref) containing data values and any attached coordinates, for the 
[`PALEOmodel.FieldRecord`](@ref) for `varname`, and records and spatial region defined by `kwargs`
    
Equivalent to `PALEOmodel.get_array(PB.get_field(output, varname), kwargs...)`
"""
function PALEOmodel.get_array(output::PALEOmodel.AbstractOutputWriter, varname::AbstractString; kwargs...)

    fr = PB.get_field(output, varname)

    fa = PALEOmodel.get_array(fr; kwargs...)

    return fa
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

Includes an additional `coords_record` (usually `:tmodel`, when storing output vs time).

# Implementation
`data::DataFrame` contains columns of same type as `FieldRecord.records` for each Variable.
"""
mutable struct OutputMemoryDomain
    "Domain name"
    name::String
    "Model output for this Domain"
    data::DataFrames.DataFrame
    "record coordinate"
    coords_record::Symbol
    "Domain data_dims"
    data_dims::Vector{PB.NamedDimension}
    "Variable metadata (attributes) (metadata[varname] -> attributes::Dict{Symbol, Any})"
    metadata::Dict{String, Dict{Symbol, Any}}
    "Domain Grid (if any)"
    grid::Union{PB.AbstractMesh, Nothing}

    # internal use only: all Variables in sorted order
    _all_vars::Vector{PB.VariableDomain}
    # current last record in preallocated data::DataFrame (may be less than length(data))
    _nrecs
end

Base.length(output::OutputMemoryDomain) = output._nrecs

"create from a PALEOboxes::Domain"
function OutputMemoryDomain(
    dom::PB.Domain, modeldata::PB.ModelData, nrecords; 
    coords_record=:tmodel, coords_units="yr"
)
  
    odom =  OutputMemoryDomain(
        dom.name,
        DataFrames.DataFrame(),        
        coords_record,
        deepcopy(dom.data_dims),
        Dict{String, Dict{Symbol, Any}}(),
        deepcopy(dom.grid),
        PB.VariableDomain[],
        0,       
    )

    # create list of variables sorted by host dependent type, then by name
    odom._all_vars = vcat(
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
            PB.get_variables(dom, v->PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_State,)),
            by=var->var.name
        ),
        sort(
            PB.get_variables(
                dom, v-> !(PB.get_attribute(v, :vfunction, PB.VF_Undefined) in (PB.VF_StateExplicit, PB.VF_Total, PB.VF_Constraint, PB.VF_Deriv, PB.VF_State))),
            by=var->var.name
        ),
    )
    
    # add empty (ie undefined) columns to dataframe

    # add record coordinate column
    DataFrames.insertcols!(
        odom.data, 
        DataFrames.ncol(odom.data)+1, 
        coords_record => Vector{Float64}(undef, nrecords)
    )   
    odom.metadata[String(coords_record)] = Dict(
        :var_name=>String(coords_record), :domain_name=>dom.name,
        :vfunction=>PB.VF_Undefined, :description=>"output record coordinate",
        :field_data=>PB.ScalarData, :space=>PB.ScalarSpace, :data_dims=>(), :units=>coords_units,
    )

    # add variables
    for var in odom._all_vars
        # records storage type is that of FieldRecord.records
        field = PB.get_field(var, modeldata)
        if PALEOmodel.field_single_element(field)
            # if Field contains single elements, store as a Vector of elements
            records = Vector{eltype(field.values)}(undef, nrecords)
        else
            # if Field contains something else, store as a Vector of those things
            records = Vector{typeof(field.values)}(undef, nrecords)
        end
       
        DataFrames.insertcols!(
            odom.data, 
            DataFrames.ncol(odom.data)+1, 
            Symbol(var.name) => records
        )            

        attrbs = deepcopy(var.attributes)
        attrbs[:var_name] = var.name
        attrbs[:domain_name] = dom.name
        odom.metadata[var.name] = attrbs
    end

    return odom
end

"create from a DataFrames DataFrame containing scalar data"
function OutputMemoryDomain(
    name::AbstractString, data::DataFrames.DataFrame;
    metadata::Dict{String, Dict{Symbol, Any}}=Dict("tmodel"=>Dict{Symbol, Any}(:units=>"yr")),
    coords_record=:tmodel,
)
    # create minimal metadata for scalar Variables
    for vname in DataFrames.names(data)
        vmeta = get!(metadata, vname, Dict{Symbol, Any}())
        vmeta[:var_name] = vname
        vmeta[:domain_name] = name
        vmeta[:field_data] = PB.ScalarData
        vmeta[:space] = PB.ScalarSpace
        vmeta[:data_dims] = ()
    end

    return OutputMemoryDomain(
        name,
        data,
        coords_record,
        [],
        metadata,
        nothing,
        [],
        DataFrames.nrow(data)
    )

end

"""
    OutputMemoryDomain(name::AbstractString, coords_record::PALEOmodel.FieldRecord)

Create from a `coords_record` (eg defining `tmodel`). Add additional Fields with
`add_field!`.
"""
function OutputMemoryDomain(
    name::AbstractString, coords_record::PALEOmodel.FieldRecord;
    data_dims::Vector{PB.NamedDimension} = Vector{PB.NamedDimension}(),
    grid = nothing,
)
    data = DataFrames.DataFrame()

    haskey(coords_record.attributes, :var_name) ||
        throw(ArgumentError("FieldRecord has no :var_name attribute"))
    varname = coords_record.attributes[:var_name]
   
    DataFrames.insertcols!(data, Symbol(varname)=>coords_record.records)
    metadata = Dict(varname=>deepcopy(coords_record.attributes))
    # update domain name 
    metadata[varname][:domain_name] = name

    return OutputMemoryDomain(
        name,
        data,
        Symbol(varname),
        data_dims,
        metadata,
        grid,
        [],
        DataFrames.nrow(data)
    )

end


function add_record!(odom::OutputMemoryDomain, modeldata, rec_coord)
        
    odom._nrecs +=1       
    df = odom.data

    df[!, odom.coords_record][odom._nrecs] = rec_coord

    for var in odom._all_vars
        field = PB.get_field(var, modeldata)
        values = PB.get_values_output(field)
        
        if PALEOmodel.field_single_element(field)
            df[!, Symbol(var.name)][odom._nrecs] = values[]
        else
            # copy array(s) data                           
            df[!, Symbol(var.name)][odom._nrecs] = copy(values)
        end
    end

    return nothing
end

function PB.add_field!(odom::OutputMemoryDomain, fr::PALEOmodel.FieldRecord)
    
    length(fr) == length(odom) ||
        throw(ArgumentError("FieldRecord length $(length(fr)) != OutputMemoryDomain length $(length(odom))"))

    haskey(fr.attributes, :var_name) ||
        throw(ArgumentError("FieldRecord has no :var_name attribute"))
    varname = fr.attributes[:var_name]
    !(Symbol(varname) in names(odom.data)) ||
        throw(ArgumentError("Variable $varname already exists"))

    DataFrames.insertcols!(odom.data, Symbol(varname)=>fr.records)
    odom.metadata[varname] = deepcopy(fr.attributes)
    # update domain name 
    odom.metadata[varname][:domain_name] = odom.name

    return nothing
end

function PB.get_field(odom::OutputMemoryDomain, varname)

    df = odom.data
    varname in DataFrames.names(df) || 
        error("Variable $varname not found in output (no column '$varname' in Dataframe output.domains[\"$(odom.name)\"].data)")

    vdata = df[!, Symbol(varname)]

    attributes = get(odom.metadata, varname, nothing)

    !isnothing(attributes) ||
        @error "$(odom.name).$varname has no attributes"

    grid = odom.grid

    data_dims = []
    for dimname in attributes[:data_dims]
        idx = findfirst(d -> d.name==dimname, odom.data_dims)
        !isnothing(idx) ||
            error("Domain $(odom.name) has no dimension='$dimname' (available dimensions: $(odom.data_dims)")
        push!(data_dims, odom.data_dims[idx])
    end

    field_data = attributes[:field_data]
    space = attributes[:space]

    fr = PALEOmodel.wrap_fieldrecord(
        vdata, field_data, Tuple(data_dims), missing, space, grid, attributes,
        coords_record=[
            PB.FixedCoord(
                String(odom.coords_record),
                df[!, odom.coords_record],
                odom.metadata[String(odom.coords_record)]
            ),
        ]
    )

    return fr
end


function PB.show_variables(
    odom::OutputMemoryDomain; 
    attributes=[:units, :vfunction, :space, :field_data, :description],
    filter = attrb->true, 
)
    shownames = []
    for vn in names(odom.data)
        if filter(odom.metadata[vn])
            push!(shownames, vn)
        end
    end
    sort!(shownames)
    
    df = DataFrames.DataFrame()
    df.name = shownames
    for att in attributes
        DataFrames.insertcols!(df, att=>[get(odom.metadata[vn], att, missing) for vn in shownames])
    end   

    return df
end

########################################
# OutputMemory
##########################################

"""
    OutputMemory

In-memory container for model output, organized by model Domains.

Implements the [`PALEOmodel.AbstractOutputWriter`](@ref) interface, with additional methods
[`save_jld2`](@ref) and [`load_jld2!`](@ref) to save and load data.


# Implementation
Field `domains::Dict{String, OutputMemoryDomain}` contains per-Domain model output.
"""
struct OutputMemory <: PALEOmodel.AbstractOutputWriter
    
    domains::Dict{String, OutputMemoryDomain}
end

function OutputMemory()
    return OutputMemory(Dict{String, OutputMemoryDomain}())
end

"create from collection of OutputMemoryDomain"
function OutputMemory(output_memory_domains::Union{Vector, Tuple})
    om = OutputMemory(Dict(om.name => om for om in output_memory_domains))
    return om
end

function Base.length(output::OutputMemory)
    lengths = unique([length(omd) for (k, omd) in output.domains])
    length(lengths) == 1 ||
        error("output $output has Domains of different lengths")

    return lengths[]
end


function PB.get_table(output::OutputMemory, domainname::AbstractString)
    haskey(output.domains, domainname) ||
        throw(ArgumentError("no Domain $domainname"))

    return output.domains[domainname].data
end

function PB.get_table(output::OutputMemory, varnames::Vector{<:AbstractString})
    df = DataFrames.DataFrame()

    for varname in varnames
        if contains(varname, ".")
            vdom, vname = split(varname, ".")
            if haskey(output.domains, vdom)
                dfdom = output.domains[vdom].data
                if vname in names(dfdom)
                    vardata = dfdom[:, Symbol(vname)]
                    df = DataFrames.insertcols!(df, varname=>vardata)
                else
                    @warn "no Variable found for $varname"
                end
            else
                @warn "no Domain found for $varname"
            end
        else
            @warn "Variable $varname is not of form <Domain>.<Name>"
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

Save to `filename` in JLD2 format (NB: filename must either have no extension or have extension `.jld2`)
"""
function save_jld2(output::OutputMemory, filename)

    filename = _check_filename_jld2(filename)

    # create a temporary copy to omit _all_vars
    output_novars = copy(output.domains)
    for (k, omd) in output_novars
        output_novars[k] = OutputMemoryDomain(
            omd.name,
            omd.data,
            omd.coords_record,
            omd.data_dims,
            omd.metadata,
            omd.grid,
            [],  # omit _allvars
            omd._nrecs,
        )        
    end

    @info "saving to $filename ..."
    FileIO.save(filename, output_novars)
    @info "done"

    return nothing
end

"""
    load_jld2!(output::OutputMemory, filename)

Load from `filename` in JLD2 format, replacing any existing content in `output`.
(NB: filename must either have no extension or have extension `.jld2`).

# Example
```julia
julia> output = PALEOmodel.OutputWriters.load_jld2!(PALEOmodel.OutputWriters.OutputMemory(), "savedoutput.jld2")
```
"""
function load_jld2!(output::OutputMemory, filename)

    filename = _check_filename_jld2(filename)

    @info "loading from $filename ..."

    jld2data = FileIO.load(filename) # returns Dict

    empty!(output.domains)
    for (domainname, odom) in jld2data
        output.domains[domainname] = odom
    end

    return output
end


function _check_filename_jld2(filename)
    froot, fext = splitext(filename)
    if isempty(fext)
        fext = ".jld2"
    elseif fext != ".jld2"
        error("filename '$filename' must have extension .jld2")
    end
    filename = froot*fext

    return filename
end

"append output2 to the end of output1"
function Base.append!(output1::OutputMemory, output2::OutputMemory)
    for domname in keys(output1.domains)
        o1dom, o2dom = output1.domains[domname], output2.domains[domaname]
        append!(o1dom.data, o2dom.data)
        o1dom.nrecs += o2dom.nrecs
    end
  
    return output1
end


function initialize!(
    output::OutputMemory, model::PB.Model, modeldata::PB.ModelData, nrecords;
    rec_coord::Symbol=:tmodel
)

    # Create Dict of DataFrames with output
    empty!(output.domains)
    for dom in model.domains
        output.domains[dom.name] = OutputMemoryDomain(
            dom, modeldata, nrecords,
            coords_record=rec_coord,
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
        && varname in DataFrames.names(output.domains[domainname].data)
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
    df = odom.data
    varname in DataFrames.names(df) || 
        error("Variable $varname not found in output (no column '$varname' in Dataframe output.domains[\"$domainname\"].data)")
    
    if isnothing(records)
        output = df[!, Symbol(varname)]
    else
        output = df[records, Symbol(varname)]
        # bodge - fix scalar data
        if isa(records, Integer) && !isa(output, AbstractVector)
            output =[output]
        end
    end

    return output
end    


###########################
# Pretty printing
############################

"compact form"
function Base.show(io::IO, odom::OutputMemoryDomain)
    print(io, 
        "OutputMemoryDomain(name=", odom.name,
        ", data_dims=", odom.data_dims, 
        ", length=", length(odom), 
    ")")
end

"compact form"
function Base.show(io::IO, output::OutputMemory)
    print(io, "OutputMemory(domains=", keys(output.domains), ")")
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


end # module
