"""
    ReactionNetwork

Functions to analyze a PALEOboxes.Model or PALEOmodel output that contains a reaction network

Compiles reaction stoichiometry and rate information from attributes attached to reaction rate variables:

- rate_processname::String: a process name (eg "photolysis", "reaction", ...)
- rate_species::Vector{String} names of reactant and product species
- rate_stoichiometry::Vector{Float64} stoichiometry of reactant and product species

"""
module ReactionNetwork

import Printf
import Requires
import PALEOboxes as PB
import PALEOmodel
import DataFrames

"""
    add_equations!(ratetable)

Add a column `equation` with user-friendly chemical equation to `ratetable`
"""
function add_equations!(
    ratetable;
    species_root_only=true
)
    equation = String[]
    for rv in eachrow(ratetable)
        push!(
            equation,
            stoich_to_equation(
                    Dict(PB.IteratorUtils.zipstrict(rv.rate_species, rv.rate_stoichiometry)), 
                    sourcename=rv.name,
                    sinkname=rv.name,
                    species_root_only=species_root_only,
                )
        )
    end

    ratetable.equation = equation

    return nothing
end

"return species only from string of form  species::isotope eg CH4::CIsotope"
get_species_root(species) = split(species, "::")[1]

"""
    stoich_to_equations(stoichdict, sinkname) -> equation::String

Turn a `stoichdict` Dict(speciesname => stoichiometry, ...) into a user-friendly equation as a string.
Optional `sourcename`, `sinkname` are used as LHS or RHS if no species present.
"""
function stoich_to_equation(
    stoichdict; 
    sourcename="source", 
    sinkname="sink",
    species_root_only=true
)

    lhs = ""
    rhs = ""
    for (sp, stoich) in stoichdict
        if species_root_only
            sp = get_species_root(sp)
        end
        stoichprefix = ""
        if abs(stoich) != 1
            stoichprefix = string(abs(stoich))*" "
        end
        if stoich <= 0
            if !isempty(lhs)
                lhs *= " + "
            end
            lhs *= stoichprefix*sp
        end
        if stoich >= 0
            if !isempty(rhs)
                rhs *= " + "
            end
            rhs *= stoichprefix*sp
        end
    end
    if isempty(lhs)
        lhs = sourcename
    end 
    if isempty(rhs)
        rhs = sinkname
    end 

    eqn = lhs * " -> " * rhs
    return eqn
end

function reverse_equation(eqn)
    lhs, rhs = strip.(split(eqn, "->"))
    return rhs*" -> "*lhs
end




"""
    get_ratetable(model, domainname) -> DataFrame
    get_ratetable(output, domainname) -> DataFrame

Get table of rate Variables and reactions

Returns a DataFrame with columns `:name`, `:rate_processname`, `:rate_species`, `:rate_stoichiometry`
"""
function get_ratetable(output::PALEOmodel.AbstractOutputWriter, domainname; add_equations=true, species_root_only=true)
    ratetable = PB.show_variables(
        output, domainname; 
        filter=d->(:rate_processname in keys(d)), attributes=[:rate_processname, :rate_species, :rate_stoichiometry]
    )

    if add_equations
        add_equations!(ratetable; species_root_only)
    end

    sort!(ratetable, [:name])

    return ratetable
end

function get_ratetable(model::PB.Model, domainname; add_equations=true, species_root_only=true)

    domain = PB.get_domain(model, domainname)

    ratevars = PB.get_variables(domain, v->PB.has_attribute(v, :rate_processname))

    ratetable = DataFrames.DataFrame(
        name = [rv.name for rv in ratevars],
        rate_processname = [PB.get_attribute(rv, :rate_processname) for rv in ratevars],
        rate_species = [PB.get_attribute(rv, :rate_species) for rv in ratevars],
        rate_stoichiometry = [PB.get_attribute(rv, :rate_stoichiometry) for rv in ratevars],
    )

    if add_equations
        add_equations!(ratetable; species_root_only)
    end

    sort!(ratetable, [:name])

    return ratetable
end

"""
    get_all_species_ratevars(model, domainname) -> OrderedDict(speciesname => [(stoich, ratevarname, processname), ...])
    get_all_species_ratevars(output, domainname) -> OrderedDict(speciesname => [(stoich, ratevarname, processname), ...])
    get_all_species_ratevars(ratetable::DataFrame) -> OrderedDict(speciesname => [(stoich, ratevarname, processname), ...])

Get all species and contributing reaction rate Variable names as Dict of Tuples (stoich, ratevarname, processname) where
`ratevarname` is the name of an output Variable with a reaction rate, `stoich` is the stoichiometry of that rate
applied to `species`, and `processname` identifies the nature of the reaction.
"""
function get_all_species_ratevars(
    ratetable;
    species_root_only=true
)
    species_rates = Dict{String, Vector{Tuple{Float64,String, String}}}()

    for rv in eachrow(ratetable)      
        for (speciesname, s) in PB.IteratorUtils.zipstrict(rv.rate_species, rv.rate_stoichiometry)
            if species_root_only
                speciesname = get_species_root(speciesname)
            end
            sr = get!(species_rates, speciesname, [])
            push!(sr, (s, rv.name, rv.rate_processname))
        end
    end

    # for each species, sort rates first by stoichiometry and then by name
    for (species, rates) in species_rates
        sort!(rates)
    end

    # return Dict sorted by species name
    return sort(species_rates)
end

function get_all_species_ratevars(output::PALEOmodel.AbstractOutputWriter, domainname; kwargs...)
    ratetable = get_ratetable(output, domainname; add_equations=false)
    return get_all_species_ratevars(ratetable; kwargs...)
end

function get_all_species_ratevars(model::PB.Model, domainname; kwargs...)
    ratetable = get_ratetable(model, domainname; add_equations=false)
    return get_all_species_ratevars(ratetable; kwargs...)
end

"""
    get_rates(output, domainname [, outputrec] [, indices] [, scalefac] [, add_equations] [, ratetable_source]) -> DataFrame

Get all reaction rates as column `rate_total` for `domainname` from `output` record `outputrec` (defaults to last time record),
for subset of cells in `indices` (defaults to whole domain).

Set optional `ratetable_source = model` to use with older output that doesn't include rate variable attributes.
"""
function get_rates(
    output::PALEOmodel.AbstractOutputWriter, domainname;
    outputrec=length(output.domains[domainname]), 
    indices=Colon(),
    scalefac=1.0,
    add_equations=true,
    species_root_only=true,
    ratetable_source=output,
)

    ratetable = get_ratetable(ratetable_source, domainname; add_equations, species_root_only)

    rate_total = Float64[]
    for rv in eachrow(ratetable)
        rate = PB.get_data(output, domainname*"."*rv.name; records=outputrec)
        rate_tot = sum(rate[indices])        
        push!(rate_total, rate_tot*scalefac)
    end
       
    ratetable.rate_total = rate_total

    return ratetable
end

"""
    get_all_species_ratesummaries(output, domainname [, outputrec] [, indices] [, scalefac] [, ratetable_source]) 
        -> OrderedDict(speciesname => (source, sink, net, source_rxs, sink_rxs))

Get `source`, `sink`, `net` rates and rates of `source_rxs` and `sink_rxs` 
for all species in `domainname` from `output` record `outputrec` (defaults to last record), 
cells in `indices` (defaults to whole domain),

Optional `scalefac` to convert units, eg `scalefac`=1.90834e12 to convert mol m-2 yr-1 to molecule cm-2 s-1

Set optional `ratetable_source = model` to use with older output that doesn't include rate variable attributes.
"""
function get_all_species_ratesummaries(
    output, domainname;
    outputrec=length(output.domains[domainname]), 
    indices=Colon(),
    scalefac=1.0,
    species_root_only=true,
    ratetable_source=output,
)

    ratetable = get_rates(output, domainname; outputrec, indices, scalefac, add_equations=true, species_root_only, ratetable_source)

    species_ratevars = get_all_species_ratevars(ratetable; species_root_only)

    rate_summaries = Dict()
    for (species, ratevars) in species_ratevars
        source, sink = 0.0, 0.0
        source_rxs, sink_rxs = [], []
        for (stoich, ratevarname, processname) in ratevars
            rate = ratetable[ratetable.name .== ratevarname, :rate_total][]
            eqn = ratetable[ratetable.name .== ratevarname, :equation][]
            if rate < 0
                stoich = -stoich
                rate = -rate
                eqn = reverse_equation(eqn)
            end
            if stoich == 0
                rxrate = rate  # assume species appears once on each side of reaction
            else
                rxrate = abs(stoich)*rate
            end
            if stoich <= 0  # add a rate with stoich = 0 to both source and sink 
                sink += rxrate
                push!(sink_rxs, (ratevarname, rxrate, eqn, processname))
            end
            if stoich >= 0
                source += rxrate
                push!(source_rxs, (ratevarname, rxrate, eqn, processname))
            end
            net = source - sink
            rate_summaries[species] = (;source, sink, net, source_rxs, sink_rxs)
        end
    end

    return sort(rate_summaries)
end

"""
    show_ratesummaries([io::IO = stdout], ratesummaries [; select_species=[]])

Print per-species reaction rates from `ratesummaries` to output stream `io` (defaults to `stdout`), 
optionally selecting species to print.

# Example

    ratesummaries = PALEOmodel.ReactionNetwork.get_all_species_ratesummaries(output, "atm")
    PALEOmodel.ReactionNetwork.show_ratesummaries(ratesummaries)

"""
function show_ratesummaries(io::IO, ratesummaries; select_species=[])
    for (species, rates) in ratesummaries
        if isempty(select_species) || species in select_species
            Printf.@printf(io, "\n")
            Printf.@printf(io, "%-8s                                                           net:   %g\n", species, rates.net)
            Printf.@printf(io, "\n")
            Printf.@printf(io, "%-8sProduction reactions                          rate         total: %g\n", species, rates.source)
            for (ratevarname, rxrate, equation, processname) in rates.source_rxs
                Printf.@printf(io, "        %-40s %16g            %-16s%s\n", equation, rxrate, "["*processname*"]", ratevarname)
            end
            Printf.@printf(io, "\n")
            Printf.@printf(io, "%-8sLoss reactions                                rate         total: %g\n", species, rates.sink)
            for (ratevarname, rxrate, equation, processname) in rates.sink_rxs
                Printf.@printf(io, "        %-40s %16g            %-16s%s\n", equation, rxrate, "["*processname*"]", ratevarname)
            end
        end
    end

    return nothing
end

show_ratesummaries(ratesummaries; kwargs...) = show_ratesummaries(stdout, ratesummaries; kwargs...)

function __init__()
    # If PyCall is available, include additional functions using Python pydot.
    Requires.@require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("ReactionNetworkVis.jl")
end

end # module
