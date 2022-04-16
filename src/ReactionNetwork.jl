"""
    ReactionNetwork

Functions to analyize a PALEOboxes.Model that contains a reaction network
"""
module ReactionNetwork

using PyCall
import Printf

import PALEOboxes as PB
import PALEOmodel


"""
    get_equations(model, domainname) -> OrderedDict(ratevarname => reactioneqn)

Get all rate variables and a string with a user-friendly chemical equation for `domainname` in `model`
"""
function get_equations(
    model::PB.Model, domainname;
    species_root_only=true
)
    
    dom = PB.get_domain(model, domainname)

    equations = Dict{String, Any}()
    for rj in dom.reactions
        rvs = PB.get_rate_stoichiometry(rj)
        if !isnothing(rvs)
            for (ratevarname, processname, stoich) in rvs   
                equations[ratevarname] = stoich_to_equation(
                    stoich, 
                    sourcename=ratevarname,
                    sinkname=ratevarname,
                    species_root_only=species_root_only
                )
            end
        end
    end
    
    return sort(equations)
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
    get_all_species_ratevars(model, domainname) -> OrderedDict(speciesname => [(stoich, ratevarname, processname), ...])

Get all species and contributing reaction rate Variable names as Dict of Tuples (stoich, ratevarname) where
`ratevarname` is the name of an output Variable with a reaction rate, `stoich` is the stoichiometry of that rate
applied to `species`.
"""
function get_all_species_ratevars(
    model::PB.Model, domainname;
    species_root_only=true
)

    species_rates = Dict{String, Vector{Tuple{Float64,String, String}}}()

    dom = PB.get_domain(model, domainname)

    for rj in dom.reactions
        rvs = PB.get_rate_stoichiometry(rj)
        if !isnothing(rvs)
            for (ratevarname, processname, stoich) in rvs           
                for (speciesname, s) in stoich
                    if species_root_only
                        speciesname = get_species_root(speciesname)
                    end
                    sr = get!(species_rates, speciesname, [])
                    push!(sr, (s, ratevarname, processname))
                end
            end
        end
    end

    # for each species, sort rates first by stoichiometry and then by name
    for (species, rates) in species_rates
        sort!(rates)
    end

    # return Dict sorted by species name
    return sort(species_rates)
end

"""
    get_rates(model, output, domainname [, outputrec] [, indices]) -> OrderedDict(ratevarname => rate)

Get all reaction rates for `domainname` from `output` record `outputrec` (defaults to last time record),
for subset of cells in `indices` (defaults to whole domain).
"""
function get_rates(
    model::PB.Model, output, domainname;
    outputrec=length(output.domains[domainname]), 
    indices=nothing,
    scalefac=1.0
)
    
    dom = PB.get_domain(model, domainname)

    if isnothing(indices)
        cellrange = PB.Grids.create_default_cellrange(dom, dom.grid)
        indices = cellrange.indices
    end

    rate_totals = Dict()
    for rj in dom.reactions
        rvs = PB.get_rate_stoichiometry(rj)
        if !isnothing(rvs)
            for (ratevarname, processname, stoich) in rvs   
                rate = PB.get_data(output, domainname*"."*ratevarname, records=outputrec)
                rate_tot = sum(rate[indices])        
                rate_totals[ratevarname] = rate_tot*scalefac
            end
        end
    end
    
    return sort(rate_totals)
end

"""
    get_all_species_ratesummaries(model, output, domainname [, outputrec] [, indices]) 
        -> OrderedDict(speciesname => (source, sink, net, source_rxs, sink_rxs))

Get `source`, `sink`, `net` rates and rates of `source_rxs` and `sink_rxs` 
for all species in `domainname` from `output` record `outputrec` (defaults to last record), 
cells in `indices` (defaults to whole domain),

Optional `scalefac` to convert units, eg `scalefac`=1.90834e12 to convert mol m-2 yr-1 to molecule cm-2 s-1
"""
function get_all_species_ratesummaries(
    model::PB.Model, output, domainname;
    outputrec=length(output.domains[domainname]), 
    indices=nothing,
    scalefac=1.0,
    species_root_only=true
)

    rate_totals = get_rates(
        model, output, domainname,
        outputrec=outputrec, 
        indices=indices,
        scalefac=scalefac
    )

    equations = get_equations(
        model, domainname,
        species_root_only=species_root_only
    )

    species_ratevars = get_all_species_ratevars(
        model, domainname,
        species_root_only=species_root_only
    )

    rate_summaries = Dict()
    for (species, ratevars) in species_ratevars
        source, sink = 0.0, 0.0
        source_rxs, sink_rxs = [], []
        for (stoich, ratevarname, processname) in ratevars
            rate = rate_totals[ratevarname]
            eqn = equations[ratevarname]
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
    show_ratesummaries(ratesummaries [,select_species=[]])

Print `ratesummaries` to terminal, optionally selecting species to print
"""
function show_ratesummaries(ratesummaries; select_species=[])
    for (species, rates) in ratesummaries
        if isempty(select_species) || species in select_species
            Printf.@printf("\n")
            Printf.@printf("%-8s                                                           net:   %g\n", species, rates.net)
            Printf.@printf("\n")
            Printf.@printf("%-8sProduction reactions                          rate         total: %g\n", species, rates.source)
            for (ratevarname, rxrate, equation, processname) in rates.source_rxs
                Printf.@printf("        %-40s %16g            %-16s%s\n", equation, rxrate, "["*processname*"]", ratevarname)
            end
            Printf.@printf("\n")
            Printf.@printf("%-8sLoss reactions                                rate         total: %g\n", species, rates.sink)
            for (ratevarname, rxrate, equation, processname) in rates.sink_rxs
                Printf.@printf("        %-40s %16g            %-16s%s\n", equation, rxrate, "["*processname*"]", ratevarname)
            end
        end
    end

    return nothing
end

"""
    show_network_rate_nodes(model::PB.Model, domainname, output [, outputrec] [, indices] [.rate_thresh=0.0]) -> graph

Display reaction network as a `pydot` `graph`, including nodes for reactions.

NB: requires the Python `pydot` package. See <https://github.com/JuliaPy/PyCall.jl>, 
and the docstring for `pyimport_conda`. Assuming that a conda-based Python is being used,

    julia> using PyCall
    julia> pyimport_conda("pydot", "pydot")

will download and install the Python package.
"""
function show_network_rate_nodes(
    model::PB.Model, domainname, output;
    outputrec=length(output.domains[domainname]), 
    indices=nothing,
    rate_thresh=0.0,
    species_root_only=true
)

    rate_totals = get_rates(model, output, domainname;
        outputrec=outputrec, 
        indices=indices)

    rate_max = 0.0
    for (rname, r) in rate_totals
        rate_max = max(rate_max, r)
    end

    # species_ratevars = get_all_species_ratevars(model, domainname)

    ratesummaries = get_all_species_ratesummaries(
        model, output, domainname,
        outputrec=outputrec,
        indices=indices,
        species_root_only=species_root_only
    )

    pydot  = pyimport_conda("pydot", "pydot")
    graph = pydot.Dot("Reactions", graph_type="digraph", bgcolor="white", layout="dot", rank="same", splines="line")

    for (species, ratesummary) in ratesummaries

        node = pydot.Node(species, label=species)
        graph.add_node(node)

        net = ratesummary.net
        if abs(net) >= rate_thresh
            node = pydot.Node(species*"_net", label=string(abs(net)), shape="diamond")
            graph.add_node(node)
            if net > 0
                from, to = species, species*"_net"
            else
                from, to = species*"_net", species
            end
            net_norm = abs(net)/rate_max
            edge = pydot.Edge(
                from, to,
                color="black",
                weight=Int(floor(100*net_norm))+1,
                penwidth=max(1.0, 20.0*net_norm)
            )
            graph.add_edge(edge)
        end
    end


    dom = PB.get_domain(model, domainname)
  
    colourlist = ["green", "black", "blue"]
    rate_idx = 1
    for rj in dom.reactions
        rvs = PB.get_rate_stoichiometry(rj)
        if !isnothing(rvs)
            
           
            for (ratevarname, processname, stoichvec) in rvs
                if abs(rate_totals[ratevarname]) >= rate_thresh
                    if processname == "photolysis"
                        colour="red"
                    else
                        colour=colourlist[mod(rate_idx, length(colourlist))+1]                      
                    end

                    rate_norm = abs(rate_totals[ratevarname])/rate_max
                    node = pydot.Node(ratevarname, shape="box", color=colour)
                    graph.add_node(node)        
                    for (speciesname, stoich) in stoichvec
                        if species_root_only
                            speciesname = get_species_root(speciesname)
                        end
                        if stoich*rate_totals[ratevarname] > 0
                            from, to = ratevarname, speciesname
                        else
                            from, to = speciesname, ratevarname
                        end
                        if stoich != 0
                            rate_stoich_norm = rate_norm * abs(stoich)
                            edge = pydot.Edge(
                                from, to,
                                color=colour,
                                weight=Int(floor(100*rate_stoich_norm))+1,
                                penwidth=max(1.0, 20.0*rate_stoich_norm),
                            )
                            graph.add_edge(edge)
                        end
                    end
                    rate_idx += 1
                end
            end
        end
    end  

    return graph
end


function display_svg(graph)
    output_graphviz_svg = graph.create_svg()
    display("image/svg+xml", output_graphviz_svg)
end

#=
function show_network(model::PB.Model, domainname, output;
    outputrec=length(output.domains[domainname]), 
    indices=nothing,
    rate_thresh=0.0)

    rate_totals = get_rates(model, output, domainname;
        outputrec=outputrec, 
        indices=indices)

    rate_max = 0.0
    for (rname, r) in rate_totals
        rate_max = max(rate_max, r)
    end

    species_ratevars = get_all_species_ratevars(model, domainname)

    pydot  = pyimport_conda("pydot", "pydot")
    graph = pydot.Dot("Reactions", graph_type="digraph", bgcolor="white", layout="dot", splines="line")

    for (species, rates) in species_ratevars

        node = pydot.Node(species, label=species)
        graph.add_node(node)
    end


    dom = PB.get_domain(model, domainname)

    rate_idx = 1
    for rj in dom.reactions
        rvs = PB.get_rate_stoichiometry(rj)
        if !isnothing(rvs)
            
            for (ratevarname, processname, stoichvec) in rvs
                if rate_totals[ratevarname] >= rate_thresh
                    rate_norm= rate_totals[ratevarname]/rate_max
                    reactants = []
                    products = []
                    for (speciesname, stoich) in stoichvec
                        if stoich < 0
                            push!(reactants, speciesname)
                        end
                        if stoich > 0
                            push!(products, speciesname)
                        end
                    end
                    for r in reactants
                        for p in products
                            edge = pydot.Edge(r, p, color="blue", weight=Int(floor(100*rate_norm)), penwidth=max(1.0, 20.0*rate_norm))
                            graph.add_edge(edge)
                        end
                    end
                
                    rate_idx += 1
                end
            end
        end
    end  

    return graph
end

=#

"""
    test_pydot()

Test python pydot package and graphviz command-line tools installed and working
"""
function test_pydot()
    pydot  = pyimport_conda("pydot", "pydot")
    graph = pydot.Dot("my_graph", graph_type="graph", bgcolor="white")
    my_node = pydot.Node("a", label="Foo")
    graph.add_node(my_node)
    graph.add_node(pydot.Node("b", shape="circle"))
    my_edge = pydot.Edge("a", "b", color="blue")
    graph.add_edge(my_edge)

    output_graphviz_svg = graph.create_svg()
    display("image/svg+xml", output_graphviz_svg)

    return graph
end

end # module
