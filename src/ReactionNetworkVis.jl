"""
   
Additional functions for ReactionNetwork module to display a reaction network from a PALEOboxes.Model

NB: uses Julia PyCall for pydot package, Project environment must have PyCall package installed and configured.
"""

using .PyCall


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

