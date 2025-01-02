import RecipesBase
import Infiltrator

###########################
# PlotPager
##############################


"""
    PlotPager(layout [, kwargs=NamedTuple()][; displayfunc=(plot, nplot)->display(plot)])

Accumulate plots into subplots.

`layout` is supplied to `Plots.jl` `layout` keyword, may be an Int or a Tuple (ny, nx),
see <https://docs.juliaplots.org/latest/>.

Optional argument `kwargs::NamedTuple` provides additional keyword arguments passed through to `plot`
(eg `(legend_background_color=nothing, )` to set all subplot legends to transparent backgrounds)

Optional keyword argument `displayfunc::(plot, nplot)->display(plot)` provides the function used to display a screen of plots,
where `plot` is a Plot object and `nplot::Integer` is the sequential number of this screen.

# Usage

    julia> pp = PlotPager((2,2))  # 4 panels per screen (2 down, 2 across)
    julia> pp(plot(1:3))  # Accumulate
    julia> pp(:skip, plot(1:4), plot(1:5), plot(1:6))  # add multiple panels in one command
    julia> pp(:newpage) # flush any partial screen and start new page (NB: always add this at end of a sequence!)

    julia> pp = PlotPager((2, 2); displayfunc=(plot, nplot)->savefig(plot, "plot_\$nplot.png")) # save to file instead of default display

# Commands
- `pp(p)`: accumulate plot p
- `pp(:skip)`: leave a blank panel
- `pp(:newpage)`: fill with blank panels and start new page
- `pp(p1, p2, ...)`: multiple plots/commands in one call 
"""
struct PlotPager
    layout
    kwargs::NamedTuple
    _current_plots::Vector
    nplots::Ref{Int64}
    displayfunc
    function PlotPager(
        layout, kwargs=NamedTuple();
        displayfunc=(plot, nplot)->display(plot),
    )
        return new(layout, kwargs, [], Ref(0), displayfunc)
    end
end

_plots_per_screen(layout::Int) = layout
_plots_per_screen(layout::Tuple{Int, Int}) = layout[1]*layout[2]

function (pp::PlotPager)(p)
    push!(pp._current_plots, p)

    length(pp._current_plots) <= _plots_per_screen(pp.layout) ||
        error("length(_current_plots) > plots_per_screen")
    if length(pp._current_plots) == _plots_per_screen(pp.layout)
        plot_ref = RecipesBase.plot(pp._current_plots...; layout=pp.layout, pp.kwargs...)
        empty!(pp._current_plots)

        pp.nplots[] += 1
        pp.displayfunc(plot_ref, pp.nplots[])
        
    end

    return nothing
end

function (pp::PlotPager)(p::Symbol)
    if p == :skip
        pp(RecipesBase.plot(foreground_color_subplot=:white)) # empty panel
    elseif p == :newpage
        if !isempty(pp._current_plots)
            for _ in 1:(_plots_per_screen(pp.layout) - length(pp._current_plots))
                pp(:skip)
            end
        end
    else
        throw(ArgumentError("unrecognized PlotPager command :$p"))
    end

    return nothing
end

function (pp::PlotPager)(p, ps... )
    pp(p)
    pp.(ps)
    return nothing
end

"""
    DefaultPlotPager()

Dummy version of [`PlotPager`](@ref) that just calls `display` for each plot.
"""
struct DefaultPlotPager
end

function (pp::DefaultPlotPager)(p)
    display(p)
    return nothing
end

function (pp::DefaultPlotPager)(p::Symbol)
    return nothing
end

function (pp::DefaultPlotPager)(p, ps... )
    pp(p)
    pp.(ps)
    return nothing
end

"""
    NoPlotPager()

Dummy version of [`PlotPager`](@ref) that doesn't display the plot
"""
struct NoPlotPager
end

(pp::NoPlotPager)(p...) = nothing

##############################
# Plot recipes
###############################

"""
    plot(output::AbstractOutputWriter, vars::Union{AbstractString, Vector{<:AbstractString}} [, selectargs::NamedTuple] [; coords::AbstractVector])
    heatmap(output::AbstractOutputWriter, var::AbstractString [, selectargs::NamedTuple] [; coords::AbstractVector])
    plot(outputs::Vector{<:AbstractOutputWriter}, vars::Union{AbstractString, Vector{<:AbstractString}} [, selectargs::NamedTuple] [; coords::AbstractVector])

Plot recipe that calls `PB.get_field(output, var)`, and passes on to `plot(fr::FieldRecord, selectargs)`
(see [`RecipesBase.apply_recipe(::Dict{Symbol, Any}, fr::FieldRecord, selectargs::NamedTuple)`](@ref))

Vector-valued `outputs` or `vars` are "broadcast" to create a plot series for each element.
A `labelprefix` (index in `outputs` Vector) is added to identify each plot series produced.

If `var` is of form `<domain>.<name>.<structfield>`, then sets the `structfield` keyword argument
to take a single field from a `struct` Variable.

Optional argument `coords` can be used to supply plot coordinates from Variable in output.
Format is a Vector of Pairs of "coord_name"=>("var_name1", "var_name2", ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>("atm.zmid", "atm.zlower", "atm.zupper")]

"""
RecipesBase.@recipe function f(
    output::AbstractOutputWriter,
    vars::Union{AbstractString, AbstractVector{<:AbstractString}},
    selectargs::NamedTuple=NamedTuple();
    coords=nothing, # NB: PlotRecipes doesn't support type annotation
)

    if isa(vars, AbstractString)
        vars = [vars]
    end

    delete!(plotattributes, :coords)
            
    for var in vars
        RecipesBase.@series begin
             # if var is of form <domain>.<name>.<structfield>, set :structfield to take a single field from struct-valued Var
            var, var_structfield = _split_var_structfield(var)
            if !isnothing(var_structfield)
                structfield := var_structfield
            end
        
            # pass through to plot recipe for FieldRecord
            PB.get_field(output, var), selectargs, coords
        end
    end

    return nothing
end

function _split_var_structfield(var)
    # if var is of form <domain>.<name>.<structfield>, split off <structfield>
    varsplit = split(var, ".")
    if length(varsplit) == 3        
        return join(varsplit[1:2], "."), Symbol(varsplit[3])
    else
        return var, nothing
    end
end

"""
    plot(outputs::Vector{<:AbstractOutputWriter}, vars::Union{AbstractString, Vector{<:AbstractString}}, selectargs::NamedTuple=NamedTuple())

Pass through (ie "broadcast") each element `output` of `outputs` to
`plot(output::AbstractOutputWriter, vars, selectargs)`,
adding a `labelprefix` (index in `outputs` Vector) to identify each plot series produced.
"""
RecipesBase.@recipe function f(
    outputs::Vector{<:AbstractOutputWriter}, 
    vars::Union{AbstractString, Vector{<:AbstractString}},
    selectargs::NamedTuple=NamedTuple()
)
    for (i, output) in enumerate(outputs)
        RecipesBase.@series begin
            labelprefix --> "$i: "
            output, vars, selectargs
        end
    end
 
    return nothing
end


"""
    plot(fr::FieldRecord, selectargs::NamedTuple)
    heatmap(fr::FieldRecord, selectargs::NamedTuple)

Plot recipe to plot a single [`FieldRecord`](@ref)

Calls [`get_array(fr, selectargs)`](@ref) and passes on to `plot(fa::FieldArray)`
(see [`RecipesBase.apply_recipe(::Dict{Symbol, Any}, fa::FieldArray)`](@ref)).

Vector-valued fields in `selectargs` are "broadcasted" (generating a separate plot series for each combination)

Optional argument `coords` can be used to supply plot coordinates from Variable in output.
Format is a Vector of Pairs of "coord_name"=>("var_name1", "var_name2", ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>("atm.zmid", "atm.zlower", "atm.zupper")]

"""
RecipesBase.@recipe function f(@nospecialize(fr::FieldRecord), selectargs::NamedTuple, coords=nothing)

    # broadcast any Vector-valued argument in selectargs
    bcastargs = broadcast_dict([Dict{Symbol, Any}(pairs(selectargs))])

    for sa in bcastargs
        sa_nt = NamedTuple(sa)
        RecipesBase.@series begin
            get_array(fr, sa_nt; coords)
        end
    end
end


"""
    plot(fa::FieldArray; kwargs...)
    heatmap(fa::FieldArray; kwargs...)
    plot(fas::Vector{<:FieldArray}; kwargs...)

Plot recipe that plots a single [`FieldArray`] or Vector of [`FieldArray`]. 

If `fa` has a single dimension, this is suitable for a line-like plot, if two dimensions, a heatmap.

If `fas::Vector` is supplied, this is "broadcast" generating one plot series for each element, 
with the Vector index prepended to `labelprefix` to identify the plot series (unless overridden by `labellist` or `labelattribute`)

# Keywords
- `swap_xy::Bool=false`: true to swap x and y axes 
- `mult_y_coord=1.0`: workaround for bugs in Plots.jl heatmap `yflip` - multiply y coordinate by constant factor.
- `structfield::Union{Symbol, Nothing}=nothing`: use field `structfield` from a struct-valued array.
- `map_values=PB.get_total`: function to apply to y (for a 1D series) or z (for a 2D heatmap etc) before plotting 
- `labelprefix=""`: prefix for plot label.
- `labellist=[]`: list of labels to override defaults
- `labelattribute=nothing`: FieldArray attribute to use as label
"""
RecipesBase.@recipe function f(
    fa::FieldArray;
    swap_xy=false,
    mult_y_coord=1.0, 
    structfield=nothing,
    map_values=PB.get_total,
    labelprefix="",
    labellist=[],
    labelattribute=nothing,
)
    isa(swap_xy, Bool) || throw(ArgumentError("keyword argument 'swap_xy=$swap_xy' is not a Bool"))
    isa(structfield, Union{Nothing, Symbol}) || throw(ArgumentError("keyword argument 'structfield=$structfield' is not a Symbol"))
  
    do_swap_xy = swap_xy  # true to swap x, y
    delete!(plotattributes, :swap_xy)

    function get_attribute(fa, attrb, default)
        if isnothing(fa.attributes)
            return default
        else
            return get(fa.attributes, attrb, default)
        end
    end

    function swap_coords_values(c_values, f_values, c_label, f_label)
        if do_swap_xy
            xlabel --> f_label
            ylabel --> c_label
            return f_values, c_values
        else
            xlabel --> c_label
            ylabel --> f_label
            return c_values, f_values
        end
    end

    #                                          f(i, j)
    function swap_coords_xy(i_values, j_values, f_values, i_label, j_label)
        if do_swap_xy
            xlabel --> i_label
            ylabel --> j_label
            return i_values, j_values, transpose(f_values)
        else
            xlabel --> j_label
            ylabel --> i_label
            return j_values, i_values, f_values
        end
    end


    values = fa.values
    name = fa.name
    # take specified field 
    if !isnothing(structfield)
        values = getproperty.(values, structfield)
        name=name*".$structfield"
    end
    delete!(plotattributes, :structfield)

    # apply transform
    values = map_values.(values)
    delete!(plotattributes, :map_values)

    if length(fa.dims_coords) == 1
        if !isempty(labellist)
            label --> popfirst!(labellist)
        elseif !isnothing(labelattribute)
            label --> string(get_attribute(fa, labelattribute, ""))
        else
            label --> labelprefix*name
        end
        f_label = append_units(get_attribute(fa, :var_name, ""), fa.attributes)
        nameddim, coords_vec = fa.dims_coords[1]

        if length(coords_vec) == 1 || length(coords_vec) > 3
            co = first(coords_vec)            
            return swap_coords_values(
                co.values, values,
                append_units(co.name, co.attributes), f_label,
            )
            
        elseif length(coords_vec) in (2, 3)
            co_lower = coords_vec[end-1]
            co_upper = coords_vec[end]
            c_label = append_units(co_lower.name*", "*co_upper.name, co_lower.attributes)           
            return swap_coords_values(
                create_stepped(co_lower.values, co_upper.values, values)...,
                c_label, f_label,
            )
        else            
            return swap_coords_values(
                1:length(values), values,
                "", f_label,
            )
        end

    elseif length(fa.dims_coords) == 2
        title --> append_units(fa.name, fa.attributes)        
       
        i_values, i_label = build_coords_edges(fa.dims_coords[1]...)
        j_values, j_label = build_coords_edges(fa.dims_coords[2]...)

        # workaround for Plots.jl heatmap: needs x, y to either both be midpoints, or both be edges
        if length(i_values) == size(values, 1) && length(j_values) == size(values, 2)+1
            @warn "$(fa.name) guessing '$i_label' coordinate edges from midpoints assuming uniform spacing"
            i_values = guess_coords_edges(i_values)
        end
        if length(i_values) == size(values, 1)+1 && length(j_values) == size(values, 2)
            @warn "$(fa.name) guessing '$j_label' coordinate edges from midpoints assuming uniform spacing"
            j_values = guess_coords_edges(j_values)
        end

        x_values, y_values, f_values = swap_coords_xy(
            i_values, j_values, values, 
            i_label, j_label
        )

        # heatmap appears to require coordinates in ascending order
        if x_values[end] < x_values[1]
            x_values = reverse(x_values)
            f_values = reverse(f_values, dims=2)
        end

        # workaround for Plots.jl heatmap (for GR backend at least) yflip exposes bugs (offsets etc):
        # multiply y by -1 instead of yflip
        y_values = mult_y_coord .* y_values
        delete!(plotattributes, :mult_y_coord)

        if y_values[end] < y_values[1]
            y_values = reverse(y_values)
            f_values = reverse(f_values, dims=1)
            # yflip --> true
        end

        return x_values, y_values, f_values
    else
        throw(ArgumentError("unknown number of dimensions $(length(fa.dims_coords)), can't plot $fa"))
    end
end

"""
    plot(fas::Vector{<:FieldArray}; labelprefix="")

Pass through (ie "broadcast") each element of `fas` to `plot(fa::FieldArray)`,
generating one plot series for each.  Adds Vector index to `labelprefix`.
"""
RecipesBase.@recipe function f(
    fas::Vector{<:FieldArray};
    labelprefix="",
)
    for (i, fa) in enumerate(fas)
        RecipesBase.@series begin
            labelprefix --> "$i: "*labelprefix
            fa
        end
    end

    return nothing
end

##############################
# Utility functions
##############################

"create d, z for a stepped plot, repeating `data` at `zlower` and `zupper`"
function create_stepped(z1, z2, data)
    z = Float64[]
    d = Float64[]
    
    if (z1[end] > z1[begin]) == (z2[begin] > z1[begin])
        zfirst, zsecond = z1, z2
    else
        zfirst, zsecond = z2, z1
    end

    for (dval, zl, zu) in zip(data, zfirst, zsecond)
        push!(z, zl)
        push!(d, dval)
        push!(z, zu)
        push!(d, dval)
    end
    return z, d
end

"broadcast vector-valued keys into multiple copies of Dict"
function broadcast_dict(dv::Vector{<:Dict})
   
    vector_key = false
    dvout = Dict[]
    for d in dv
        for (k, v) in  d
            if isa(v, Vector)               
                vector_key = true
                for sv in v
                    c = copy(d)
                    c[k] = sv
                    push!(dvout, c)
                end
                break
            end
        end
        if !vector_key
            push!(dvout, d)
        end
    end
  
    if vector_key
        return broadcast_dict(dvout)
    else
        return dvout
    end
end


########################################################
# Helper functions for output visualisation
#########################################################

"""
    build_coords_edges(coords_vec::Vector{FieldArray}) -> Vector{Float64}

Build a vector of coordinate edges (length `n+1``) from `coords_vec`, assuming the PALEO
convention that `coords_vec` contains three elements with 
cell midpoints, lower edges, upper edges each of length `n`, in that order.

Falls back to just returning the first entry in `coords_vec` for other cases.
"""
function build_coords_edges(dim::PB.NamedDimension, coords_vec::Vector{FieldArray})

    if isempty(coords_vec)
        # no coordinate - just return indices
        co_values = collect(1:(dim.size+1))
        co_label = dim.name
    elseif length(coords_vec) == 1 && is_boundary_coordinate(coords_vec[1])
        co = coords_vec[1]
        co_values = vcat(co.values[1, :], co.values[2, end])
        co_label = append_units(co.name, co.attributes)
    elseif length(coords_vec) == 1 || length(coords_vec) > 3
        # 1 coordinate or something we don't understand - take first
        co = first(coords_vec)
        co_values = co.values
        co_label = append_units(co.name, co.attributes)
    elseif length(coords_vec) == 2 && is_boundary_coordinate(coords_vec[2])
        # assume mid, boundary
        co = coords_vec[2]
        co_values = vcat(co.values[1, :], co.values[2, end])
        co_label = append_units(co.name, co.attributes)
    elseif length(coords_vec) in (2, 3)
        # 2 coordinates assume lower, upper edges
        # 3 coordinates assume mid, lower, upper
        co_lower = coords_vec[end-1]
        co_upper = coords_vec[end]
        co_label = append_units(co_lower.name*", "*co_upper.name, co_lower.attributes)
        first(co_lower.values) < first(co_upper.values) ||
            @warn "build_coords_edges: $co_label co_lower is > co_upper - check model grid"            
        if co_lower.values[end] > co_lower.values[1] # ascending order
            co_lower.values[2:end] == co_upper.values[1:end-1] ||
                @warn "build_coords_edges: $co_label lower and upper edges don't match"
            co_values = [co_lower.values; co_upper.values[end]]
        else # descending order
            co_lower.values[1:end-1] == co_upper.values[2:end] ||
                @warn "build_coords_edges: $co_label lower and upper edges don't match"
            co_values = [co_upper.values[1]; co_lower.values]
        end

    end

    return co_values, co_label
end

"guess coordinate edges from midpoints, assuming uniform spacing"
function guess_coords_edges(x_midpoints)
    first_x = x_midpoints[1] - 0.5*(x_midpoints[2] - x_midpoints[1])
    last_x = x_midpoints[end] + 0.5*(x_midpoints[end] - x_midpoints[end-1])
    return [first_x; 0.5.*(x_midpoints[1:end-1] .+ x_midpoints[2:end]); last_x]
end


"""
    append_units(name::AbstractString, attributes) -> "name (units)"

Utility function to append variable units string to a variable name for display.
"""
function append_units(name::AbstractString, attributes::Dict{Symbol, Any})
    units = get(attributes, :units, "")
    if isempty(units)
        return name
    else
        return name*" ($units)"
    end
end

append_units(name::AbstractString, attributes::Nothing) = name
