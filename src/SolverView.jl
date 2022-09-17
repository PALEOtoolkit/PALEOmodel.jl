"""
    SolverView

Provides a view on the whole or some part of the Model for a numerical solver.
Contains `PALEOboxes.VariableAggregator`s for a subset of spatial locations 
(Domains, indices within spatial Domains) and Variables, with:
- ODE paired `stateexplicit` (S) and `stateexplicit_deriv` (dS/dt), where dS/dt = F(S).
- Implicit-ODE paired `total` (T) and `total_deriv` (dT/dt), where dT(S)/dt = F(T(S)) with
  `total` a function of explicit and implicit state Variables `statexplicit` and `state` (S).
- Algebraic `constraint`s (C), where C(S) = 0 with C a function of explicit and implicit state Variables `statexplicit` and `state` (S),

The number of `total` + number of `constraint` Variables must equal the number of implicit `state` Variables.

Optional access methods provide an `ODE/DAE solver` view with composite `statevar` and `statevar_sms`,
where:

- `statevar` is a concatenation of `stateexplicit` and `state` ([`set_statevar!`](@ref))
- `statevar_sms` is a concatenation of `stateexplicit_deriv`, `total_deriv`, `constraints` ([`get_statevar_sms!`](@ref))
"""
mutable struct SolverView{
    T, 
    VA1 <: PB.VariableAggregator, VA2 <: PB.VariableAggregator, VA3 <: PB.VariableAggregator, VA4 <: PB.VariableAggregator, VA5 <: PB.VariableAggregator, VA6 <: PB.VariableAggregator,
    VA7 <: PB.VariableAggregatorNamed
}
    stateexplicit::VA1
    stateexplicit_deriv::VA2
    stateexplicit_norm::Vector{Float64}
    
    total::VA3
    total_deriv::VA4
    total_norm::Vector{Float64}

    constraints::VA5
    constraints_norm::Vector{Float64}

    state::VA6
    state_norm::Vector{Float64}

    hostdep::VA7

    function SolverView(
        etype::Type{T};  # eltype(modeldata)
        stateexplicit::A1, stateexplicit_deriv::A2, total::A3, total_deriv::A4, constraints::A5, state::A6, hostdep::A7
    ) where {T, A1, A2, A3, A4, A5, A6, A7}
        return new{T, A1, A2, A3, A4, A5, A6, A7}(
            stateexplicit,
            stateexplicit_deriv,
            Vector{Float64}(),
            total,
            total_deriv,
            Vector{Float64}(),
            constraints,
            Vector{Float64}(),
            state,
            Vector{Float64}(),
            hostdep
        )
    end
end

Base.eltype(::Type{SolverView{T, A1, A2, A3, A4, A5, A6, A7}}) where {T, A1, A2, A3, A4, A5, A6, A7} = T

"compact form"
function Base.show(io::IO, sv::SolverView)
    print(
        io,
        "SolverView{$(eltype(sv)), ...}",
        "(stateexplicit $(length(sv.stateexplicit)), "*
        "total $(length(sv.total)), "*
        "constraints $(length(sv.constraints)), "*
        "state $(length(sv.state)), "*
        "hostdep $(length(sv.hostdep)))"
    )
    return nothing
end
"multiline form"
function Base.show(io::IO, m::MIME"text/plain", sv::SolverView)
    println(io, "SolverView{$(eltype(sv)), ...}:")
    for f in (:stateexplicit, :stateexplicit_deriv, :total, :total_deriv, :constraints, :state, :hostdep)
        print(io, String(f)*":  ")
        va = getfield(sv, f)
        show(io, m, va)
    end
    return nothing
end


"get number of algebraic constraints (0 if problem is an ODE)"
function num_algebraic_constraints(sv::SolverView)
    return length(sv.constraints)
end

"get number of implicit total variables with time derivatives"
function num_total(sv::SolverView)
    return length(sv.total)
end

"""
    set_statevar!(sv::SolverView, u)

Set combined stateexplicit, state variables from u
"""
function set_statevar!(sv::SolverView, u::AbstractVector)
   
    l_ts = copyto!(sv.stateexplicit, u, sof=1)
    copyto!(sv.state, u, sof=l_ts+1)      
   
    return nothing
end

"statevar += a*u"
function add_statevar!(sv::SolverView, a, u::AbstractVector)
   
    l_ts = length(sv.stateexplicit)   
    PB.add_data!(sv.stateexplicit, a, view(u, 1:l_ts))
    PB.add_data!(sv.state, a, view(u, (l_ts+1):length(u)))      
   
    return nothing
end

function get_statevar!(u, sv::SolverView)
    l_ts = copyto!(u, sv.stateexplicit, dof=1)
    copyto!(u, sv.state, dof=1+l_ts)
    return nothing
end



function state_vars_isdifferential(sv::SolverView)
    isdifferential = trues(length(sv.stateexplicit) + length(sv.total) + length(sv.constraints))
    isdifferential[end-length(sv.constraints)+1:end] .= false
    return isdifferential
end

"""
    get_statevar_sms!(du, sv::SolverView)

Get combined derivatives and constraints, eg for an ODE solver
"""
function get_statevar_sms!(du::AbstractVector, sv::SolverView)
    l_ts = copyto!(du, sv.stateexplicit_deriv, dof=1)

    # TODO get total_deriv  
    # only used for initialisation, doesn't really make sense for an ODE solver ??
    l_ti = copyto!(du, sv.total_deriv, dof=l_ts+1)

    # add constraints to derivative (for ODE solvers that handle this using mass_matrix = 0)
    copyto!(du, sv.constraints, dof=l_ts+l_ti+1)

    return nothing
end

function get_statevar_sms(sv::SolverView)
    du = Vector{eltype(sv)}(undef, length(sv.stateexplicit_deriv) + length(sv.total_deriv) + length(sv.constraints))
    
    get_statevar_sms!(du, sv)

    return du
end

function get_statevar(sv::SolverView)
    return vcat(PB.get_data(sv.stateexplicit),
                PB.get_data(sv.state))
end

function PB.get_statevar(sv::SolverView) 
    Base.depwarn("PB.get_statevar is deprecated, use PALEOmodel.get_statevar instead", :get_statevar_norm, force=true)
    return get_statevar(sv::SolverView)
end

function get_statevar_norm(sv::SolverView)
    return vcat(sv.stateexplicit_norm,
                sv.state_norm)
end

function PB.get_statevar_norm(sv::SolverView) 
    Base.depwarn("PB.get_statevar_norm is deprecated, use PALEOmodel.get_statevar_norm instead", :get_statevar_norm, force=true)
    return get_statevar_norm(sv::SolverView)
end

function get_statevar_sms_norm(sv::SolverView)
    # use stateexplicit and total as estimates of the time derivative normalisation value
    return vcat(sv.stateexplicit_norm, sv.total_norm, sv.constraints_norm)
end

"copy norm values from state variable etc data"
function copy_norm!(sv::SolverView)
    sv.stateexplicit_norm   = PB.value_ad.(PB.get_data(sv.stateexplicit))
    sv.state_norm           = PB.value_ad.(PB.get_data(sv.state))
    sv.total_norm           = PB.value_ad.(PB.get_data(sv.total))
    sv.constraints_norm     = PB.value_ad.(PB.get_data(sv.constraints))
    return nothing
end

"copy norm values back into state variable etc data"
function uncopy_norm!(sv::SolverView)
    copyto!(sv.stateexplicit, sv.stateexplicit_norm)
    copyto!(sv.state, sv.state_norm)
    copyto!(sv.total, sv.total_norm)
    copyto!(sv.constraints, sv.constraints_norm)
   
    return nothing
end

"""
    set_tforce!(sv::SolverView, t)

Set `global.tforce` model time (if defined) from t.
"""
set_tforce!(sv::SolverView, t) = PB.set_values!(sv.hostdep, Val(:global), Val(:tforce), t; allow_missing=true)


"""
    create_solver_view(model, modeldata; [verbose=true]) 
        -> SolverView
    create_solver_view(model, modeldata, cellranges; [verbose=false], [indices_from_cellranges=true])
        -> SolverView

Create a [`SolverView`](@ref) for the entire model, or for a subset of Model Variables defined by
the Domains and operatorIDs of `cellranges`. 

# Keywords
- `indices_from_cellranges=true`: true to restrict to the index ranges from `cellranges`, false to just use `cellranges` to define Domains
and take the whole of each Domain.
- `hostdep_all=true`: true to include host dependent not-state Variables from all Domains
- `reallocate_hostdep_eltype=Float64`: a data type to reallocate `hostdep` Variables eg to replace any
AD types.
"""
create_solver_view(
    model, modeldata::PB.AbstractModelData;
    verbose=true,
) = create_solver_view(
        model, modeldata, modeldata.cellranges_all,
        indices_from_cellranges=false, verbose=verbose,
    )

function create_solver_view(
    model, modeldata::PB.AbstractModelData, cellranges;
    verbose=false,
    indices_from_cellranges=true,
    exclude_var_nameroots=[],
    hostdep_all=true,
    reallocate_hostdep_eltype=Float64,
)
 
    io = IOBuffer() # only displayed if verbose==true
    println(io, "create_solver_view:")
    
    check_domains = [cr.domain for cr in cellranges]
    length(check_domains) == length(unique(check_domains)) ||
        throw(ArgumentError("create_solver_view: cellranges contain duplicate Domains"))

    stateexplicit, stateexplicit_deriv, stateexplicit_cr = PB.VariableDomain[], PB.VariableDomain[], []
    total, total_deriv, total_cr = PB.VariableDomain[], PB.VariableDomain[], []
    constraint, constraint_cr = PB.VariableDomain[], []
    state, state_cr = PB.VariableDomain[], []
    hostdep = PB.VariableDomain[]

    report = []
    for cr in cellranges
        dom = cr.domain              
        crreport = Any[dom.name, cr.operatorID]
        for (vfunction, var_vec, cr_vec, var_deriv_vec, deriv_suffix) in [
            (PB.VF_StateExplicit,  stateexplicit,  stateexplicit_cr,   stateexplicit_deriv,    "_sms"),
            (PB.VF_Total,          total,          total_cr,           total_deriv,            "_sms"),
            (PB.VF_Constraint,     constraint,     constraint_cr,      nothing,                ""),
            (PB.VF_State,          state,          state_cr,           nothing,                ""),
            (PB.VF_Undefined,      hostdep,        nothing,            nothing,                ""),
        ]

            (vars, vars_deriv) = 
                PB.get_host_variables(
                    dom, vfunction,
                    match_deriv_suffix=deriv_suffix,
                    operatorID=cr.operatorID,
                    exclude_var_nameroots=exclude_var_nameroots
                )

            if isnothing(var_deriv_vec)
                # sort by name: VF_Constraint and VF_State are unpaired in general, but in some cases will have a convenient ordering based on name
                sort!(vars; by=v->PB.fullname(v))
                append!(var_vec, vars)
            else
                append!(var_vec, vars)
                append!(var_deriv_vec, vars_deriv)
            end
            if !isnothing(cr_vec)
                append!(cr_vec, [indices_from_cellranges ? cr : nothing for v in vars])
            end

            push!(crreport, length(vars))
        end
        push!(report, crreport)    
    end    
    push!(report, ["Total", "-", length(stateexplicit), length(total), length(constraint), length(state), length(hostdep)])

    n_state_vars = length(stateexplicit) + length(state)
    n_equations = length(stateexplicit) + length(total) + length(constraint)

    if verbose        
        colnames = ["Domain", "operatorID", "VF_StateExplicit", "VF_Total", "VF_Constraint", "VF_State", "VF_Undefined"]
        colwidths = [24,       12,          18,                 18,         18,              18,         18]
        sep = rpad("", sum(colwidths), "-")
        println(io, "    host-dependent Variables:")
        println(io, "    ", sep)
        println(io, "    ", join(rpad.(colnames, colwidths)))
        for r in report
            r == last(report) && println(io, "    ", sep)
            println(io, "    ", join(rpad.(string.(r), colwidths)))
        end
        println(io, "    ", sep)
        println(io, "  n_state_vars $n_state_vars  (stateexplicit $(length(stateexplicit)) "*
            "+ state $(length(state)))")
        println(io, "  n_equations $n_equations  (stateexplicit $(length(stateexplicit)) "*
            "+ total $(length(total)) + constraint $(length(constraint)))")             
    end

    n_state_vars == n_equations || 
        error("create_solver_view: n_state_vars != n_equations")

    if hostdep_all
        println(io, "  including all host-dependent non-state Variables")
        empty!(hostdep)
        for dom in model.domains
            dv, _ = PB.get_host_variables(dom, PB.VF_Undefined)
            append!(hostdep, dv )
        end
    end
    println(io, "  host-dependent non-state Variables (:vfunction PB.VF_Undefined): $([PB.fullname(v) for v in hostdep])")

    if !isnothing(reallocate_hostdep_eltype)
         # If requested, change data type eg to remove AD type    
        reallocated_variables = PB.reallocate_variables!(hostdep, modeldata, reallocate_hostdep_eltype)
        if !isempty(reallocated_variables)
            println(io, "  reallocating host-dependent Variables to eltype $reallocate_hostdep_eltype:")        
            for (v, old_eltype) in reallocated_variables
                println(io, "    $(PB.fullname(v)) data $old_eltype -> $reallocate_hostdep_eltype")
            end
        end
    end

    verbose && @info String(take!(io))  

    sv = SolverView(
        eltype(modeldata),
        stateexplicit = PB.VariableAggregator(stateexplicit, stateexplicit_cr, modeldata),
        stateexplicit_deriv = PB.VariableAggregator(stateexplicit_deriv, stateexplicit_cr, modeldata),
        total = PB.VariableAggregator(total, total_cr, modeldata),
        total_deriv = PB.VariableAggregator(total_deriv, total_cr, modeldata),
        constraints = PB.VariableAggregator(constraint, constraint_cr, modeldata),
        state = PB.VariableAggregator(state, state_cr, modeldata),
        hostdep = PB.VariableAggregatorNamed(hostdep, modeldata)
    )

    return sv
end

"""
    set_default_solver_view!(model, modeldata)

(Optional, used to set `modeldata.solver_view_all` to a [`SolverView`](@ref)) for the whole
model, and set `modeldata.hostdep_data` to any non-state-variable host dependent Variables)

`reallocate_hostdep_eltype` a data type to reallocate `hostdep_data` eg to replace any
AD types.
"""
function set_default_solver_view!(
    model::PB.Model, modeldata::PB.AbstractModelData,
)
    PB.check_modeldata(model, modeldata)  

    # create a default SolverView for the entire model (from modeldata.cellranges_all)
    sv = create_solver_view(model, modeldata)
   
    modeldata.solver_view_all = sv
    
    return nothing
end
