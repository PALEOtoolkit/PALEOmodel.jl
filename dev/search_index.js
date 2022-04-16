var documenterSearchIndex = {"docs":
[{"location":"References/#References","page":"References","title":"References","text":"","category":"section"},{"location":"References/","page":"References","title":"References","text":"","category":"page"},{"location":"indexpage/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"indexpage/","page":"Index","title":"Index","text":"","category":"page"},{"location":"MathematicalFormulation/#Mathematical-formulation-of-the-reaction-transport-problem","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"","category":"section"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"The PALEO models define various special cases of a general DAE problem (these can be combined, providing the number  of implicit state variables S_impl is equal to the number of algebraic constraints G plus the number of total variables U):","category":"page"},{"location":"MathematicalFormulation/#Explicit-ODE","page":"Mathematical formulation of the reaction-transport problem","title":"Explicit ODE","text":"","category":"section"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"The time derivative of explicit state variables S_explicit (a subset of all state variables S_all) are an explicit function of time t:","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"fracdS_explicitdt = F(S_all t)","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"where explicit state variables S_explicit are identified by PALEO attribute :vfunction = PALEOboxes.VF_StateExplicit and paired time derivatives F by :vfunction = PALEOboxes.VF_Deriv along with the naming convention <statevarname>, <statevarname>_sms.","category":"page"},{"location":"MathematicalFormulation/#Algebraic-constraints","page":"Mathematical formulation of the reaction-transport problem","title":"Algebraic constraints","text":"","category":"section"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"State variables S_impl (a subset of all state variables S_all) are defined by algebraic constraints G:","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"0 = G(S_all t)","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"where implicit state variables S_impl are identified by PALEO attribute :vfunction = PALEOboxes.VF_State and algebraic constaints G by :vfunction = PALEOboxes.VF_Constraint (these are not paired).","category":"page"},{"location":"MathematicalFormulation/#ODE-with-variable-substitution","page":"Mathematical formulation of the reaction-transport problem","title":"ODE with variable substitution","text":"","category":"section"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"State variables S_impl (a subset of all state variables S_all) are defined the time evolution of total variables U(S_all) (this case is common in biogeochemistry where the total variables U represent conserved chemical elements, and the state variables eg primary species):","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"fracdU(S_all)dt = F(U(S_all) t)","category":"page"},{"location":"MathematicalFormulation/","page":"Mathematical formulation of the reaction-transport problem","title":"Mathematical formulation of the reaction-transport problem","text":"where total variables U are identified by PALEO attribute :vfunction = PALEOboxes.VF_Total and paired time derivatives F by :vfunction = PALEOboxes.VF_Deriv along with the naming convention <totalvarname>, <totalvarname>_sms, and implicit state variables S_impl are identified by PALEO attribute :vfunction = PALEOboxes.VF_State.","category":"page"},{"location":"PALEOmodel/#PALEOmodel","page":"PALEOmodel","title":"PALEOmodel","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"Run","category":"page"},{"location":"PALEOmodel/#PALEOmodel.Run","page":"PALEOmodel","title":"PALEOmodel.Run","text":"Run\n\nContainer for model and output.\n\nFields\n\nmodel::Union{Nothing, PB.Model}: The model instance.\noutput::Union{Nothing, AbstractOutputWriter}: model output\n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#Create-and-initialize","page":"PALEOmodel","title":"Create and initialize","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"initialize!","category":"page"},{"location":"PALEOmodel/#PALEOmodel.initialize!","page":"PALEOmodel","title":"PALEOmodel.initialize!","text":"initialize!(model::PB.Model; kwargs...) -> (initial_state, modeldata)\n\nInitialize model and return initial_state Vector and modeldata struct\n\nKeywords:\n\neltype::Type=Float64: default data type to use for model arrays\neltypemap=Dict{String, DataType}: Dict of data types to look up Variable :datatype attribute\npickup_output=nothing: OutputWriter with pickup data to initialise from\nthreadsafe=false: true to create thread safe Atomic Variables where :atomic attribute = true\nmethod_barrier=nothing: thread barrier to add to dispatch lists if threadsafe==true\nexpect_hostdep_varnames=[\"global.tforce\"]: non-state-Variable host-dependent Variable names expected\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Integrate","page":"PALEOmodel","title":"Integrate","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.ODE","category":"page"},{"location":"PALEOmodel/#DifferentialEquations-solvers","page":"PALEOmodel","title":"DifferentialEquations solvers","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"integrate\nintegrateDAE\nintegrateForwardDiff\nintegrateDAEForwardDiff","category":"page"},{"location":"PALEOmodel/#PALEOmodel.ODE.integrate","page":"PALEOmodel","title":"PALEOmodel.ODE.integrate","text":"integrate(run, initial_state, modeldata, tspan [; kwargs...] )\n\nIntegrate run.model as an ODE or as a DAE with constant mass matrix, and write to outputwriter\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntspan:  (tstart, tstop) integration start and stop times\n\nKeywords\n\nalg=Sundials.CVODE_BDF():  ODE algorithm to use\noutputwriter=run.output: PALEOmodel.AbstractOutputWriter instance to hold output\nsolvekwargs=NamedTuple(): NamedTuple of keyword arguments passed through to DifferentialEquations.jl solve  (eg to set abstol, reltol, saveat,  see https://diffeq.sciml.ai/dev/basics/common_solver_opts/)\njac_ad=:NoJacobian: Jacobian to generate and use (:NoJacobian, :ForwardDiffSparse, :ForwardDiff)\njac_ad_t_sparsity=tspan[1]: model time at which to calculate Jacobian sparsity pattern\nsteadystate=false: true to use DifferentialEquations.jl SteadyStateProblem (not recommended, see PALEOmodel.SteadyState.steadystate).\nBLAS_num_threads=1: number of LinearAlgebra.BLAS threads to use\ninit_logger=Logging.NullLogger(): default value omits logging from (re)initialization to generate Jacobian modeldata, Logging.CurrentLogger() to include\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODE.integrateDAE","page":"PALEOmodel","title":"PALEOmodel.ODE.integrateDAE","text":"integrateDAE(run, initial_state, modeldata, tspan;alg=IDA())\n\nIntegrate run.model as a DAE and copy output to outputwriter.  Arguments as integrate.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODE.integrateForwardDiff","page":"PALEOmodel","title":"PALEOmodel.ODE.integrateForwardDiff","text":"integrate with argument defaults to  use ForwardDiff AD Jacobian\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODE.integrateDAEForwardDiff","page":"PALEOmodel","title":"PALEOmodel.ODE.integrateDAEForwardDiff","text":"integrateDAE with argument defaults to use ForwardDiff AD Jacobian\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Fixed-timestep-solvers","page":"PALEOmodel","title":"Fixed timestep solvers","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.ODEfixed","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"integrateEuler\nintegrateSplitEuler\nintegrateEulerthreads\nintegrateSplitEulerthreads","category":"page"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateEuler","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateEuler","text":"integrateEuler(run, initial_state, modeldata, tspan, Δt; [,outputwriter])\n\nIntegrate run.model using first-order Euler with fixed timestep.\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntspan:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times\nΔt: (yr) fixed timestep \n[outputwriter: PALEOmodel.AbstractOutputWriter instance to write model output to]\n[report_interval: number of timesteps between progress update to console]\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateSplitEuler","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateSplitEuler","text":"integrateSplitEuler(run, initial_state, modeldata, tspan, Δt_outer, n_inner;\n                        cellranges_outer, cellranges_inner,\n                        [,outputwriter])\n\nIntegrate run.model representing:\n\nfracdSdt =  f_outer(t S) + f_inner(t S)\n\nusing split first-order Euler with Δt_outer for f_outer and a shorter timestep Δt_outer/n_inner for f_inner.\n\nf_outer is defined by calling PALEOboxes.do_deriv with cellranges_outer hence corresponds to those Reactions with operatorID of cellranges_outer. f_inner is defined by calling PALEOboxes.do_deriv with cellranges_inner hence corresponds to those Reactions with operatorID of cellranges_inner.\n\nNB: the combined time derivative is written to outputwriter.\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntspan:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times\nΔt_outer: (yr) fixed outer timestep \nn_inner: number of inner timesteps per outer timestep\n\nKeywords\n\ncellranges_outer: Vector of CellRange with operatorID defining f_outer.\ncellranges_inner: Vector of CellRange with operatorID defining f_inner.\n[outputwriter: PALEOmodel.AbstractOutputWriter instance to write model output to]\n[report_interval: number of outer timesteps between progress update to console]\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateEulerthreads","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateEulerthreads","text":"integrateEulerthreads(run, initial_state, modeldata, cellranges, tspan, Δt;\n    outputwriter=run.output, report_interval=1000)\n\nIntegrate run.model using first-order Euler with fixed timestep Δt, with tiling over multiple threads.\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ncellranges::Vector{Vector{AbstractCellRange}}: Vector of Vector-of-cellranges, one per thread (so length(cellranges) == Threads.nthreads).\ntspan:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times\nΔt: (yr) fixed outer timestep \n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateSplitEulerthreads","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateSplitEulerthreads","text":"integrateSplitEulerthreads(run, initial_state, modeldata, tspan, Δt_outer, n_inner::Int; \n                            cellranges_outer, cellranges_inner, [,outputwriter] [, report_interval])\n\nIntegrate run.model using split first-order Euler with Δt_outer for f_outer and a shorter timestep Δt_outer/n_inner for f_inner.\n\nf_outer is defined by calling PALEOboxes.do_deriv with cellrange_outer hence corresponds to those Reactions with operatorID of cellrange_outer.  f_inner is defined by calling PALEOboxes.do_deriv with cellrange_inner hence corresponds to those Reactions with operatorID of cellrange_inner.\n\nUses Threads.nthreads threads and tiling described by cellranges_inner and cellranges_outer (each a Vector of Vector{AbstractCellRange}, one per thread).\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntspan:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times\nΔt_outer: (yr) fixed outer timestep \nn_inner: number of inner timesteps per outer timestep (0 for non-split solver)\ncellranges_outer::Vector{Vector{AbstractCellRange}}: Vector of list-of-cellranges, one list per thread (so length(cellranges) == Threads.nthreads), with operatorID defining f_outer.\ncellranges_inner::Vector{Vector{AbstractCellRange}}: As cellranges_outer, with operatorID defining f_inner.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.ODELocalIMEX","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"integrateLocalIMEXEuler","category":"page"},{"location":"PALEOmodel/#PALEOmodel.ODELocalIMEX.integrateLocalIMEXEuler","page":"PALEOmodel","title":"PALEOmodel.ODELocalIMEX.integrateLocalIMEXEuler","text":"integrateLocalIMEXEuler(run, initial_state, modeldata, tspan, Δt_outer [; kwargs...])\n\nIntegrate run.model representing:\n\nfracdSdt =  f_outer(t S) + f_inner(t S)\n\nusing first-order Euler with Δt_outer for f_outer and implicit first-order Euler for f_inner, where f_inner is local (within-cell, ie no transport), for a single Domain, and uses only StateExplicit and Deriv variables.\n\nf_outer is defined by calling PALEOboxes.do_deriv with cellranges_outer hence corresponds to those Reactions with operatorID of cellranges_outer.  f_inner is defined by calling PALEOboxes.do_deriv with cellrange_inner hence corresponds to those Reactions with operatorID of cellrange_inner.\n\nNB: the combined time derivative is written to outputwriter.\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntspan:  (tstart, toutput1, toutput2, ..., tstop) integration start, output, stop times\nΔt_outer: (yr) fixed timestep\n\nKeywords\n\ncellranges_outer: Vector of CellRange with operatorID defining f_outer.\ncellrange_inner: A single CellRange with operatorID defining f_inner.\nexclude_var_nameroots: State variables that are modified by Reactions in cellrange_inner, but not needed to find implicit solution (ie reaction rates etc don't depend on them).\n[outputwriter=run.output: PALEOmodel.AbstractOutputWriter instance to write model output to]\n[report_interval=1000: number of outer timesteps between progress update to console]\n[Lnorm_inf_max=1e-3:  normalized error tolerance for implicit solution]\n[niter_max=10]: maximum number of Newton iterations\n[request_adchunksize=4]: request ForwardDiff AD chunk size (will be restricted to an upper limit)\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Fixed-timestep-wrappers","page":"PALEOmodel","title":"Fixed timestep wrappers","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.ODEfixed","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"integrateFixed\nintegrateFixedthreads","category":"page"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateFixed","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateFixed","text":"integrateFixed(run, initial_state, modeldata, tspan, Δt_outer;\n            timesteppers, outputwriter=run.output, report_interval=1000)\n\nFixed timestep integration, with time step implemented by timesteppers,\n\n`timesteppers = [ [(timestep_function, cellranges, timestep_ctxt), ...], ...]`\n\nWhere timestep_function(model, modeldata, cellranges, timestep_ctxt, touter, Δt, barrier)\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.ODEfixed.integrateFixedthreads","page":"PALEOmodel","title":"PALEOmodel.ODEfixed.integrateFixedthreads","text":"integrateFixedthreads(run, initial_state, modeldata, tspan, Δt_outer;\n            timesteppers, outputwriter=run.output, report_interval=1000)\n\nFixed timestep integration using Threads.nthreads() threads, with time step implemented by timesteppers,\n\n`timesteppers = [ [(timestep_function, cellranges, timestep_ctxt), ... (1 per thread)], ...]`\n\nWhere timestep_function(model, modeldata, cellranges, timestep_ctxt, touter, Δt, barrier).\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Steady-state-solvers-(Julia-NLsolve-based)","page":"PALEOmodel","title":"Steady-state solvers (Julia NLsolve based)","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.SteadyState","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"steadystate\nsteadystateForwardDiff\nsteadystate_ptc\nsteadystate_ptcForwardDiff","category":"page"},{"location":"PALEOmodel/#PALEOmodel.SteadyState.steadystate","page":"PALEOmodel","title":"PALEOmodel.SteadyState.steadystate","text":"steadystate(run, initial_state, modeldata, tss [; kwargs...] )\n\nFind steady-state solution (using NLsolve.jl package)  and write to outputwriter (two records are written, for initial_state and the steady-state solution).\n\nArguments\n\nrun::Run: struct with model::PB.Model to integrate and output field\ninitial_state::AbstractVector: initial state vector\nmodeldata::Modeldata: ModelData struct with appropriate element type for forward model\ntss:  (yr) model tforce time for steady state solution\n\nOptional Keywords\n\noutputwriter=run.output: PALEOmodel.AbstractOutputWriter instance to hold output\ninitial_time=-1.0:  tmodel to write for first output record\nsolvekwargs=NamedTuple(): NamedTuple of keyword arguments passed through to NLsolve.jl  (eg to set method, ftol, iteration, show_trace, store_trace).\njac_ad: :NoJacobian, :ForwardDiffSparse, :ForwardDiff\nuse_norm=false: true to normalize state variables using PALEO norm_value\nBLAS_num_threads=1: number of LinearAlgebra.BLAS threads to use\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.SteadyState.steadystateForwardDiff","page":"PALEOmodel","title":"PALEOmodel.SteadyState.steadystateForwardDiff","text":"steadystate with argument defaults to  use ForwardDiff AD Jacobian\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.SteadyState.steadystate_ptc","page":"PALEOmodel","title":"PALEOmodel.SteadyState.steadystate_ptc","text":"steadystate_ptc(run, initial_state, modeldata, tss, deltat_initial, tss_max [; kwargs...])\n\nFind steady-state solution and write to outputwriter, using naive pseudo-transient-continuation with first order implicit Euler pseudo-timesteps from tss to tss_max and NLsolve.jl as the non-linear solver.\n\nEach pseudo-timestep solves the nonlinear system S(t+Δt) = S(t) + Δt dS/dt(t+Δt) for S(t+Δt), using a variant of Newton's method.\n\nInitial pseudo-timestep Δt is deltat_initial, this is multiplied by deltat_fac for the next iteration until pseudo-time tss_max is reached. If an iteration fails, Δt is divided by deltat_fac and the iteration retried.\n\nNB: this is a very naive initial implementation, there is currently no reliable error control to adapt pseudo-timesteps  to the rate of convergence, so requires some trial-and-error to set an appropiate deltat_fac for each problem.\n\nKeywords\n\ndeltat_fac=2.0: factor to increase pseudo-timestep on success\noutputwriter=run.output: output destination\nsolvekwargs=NamedTuple(): arguments to pass through to NLsolve\njac_ad=:NoJacobian: AD Jacobian to use\nrequest_adchunksize=10: ForwardDiff chunk size to request.\njac_cellranges=modeldata.cellranges_all: CellRanges to use for Jacobian calculation (eg to restrict to an approximate Jacobian)\nenforce_noneg=false: fail pseudo-timesteps that generate negative values for state variables.\nuse_norm=false: true to apply PALEO norm_value to state variables\nverbose=false: true to detailed output\nBLAS_num_threads=1: restrict threads used by Julia BLAS (likely irrelevant if using sparse Jacobian?)\n\nSee steadystate for more details of arguments.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.SteadyState.steadystate_ptcForwardDiff","page":"PALEOmodel","title":"PALEOmodel.SteadyState.steadystate_ptcForwardDiff","text":"steadystate_ptc with argument defaults to  use ForwardDiff AD Jacobian\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Steady-state-solvers-(Sundials-Kinsol-based):","page":"PALEOmodel","title":"Steady-state solvers (Sundials Kinsol based):","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.SteadyStateKinsol","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"steadystate_ptc","category":"page"},{"location":"PALEOmodel/#PALEOmodel.SteadyStateKinsol.steadystate_ptc","page":"PALEOmodel","title":"PALEOmodel.SteadyStateKinsol.steadystate_ptc","text":"steadystate_ptc(run, initial_state, modeldata, tss, deltat_initial, tss_max; \n    [,deltat_fac=2.0] [,tss_output] [,outputwriter] [,createkwargs] [,solvekwargs]\n    [,jac_cellranges] [, use_directional_ad] [, directional_ad_eltypestomap] [,verbose] [,  BLAS_num_threads] )\n\nFind steady-state solution and write to outputwriter, using naive pseudo-transient-continuation with first order implicit Euler pseudo-timesteps and PALEOmodel.Kinsol as the non-linear solver.\n\nEach pseudo-timestep solves the nonlinear system S(t+Δt) = S(t) + Δt dS/dt(t+Δt) for S(t+Δt), using a variant of Newton's method (preconditioned Newton-Krylov, with the Jacobian as preconditioner)\n\nInitial pseudo-timestep Δt is deltat_initial, this is multiplied by deltat_fac for the next iteration until pseudo-time tss_max is reached. If an iteration fails, Δt is divided by deltat_fac and the iteration retried.\n\nNB: this is a very naive initial implementation, there is currently no reliable error control to adapt pseudo-timesteps  to the rate of convergence, so requires some trial-and-error to set an appropiate deltat_fac for each problem.\n\nSolver PALEOmodel.Kinsol options are set by arguments createkwargs (passed through to PALEOmodel.Kinsol.kin_create) and solvekwargs (passed through to PALEOmodel.Kinsol.kin_solve).\n\nPreconditioner (Jacobian) calculation can be modified by jac_cellranges, to specify a operatorIDs  so use only a subset of Reactions in order to  calculate an approximate Jacobian to use as the preconditioner.\n\nIf use_directional_ad is true, the Jacobian-vector product will be calculated using automatic differentiation (instead of  the default finite difference approximation).  directional_ad_eltypestomap can be used to specify Variable :datatype tags (strings) that should be mapped to the AD directional derivative datatype hence included in the AD directional derivative.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"Kinsol\nKinsol.kin_create\nKinsol.kin_solve","category":"page"},{"location":"PALEOmodel/#PALEOmodel.Kinsol","page":"PALEOmodel","title":"PALEOmodel.Kinsol","text":"Kinsol\n\nMinimal Julia wrapper for the Sundials kinsol nonlinear system solver https://computing.llnl.gov/projects/sundials/kinsol\n\nThis closely follows the native C interface, as documented in the Kinsol manual, with conversion to-from native Julia types.\n\nThe main user-facing functions are Kinsol.kin_create and Kinsol.kin_solve.\n\n\n\n\n\n","category":"module"},{"location":"PALEOmodel/#PALEOmodel.Kinsol.kin_create","page":"PALEOmodel","title":"PALEOmodel.Kinsol.kin_create","text":"kin_create(f, y0 [; kwargs...]) -> kin\n\nCreate and return a kinsol solver context kin, which can then be passed to kin_solve\n\nArguments\n\nf: Function of form f(fy::Vector{Float64}, y::Vector{Float64}, userdata)\ny0::Vector template Vector of initial values (used only to define problem dimension)\n\nKeywords\n\nuserdata: optional user data, passed through to f etc.\nlinear_solver: linear solver to use (only partially implemented, supports :Dense, :Band, :FGMRES)\npsolvefun: optional preconditioner solver function (for :FGMRES)\npsetupfun: optional preconditioner setup function\njvfun: optional Jacobian*vector  function (for :FGMRES)\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.Kinsol.kin_solve","page":"PALEOmodel","title":"PALEOmodel.Kinsol.kin_solve","text":"kin_solve(\n    kin, y0::Vector;\n    [strategy] [, fnormtol] [, mxiter] [, print_level] [,y_scale] [, f_scale] [, noInitSetup]\n) -> (y, kin_stats)\n\nSolve nonlinear system using kinsol solver context kin (created by kin_create) and initial conditions y0. Returns solution y and solver statistics kinstats. kinstats.returnflag indicates success/failure.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Field-Array","page":"PALEOmodel","title":"Field Array","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"FieldArray provides a generic array type with named dimensions PALEOboxes.NamedDimension and optional coordinates PALEOboxes.FixedCoord for processing of model output.","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"FieldArray\nget_array","category":"page"},{"location":"PALEOmodel/#PALEOmodel.FieldArray","page":"PALEOmodel","title":"PALEOmodel.FieldArray","text":"FieldArray\n\nA generic xarray-like or  IRIS-like  Array with named dimensions and optional coordinates.\n\nNB: this aims to be simple and generic, not efficient !!! Intended for representing model output, not for numerically-intensive calculations.\n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#PALEOmodel.get_array","page":"PALEOmodel","title":"PALEOmodel.get_array","text":"get_array(obj, ...) -> FieldArray\n\nGet FieldArray from PALEO object obj\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"FieldRecord","category":"page"},{"location":"PALEOmodel/#PALEOmodel.FieldRecord","page":"PALEOmodel","title":"PALEOmodel.FieldRecord","text":"FieldRecord{D <: AbstractData, S <: AbstractSpace, ...}\n\nA series of records each containing a PALEOboxes.Field.\n\nImplementation\n\nFields with array values are stored in records as a Vector of arrays. Fields with single values (fieldsingleelement true) are stored as a Vector of eltype(Field.values). \n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#Output","page":"PALEOmodel","title":"Output","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"OutputWriters","category":"page"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters","page":"PALEOmodel","title":"PALEOmodel.OutputWriters","text":"OutputWriters\n\nData structures and methods to hold and manage model output.\n\n\n\n\n\n","category":"module"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel.OutputWriters","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"OutputMemory\nOutputMemoryDomain\nsave_jld2\nload_jld2!\ninitialize!\nadd_record!","category":"page"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.OutputMemory","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.OutputMemory","text":"OutputMemory\n\nIn-memory model output, organized by model Domains.\n\nField domains::Dict{String, OutputMemoryDomain} contains per-Domain model output.\n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.OutputMemoryDomain","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.OutputMemoryDomain","text":"OutputMemoryDomain\n\nIn-memory model output, for one model Domain.\n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.save_jld2","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.save_jld2","text":"save_jld2(output::OutputMemory, filename)\n\nSave to filename in JLD2 format (NB: filename must either have no extension or have extension .jld2)\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.load_jld2!","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.load_jld2!","text":"load_jld2!(output::OutputMemory, filename)\n\nLoad from filename in JLD2 format, replacing any existing content in output. (NB: filename must either have no extension or have extension .jld2).\n\nExample\n\njulia> output = PALEOmodel.OutputWriters.load_jld2!(PALEOmodel.OutputWriters.OutputMemory(), \"savedoutput.jld2\")\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.initialize!","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.initialize!","text":"initialize!(output::OutputMemory, model, modeldata, nrecords [;rec_coord=:tmodel])\n\nInitialize from a PALEOboxes::Model, reserving memory for an assumed output dataset of nrecords.\n\nThe default for rec_coord is :tmodel, for a sequence of records following the time evolution of the model.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#PALEOmodel.OutputWriters.add_record!","page":"PALEOmodel","title":"PALEOmodel.OutputWriters.add_record!","text":"add_record!(output::OutputMemory, model, modeldata, rec_coord)\n\nAdd an output record for current state of model at record coordinate rec_coord. The usual case (set by initialize!) is that the record coordinate is model time tmodel.\n\n\n\n\n\n","category":"function"},{"location":"PALEOmodel/#Plot-output","page":"PALEOmodel","title":"Plot output","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"Plot recipes for PALEOboxes.FieldArray TODO","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"PlotPager\nPlot.test_heatmap_edges","category":"page"},{"location":"PALEOmodel/#PALEOmodel.PlotPager","page":"PALEOmodel","title":"PALEOmodel.PlotPager","text":"PlotPager(layout, [, kwargs=NamedTuple()])\n\nAccumulate plots into subplots.\n\nlayout is supplied to Plots.jl layout keyword, may be an Int or a Tuple (ny, nx), see https://docs.juliaplots.org/latest/\n\nOptional kwargs provides keyword arguments supplied to plot (eg (legendbackgroundcolor=nothing, ) to set all subplot legends to transparent backgrounds)\n\nUsage\n\njulia> pp = PlotPager((2,2))  # 4 panels per screen (2 down, 2 across)\njulia> pp(plot(1:3))  # Accumulate\njulia> pp(:skip, plot(1:4), plot(1:5), plot(1:6))  # add multiple panels in one command\njulia> pp(:newpage) # flush any partial screen and start new page (NB: always add this at end of a sequence!)\n\nCommands\n\npp(p::AbstractPlot): accumulate plot p\npp(:skip): leave a blank panel\npp(:newpage): fill with blank panels and start new page\npp(p1, p2, ...): multiple plots/commands in one call \n\n\n\n\n\n","category":"type"},{"location":"PALEOmodel/#Analyze-reaction-network","page":"PALEOmodel","title":"Analyze reaction network","text":"","category":"section"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"PALEOmodel/","page":"PALEOmodel","title":"PALEOmodel","text":"ReactionNetwork","category":"page"},{"location":"PALEOmodel/#PALEOmodel.ReactionNetwork","page":"PALEOmodel","title":"PALEOmodel.ReactionNetwork","text":"ReactionNetwork\n\nFunctions to analyize a PALEOboxes.Model that contains a reaction network\n\n\n\n\n\n","category":"module"},{"location":"#PALEOmodel.jl","page":"PALEOmodel.jl","title":"PALEOmodel.jl","text":"","category":"section"},{"location":"","page":"PALEOmodel.jl","title":"PALEOmodel.jl","text":"CurrentModule = PALEOmodel","category":"page"},{"location":"","page":"PALEOmodel.jl","title":"PALEOmodel.jl","text":"The PALEOmodel Julia package provides modules to create and solve a standalone PALEOboxes.Model, and to analyse output interactively from the Julia REPL. It implements:","category":"page"},{"location":"","page":"PALEOmodel.jl","title":"PALEOmodel.jl","text":"Numerical solvers (see Integrate)\nData structures in the OutputWriters submodule, eg  OutputWriters.OutputMemory to hold model output\nOutput plotting (see Plot output).\nRun (a container for a PALEOboxes.Model model and output).","category":"page"},{"location":"","page":"PALEOmodel.jl","title":"PALEOmodel.jl","text":"PALEO documentation follows the recommendations from https://documentation.divio.com/","category":"page"}]
}
