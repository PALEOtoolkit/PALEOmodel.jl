"""
    Kinsol

Minimal Julia wrapper for the Sundials kinsol nonlinear system solver <https://computing.llnl.gov/projects/sundials/kinsol>

This closely follows the native C interface, as documented in the Kinsol manual, with conversion to-from native Julia types.

The main user-facing functions are [`Kinsol.kin_create`](@ref) and [`Kinsol.kin_solve`](@ref).
"""
module Kinsol

import Sundials

# import Infiltrator

###########################################################
# Internals: Julia <-> C wrapper functions
###########################################################

# wrapper with Julia user-supplied functions and data
# this is passed as an opaque pointer into the kinsol C code
mutable struct UserFunctionAndData{F1, F2, F3, F4}
    func::F1
    psetup::F2
    psolve::F3
    jv::F4
    data::Any
end

# Julia adaptor function with C types, passed in to Kinsol C code as a callback
# wraps C types and forwards to the Julia user function
function kinsolfun(
    y::Sundials.N_Vector,
    fy::Sundials.N_Vector, 
    userfun::UserFunctionAndData
)
    userfun.func(convert(Vector, fy), convert(Vector, y), userfun.data)
    return Sundials.KIN_SUCCESS
end

function kinprecsetup(
    u::Sundials.N_Vector,
    uscale::Sundials.N_Vector, 
    fval::Sundials.N_Vector,
    fscale::Sundials.N_Vector,
    userfun::UserFunctionAndData
)
    retval = userfun.psetup(
        convert(Vector, u),
        convert(Vector, uscale), 
        convert(Vector, fval),
        convert(Vector, fscale),
        userfun.data
    )
       
    return Cint(retval)
end

function kinprecsolve(
    u::Sundials.N_Vector,
    uscale::Sundials.N_Vector, 
    fval::Sundials.N_Vector,
    fscale::Sundials.N_Vector,
    v::Sundials.N_Vector,
    userfun::UserFunctionAndData
)
    retval = userfun.psolve(
        convert(Vector, u),
        convert(Vector, uscale), 
        convert(Vector, fval),
        convert(Vector, fscale),
        convert(Vector, v), userfun.data
    )

    return Cint(retval)
end

function kinjactimesvec(
    v::Sundials.N_Vector,
    Jv::Sundials.N_Vector, 
    u::Sundials.N_Vector,
    new_u::Ptr{Cint},
    userfun::UserFunctionAndData
) 
    retval = userfun.jv(
        convert(Vector, v),
        convert(Vector, Jv), 
        convert(Vector, u),
        unsafe_wrap(Array, new_u, (1, )),
        userfun.data
    )

    return Cint(retval)
end

"""
    kin_create(f, y0 [; kwargs...]) -> kin

Create and return a kinsol solver context `kin`, which can then be passed to [`kin_solve`](@ref)

# Arguments
- `f`: Function of form f(fy::Vector{Float64}, y::Vector{Float64}, userdata)
- `y0::Vector` template Vector of initial values (used only to define problem dimension)

# Keywords
- `userdata`: optional user data, passed through to `f` etc.
- `linear_solver`: linear solver to use (only partially implemented, supports :Dense, :Band, :FGMRES)
- `psolvefun`: optional preconditioner solver function (for :FGMRES)
- `psetupfun`: optional preconditioner setup function
- `jvfun`: optional Jacobian*vector  function (for :FGMRES)
"""
function kin_create(
    f, y0::Vector{Float64};
    userdata::Any = nothing,
    linear_solver = :Dense,
    jac_upper = 0,
    jac_lower = 0,
    krylov_dim = 0,
    psetupfun = nothing,
    psolvefun = nothing,
    jvfun = nothing,
)
    
    mem_ptr = Sundials.KINCreate()
    (mem_ptr == C_NULL) && error("Failed to allocate KINSOL solver object")
    kmem = Sundials.Handle(mem_ptr)

    handles = []

    # use the user_data field to pass a function
    #   see: https://github.com/JuliaLang/julia/issues/2554
    userfun = UserFunctionAndData(f, psetupfun, psolvefun, jvfun, userdata)
    push!(handles, userfun) # prevent userfun from being garbage collected (required for julia 1.8)
    function getkinsolfun(userfun::T) where {T}
        @cfunction(kinsolfun, Cint, (Sundials.N_Vector, Sundials.N_Vector, Ref{T}))
    end
    function getpsetupfun(userfun::T) where {T}
        @cfunction(kinprecsetup, Cint, (Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Ref{T}))
    end
    function getpsolvefun(userfun::T) where {T}
        @cfunction(kinprecsolve, Cint, (Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Ref{T}))
    end
    function getkinjactimesvec(userfun::T) where {T}
        @cfunction(kinjactimesvec, Cint, (Sundials.N_Vector, Sundials.N_Vector, Sundials.N_Vector, Ptr{Cint}, Ref{T}))
    end

    flag = Sundials.@checkflag Sundials.KINInit(kmem, getkinsolfun(userfun), Sundials.NVector(y0)) true

    if linear_solver == :Dense
        A = Sundials.SUNDenseMatrix(length(y0), length(y0))
        push!(handles, Sundials.MatrixHandle(A, Sundials.DenseMatrix()))
        LS = Sundials.SUNLinSol_Dense(y0, A)
        push!(handles, Sundials.LinSolHandle(LS, Sundials.Dense()))
    elseif linear_solver == :Band
        A = Sundials.SUNBandMatrix(length(y0), jac_upper, jac_lower)
        push!(handles, Sundials.MatrixHandle(A, Sundials.BandMatrix()))
        LS = Sundials.SUNLinSol_Band(y0, A)
        push!(handles, Sundials.LinSolHandle(LS, Sundials.Band()))
    elseif linear_solver == :FGMRES
        A = nothing
        prec_side = isnothing(psolvefun) ? 0 : 2 # right preconditioning only       
        LS = Sundials.SUNLinSol_SPFGMR(y0, prec_side, krylov_dim)
        push!(handles, Sundials.LinSolHandle(LS, Sundials.SPFGMR()))
    end

    flag = Sundials.@checkflag Sundials.KINSetUserData(kmem, userfun) true

    flag = Sundials.@checkflag Sundials.KINSetLinearSolver(kmem, LS, A === nothing ? C_NULL : A) true
    # flag = Sundials.@checkflag Sundials.KINDlsSetLinearSolver(kmem, LS, A === nothing ? C_NULL : A) true    

    if !isnothing(psolvefun)        
        flag = Sundials.@checkflag Sundials.KINSetPreconditioner(kmem, 
                                        psetupfun === nothing ? C_NULL : getpsetupfun(userfun), 
                                        getpsolvefun(userfun)) true        
    end

    if !isnothing(jvfun)
        flag = Sundials.@checkflag Sundials.KINSetJacTimesVecFn(kmem, getkinjactimesvec(userfun)) true
    end


    return (;kmem, handles)
end

"""
    kin_solve(
        kin, y0::Vector;
        [strategy] [, fnormtol] [, mxiter] [, print_level] [,y_scale] [, f_scale] [, noInitSetup]
    ) -> (y, kin_stats)

Solve nonlinear system using kinsol solver context `kin` (created by [`kin_create`](@ref)) and initial conditions `y0`.
Returns solution `y` and solver statistics `kinstats`. `kinstats.returnflag` indicates success/failure.
"""
function kin_solve(
    kin, y0::Vector{Float64};
    strategy = Sundials.KIN_NONE,
    fnormtol::Float64 = 0.0,
    mxiter = 200,
    print_level = 0,
    y_scale = ones(length(y0)),
    f_scale = ones(length(y0)),
    noInitSetup = false,
)
    
    y = copy(y0)

    kmem = kin.kmem

    flag = Sundials.@checkflag Sundials.KINSetPrintLevel(kmem, print_level) true
    
    flag = Sundials.@checkflag Sundials.KINSetFuncNormTol(kmem, fnormtol) true

    flag = Sundials.@checkflag Sundials.KINSetNumMaxIters(kmem, mxiter) true

    flag = Sundials.@checkflag Sundials.KINSetNoInitSetup(kmem, noInitSetup) true

    ## Solve problem    
    returnflag = Sundials.KINSol(kmem, y, strategy, y_scale, f_scale)

    ## Get stats
    nfevals = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumFuncEvals(kmem, nfevals)
    nniters = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumNonlinSolvIters(kmem, nniters)
    nbcfails = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumBetaCondFails(kmem, nbcfails)
    nbacktr = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumBacktrackOps(kmem, nbacktr)
    fnorm = [NaN]
    flag = Sundials.@checkflag Sundials.KINGetFuncNorm(kmem, fnorm)
    steplength = [NaN]
    flag = Sundials.@checkflag Sundials.KINGetStepLength(kmem, steplength)
    njevals = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumJacEvals(kmem, njevals)
    nfevalsLS = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumLinFuncEvals(kmem, nfevalsLS)
    nliters = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumLinIters(kmem, nliters)
    nlcfails = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumLinConvFails(kmem, nlcfails)
    npevals = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumPrecEvals(kmem, npevals)
    npsolves = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumPrecSolves(kmem, npsolves)
    njvevals = [0]
    flag = Sundials.@checkflag Sundials.KINGetNumJtimesEvals(kmem, njvevals)

    kinstats = (
        ReturnFlag             = returnflag,
        NumFuncEvals           = nfevals[],
        NumNonlinSolvIters     = nniters[],
        NumBetaCondFails       = nbcfails[],
        NumBacktrackOps        = nbacktr[],
        FuncNorm               = fnorm[],
        StepLength             = steplength[],
        NumJacEvals            = njevals[],
        NumLinFuncEvals        = nfevalsLS[],
        NumLinIters            = nliters[],
        NumLinConvFails        = nlcfails[],
        NumPrecEvals           = npevals[],
        NumPrecSolves          = npsolves[],
        NumJtimesEvals         = njvevals[],
    )

    return (y, kinstats)
end




end # module
