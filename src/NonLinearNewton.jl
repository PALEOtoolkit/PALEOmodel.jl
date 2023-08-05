module NonLinearNewton

# import Infiltrator

import LinearAlgebra

"""
    solve(
        func, jac, u0::AbstractVector;
        reltol=1e-5,
        miniters=0,
        maxiters=100,
        verbose=0,
        jac_constant::Bool=false,
        u_min=-Inf,
    ) -> (u, Lnorm_2, Lnorm_inf, niter)

Minimal naive Newton solver for nonlinear function `func(u)` with jacobian `jac(u)`, starting at `u0`.


Stopping criteria is `norm(func(u), Inf) < reltol`.

Non-allocating if `u0` and hence `u` and `func(u)` are `StaticArrays.SVector`s.

Set `verbose` to 1, 2, 3, 4 for increasing levels of output.
"""
function solve(
    func::F,
    jac::J, 
    u0::AbstractVector;
    reltol=1e-5,
    miniters::Integer=0,
    maxiters::Integer=100,
    verbose::Integer=0,
    jac_constant::Bool=false,
    project_region = x->x,
) where {F, J}

    u = copy(u0)
    residual = func(u0)
    Lnorm_2 = LinearAlgebra.norm(residual, 2)
    Lnorm_inf = LinearAlgebra.norm(residual, Inf)
    iters = 0
    verbose >= 2 && @info "iters $iters Lnorm_2 $Lnorm_2 Lnorm_inf $Lnorm_inf u $u residual $residual"
    
    if (Lnorm_inf > reltol || iters < miniters)
        jacobianfac = jac(u)
        while (Lnorm_inf > reltol || iters < miniters) && iters < maxiters
            
            verbose >= 4 && @info "iters $iters jac:" jacobianfac
            u = u - jacobianfac \ residual
            u = project_region(u)
            iters += 1
            residual = func(u)
            Lnorm_2 = LinearAlgebra.norm(residual, 2)
            Lnorm_inf = LinearAlgebra.norm(residual, Inf)

            if !jac_constant && Lnorm_inf > reltol
                jacobianfac = jac(u)
            end
            
            verbose >= 3 && @info "iters $iters Lnorm_2 $Lnorm_2 Lnorm_inf $Lnorm_inf"
            verbose >= 4 && @info "    u $u residual $residual"
        end
    end

    verbose >= 2 && @info "iters $iters Lnorm_2 $Lnorm_2 Lnorm_inf $Lnorm_inf u $u residual $residual"

    iters < maxiters || throw(ErrorException("NonLinearNewton.solve: maxiters $maxiters reached"))

    return (u, Lnorm_2, Lnorm_inf, iters)
end

end