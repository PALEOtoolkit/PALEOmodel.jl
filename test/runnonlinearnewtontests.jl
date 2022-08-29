using Test
using Logging

import PALEOmodel
import StaticArrays
import ForwardDiff
import LinearAlgebra

@testset "NonLinearNewton" begin

    function ftest(u)
        # println("ftest u: $u")
        return u.*u .- 1
    end

    function jtest(u)
        return ForwardDiff.jacobian(ftest, u)
    end

    u0 = StaticArrays.SVector{10}(1.0:10.0)

    (u, Lnorm_2, Lnorm_inf, niter) = PALEOmodel.NonLinearNewton.solve(ftest, jtest, u0; reltol=1e-5, verbose=0)

    @test niter == 7
    @test LinearAlgebra.norm(ftest(u), Inf) < 1e-5
    @test LinearAlgebra.norm(u .- 1, Inf) < 1e-10
    
end