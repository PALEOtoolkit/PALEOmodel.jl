import PALEOmodel
import PALEOboxes as PB

using SparseArrays
using LinearAlgebra  # for I

using Test
using Sundials


@testset "Kinsol" begin

    @testset "kinsol_prec" begin
        include("kinsol_prec.jl")
    end
    
    return nothing
end
  
