
using Test
using Logging

@testset "All tests" begin

include("runfieldtests.jl")

include("runthreadingtests.jl")

include("runkinsoltests.jl")

include("runoutputwritertests.jl")

end