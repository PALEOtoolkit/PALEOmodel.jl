
using Test
using Logging

@testset "All tests" begin

include("runfieldtests.jl")

include("runthreadingtests.jl")

include("runnonlinearnewtontests.jl")

include("runkinsoltests.jl")

include("runoutputwritertests.jl")

include("testsparseutils.jl")

end