using Test
using Logging

import SparseArrays
import PALEOmodel

const SU = PALEOmodel.SparseUtils

@testset "SparseUtils" begin
    
@testset "add_sparse_fixed!" begin
    A = SparseArrays.sprand(10, 10, 1e-1)

    A_copy = copy(A)
    SU.add_sparse_fixed!(A_copy, A)
    @test A_copy == 2*A

    A_copy = copy(A)
    A_full = similar(A)
    fill!(A_full, 1.0)
    @test_throws ErrorException SU.add_sparse_fixed!(A_copy, A_full)

    A_copy = A[2:5, 3:5]
    SU.add_sparse_fixed!(A_copy, view(A, 2:5, 3:5))
    @test A_copy == 2*A[2:5, 3:5]
end

@testset "get_column_sparse!" begin
    A = SparseArrays.sprand(10, 10, 1e-1)

    x = ones(10)
    for j in 1:10
        num_nonzero = SU.get_column_sparse!(x, view(A, :, j))
        @test x == A[:, j]
        @test num_nonzero == count(!iszero, A[:, j])
    end

    x = ones(7)
    for j in 1:10
        num_nonzero = SU.get_column_sparse!(x, view(A, 2:8, j))
        @test x == A[2:8, j]
        @test num_nonzero == count(!iszero, A[2:8, j])
    end
end

@testset "add_column_sparse_fixed!" begin
    A = SparseArrays.sprand(10, 10, 1e-1)

    for j in 1:10
        x = Vector(A[:, j])
        A_copy = copy(A)
        SU.add_column_sparse_fixed!(view(A_copy, :, j), x)
        @test A_copy[:, j] == 2*A[:, j]
    end
    
end

@testset "mult_sparse_vec!" begin
    A = SparseArrays.sprand(10, 10, 1e-1)

    x = ones(6)
    y = ones(3)

    SU.mult_sparse_vec!(y, view(A, 5:7, 2:7), x)
    @test y == A[5:7, 2:7]*x

    SU.mult_sparse_vec!(y, (@view A[5:7, 2:7]), x)
    @test y == A[5:7, 2:7]*x

end

end