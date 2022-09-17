"""
    SparseUtils

Contains convenience functions and optimised special cases for sparse matrices.
"""
module SparseUtils

import SparseArrays
import LinearAlgebra
import Infiltrator

"""
    add_sparse_fixed!(A::SparseMatrixCSC, B::SparseMatrixCSC)
    add_sparse_fixed!(A::SparseMatrixCSC, B::SparseMatrixCSC[iA, jA])

Sparse matrix `A .+= B`, without modifying sparsity pattern of `A`.

Errors if `B` contains elements not in sparsity pattern of `A`

    julia> A = SparseArrays.sparse([1, 4], [2, 3], [1.0, 1.0], 4, 4)
    julia> B = SparseArrays.sparse([1], [2], [2.0], 4, 4)
    julia> SparseUtils.add_sparse_fixed!(A, B) # OK
    julia> C = SparseArrays.sparse([2], [2], [2.0], 4, 4)
    julia> SparseUtils.add_sparse_fixed!(A, C) # errors

`B` can be a `view`, with the restriction that the first index subset `iA` must be a `UnitRange`, eg
`B[2:3, 3:4]` or `B[2:3, [3, 4]]` is OK, `B[[2, 3], 3:4]` is not supported.
"""
function add_sparse_fixed!(A::SparseArrays.SparseMatrixCSC, B::SparseArrays.SparseMatrixCSC)
    size(A) == size(B) || throw(DimensionMismatch("A and B are not the same size"))

    @inbounds for j in 1:size(B, 2)
        idx_A = A.colptr[j]
        for idx_B in B.colptr[j]:(B.colptr[j+1]-1)
            i_B = B.rowval[idx_B]
            while idx_A < A.colptr[j+1] && A.rowval[idx_A] != i_B
                idx_A += 1
            end
            idx_A < A.colptr[j+1] || error("element [$i_B, $j] in B is not present in A")
            A.nzval[idx_A] += B.nzval[idx_B]
        end
    end
    return A
end

function add_sparse_fixed!(A::SparseArrays.SparseMatrixCSC, B::SubArray{T, 2, <:SparseArrays.SparseMatrixCSC, <:Tuple{UnitRange, I2}, false}) where {T, I2}
    return add_sparse_fixed!(A, B.parent, B.indices[1], B.indices[2])
end

# A .+= B[iB, jB]
function add_sparse_fixed!(A::SparseArrays.SparseMatrixCSC, B::SparseArrays.SparseMatrixCSC, iB::UnitRange, jB)
    size(A) == (length(iB), length(jB)) || throw(DimensionMismatch("A and B[iB, jB] are not the same size"))

    @inbounds for (j_A, j_B) in zip(1:size(A, 2), jB)
        idx_A = A.colptr[j_A]
        for idx_B in B.colptr[j_B]:(B.colptr[j_B+1]-1)
            i_B = B.rowval[idx_B]
            if i_B in iB
                while idx_A < A.colptr[j_A+1] && iB[A.rowval[idx_A]] != i_B
                    idx_A += 1
                end
                idx_A < A.colptr[j_A+1] || error("element B[$i_B, $j_B] in B is not present in A")
                A.nzval[idx_A] += B.nzval[idx_B]
            end
        end
    end
    return A
end

"""
    get_column_sparse!(x::AbstractVector, A::SparseMatrixCSC[iA, jA::Int64]) -> num_nonzero::Int

Get a Vector `x` from a column of sparse matrix `A`

Calculates x .= A[iA, jA::Int] where `iA` is a range of indices and `jA` specifies column.

Returns the number of non-zero elements in x
"""
function get_column_sparse!(x::AbstractVector, A::SubArray{T, 1, <:SparseArrays.SparseMatrixCSC, <:Tuple{I1, Int}, false}) where {T, I1}
    return get_column_sparse!(x, parent(A), A.indices[1], A.indices[2])
end

function get_column_sparse!(x::AbstractVector, A::SparseArrays.SparseMatrixCSC, iA, jA::Int64)
    length(x) == length(iA) || throw(DimensionMismatch("x and iA are not the same length"))
    x .= 0.0

    n_nonzero = 0
    i = 1
    @inbounds for idx_A in A.colptr[jA]:(A.colptr[jA+1]-1)
        i_A = A.rowval[idx_A]
        if i_A < first(iA)
            continue
        end
        if i_A > last(iA)
            break
        end
        i = _find_i(i, iA, i_A)
        if i_A == iA[i]
            x[i] = A.nzval[idx_A]
            n_nonzero += 1
        end
    end
    return n_nonzero
end

# find ivals[i] == iv, starting at i
# returns i which may or may not be correct
@inline function _find_i(i::Integer, ivals, iv)
    @inbounds while iv > ivals[i] && i < length(ivals)
        i += 1
    end
    return i
end

function check_all_zero(x::AbstractVector, ifrom, ito)
    num_zeros = 0
    @inbounds for i in ifrom:ito
        num_zeros += !iszero(x[i])
    end
    num_zeros == 0 || error("x[$ifrom:$ito] != 0")
    return nothing
end

"""
    add_column_sparse_fixed!(A::SparseMatrixCSC[:, jA], y)

Add a column y to A[:, jA], checking sparsity pattern
"""
function add_column_sparse_fixed!(A::SubArray{T, 1, <:SparseArrays.SparseMatrixCSC, <:Tuple{Base.Slice, Int}, false}, y::AbstractVector) where {T}
    return add_column_sparse_fixed!(parent(A), A.indices[2], y)
end

function add_column_sparse_fixed!(A::SparseArrays.SparseMatrixCSC, jA::Int64, y::AbstractVector)
    size(A, 1) == length(y) || throw(DimensionMismatch("size(A, 1) != length(y)"))

    last_i = 1
    @inbounds for idx_A in A.colptr[jA]:(A.colptr[jA+1]-1)
        i = A.rowval[idx_A]
        check_all_zero(y, last_i+1, i-1)
        A.nzval[idx_A] += y[i]
        last_i = i
    end
    check_all_zero(y, last_i+1, length(y))

    # idx_A = A.colptr[jA]
    # i_A = A.rowval[idx_A]
    # @inbounds for (i_y, y) in zip(eachindex(y), y)
    #     if !iszero(y)
    #         while i_A != i_y && idx_A < A.colptr[jA+1]
    #             idx_A +=1
    #             i_A = A.rowval[idx_A]
    #         end
    #         i_A == i_y || error("element y[$i_y] in y is not present in A[:, $jA]")
    #         A.nzval[idx_A] += y
    #     end    
    # end
    
    return A
end

"""
    mult_sparse_vec!(y::AbstractVector, A::SparseMatrixCSC[iA, jA], x::AbstractVector)

Calculate `y .= A[iA, jA] * x`, where `iA` and `jA` are index ranges defining a part of `A`
"""
function mult_sparse_vec!(y::AbstractVector, A::SubArray{T, 2, <:SparseArrays.SparseMatrixCSC, I, false}, x::AbstractVector) where {T, I}
    return mult_sparse_vec!(y, parent(A), A.indices[1], A.indices[2], x)
end

function mult_sparse_vec!(y::AbstractVector, A::SparseArrays.SparseMatrixCSC, iA, jA, x::AbstractVector)
    length(iA) == length(y) || throw(DimensionMismatch("length(iA) != length(y)"))
    length(jA) == length(x) || throw(DimensionMismatch("length(jA) != length(x)"))

    y .= 0.0

    @inbounds for (j, j_A) in enumerate(jA)
        i = 1
        for idx_A in A.colptr[j_A]:(A.colptr[j_A+1]-1)
            i_A = A.rowval[idx_A]
            i = _find_i(i, iA, i_A)
            if i_A == iA[i]
                y[i] += A.nzval[idx_A]*x[j]
            end
            
        end
    end
    return nothing
end



"""
    get_sparse_inverse(A::AbstractMatrix) -> A_inv::SparseArrays.SparseMatrixCSC

Horrible hack to get a sparse inverse via a dense inverse (very slow, allocates a huge dense matrix)

Intended for use when calculating sparsity patterns only.
Workaround for missing functions to calculate A \\ x for sparse x ie preserving sparsity.
"""
function get_sparse_inverse(A::AbstractMatrix)

    # find dense inverse
    A_inv_dense = LinearAlgebra.inv(Matrix(A))
    # reconstruct a sparse matrix
    ij = Tuple.(findall(!iszero, A_inv_dense))
    I = [i for (i,j) in ij]
    J = [j for (i,j) in ij]
    V = [A_inv_dense[i, j] for (i, j) in ij]
    A_inv = SparseArrays.sparse(I, J, V)

    return A_inv
end


"fill structural non-zeros: return sparse matrix with all stored elements filled with `val`"
function fill_sparse_jac(initial_jac; val=1.0, fill_diagonal=true)
    num_elements = size(initial_jac, 1)*size(initial_jac, 2)

    @info "  fill_sparse_jac: initial_jac nnz $(SparseArrays.nnz(initial_jac)) "*
        "($(round(100*SparseArrays.nnz(initial_jac)/num_elements, sigdigits=3))%) non-zero $(count(!iszero, initial_jac))"
    I, J, _ = SparseArrays.findnz(initial_jac)

    jac_filled = SparseArrays.sparse(I, J, val, size(initial_jac, 1), size(initial_jac, 2) )
    if fill_diagonal
        size(initial_jac, 1) == size(initial_jac, 2) ||
            error("fill_sparse_jac: fill_diagonal == true and Jacobian is not square")
        for i = 1:size(initial_jac, 1)
            jac_filled[i, i] = val
        end
        @info "  fill_sparse_jac: after filling diagonal nnz $(SparseArrays.nnz(jac_filled))"     
    end

    return jac_filled
end

end