struct Identity end
precondition_Nothing(args...; kwargs...) = Identity()

LinearAlgebra.ldiv!(::Identity, b) = b
# Base.:*(::Identity, x) = x

struct CUDA_Jacobi
    jac_vec
end

function precondition_CUDA_Jacobi(A::CuSparseMatrixCSR{T}) where T
    J_ptr = A.rowPtr
    Js = A.colVal
    Ks = A.nzVal

    jac_vec = CUDA.ones(T, size(A, 2))
    CUDA.@sync @Dumb_CUDA_Batch 256 find_Jac_Preconditioner(J_ptr, Js, Ks, jac_vec)
    return CUDA_Jacobi(jac_vec)
end

@Dumb_Kernel find_Jac_Preconditioner(J_ptr, Js, Ks, jac_vec) begin
    j_start_pos = J_ptr[thread_idx]
    j_final_pos = J_ptr[thread_idx + 1] - 1
    for j_pos = j_start_pos:j_final_pos 
        if Js[j_pos] == thread_idx
            jac_vec[thread_idx] = abs(Ks[j_pos])
        end
    end
end

function LinearAlgebra.ldiv!(Pl::CUDA_Jacobi, b) 
    b ./= Pl.jac_vec
end

struct CUDA_ILU
    ilu_mat
end

precondition_CUDA_ILU(A) = CUDA_ILU(ilu02!(copy(A), 'O'))
function LinearAlgebra.ldiv!(Pl::CUDA_ILU, b) 
    T = eltype(Pl.ilu_mat)
    LinearAlgebra.ldiv!(UnitLowerTriangular(Pl.ilu_mat), b)
    LinearAlgebra.ldiv!(UpperTriangular(Pl.ilu_mat), b)
end 