using CUDA.CUSOLVER: csrlsvqr!, csrlsvlu!
using LinearAlgebra: lu
using SparseArrays: sparse

"""
    solver_LU_CPU(globalfield)

CPU LU solver.
"""
function solver_LU_CPU(globalfield::GlobalField{ArrayType}) where {ArrayType}
    @Takeout (K_I, K_J, K_val_ids, K_total, residue) FROM globalfield
    
    println("CPU LU direct solver")
    K_vals = K_total[K_val_ids]

    cpu_K_I = FEM_convert(Array, K_I)
    cpu_K_J = FEM_convert(Array, K_J)
    cpu_K_total = FEM_convert(Array, K_vals)
    cpu_res = FEM_convert(Array, residue)

    cpu_K = sparse(cpu_K_I, cpu_K_J, cpu_K_total)
    cpu_x = lu(cpu_K) \ cpu_res # CPU direct solver is faster than csrlsvluHost on my machine
    FEM_convert(ArrayType, cpu_x)
end

"""
    solver_LU_CPU(globalfield; reorder::Integer = 1, singular_tol::Number) 

GPU LU solver with `CUDA`'s `csrlsvlu!`. Note, `csrlsvlu!` is essentially a host function and the device part is not accessible.
"""
function solver_LU_GPU(globalfield::GlobalField{ArrayType}; reorder::Integer = 1, singular_tol::Number) where {ArrayType}
    @Takeout (K_I, K_J, K_val_ids, K_total, residue) FROM globalfield
    
    println("GPU LU solver (csrlsvlu) with initial res = $(normalized_norm(b)) and target res = $converge_tol")
    K_vals = K_total[K_val_ids]

    cpu_K_I = FEM_convert(Array, K_I)
    cpu_K_J = FEM_convert(Array, K_J)
    cpu_K_total = FEM_convert(Array, K_vals)
    cpu_res = FEM_convert(Array, residue)

    cpu_K = sparse(cpu_K_I, cpu_K_J, cpu_K_total)

    cpu_x = zeros(FEM_Float, length(residue))
    csrlsvlu!(cpu_K, cpu_res, cpu_x, eltype(cpu_x)(singular_tol), Cint(reorder), 'O')

    FEM_convert(ArrayType, cpu_x)
end

"""
    solver_QR_GPU(globalfield; reorder::Integer = 1, singular_tol::Number)

GPU QR solver with `CUDA`'s `csrlsvqr!`.
"""
function solver_QR_GPU(globalfield::GlobalField{ArrayType}; reorder::Integer = 1, singular_tol::Number) where {ArrayType <: CuArray}
    @Takeout (K_I, K_J, K_val_ids, K_total, residue, converge_tol) FROM globalfield
    
    K_vals = K_total[K_val_ids]
    A = CuSparseMatrixCSR(CuSparseMatrixCOO{FEM_Float}(K_I, K_J, K_vals))
    b = residue
    x = FEM_buffer(ArrayType, FEM_Float, length(residue))

    println("GPU QR solver (csrlsvqr) with initial res = $(normalized_norm(b))and target res = $converge_tol")
    csrlsvqr!(A, b, x, eltype(x)(singular_tol), Cint(reorder), 'O')
    x
end

