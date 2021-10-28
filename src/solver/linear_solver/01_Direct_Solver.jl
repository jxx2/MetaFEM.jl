using CUDA.CUSOLVER: csrlsvqr!, csrlsvlu!

function solver_LU_CPU(globalfield) 
    @Takeout (K_I, K_J, K_val_ids, K_total, residue) FROM globalfield
    
    println("CPU LU direct solver")
    K_vals = K_total[K_val_ids]

    cpu_K_I = collect(K_I)
    cpu_K_J = collect(K_J)
    cpu_K_total = collect(K_vals)
    cpu_res = collect(residue)

    cpu_K = sparse(cpu_K_I, cpu_K_J, cpu_K_total)
    x_cpu = LinearAlgebra.lu(cpu_K) \ cpu_res # CPU direct solver is faster than csrlsvluHost on my machine
    cu(x_cpu)
end

function solver_LU(globalfield; reorder::Integer = 1, singular_tol::Number) 
    @Takeout (K_I, K_J, K_val_ids, K_total, residue) FROM globalfield
    
    println("GPU LU solver (csrlsvlu) with initial res = ", normalized_norm(b), " and target res = ", converge_tol)
    K_vals = K_total[K_val_ids]
    x = CUDA.zeros(FEM_Float, length(residue))

    cpu_K_I = collect(K_I)
    cpu_K_J = collect(K_J)
    cpu_K_total = collect(K_vals)
    cpu_res = collect(residue)

    cpu_K = sparse(cpu_K_I, cpu_K_J, cpu_K_total)

    cpu_x = zeros(FEM_Float, length(residue))
    csrlsvlu!(cpu_K, cpu_res, cpu_x, eltype(x)(singular_tol), Cint(reorder), 'O')
    cu(cpu_x)
end

function solver_QR(globalfield; reorder::Integer = 1, singular_tol::Number)
    @Takeout (K_I, K_J, K_val_ids, K_total, residue, converge_tol) FROM globalfield
    
    K_vals = K_total[K_val_ids]
    A = CuSparseMatrixCSR(CuSparseMatrixCOO{FEM_Float}(K_I, K_J, K_vals))
    b = residue
    x = CUDA.zeros(FEM_Float, length(residue))

    println("GPU QR solver (csrlsvqr) with initial res = ", normalized_norm(b), " and target res = ", converge_tol)
    csrlsvqr!(A, b, x, eltype(x)(singular_tol), Cint(reorder), 'O')
    x
end

